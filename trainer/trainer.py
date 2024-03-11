import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker, log_audio, log_waveform, log_spectrogram
from tqdm import tqdm
from data_loader import preprocessing, postprocessing


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = self.len_epoch

        self.train_metrics = MetricTracker(
            "loss",
            "mag_loss",
            "phase_loss",
            *[m.__name__ for m in self.metric_ftns],
            writer=self.writer,
        )
        self.valid_metrics = MetricTracker(
            "loss",
            "mag_loss",
            "phase_loss",
            *[m.__name__ for m in self.metric_ftns],
            writer=self.writer,
        )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # Set the model to training mode
        self.model.train()
        # Reset the train metrics
        self.train_metrics.reset()
        # Set the progress bar for the epoch
        with tqdm(
            self.data_loader,
            desc=f"Epoch {epoch} {self._progress(0)}",
            unit="batch",
        ) as tepoch:
            # Save losses and metrics for this epoch
            epoch_log = {
                "losses": {"total_loss": [], "mag_loss": [], "phase_loss": []},
                "metrics": {m.__name__: [] for m in self.metric_ftns},
            }
            # Iterate through the batches
            for batch_idx, (data, target) in enumerate(tepoch):
                # Set description for the progress bar
                tepoch.set_description(f"Epoch {epoch} {self._progress(batch_idx)}")
                # Both data and target are in the shape of (batch_size, 2 (mag and phase), num_chunks, frequency_bins, frames)
                data, target = data.to(self.device), target.to(self.device)
                # Set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes
                self.optimizer.zero_grad()
                # Initialize the chunk ouptut and losses
                chunk_outputs = {"mag": [], "phase": []}
                total_loss = 0
                chunk_losses = []

                # Iterate through the chunks and calculate the loss for each chunk
                for chunk_idx in range(data.size(2)):
                    # Get current chunk data (and unsqueeze the chunk dimension)
                    chunk_data = data[:, :, chunk_idx, :, :].unsqueeze(2)
                    # Get current chunk target (and unsqueeze the chunk dimension)
                    chunk_target = target[:, :, chunk_idx, :, :].unsqueeze(2)
                    # Forward pass
                    chunk_mag, chunk_phase = self.model(chunk_data)
                    # Calculate the chunk loss
                    chunk_mag_loss = self.criterion(chunk_mag, chunk_target[:, 0, ...])
                    chunk_phase_loss = self.criterion(
                        chunk_phase, chunk_target[:, 1, ...]
                    )
                    # print(f'chunk_mag.shape: {chunk_mag.shape}, chunk_target[:, 0, ...].shape: {chunk_target[:, 0, ...].shape}')
                    # Accumulate the chunk loss
                    chunk_losses.append(chunk_mag_loss + chunk_phase_loss)
                    # Store the chunk output
                    chunk_outputs["mag"].append(chunk_mag)
                    chunk_outputs["phase"].append(chunk_phase)

                # Concatenate the outputs along the chunk dimension
                chunk_outputs["mag"] = torch.cat(chunk_outputs["mag"], dim=1).unsqueeze(
                    1
                )
                chunk_outputs["phase"] = torch.cat(
                    chunk_outputs["phase"], dim=1
                ).unsqueeze(1)
                # print(f'chunk_outputs["mag"].shape: {chunk_outputs["mag"].shape}, chunk_outputs["phase"].shape: {chunk_outputs["phase"].shape}')
                # Reconstruct the waveform from the concatenated output and target
                output_waveform = postprocessing.reconstruct_from_stft_chunks(
                    mag=chunk_outputs["mag"],
                    phase=chunk_outputs["phase"],
                    batch_input=True,
                    crop=True,
                )
                target_waveform = postprocessing.reconstruct_from_stft_chunks(
                    mag=target[:, 0, ...].unsqueeze(1),
                    phase=target[:, 1, ...].unsqueeze(1),
                    batch_input=True,
                    crop=True,
                )
                # print(f'output_waveform.shape: {output_waveform.shape}, target_waveform.shape: {target_waveform.shape}')
                # Compute the STFT of the output and target waveforms
                output_mag, output_phase = preprocessing.get_mag_phase(
                    output_waveform, chunk_wave=False, batch_input=True
                )
                target_mag, target_phase = preprocessing.get_mag_phase(
                    target_waveform, chunk_wave=False, batch_input=True
                )

                # Calculate the mag and phase loss
                mag_loss = self.criterion(output_mag, target_mag)
                phase_loss = self.criterion(output_phase, target_phase)
                # Calculate total loss
                total_loss = torch.stack(chunk_losses).mean() + mag_loss + phase_loss
                # Backward pass
                total_loss.backward()
                # Update the weights
                self.optimizer.step()

                # Save losses and metrics for this batch
                epoch_log["losses"]["total_loss"].append(total_loss.item())
                epoch_log["losses"]["mag_loss"].append(mag_loss.item())
                epoch_log["losses"]["phase_loss"].append(phase_loss.item())
                for met in self.metric_ftns:
                    epoch_log["metrics"][met.__name__].append(
                        met(output_mag, target_mag)
                    )

                # Update the progress bar
                tepoch.set_postfix(
                    loss=total_loss.item(),
                    mag_loss=mag_loss.item(),
                    phase_loss=phase_loss.item(),
                )

                # Log mean losses and metrics of this epoch to the tensorboard
                if batch_idx == (self.len_epoch - 1):
                    # Update step for the tensorboard
                    self.writer.set_step(epoch)
                    # Get the input waveform and stft of the output and target waveforms
                    input_waveform = postprocessing.reconstruct_from_stft_chunks(
                        mag=data[0, 0, ...].unsqueeze(0),
                        phase=data[0, 1, ...].unsqueeze(0),
                        batch_input=False,
                        crop=True,
                    )
                    # Name the audio files
                    name_list = ["input", "output", "target"]
                    # Store the waveforms
                    waveforms = [input_waveform, output_waveform[0], target_waveform[0]]
                    # Add audio to the tensorboard
                    log_audio(self.writer, name_list, waveforms, 48000)
                    # Add the waveforms to the tensorboard
                    log_waveform(self.writer, name_list, waveforms)
                    # Add the spectrograms to the tensorboard
                    log_spectrogram(self.writer, name_list, waveforms, stft=False)
                    # Add the STFT spectrograms to the tensorboard
                    log_spectrogram(self.writer, name_list, waveforms, stft=True)

            # Update the epoch loss and metrics from epoch_log
            self.train_metrics.update(
                "loss", np.mean(epoch_log["losses"]["total_loss"])
            )
            self.train_metrics.update(
                "mag_loss", np.mean(epoch_log["losses"]["mag_loss"])
            )
            self.train_metrics.update(
                "phase_loss", np.mean(epoch_log["losses"]["phase_loss"])
            )
            for met in self.metric_ftns:
                self.train_metrics.update(
                    met.__name__, np.mean(epoch_log["metrics"][met.__name__])
                )

            # Add histogram of model parameters to the tensorboard
            log = self.train_metrics.result()
            # Validate after each epoch
            if self.do_validation:
                val_log = self._valid_epoch(epoch)
                log.update(**{"val_" + k: v for k, v in val_log.items()})
            # Step the learning rate scheduler after each epoch
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                # Log the learning rate to the tensorboard
                self.writer.add_scalar(
                    "learning_rate", self.lr_scheduler.get_last_lr()[0]
                )

        # Return the log
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        # Save valid losses and metrics for this epoch
        epoch_log_valid = {
            "losses": {"total_loss": [], "mag_loss": [], "phase_loss": []},
            "metrics": {m.__name__: [] for m in self.metric_ftns},
        }
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Initialize the chunk ouptut and losses
                chunk_outputs = {"mag": [], "phase": []}
                total_loss = 0
                chunk_losses = []

                # Iterate through the chunks and calculate the loss for each chunk
                for chunk_idx in range(data.size(2)):
                    # Get current chunk data (and unsqueeze the chunk dimension)
                    chunk_data = data[:, :, chunk_idx, :, :].unsqueeze(2)
                    # Get current chunk target (and unsqueeze the chunk dimension)
                    chunk_target = target[:, :, chunk_idx, :, :].unsqueeze(2)
                    # Forward pass
                    chunk_mag, chunk_phase = self.model(chunk_data)
                    # Calculate the chunk loss
                    chunk_mag_loss = self.criterion(chunk_mag, chunk_target[:, 0, ...])
                    chunk_phase_loss = self.criterion(
                        chunk_phase, chunk_target[:, 1, ...]
                    )
                    # Accumulate the chunk loss
                    chunk_losses.append(chunk_mag_loss + chunk_phase_loss)
                    # Store the chunk output
                    chunk_outputs["mag"].append(chunk_mag)
                    chunk_outputs["phase"].append(chunk_phase)

                # Concatenate the outputs along the chunk dimension
                chunk_outputs["mag"] = torch.cat(chunk_outputs["mag"], dim=1).unsqueeze(
                    1
                )
                chunk_outputs["phase"] = torch.cat(
                    chunk_outputs["phase"], dim=1
                ).unsqueeze(1)
                # Reconstruct the waveform from the concatenated output and target
                output_waveform = postprocessing.reconstruct_from_stft_chunks(
                    mag=chunk_outputs["mag"],
                    phase=chunk_outputs["phase"],
                    batch_input=True,
                    crop=True,
                )
                target_waveform = postprocessing.reconstruct_from_stft_chunks(
                    mag=target[:, 0, ...].unsqueeze(1),
                    phase=target[:, 1, ...].unsqueeze(1),
                    batch_input=True,
                    crop=True,
                )
                # Compute the STFT of the output and target waveforms
                output_mag, output_phase = preprocessing.get_mag_phase(
                    output_waveform, chunk_wave=False, batch_input=True
                )
                target_mag, target_phase = preprocessing.get_mag_phase(
                    target_waveform, chunk_wave=False, batch_input=True
                )
                # Calculate the mag and phase loss
                mag_loss = self.criterion(output_mag, target_mag)
                phase_loss = self.criterion(output_phase, target_phase)
                # Calculate total loss
                total_loss = torch.stack(chunk_losses).mean() + mag_loss + phase_loss

                # Save valid losses and metrics for this batch
                epoch_log_valid["losses"]["total_loss"].append(total_loss.item())
                epoch_log_valid["losses"]["mag_loss"].append(mag_loss.item())
                epoch_log_valid["losses"]["phase_loss"].append(phase_loss.item())
                for met in self.metric_ftns:
                    epoch_log_valid["metrics"][met.__name__].append(
                        met(output_mag, target_mag)
                    )

                # Log mean losses and metrics of this epoch to the tensorboard
                if batch_idx == (len(self.valid_data_loader) - 1):
                    # Update step for the tensorboard
                    self.writer.set_step(epoch, "valid")
                    # Get the input waveform and stft of the output and target waveforms
                    input_waveform = postprocessing.reconstruct_from_stft_chunks(
                        mag=data[0, 0, ...].unsqueeze(0),
                        phase=data[0, 1, ...].unsqueeze(0),
                        batch_input=False,
                        crop=True,
                    )
                    # Name the audio files
                    name_list = ["input", "output", "target"]
                    # Store the waveforms
                    waveforms = [input_waveform, output_waveform[0], target_waveform[0]]
                    # Add audio to the tensorboard
                    log_audio(self.writer, name_list, waveforms, 48000)
                    # Add the waveforms to the tensorboard
                    log_waveform(self.writer, name_list, waveforms)
                    # Add the spectrograms to the tensorboard
                    log_spectrogram(self.writer, name_list, waveforms, stft=False)
                    # Add the STFT spectrograms to the tensorboard
                    log_spectrogram(self.writer, name_list, waveforms, stft=True)

        # Update the mean valid loss and metrics from epoch_log
        self.valid_metrics.update(
            "loss", np.mean(epoch_log_valid["losses"]["total_loss"])
        )
        self.valid_metrics.update(
            "mag_loss", np.mean(epoch_log_valid["losses"]["mag_loss"])
        )
        self.valid_metrics.update(
            "phase_loss", np.mean(epoch_log_valid["losses"]["phase_loss"])
        )
        for met in self.metric_ftns:
            self.valid_metrics.update(
                met.__name__,
                np.mean(epoch_log_valid["metrics"][met.__name__]),
            )

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
