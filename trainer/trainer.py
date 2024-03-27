import torch
import numpy as np
from tqdm import tqdm
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from utils.output_logger import log_audio, log_waveform, log_spectrogram
from data_loader import preprocessing, postprocessing

# TODO: Set the discriminator in the config file and init in train.py
from model.model import MultiPeriodDiscriminator


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
        self.config_dataloader = self.config["data_loader"]["args"]
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

        self.MPD = MultiPeriodDiscriminator().to(self.device)
        # Get trainables parameters of MPD
        self.trainable_params_D = filter(
            lambda p: p.requires_grad, self.MPD.parameters()
        )
        self.optimizer_D = torch.optim.Adam(self.trainable_params_D, lr=0.0001)
        # Print the number of trainable parameters of the MPD
        print(
            f"Trainable parameters of MPD: {sum(p.numel() for p in self.MPD.parameters() if p.requires_grad)}"
        )
        self.lr_scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_D, step_size=25, gamma=0.1
        )

        self.train_metrics = MetricTracker(
            "total_loss",
            "global_mag_loss",
            "global_phase_loss",
            "local_loss",
            "local_mag_loss",
            "local_phase_loss",
            "generator_loss",
            "discriminator_loss",
            "feature_loss",
            *[m.__name__ for m in self.metric_ftns],
            writer=self.writer,
        )
        self.valid_metrics = MetricTracker(
            "total_loss",
            "global_mag_loss",
            "global_phase_loss",
            "local_loss",
            "local_mag_loss",
            "local_phase_loss",
            "generator_loss",
            "discriminator_loss",
            "feature_loss",
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
        self.MPD.train()
        # Reset the train metrics
        self.train_metrics.reset()
        # Grad scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler()
        scaler_D = torch.cuda.amp.GradScaler()
        # Set the progress bar for the epoch
        with tqdm(
            self.data_loader,
            desc=f"Epoch {epoch} {self._progress(0)}",
            unit="batch",
        ) as tepoch:
            # Save losses and metrics for this epoch
            epoch_log = {
                "losses": {
                    "total_loss": [],
                    "global_mag_loss": [],
                    "global_phase_loss": [],
                    "local_loss": [],
                    "local_mag_loss": [],
                    "local_phase_loss": [],
                    "generator_loss": [],
                    "discriminator_loss": [],
                    "feature_loss": [],
                },
                "metrics": {m.__name__: [] for m in self.metric_ftns},
            }
            # Iterate through the batches
            for batch_idx, (data, target) in enumerate(tepoch):
                # Reset the peak memory stats for the GPU
                torch.cuda.reset_peak_memory_stats()
                # Set description for the progress bar
                tepoch.set_description(f"Epoch {epoch} {self._progress(batch_idx)}")
                # Both data and target are in the shape of (batch_size, 2 (mag and phase), num_chunks, frequency_bins, frames)
                data, target = data.to(self.device, non_blocking=True), target.to(
                    self.device, non_blocking=True
                )

                # Enables autocasting for the forward pass (model + loss)
                with torch.autocast(device_type="cuda"):
                    # Initialize the chunk ouptut and losses
                    chunk_inputs = {"mag": [], "phase": []}
                    chunk_outputs = {"mag": [], "phase": []}
                    total_loss = 0
                    chunk_losses = {"mag": [], "phase": []}

                    # Iterate through the chunks and calculate the loss for each chunk
                    for chunk_idx in range(data.size(2)):
                        # Get current chunk data (and unsqueeze the chunk dimension)
                        chunk_data = data[:, :, chunk_idx, :, :].unsqueeze(2)
                        # Get current chunk target (and unsqueeze the chunk dimension)
                        chunk_target = target[:, :, chunk_idx, :, :].unsqueeze(2)

                        # Save the chunk input if batch_idx == (self.len_epoch - 1) (the last batch of the epoch)
                        if batch_idx == (self.len_epoch - 1):
                            chunk_inputs["mag"].append(chunk_data[:, 0, ...])
                            chunk_inputs["phase"].append(chunk_data[:, 1, ...])

                        # Forward pass
                        chunk_mag, chunk_phase = self.model(chunk_data)
                        # Calculate the chunk loss
                        chunk_mag_loss = self.criterion["mse_loss"](
                            chunk_mag, chunk_target[:, 0, ...]
                        )
                        chunk_phase_loss = self.criterion["mse_loss"](
                            chunk_phase, chunk_target[:, 1, ...]
                        )
                        # print(f'chunk_mag.shape: {chunk_mag.shape}, chunk_target[:, 0, ...].shape: {chunk_target[:, 0, ...].shape}')
                        # Accumulate the chunk loss
                        chunk_losses["mag"].append(chunk_mag_loss)
                        chunk_losses["phase"].append(chunk_phase_loss)
                        # Store the chunk output
                        chunk_outputs["mag"].append(chunk_mag)
                        chunk_outputs["phase"].append(chunk_phase)

                    # Concatenate the outputs along the chunk dimension
                    chunk_outputs["mag"] = torch.cat(
                        chunk_outputs["mag"], dim=1
                    ).unsqueeze(1)
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
                        config_dataloader=self.config_dataloader,
                    )
                    target_waveform = postprocessing.reconstruct_from_stft_chunks(
                        mag=target[:, 0, ...].unsqueeze(1),
                        phase=target[:, 1, ...].unsqueeze(1),
                        batch_input=True,
                        crop=True,
                        config_dataloader=self.config_dataloader,
                    )

                    # print(f'output_waveform.shape: {output_waveform.shape}, target_waveform.shape: {target_waveform.shape}')
                    # Compute the STFT of the output and target waveforms
                    output_mag, output_phase = preprocessing.get_mag_phase(
                        output_waveform,
                        chunk_wave=False,
                        batch_input=True,
                        stft_params=self.config_dataloader["stft_params"],
                    )
                    target_mag, target_phase = preprocessing.get_mag_phase(
                        target_waveform,
                        chunk_wave=False,
                        batch_input=True,
                        stft_params=self.config_dataloader["stft_params"],
                    )

                    # Detach for discriminator (prevent graph modifications)
                    target_waveform = target_waveform.detach()
                    output_waveform = output_waveform.detach()

                    # Discriminator training
                    self.optimizer_D.zero_grad()

                    # Get the discriminator output
                    y_real, y_gen, _, _ = self.MPD(target_waveform, output_waveform)
                    # Calculate the discriminator loss
                    loss_MPD, _, _ = self.criterion["discriminator_loss"](y_real, y_gen)

                # Backward pass
                scaler_D.scale(loss_MPD).backward()
                # Update the weights
                scaler_D.step(self.optimizer_D)
                # Update the scaler for the next iteration
                scaler_D.update()

                self.optimizer.zero_grad()
                with torch.autocast(device_type="cuda"):
                    # Get the discriminator output
                    y_real, y_gen, feature_map_real, feature_map_gen = self.MPD(
                        target_waveform, output_waveform
                    )
                    # Calculate the feature loss
                    loss_feature = self.criterion["feature_loss"](
                        feature_map_real, feature_map_gen
                    )
                    # Calculate the generator loss
                    loss_G, _ = self.criterion["generator_loss"](y_gen)
                    # Calculate the mag and phase loss
                    local_mag_loss = torch.stack(chunk_losses["mag"]).mean()
                    local_phase_loss = torch.stack(chunk_losses["phase"]).mean()
                    global_mag_loss = self.criterion["mse_loss"](output_mag, target_mag)
                    global_phase_loss = self.criterion["mse_loss"](
                        output_phase, target_phase
                    )
                    # Get global loss and local loss
                    global_loss = global_mag_loss + global_phase_loss
                    local_loss = local_mag_loss + local_phase_loss
                    # Calculate total loss
                    total_loss = global_loss + local_loss + loss_feature + loss_G

                # Backward pass
                scaler.scale(total_loss).backward()
                # Update the weights
                scaler.step(self.optimizer)
                # Update the scaler for the next iteration
                scaler.update()

                # Save losses and metrics for this batch
                epoch_log["losses"]["total_loss"].append(total_loss.item())
                epoch_log["losses"]["global_mag_loss"].append(global_mag_loss.item())
                epoch_log["losses"]["global_phase_loss"].append(
                    global_phase_loss.item()
                )
                epoch_log["losses"]["local_loss"].append(local_loss.item())
                epoch_log["losses"]["local_mag_loss"].append(local_mag_loss.item())
                epoch_log["losses"]["local_phase_loss"].append(local_phase_loss.item())

                epoch_log["losses"]["generator_loss"].append(loss_G.item())
                epoch_log["losses"]["discriminator_loss"].append(loss_MPD.item())
                epoch_log["losses"]["feature_loss"].append(loss_feature.item())

                for met in self.metric_ftns:
                    epoch_log["metrics"][met.__name__].append(
                        met(output_mag, target_mag)
                    )

                # Update the progress bar
                tepoch.set_postfix(
                    {
                        "total_loss": total_loss.item(),
                        "local_loss": local_loss.item(),
                        "loss_G": loss_G.item(),
                        "loss_D": loss_MPD.item(),
                        "feat_loss": loss_feature.item(),
                        "mem": torch.cuda.max_memory_allocated() / (1024.0 * 1024.0),
                    }
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
                        config_dataloader=self.config_dataloader,
                    )
                    # Concatenate the inputs along the chunk dimension
                    chunk_inputs["mag"] = torch.cat(
                        chunk_inputs["mag"], dim=1
                    ).unsqueeze(1)
                    chunk_inputs["phase"] = torch.cat(
                        chunk_inputs["phase"], dim=1
                    ).unsqueeze(1)
                    # Name the audio files
                    name_list = ["input", "output", "target"]
                    # Store the waveforms
                    waveforms = [input_waveform, output_waveform[0], target_waveform[0]]
                    # Add audio to the tensorboard
                    log_audio(
                        self.writer, name_list, waveforms, self.config["source_sr"]
                    )
                    # Add the waveforms to the tensorboard
                    log_waveform(self.writer, name_list, waveforms)
                    # Add the spectrograms to the tensorboard
                    log_spectrogram(
                        self.writer,
                        name_list,
                        waveforms,
                        stft=False,
                        sample_rate=self.config["source_sr"],
                    )
                    # Add the STFT spectrograms to the tensorboard
                    log_spectrogram(
                        self.writer,
                        name_list,
                        waveforms,
                        stft=True,
                        sample_rate=self.config["source_sr"],
                    )
                    # Add the STFT spectrograms of chunks to the tensorboard
                    log_spectrogram(
                        self.writer,
                        name_list,
                        [
                            [chunk_inputs["mag"][0], chunk_inputs["phase"][0]],
                            [chunk_outputs["mag"][0], chunk_outputs["phase"][0]],
                            [
                                target[:, 0, ...].unsqueeze(1)[0],
                                target[:, 1, ...].unsqueeze(1)[0],
                            ],
                        ],
                        stft=True,
                        chunks=True,
                    )

            # Update the epoch loss and metrics from epoch_log
            self.train_metrics.update(
                "total_loss", np.mean(epoch_log["losses"]["total_loss"])
            )
            self.train_metrics.update(
                "global_mag_loss", np.mean(epoch_log["losses"]["global_mag_loss"])
            )
            self.train_metrics.update(
                "global_phase_loss", np.mean(epoch_log["losses"]["global_phase_loss"])
            )
            self.train_metrics.update(
                "local_loss", np.mean(epoch_log["losses"]["local_loss"])
            )
            self.train_metrics.update(
                "local_mag_loss", np.mean(epoch_log["losses"]["local_mag_loss"])
            )
            self.train_metrics.update(
                "local_phase_loss", np.mean(epoch_log["losses"]["local_phase_loss"])
            )
            self.train_metrics.update(
                "generator_loss", np.mean(epoch_log["losses"]["generator_loss"])
            )
            self.train_metrics.update(
                "discriminator_loss", np.mean(epoch_log["losses"]["discriminator_loss"])
            )
            self.train_metrics.update(
                "feature_loss", np.mean(epoch_log["losses"]["feature_loss"])
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
                # Log the learning rate of the discriminator to the tensorboard
                self.lr_scheduler_D.step()
                self.writer.add_scalar(
                    "learning_rate_D", self.lr_scheduler_D.get_last_lr()[0]
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
        epoch_log = {
            "losses": {
                "total_loss": [],
                "global_mag_loss": [],
                "global_phase_loss": [],
                "local_loss": [],
                "local_mag_loss": [],
                "local_phase_loss": [],
                "generator_loss": [],
                "discriminator_loss": [],
                "feature_loss": [],
            },
            "metrics": {m.__name__: [] for m in self.metric_ftns},
        }
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device, non_blocking=True), target.to(
                    self.device, non_blocking=True
                )

                # Enables autocasting for the forward pass (model + loss)
                with torch.autocast(device_type="cuda"):
                    # Initialize the chunk ouptut and losses
                    chunk_inputs = {"mag": [], "phase": []}
                    chunk_outputs = {"mag": [], "phase": []}
                    total_loss = 0
                    chunk_losses = {"mag": [], "phase": []}

                    # Iterate through the chunks and calculate the loss for each chunk
                    for chunk_idx in range(data.size(2)):
                        # Get current chunk data (and unsqueeze the chunk dimension)
                        chunk_data = data[:, :, chunk_idx, :, :].unsqueeze(2)
                        # Get current chunk target (and unsqueeze the chunk dimension)
                        chunk_target = target[:, :, chunk_idx, :, :].unsqueeze(2)

                        # Save the chunk input if batch_idx == (self.len_epoch - 1) (the last batch of the epoch)
                        if batch_idx == (len(self.valid_data_loader) - 1):
                            chunk_inputs["mag"].append(chunk_data[:, 0, ...])
                            chunk_inputs["phase"].append(chunk_data[:, 1, ...])

                        # Forward pass
                        chunk_mag, chunk_phase = self.model(chunk_data)
                        # Calculate the chunk loss
                        chunk_mag_loss = self.criterion["mse_loss"](
                            chunk_mag, chunk_target[:, 0, ...]
                        )
                        chunk_phase_loss = self.criterion["mse_loss"](
                            chunk_phase, chunk_target[:, 1, ...]
                        )
                        # Accumulate the chunk loss
                        chunk_losses["mag"].append(chunk_mag_loss)
                        chunk_losses["phase"].append(chunk_phase_loss)
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
                    config_dataloader=self.config_dataloader,
                )
                target_waveform = postprocessing.reconstruct_from_stft_chunks(
                    mag=target[:, 0, ...].unsqueeze(1),
                    phase=target[:, 1, ...].unsqueeze(1),
                    batch_input=True,
                    crop=True,
                    config_dataloader=self.config_dataloader,
                )
                # Compute the STFT of the output and target waveforms
                output_mag, output_phase = preprocessing.get_mag_phase(
                    output_waveform,
                    chunk_wave=False,
                    batch_input=True,
                    stft_params=self.config_dataloader["stft_params"],
                )
                target_mag, target_phase = preprocessing.get_mag_phase(
                    target_waveform,
                    chunk_wave=False,
                    batch_input=True,
                    stft_params=self.config_dataloader["stft_params"],
                )

                with torch.autocast(device_type="cuda"):
                    # Get the discriminator output
                    y_real, y_gen, feature_map_real, feature_map_gen = self.MPD(
                        target_waveform, output_waveform
                    )
                    # Calculate the discriminator loss
                    loss_MPD, _, _ = self.criterion["discriminator_loss"](y_real, y_gen)
                    # Calculate the feature loss
                    loss_feature = self.criterion["feature_loss"](
                        feature_map_real, feature_map_gen
                    )
                    # Calculate the generator loss
                    loss_G, _ = self.criterion["generator_loss"](y_gen)
                    # Calculate the mag and phase loss
                    local_mag_loss = torch.stack(chunk_losses["mag"]).mean()
                    local_phase_loss = torch.stack(chunk_losses["phase"]).mean()
                    global_mag_loss = self.criterion["mse_loss"](output_mag, target_mag)
                    global_phase_loss = self.criterion["mse_loss"](
                        output_phase, target_phase
                    )
                    # Get global loss and local loss
                    global_loss = global_mag_loss + global_phase_loss
                    local_loss = local_mag_loss + local_phase_loss
                    # Calculate total loss
                    total_loss = global_loss + local_loss + loss_feature + loss_G

                # Save valid losses and metrics for this batch
                epoch_log["losses"]["total_loss"].append(total_loss.item())
                epoch_log["losses"]["global_mag_loss"].append(global_mag_loss.item())
                epoch_log["losses"]["global_phase_loss"].append(
                    global_phase_loss.item()
                )
                epoch_log["losses"]["local_loss"].append(local_loss.item())
                epoch_log["losses"]["local_mag_loss"].append(local_mag_loss.item())
                epoch_log["losses"]["local_phase_loss"].append(local_phase_loss.item())

                epoch_log["losses"]["generator_loss"].append(loss_G.item())
                epoch_log["losses"]["discriminator_loss"].append(loss_MPD.item())
                epoch_log["losses"]["feature_loss"].append(loss_feature.item())

                for met in self.metric_ftns:
                    epoch_log["metrics"][met.__name__].append(
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
                        config_dataloader=self.config_dataloader,
                    )
                    # Concatenate the inputs along the chunk dimension
                    chunk_inputs["mag"] = torch.cat(
                        chunk_inputs["mag"], dim=1
                    ).unsqueeze(1)
                    chunk_inputs["phase"] = torch.cat(
                        chunk_inputs["phase"], dim=1
                    ).unsqueeze(1)
                    # Name the audio files
                    name_list = ["input", "output", "target"]
                    # Store the waveforms
                    waveforms = [input_waveform, output_waveform[0], target_waveform[0]]
                    # Add audio to the tensorboard
                    log_audio(
                        self.writer, name_list, waveforms, self.config["source_sr"]
                    )
                    # Add the waveforms to the tensorboard
                    log_waveform(self.writer, name_list, waveforms)
                    # Add the spectrograms to the tensorboard
                    log_spectrogram(
                        self.writer,
                        name_list,
                        waveforms,
                        stft=False,
                        sample_rate=self.config["source_sr"],
                    )
                    # Add the STFT spectrograms to the tensorboard
                    log_spectrogram(
                        self.writer,
                        name_list,
                        waveforms,
                        stft=True,
                        sample_rate=self.config["source_sr"],
                    )
                    # Add the STFT spectrograms of chunks to the tensorboard
                    log_spectrogram(
                        self.writer,
                        name_list,
                        [
                            [chunk_inputs["mag"][0], chunk_inputs["phase"][0]],
                            [chunk_outputs["mag"][0], chunk_outputs["phase"][0]],
                            [
                                target[:, 0, ...].unsqueeze(1)[0],
                                target[:, 1, ...].unsqueeze(1)[0],
                            ],
                        ],
                        stft=True,
                        chunks=True,
                    )

        # Update the mean valid loss and metrics from epoch_log
        self.valid_metrics.update(
            "total_loss", np.mean(epoch_log["losses"]["total_loss"])
        )
        self.valid_metrics.update(
            "global_mag_loss", np.mean(epoch_log["losses"]["global_mag_loss"])
        )
        self.valid_metrics.update(
            "global_phase_loss", np.mean(epoch_log["losses"]["global_phase_loss"])
        )
        self.valid_metrics.update(
            "local_loss", np.mean(epoch_log["losses"]["local_loss"])
        )
        self.valid_metrics.update(
            "local_mag_loss", np.mean(epoch_log["losses"]["local_mag_loss"])
        )
        self.valid_metrics.update(
            "local_phase_loss", np.mean(epoch_log["losses"]["local_phase_loss"])
        )
        self.valid_metrics.update(
            "generator_loss", np.mean(epoch_log["losses"]["generator_loss"])
        )
        self.valid_metrics.update(
            "discriminator_loss", np.mean(epoch_log["losses"]["discriminator_loss"])
        )
        self.valid_metrics.update(
            "feature_loss", np.mean(epoch_log["losses"]["feature_loss"])
        )

        for met in self.metric_ftns:
            self.valid_metrics.update(
                met.__name__,
                np.mean(epoch_log["metrics"][met.__name__]),
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
