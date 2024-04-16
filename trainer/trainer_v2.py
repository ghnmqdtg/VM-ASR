import torch
import numpy as np
from tqdm import tqdm
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from utils.output_logger import log_audio, log_waveform, log_spectrogram
from data_loader import preprocessing, postprocessing
from tqdm.contrib.logging import logging_redirect_tqdm


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
        amp=False,
        gan=False,
        update_mode="v1",  # "v1" or "v2"
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.config_dataloader = self.config["data_loader"]["args"]
        self.device = device
        # Check if "mse_loss" or "mae_loss" is in criterion, if one of them is, set it to be the loss function
        if "mse_loss" in criterion:
            self.loss_for_chunk = criterion["mse_loss"]
        elif "mae_loss" in criterion:
            self.loss_for_chunk = criterion["mae_loss"]
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.epoch_log = {}  # Log for metrics each epoch (both training and validation)
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.amp = amp  # Automatic Mixed Precision
        self.gan = gan  # Generative Adversarial Network
        self.update_mode = update_mode
        self.init_gan() if gan else None
        self.init_metrics()
        self._runner = self.init_runner(
            self.config_dataloader["stft_enabled"],
            self.config_dataloader["chunking_enabled"],
        )
        # Show baisc information
        # Show model summary
        if (
            self.config_dataloader["stft_enabled"]
            and self.config_dataloader["chunking_enabled"]
        ):
            self.logger.info(model.flops())

        self.logger.info(
            f'Training with scale: {self.config_dataloader["scale"]}, random LPS: {self.config_dataloader["random_lpf"]}, AMP: {self.amp}, GAN: {self.gan}\n'
        )

    def init_gan(self):
        # Import the multi period discriminator and the cosine annealing warmup restarts
        from model.model import MultiPeriodDiscriminator
        from utils.scheduler import CosineAnnealingWarmupRestarts

        # Initialize the discriminator
        self.MPD = MultiPeriodDiscriminator().to(self.device)
        # Initialize the optimizer for the discriminator
        # Get trainables parameters of MPD
        self.trainable_params_D = filter(
            lambda p: p.requires_grad, self.MPD.parameters()
        )
        self.optimizer_D = torch.optim.Adam(self.trainable_params_D, lr=0.0001)
        # Initialize the learning rate scheduler
        if self.config["lr_scheduler"]["type"] == "CosineAnnealingWarmupRestarts":
            self.lr_scheduler_D = CosineAnnealingWarmupRestarts(
                self.optimizer_D,
                **self.config["lr_scheduler"]["args"],
            )
        else:
            self.lr_scheduler_D = self.config.init_obj(
                "lr_scheduler", torch.optim.lr_scheduler, self.optimizer_D
            )
        # Print the number of parameters and FLOPs of the MPD
        self.logger.info(self.MPD.flops(shape=(1, self.config_dataloader["length"])))

    def init_metrics(self):
        # Initialize the metrics
        self.metrics = [
            "total_loss",
            "global_loss",
            "global_mag_loss",
            "global_phase_loss",
            "local_loss",
            "local_mag_loss",
            "local_phase_loss",
        ]
        # Add the discriminator and generator losses if training with GAN
        if self.gan:
            self.metrics += ["loss_G", "loss_D", "loss_F"]
        # Initialize the metric trackers
        self.train_metrics = MetricTracker(
            *self.metrics,
            *[m.__name__ for m in self.metric_ftns],
            writer=self.writer,
        )
        self.valid_metrics = MetricTracker(
            *self.metrics,
            *[m.__name__ for m in self.metric_ftns],
            writer=self.writer,
        )
        self.logger.info("Metrics: {}".format(self.metrics))

    def init_runner(self, stft_enabled, chunking_enabled):
        # Define the runners for each combination of stft_enabled and chunking_enabled
        runner = {
            (True, True): dict(
                v1=self.process_stft_chunks, v2=self.process_stft_chunks_v2
            ).get(self.update_mode, self.process_stft_chunks),
            (True, False): NotImplementedError,
            (False, True): NotImplementedError,
            (False, False): NotImplementedError,
        }.get((stft_enabled, chunking_enabled), self.process_stft_chunks)

        if runner == NotImplementedError:
            raise NotImplementedError(
                f"Runner for stft_enabled={stft_enabled} and chunking_enabled={chunking_enabled} is not implemented yet."
            )

        self.logger.info(f"Using {runner.__name__} as runner.")

        return runner

    def _train_epoch(self, epoch):
        # Set the model to training mode
        self.model.train()
        # Reset the train metrics
        self.train_metrics.reset()
        # Grad scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None
        # Set the progress bar for the epoch
        with logging_redirect_tqdm(loggers=[self.logger], tqdm_class=tqdm):
            with tqdm(
                self.data_loader,
                desc=f"Epoch {epoch} [TRAIN] {self._progress(0, training=True)}",
                unit="batch",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                postfix={
                    "total_loss": 0.0,
                    "global_loss": 0.0,
                    "local_loss": 0.0,
                    "mem": 0,
                },
            ) as tepoch:
                for batch_idx, (data, target, padding_length) in enumerate(tepoch):
                    # Set description for the progress bar
                    tepoch.set_description(
                        f"Epoch {epoch} [TRAIN] {self._progress(batch_idx, training=True)}"
                    )
                    # Reset the peak memory stats for the GPU
                    torch.cuda.reset_peak_memory_stats()
                    # Update step for the tensorboard
                    self.writer.set_step(epoch)
                    # Set non_blocking to True for faster data transfer
                    data, target, padding_length = (
                        data.to(self.device, non_blocking=True),
                        target.to(self.device, non_blocking=True),
                        padding_length.to(self.device, non_blocking=True),
                    )
                    # Forward pass
                    (
                        output_wave,
                        target_wave,
                        outputs_mag,
                        outputs_phase,
                        metrics_values,
                    ) = self._runner(
                        data, target, padding_length, batch_idx, training=True
                    )

                    # Update the progress bar
                    self.update_progress_bar(tepoch, metrics_values)

                    # Log the output waveforms and spectrograms
                    if batch_idx == (self.len_epoch - 1):
                        self.log_outputs(
                            output_wave=output_wave,
                            target_wave=target_wave,
                            input_mag=data[0, 0, ...].unsqueeze(0),
                            input_phase=data[0, 1, ...].unsqueeze(0),
                            output_mag=outputs_mag,
                            output_phase=outputs_phase,
                            target_mag=target[0, 0, ...].unsqueeze(0),
                            target_phase=target[0, 1, ...].unsqueeze(0),
                        )
                        # Set description for the progress bar after the last batch
                        tepoch.set_description(
                            f"Epoch {epoch} [TRAIN] {self._progress(-1, training=True)}"
                        )

                # Save the log of the epoch into dict
                self.epoch_log = self.train_metrics.result()
                # Step the learning rate scheduler after each epoch
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    # Log the learning rate to the tensorboard
                    self.writer.add_scalar(
                        "learning_rate", self.lr_scheduler.get_last_lr()[0]
                    )
                # Log the progress bar to info.log after the epoch
                self.logger.info(tepoch)

    def _valid_epoch(self, epoch):
        # Set the model to evaluation mode
        self.model.eval()
        # Reset the valid metrics
        self.valid_metrics.reset()
        # Set the progress bar for the epoch
        with logging_redirect_tqdm(loggers=[self.logger], tqdm_class=tqdm):
            with tqdm(
                self.valid_data_loader,
                desc=f"Epoch {epoch} [VALID] {self._progress(0, training=False)}",
                unit="batch",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                postfix={
                    "total_loss": 0.0,
                    "global_loss": 0.0,
                    "local_loss": 0.0,
                    "mem": 0,
                },
            ) as tepoch:
                with torch.no_grad():
                    for batch_idx, (data, target, padding_length) in enumerate(tepoch):
                        # Set description for the progress bar
                        tepoch.set_description(
                            f"Epoch {epoch} [VALID] {self._progress(batch_idx, training=False)}"
                        )
                        # Reset the peak memory stats for the GPU
                        torch.cuda.reset_peak_memory_stats()
                        # Update step for the tensorboard
                        self.writer.set_step(epoch, "valid")
                        # Set non_blocking to True for faster data transfer
                        data, target, padding_length = (
                            data.to(self.device, non_blocking=True),
                            target.to(self.device, non_blocking=True),
                            padding_length.to(self.device, non_blocking=True),
                        )
                        # Forward pass
                        (
                            output_wave,
                            target_wave,
                            outputs_mag,
                            outputs_phase,
                            metrics_values,
                        ) = self._runner(
                            data, target, padding_length, batch_idx, training=False
                        )

                        # Update the progress bar
                        self.update_progress_bar(tepoch, metrics_values)

                        # Log the output waveforms and spectrograms
                        if batch_idx == (len(self.valid_data_loader) - 1):
                            self.log_outputs(
                                output_wave=output_wave,
                                target_wave=target_wave,
                                input_mag=data[0, 0, ...].unsqueeze(0),
                                input_phase=data[0, 1, ...].unsqueeze(0),
                                output_mag=outputs_mag,
                                output_phase=outputs_phase,
                                target_mag=target[0, 0, ...].unsqueeze(0),
                                target_phase=target[0, 1, ...].unsqueeze(0),
                            )
                            # Set description for the progress bar after the last batch
                            tepoch.set_description(
                                f"Epoch {epoch} [VALID] {self._progress(-1, training=False)}"
                            )

                    # Add histogram of model parameters to the tensorboard
                    for name, p in self.model.named_parameters():
                        self.writer.add_histogram(name, p, bins="auto")

                    val_log = self.valid_metrics.result()

                    self.epoch_log.update(**{"val_" + k: v for k, v in val_log.items()})
                    # Log the progress bar to info.log after the epoch
                    self.logger.info(tepoch)

    def process_stft_chunks(
        self, data, target, padding_length, batch_idx, training=True
    ):
        """
        Update the model weights once after processing all the chunks.
        """
        # Get the number of chunks
        num_chunks = data.size(2)
        chunk_losses, chunk_outputs = {"mag": [], "phase": []}, {"mag": [], "phase": []}
        if training:
            # Zero the gradients for each chunk
            self.optimizer.zero_grad()
        # Process each chunk
        with torch.autocast(device_type="cuda", enabled=self.amp):
            for chunk_idx in range(num_chunks):
                # Get current chunk data (and unsqueeze the chunk dimension)
                chunk_data = data[:, :, chunk_idx, :, :].unsqueeze(2)
                # Get current chunk target (and unsqueeze the chunk dimension)
                chunk_target = target[:, :, chunk_idx, :, :].unsqueeze(2)
                # Forward pass
                chunk_mag, chunk_phase = self.model(chunk_data)
                # Accumulate the chunk loss
                chunk_losses["mag"].append(
                    self.loss_for_chunk(chunk_mag, chunk_target[:, 0, ...])
                )
                chunk_losses["phase"].append(
                    self.loss_for_chunk(chunk_phase, chunk_target[:, 1, ...])
                )
                # Store the chunk output
                chunk_outputs["mag"].append(chunk_mag)
                chunk_outputs["phase"].append(chunk_phase)

                # Concatenate the chunk outputs if it is the last chunk
                if chunk_idx == num_chunks - 1:
                    chunk_outputs["mag"] = torch.stack(chunk_outputs["mag"], dim=2)
                    chunk_outputs["phase"] = torch.stack(chunk_outputs["phase"], dim=2)

        # Reconstruct the waveform from the concatenated output and target
        output_wave = postprocessing.reconstruct_from_stft_chunks(
            mag=chunk_outputs["mag"],
            phase=chunk_outputs["phase"],
            batch_input=True,
            crop=True,
            config_dataloader=self.config_dataloader,
        )
        target_wave = postprocessing.reconstruct_from_stft_chunks(
            mag=target[:, 0, ...].unsqueeze(1),
            phase=target[:, 1, ...].unsqueeze(1),
            batch_input=True,
            crop=True,
            config_dataloader=self.config_dataloader,
        )
        # Compute the STFT of the output and target waveforms
        output_mag, output_phase = preprocessing.get_mag_phase(
            output_wave,
            chunk_wave=False,
            batch_input=True,
            stft_params=self.config_dataloader["stft_params"],
        )
        target_mag, target_phase = preprocessing.get_mag_phase(
            target_wave,
            chunk_wave=False,
            batch_input=True,
            stft_params=self.config_dataloader["stft_params"],
        )
        # Calculate the total loss
        local_mag_loss = torch.stack(chunk_losses["mag"]).mean()
        local_phase_loss = torch.stack(chunk_losses["phase"]).mean()
        # Calculate the global loss
        global_mag_loss = self.loss_for_chunk(output_mag, target_mag)
        global_phase_loss = self.loss_for_chunk(output_phase, target_phase)
        # Calculate the loss
        local_loss = local_mag_loss + local_phase_loss
        global_loss = global_mag_loss + global_phase_loss
        total_loss = 0.3 * global_loss + 0.7 * local_loss

        # Backward pass
        if training:
            if self.amp:
                # Scale the loss
                self.scaler.scale(total_loss).backward()
                # Update the model
                self.scaler.step(self.optimizer)
                # Update the scaler
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()

        # Create a dictionary with the losses
        metrics_values = {
            "total_loss": total_loss.item(),
            "global_loss": global_loss.item(),
            "global_mag_loss": global_mag_loss.item(),
            "global_phase_loss": global_phase_loss.item(),
            "local_loss": local_loss.item(),
            "local_mag_loss": local_mag_loss.item(),
            "local_phase_loss": local_phase_loss.item(),
        }
        for met in self.metric_ftns:
            metrics_values[met.__name__] = met(output_mag, target_mag)

        # Update the metrics
        self.update_metrics(metrics_values, training=training)

        return (
            # Return the first sample of the batch
            output_wave[0],
            target_wave[0],
            chunk_outputs["mag"][0],
            chunk_outputs["phase"][0],
            metrics_values,
        )

    def process_stft_chunks_v2(
        self, data, target, padding_length, batch_idx, training=True
    ):
        """
        Update the model weights after processing each chunk.
        """
        # Get the number of chunks
        num_chunks = data.size(2)
        # Initialize the losses and outputs
        chunk_outputs = {"mag": [], "phase": []}
        # Process each chunk
        for chunk_idx in range(num_chunks):
            if training:
                # Zero the gradients for each chunk
                self.optimizer.zero_grad()
            # Enable autocast for mixed precision training
            with torch.autocast(device_type="cuda", enabled=self.amp):
                # Get current chunk data (and unsqueeze the chunk dimension)
                chunk_data = data[:, :, chunk_idx, :, :].unsqueeze(2)
                # Get current chunk target (and unsqueeze the chunk dimension)
                chunk_target = target[:, :, chunk_idx, :, :].unsqueeze(2)
                # Forward pass
                chunk_mag, chunk_phase = self.model(chunk_data)

            # Store the chunk output
            chunk_outputs["mag"].append(chunk_mag)
            chunk_outputs["phase"].append(chunk_phase)
            # Accumulate the chunk loss
            local_mag_loss = self.loss_for_chunk(chunk_mag, chunk_target[:, 0, ...])
            local_phase_loss = self.loss_for_chunk(chunk_phase, chunk_target[:, 1, ...])
            local_loss = local_mag_loss + local_phase_loss

            # Backward pass
            if training:
                if self.amp:
                    # Scale the loss
                    self.scaler.scale(local_loss).backward()
                    # Update the model
                    self.scaler.step(self.optimizer)
                    # Update the scaler
                    self.scaler.update()
                else:
                    local_loss.backward()
                    self.optimizer.step()

            if chunk_idx == num_chunks - 1:
                # Concatenate the chunk outputs
                chunk_outputs["mag"] = torch.stack(chunk_outputs["mag"], dim=2)
                chunk_outputs["phase"] = torch.stack(chunk_outputs["phase"], dim=2)

        # Reconstruct the waveform from the concatenated output and target
        output_wave = postprocessing.reconstruct_from_stft_chunks(
            mag=chunk_outputs["mag"],
            phase=chunk_outputs["phase"],
            batch_input=True,
            crop=True,
            config_dataloader=self.config_dataloader,
        )
        target_wave = postprocessing.reconstruct_from_stft_chunks(
            mag=target[:, 0, ...].unsqueeze(1),
            phase=target[:, 1, ...].unsqueeze(1),
            batch_input=True,
            crop=True,
            config_dataloader=self.config_dataloader,
        )
        # Compute the STFT of the output and target waveforms
        output_mag, output_phase = preprocessing.get_mag_phase(
            output_wave,
            chunk_wave=False,
            batch_input=True,
            stft_params=self.config_dataloader["stft_params"],
        )
        target_mag, target_phase = preprocessing.get_mag_phase(
            target_wave,
            chunk_wave=False,
            batch_input=True,
            stft_params=self.config_dataloader["stft_params"],
        )

        # Calculate the global loss
        # We only log them to compare with the previous implementation
        # We don't use them for training
        global_mag_loss = self.loss_for_chunk(output_mag, target_mag)
        global_phase_loss = self.loss_for_chunk(output_phase, target_phase)
        global_loss = global_mag_loss + global_phase_loss
        # Calculate the total loss
        total_loss = 0.3 * global_loss + 0.7 * local_loss

        # Create a dictionary with the losses
        metrics_values = {
            "total_loss": total_loss.item(),
            "global_loss": global_loss.item(),
            "global_mag_loss": global_mag_loss.item(),
            "global_phase_loss": global_phase_loss.item(),
            "local_loss": local_loss.item(),
            "local_mag_loss": local_mag_loss.item(),
            "local_phase_loss": local_phase_loss.item(),
        }
        for met in self.metric_ftns:
            metrics_values[met.__name__] = met(output_mag, target_mag)

        # Update the metrics
        self.update_metrics(metrics_values, training=training)

        return (
            # Return the first sample of the batch
            output_wave[0],
            target_wave[0],
            chunk_outputs["mag"][0],
            chunk_outputs["phase"][0],
            metrics_values,
        )

    def update_discriminator(self):
        raise NotImplementedError

    def update_metrics(self, metrics_values, training=True):
        # Update the batch metrics
        for key, value in metrics_values.items():
            if training:
                self.train_metrics.update(key, value)
            else:
                self.valid_metrics.update(key, value)

    def log_outputs(
        self,
        output_wave,
        target_wave,
        input_mag,
        input_phase,
        output_mag,
        output_phase,
        target_mag,
        target_phase,
    ):
        # Get the input waveform and stft of the output and target waveforms
        input_waveform = postprocessing.reconstruct_from_stft_chunks(
            mag=input_mag,
            phase=input_phase,
            batch_input=False,
            crop=True,
            config_dataloader=self.config_dataloader,
        )
        # Name the audio files
        name_list = ["input", "output", "target"]
        # Store the waveforms
        waveforms = [input_waveform, output_wave, target_wave]
        # Add audio to the tensorboard
        log_audio(self.writer, name_list, waveforms, self.config["source_sr"])
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
                [
                    input_mag,
                    input_phase,
                ],
                [output_mag, output_phase],
                [
                    target_mag,
                    target_phase,
                ],
            ],
            stft=True,
            chunks=True,
        )

    @staticmethod
    def update_progress_bar(tepoch, metrics_values):
        # Update the progress bar
        # Filter out the necessary metrics to display in the progress bar
        progress_metrics = {
            k: v
            for k, v in metrics_values.items()
            if k
            in ["total_loss", "global_loss", "local_loss", "loss_G", "loss_D", "loss_F"]
        }
        # Add the memory usage to the progress bar
        progress_metrics["mem"] = (
            f"{torch.cuda.max_memory_allocated() / (1024.0 * 1024.0):.0f} MB"
        )
        tepoch.set_postfix(progress_metrics)

    def _progress(self, batch_idx, training=True):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            if training:
                total = self.data_loader.split_sample_num["train"]
            else:
                total = self.data_loader.split_sample_num["valid"]
        else:
            total = self.len_epoch

        if batch_idx == -1:
            current = total
        else:
            current = batch_idx * self.data_loader.batch_size

        return base.format(current, total, 100.0 * current / total)
