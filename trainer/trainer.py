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
        amp=False,
        gan=False,
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
        self.epoch_log = {}
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.amp = amp
        self.gan = gan
        self.log_step = self.len_epoch

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

        if self.gan:
            # Initialize the discriminator
            self.MPD = MultiPeriodDiscriminator().to(self.device)
            # Print the number of parameters and FLOPs of the MPD
            self.logger.info(
                self.MPD.flops(shape=(1, self.config_dataloader["length"]))
            )
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
            # Add the discriminator and generator losses if training with GAN
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

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # Set the model to training mode
        self.model.train()
        if self.gan:
            self.MPD.train()
        # Reset the train metrics
        self.train_metrics.reset()
        # Grad scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler()
        scaler_D = torch.cuda.amp.GradScaler()

        # Set the progress bar for the epoch
        with tqdm(
            self.data_loader,
            desc=f"Epoch {epoch} [TRAIN] {self._progress(0, training=True)}",
            unit="batch",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ) as tepoch:
            # Update the progress bar
            tepoch.set_postfix(
                {
                    "total_loss": 0.0,
                    "global_loss": 0.0,
                    "local_loss": 0.0,
                    "loss_G": 0.0,
                    "loss_D": 0.0,
                    "loss_F": 0.0,
                    "mem": 0,
                }
            )
            # Iterate through the batches
            for batch_idx, (data, target) in enumerate(tepoch):
                # Reset the peak memory stats for the GPU
                torch.cuda.reset_peak_memory_stats()
                # Both data and target are in the shape of (batch_size, 2 (mag and phase), num_chunks, frequency_bins, frames)
                data, target = data.to(self.device, non_blocking=True), target.to(
                    self.device, non_blocking=True
                )

                # Initialize the chunk ouptut and losses
                total_loss = 0
                chunk_inputs = {"mag": [], "phase": []}
                chunk_outputs = {"mag": [], "phase": []}
                chunk_losses = {"mag": [], "phase": []}

                # Enables autocasting for the forward pass (model + loss)
                with torch.autocast(device_type="cuda", enabled=self.amp):
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

                if self.gan:
                    # Discriminator training
                    # Detach for discriminator (prevent graph modifications)
                    target_waveform = target_waveform.detach()
                    output_waveform = output_waveform.detach()
                    # Zero the gradients of the discriminator
                    self.optimizer_D.zero_grad()
                    # Set amp enabled for the discriminator
                    with torch.autocast(device_type="cuda", enabled=self.amp):
                        # Get the discriminator output
                        y_real, y_gen, feature_map_real, feature_map_gen = self.MPD(
                            target_waveform, output_waveform
                        )

                    # Calculate the discriminator loss
                    loss_D, _, _ = self.criterion["discriminator_loss"](y_real, y_gen)

                    # Backward pass
                    scaler_D.scale(loss_D).backward()
                    # Update the weights
                    scaler_D.step(self.optimizer_D)
                    # Update the scaler for the next iteration
                    scaler_D.update()

                self.optimizer.zero_grad()
                if self.gan:
                    # Set amp enabled for the generator
                    with torch.autocast(device_type="cuda", enabled=self.amp):
                        # Get the discriminator output
                        y_real, y_gen, feature_map_real, feature_map_gen = self.MPD(
                            target_waveform, output_waveform
                        )

                    # Calculate the generator loss
                    loss_G, _ = self.criterion["generator_loss"](y_gen)
                    # Calculate the feature loss
                    loss_F = self.criterion["feature_loss"](
                        feature_map_real, feature_map_gen
                    )

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
                if self.gan:
                    total_loss = 0.3 * global_loss + 0.7 * local_loss + loss_F + loss_G
                else:
                    total_loss = 0.3 * global_loss + 0.7 * local_loss

                # Backward pass
                scaler.scale(total_loss).backward()
                # Update the weights
                scaler.step(self.optimizer)
                # Update the scaler for the next iteration
                scaler.update()

                # Delete the variables to free memory
                del chunk_data, chunk_target, chunk_mag, chunk_phase, chunk_losses

                # Update step for the tensorboard
                self.writer.set_step(epoch)

                # Create a dictionary with the losses
                loss_values = {
                    "total_loss": total_loss.item(),
                    "global_loss": global_loss.item(),
                    "global_mag_loss": global_mag_loss.item(),
                    "global_phase_loss": global_phase_loss.item(),
                    "local_loss": local_loss.item(),
                    "local_mag_loss": local_mag_loss.item(),
                    "local_phase_loss": local_phase_loss.item(),
                }

                # Conditionally add GAN-related metrics
                if self.gan:
                    loss_values.update(
                        {
                            "loss_G": loss_G.item(),
                            "loss_D": loss_D.item(),
                            "loss_F": loss_F.item(),
                        }
                    )

                # Update the batch metrics
                for key, value in loss_values.items():
                    self.train_metrics.update(key, value)

                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(output_mag, target_mag))

                # Update the progress bar
                # Filter out the necessary metrics to display in the progress bar
                progress_metrics = {
                    k: v
                    for k, v in loss_values.items()
                    if k in ["total_loss", "local_loss", "loss_G", "loss_D", "loss_F"]
                }
                # Add the memory usage to the progress bar
                progress_metrics["mem"] = (
                    f"{torch.cuda.max_memory_allocated() / (1024.0 * 1024.0):.0f} MB"
                )
                tepoch.set_postfix(progress_metrics)

                # Log mean losses and metrics of this epoch to the tensorboard
                if batch_idx == (self.len_epoch - 1):
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

                    # Set description for the progress bar after the last batch
                    tepoch.set_description(
                        f"Epoch {epoch} [TRAIN] {self._progress(-1, training=True)}"
                    )
                else:
                    # Set description for the progress bar
                    tepoch.set_description(
                        f"Epoch {epoch} [TRAIN] {self._progress(batch_idx, training=True)}"
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
                if self.gan:
                    # Log the learning rate of the discriminator to the tensorboard
                    self.lr_scheduler_D.step()
                    self.writer.add_scalar(
                        "learning_rate_D", self.lr_scheduler_D.get_last_lr()[0]
                    )

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        # Set the progress bar for the epoch
        with tqdm(
            self.valid_data_loader,
            desc=f"Epoch {epoch} [VALID] {self._progress(0, training=False)}",
            unit="batch",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ) as tepoch:
            # Update the progress bar
            tepoch.set_postfix(
                {
                    "total_loss": 0.0,
                    "global_loss": 0.0,
                    "local_loss": 0.0,
                    "loss_G": 0.0,
                    "loss_D": 0.0,
                    "loss_F": 0.0,
                    "mem": 0,
                }
            )

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(tepoch):
                    # Reset the peak memory stats for the GPU
                    torch.cuda.reset_peak_memory_stats()
                    data, target = data.to(self.device, non_blocking=True), target.to(
                        self.device, non_blocking=True
                    )

                    # Initialize the chunk ouptut and losses
                    total_loss = 0
                    chunk_inputs = {"mag": [], "phase": []}
                    chunk_outputs = {"mag": [], "phase": []}
                    chunk_losses = {"mag": [], "phase": []}

                    # Enables autocasting for the forward pass (model + loss)
                    with torch.autocast(device_type="cuda", enabled=self.amp):
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
                    chunk_outputs["mag"] = torch.cat(
                        chunk_outputs["mag"], dim=1
                    ).unsqueeze(1)
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

                    if self.gan:
                        # Set amp enabled for the generator
                        with torch.autocast(device_type="cuda", enabled=self.amp):
                            # Get the discriminator output
                            y_real, y_gen, feature_map_real, feature_map_gen = self.MPD(
                                target_waveform, output_waveform
                            )
                        # Calculate the discriminator loss
                        loss_D, _, _ = self.criterion["discriminator_loss"](
                            y_real, y_gen
                        )
                        # Calculate the feature loss
                        loss_F = self.criterion["feature_loss"](
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
                    if self.gan:
                        total_loss = (
                            0.3 * global_loss + 0.7 * local_loss + loss_F + loss_G
                        )
                    else:
                        total_loss = 0.3 * global_loss + 0.7 * local_loss

                    # Delete the variables to free memory
                    del chunk_data, chunk_target, chunk_mag, chunk_phase, chunk_losses

                    # Update step for the tensorboard
                    self.writer.set_step(epoch, "valid")

                    # Create a dictionary with the losses
                    loss_values = {
                        "total_loss": total_loss.item(),
                        "global_loss": global_loss.item(),
                        "global_mag_loss": global_mag_loss.item(),
                        "global_phase_loss": global_phase_loss.item(),
                        "local_loss": local_loss.item(),
                        "local_mag_loss": local_mag_loss.item(),
                        "local_phase_loss": local_phase_loss.item(),
                    }

                    # Conditionally add GAN-related metrics
                    if self.gan:
                        loss_values.update(
                            {
                                "loss_G": loss_G.item(),
                                "loss_D": loss_D.item(),
                                "loss_F": loss_F.item(),
                            }
                        )

                    # Update the batch metrics
                    for key, value in loss_values.items():
                        self.valid_metrics.update(key, value)

                    for met in self.metric_ftns:
                        self.valid_metrics.update(
                            met.__name__, met(output_mag, target_mag)
                        )

                    # Update the progress bar
                    # Filter out the necessary metrics to display in the progress bar
                    progress_metrics = {
                        k: v
                        for k, v in loss_values.items()
                        if k
                        in ["total_loss", "local_loss", "loss_G", "loss_D", "loss_F"]
                    }
                    # Add the memory usage to the progress bar
                    progress_metrics["mem"] = (
                        f"{torch.cuda.max_memory_allocated() / (1024.0 * 1024.0):.0f} MB"
                    )
                    tepoch.set_postfix(progress_metrics)

                    # Log mean losses and metrics of this epoch to the tensorboard
                    if batch_idx == (len(self.valid_data_loader) - 1):
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
                        waveforms = [
                            input_waveform,
                            output_waveform[0],
                            target_waveform[0],
                        ]
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

                        # Set description for the progress bar after the last batch
                        tepoch.set_description(
                            f"Epoch {epoch} [VALID] {self._progress(-1, training=False)}"
                        )
                    else:
                        # Set description for the progress bar
                        tepoch.set_description(
                            f"Epoch {epoch} [VALID] {self._progress(batch_idx, training=False)}"
                        )

            # Add histogram of model parameters to the tensorboard
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins="auto")

            val_log = self.valid_metrics.result()

            self.epoch_log.update(**{"val_" + k: v for k, v in val_log.items()})

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

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        self.logger.info("Start training...")
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Train the model for an epoch
            self._train_epoch(epoch)
            # Check if do validation
            if self.do_validation:
                self._valid_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(self.epoch_log)

            # print logged informations to the screen
            self._log_epoch(log)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (
                        self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best
                    ) or (
                        self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best
                    )
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
