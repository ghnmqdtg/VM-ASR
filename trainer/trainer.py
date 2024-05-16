import torch
from tqdm import tqdm
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from tqdm.contrib.logging import logging_redirect_tqdm
from model.loss import mse_loss, mae_loss, MultiResolutionSTFTLoss, HiFiGANLoss
from logger.visualization import log_audio, log_waveform, log_spectrogram


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        models,
        metric_ftns,
        optimizers,
        config,
        device,
        data_loader_train,
        data_loader_val=None,
        lr_schedulers=None,
        amp=False,
        gan=False,
        logger=None,
        len_epoch=None,
    ):
        super().__init__(models, metric_ftns, optimizers, config, logger)
        self.config = config
        self.device, self.device_ids = device
        self.data_loader = data_loader_train
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # TODO: iteration-based training
            self.data_loader = inf_loop(data_loader_train)
            self.len_epoch = len_epoch
        self.data_loader_val = data_loader_val
        self.epoch_log = {}  # Log for metrics each epoch (both training and validation)
        self.do_validation = self.config.DATA.VALID_SPLIT > 0.0
        self.amp = amp  # Automatic Mixed Precision
        self.gan = gan  # Generative Adversarial Network

        self.optimizer_G = optimizers["generator"]
        self.lr_scheduler_G = lr_schedulers["generator"]

        if self.gan:
            self.optimizer_D = optimizers["discriminator"]
            self.lr_scheduler_D = lr_schedulers["discriminator"]

        self._init_metrics()
        self._init_losses()

        # Set models to device
        for key, model in self.models.items():
            self.models[key] = model.to(self.device)

        # Log summary of models
        for key, model in self.models.items():
            if key in ["generator", "mpd", "msd"]:
                length = int(self.config.DATA.TARGET_SR * self.config.DATA.SEGMENT)
                self.logger.info(f"Model summary: {model.flops(shape=(1, length))}")
            else:
                self.logger.info(f"Model summary: {model.flops()}")

    def _init_metrics(self):
        # Initialize the metric trackers
        self.metrics = ["total_loss"]
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
        self.logger.info(f"Train metrics: {self.train_metrics.get_keys()}")

    def _init_losses(self):
        # Initialize the losses
        self.multi_resolution_stft = MultiResolutionSTFTLoss(
            factor_sc=self.config.TRAIN.LOSSES.STFT_LOSS.SC_FACTOR,
            factor_mag=self.config.TRAIN.LOSSES.STFT_LOSS.MAG_FACTOR,
            emphasize_high_freq=self.config.TRAIN.LOSSES.STFT_LOSS.EMPHASIZE_HIGH_FREQ,
        )

        self.higi_gan_loss = HiFiGANLoss(
            gan_loss_type=self.config.TRAIN.ADVERSARIAL.GAN_LOSS_TYPE,
            gp_weight=self.config.TRAIN.ADVERSARIAL.GP_LAMBDA,
        )

    def _train_epoch(self, epoch):
        # Set the model to training mode
        for model in self.models.values():
            model.train()

        # Reset the train metrics
        self.train_metrics.reset()

        self.scaler_G = torch.cuda.amp.GradScaler() if self.amp else None
        self.scaler_D = torch.cuda.amp.GradScaler() if self.amp and self.gan else None

        # Set the progress bar for the epoch
        with logging_redirect_tqdm(loggers=[self.logger], tqdm_class=tqdm):
            with tqdm(
                self.data_loader,
                desc=f"Epoch {epoch} [TRAIN] {self._progress(0, training=True)}",
                unit="batch",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                postfix={key: 0.0 for key in self.train_metrics.get_keys()},
            ) as tepoch:
                for batch_idx, (
                    wave_input,
                    wave_target,
                    highcut,
                    _,  # filename is not needed for training
                    _,  # We only trim the audio for testing (see trainer/tester.py)
                ) in enumerate(tepoch):
                    # Set description for the progress bar
                    tepoch.set_description(
                        f"Epoch {epoch} [TRAIN] {self._progress(batch_idx, training=True)}"
                    )
                    # Reset the peak memory stats for the GPU
                    torch.cuda.reset_peak_memory_stats()
                    # Update step for the tensorboard
                    self.writer.set_step(epoch)
                    # Set the wave_input and wave_target to device
                    wave_input = wave_input.to(self.device, non_blocking=True)
                    wave_target = wave_target.to(self.device, non_blocking=True)

                    # Forward pass
                    with torch.cuda.amp.autocast(enabled=self.amp):
                        wave_out = self.models["generator"](wave_input, highcut)

                    # Calculate the losses
                    losses = self._get_losses(wave_out, wave_target)
                    total_generator_loss = sum(list(losses["generator"].values()))
                    if self.gan:
                        total_disc_loss = sum(list(losses["discriminator"].values()))

                    # Accumulate the loss for the accumulation steps
                    total_generator_loss /= self.config.TRAIN.ACCUMULATION_STEPS
                    if self.gan:
                        total_disc_loss /= self.config.TRAIN.ACCUMULATION_STEPS

                    # Backward pass
                    if (batch_idx + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0:
                        self._optimize(total_generator_loss)
                        if self.gan:
                            self._optimize_adversarial(total_disc_loss)

                    metrics_values = {"total_loss": total_generator_loss.item()}
                    if self.gan:
                        metrics_values.update(
                            {"total_disc_loss": total_disc_loss.item()}
                        )

                    metrics_values.update(
                        {
                            f"generator/{loss_name}": loss.item()
                            for loss_name, loss in losses["generator"].items()
                        }
                    )
                    if self.gan:
                        metrics_values.update(
                            {
                                f"discriminator/{disc_name}": loss.item()
                                for disc_name, loss in losses["discriminator"].items()
                            }
                        )

                    # Calculate the metrics
                    for met in self.metric_ftns:
                        metrics_values[met.__name__] = met(
                            wave_out.squeeze(1), wave_target.squeeze(1), hf=highcut
                        )

                    # Update the batch metrics
                    self.update_metrics(metrics_values, training=True)

                    # Update the progress bar
                    self.update_progress_bar(tepoch, metrics_values)

                    if batch_idx == (self.len_epoch - 1):
                        # Log the outputs to tensorboard or wandb
                        self.log_outputs(wave_input[0], wave_out[0], wave_target[0])
                        # Set description for the progress bar after the last batch
                        tepoch.set_description(
                            f"Epoch {epoch} [TRAIN] {self._progress(-1, training=True)}"
                        )

            # Save the log of the epoch into dict
            self.epoch_log = self.train_metrics.result()
            # Step the learning rate scheduler after each epoch
            num_steps = len(self.data_loader) // self.config.TRAIN.ACCUMULATION_STEPS
            if self.lr_scheduler_G is not None:
                self.lr_scheduler_G.step_update(
                    (epoch * num_steps + batch_idx)
                    // self.config.TRAIN.ACCUMULATION_STEPS
                )
                # Log the learning rate
                self.writer.add_scalar(
                    "lr/generator", self.optimizer_G.param_groups[-1]["lr"]
                )
            if self.gan:
                if self.lr_scheduler_D is not None:
                    self.lr_scheduler_D.step_update(
                        (epoch * num_steps + batch_idx)
                        // self.config.TRAIN.ACCUMULATION_STEPS
                    )
                    # Log the learning rate
                    self.writer.add_scalar(
                        "lr/discriminator", self.optimizer_D.param_groups[-1]["lr"]
                    )
            # Log the progress bar
            self.logger.info(tepoch)

    def _valid_epoch(self, epoch):
        # Set the model to training mode
        for model in self.models.values():
            model.eval()
        # Reset the train metrics
        self.valid_metrics.reset()
        # Set the progress bar for the epoch
        with logging_redirect_tqdm(loggers=[self.logger], tqdm_class=tqdm):
            with tqdm(
                self.data_loader_val,
                desc=f"Epoch {epoch} [VALID] {self._progress(0, training=False)}",
                unit="batch",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
                postfix={key: 0.0 for key in self.valid_metrics.get_keys()},
            ) as tepoch:
                with torch.no_grad():
                    for batch_idx, (
                        wave_input,
                        wave_target,
                        highcut,
                        _,
                        _,
                    ) in enumerate(tepoch):
                        # Set description for the progress bar
                        tepoch.set_description(
                            f"Epoch {epoch} [VALID] {self._progress(batch_idx, training=False)}"
                        )
                        # Reset the peak memory stats for the GPU
                        torch.cuda.reset_peak_memory_stats()
                        # Update step for the tensorboard
                        self.writer.set_step(epoch, "valid")
                        # Set the wave_input and wave_target to device
                        wave_input = wave_input.to(self.device, non_blocking=True)
                        wave_target = wave_target.to(self.device, non_blocking=True)

                        # Forward pass
                        with torch.cuda.amp.autocast(enabled=self.amp):
                            wave_out = self.models["generator"](wave_input, highcut)

                        # Calculate the losses
                        losses = self._get_losses(wave_out, wave_target)
                        total_generator_loss = sum(list(losses["generator"].values()))
                        if self.gan:
                            total_disc_loss = sum(
                                list(losses["discriminator"].values())
                            )

                        metrics_values = {"total_loss": total_generator_loss.item()}
                        if self.gan:
                            metrics_values.update(
                                {"total_disc_loss": total_disc_loss.item()}
                            )

                        metrics_values.update(
                            {
                                f"generator/{loss_name}": loss.item()
                                for loss_name, loss in losses["generator"].items()
                            }
                        )
                        if self.gan:
                            metrics_values.update(
                                {
                                    f"discriminator/{disc_name}": loss.item()
                                    for disc_name, loss in losses[
                                        "discriminator"
                                    ].items()
                                }
                            )

                        for met in self.metric_ftns:
                            metrics_values[met.__name__] = met(
                                wave_out.squeeze(1), wave_target.squeeze(1), hf=highcut
                            )

                        # Update the batch metrics
                        self.update_metrics(metrics_values, training=False)

                        # Update the progress bar
                        self.update_progress_bar(tepoch, metrics_values)

                        if batch_idx == (len(self.data_loader_val) - 1):
                            # Log the outputs to tensorboard or wandb
                            self.log_outputs(wave_input[0], wave_out[0], wave_target[0])
                            # Set description for the progress bar after the last batch
                            tepoch.set_description(
                                f"Epoch {epoch} [VALID] {self._progress(-1, training=False)}"
                            )

            # Save the log of the epoch into dict
            val_log = self.valid_metrics.result()
            self.epoch_log.update(**{"val_" + k: v for k, v in val_log.items()})
            # Log the progress bar
            self.logger.info(tepoch)

    def _get_losses(self, wave_out, wave_target):
        losses = {"generator": {}, "discriminator": {}}
        with torch.autograd.set_detect_anomaly(True):
            # Generator loss
            if "l1" in self.config.TRAIN.LOSSES.GEN:
                losses["generator"].update({"l1": mae_loss(wave_out, wave_target)})
            if "l2" in self.config.TRAIN.LOSSES.GEN:
                losses["generator"].update({"l2": mse_loss(wave_out, wave_target)})
            if "multi_resolution_stft" in self.config.TRAIN.LOSSES.GEN:
                losses["generator"].update(
                    {
                        "multi_resolution_stft": self._get_stft_loss(
                            wave_out, wave_target
                        )
                    }
                )

            # Discriminator loss
            if self.gan:
                if "mpd" in self.config.TRAIN.ADVERSARIAL.DISCRIMINATORS:
                    gen_losses, disc_loss = self._get_mpd_loss(wave_out, wave_target)
                    if not self.config.TRAIN.ADVERSARIAL.ONLY_FEATURE_LOSS:
                        losses["generator"].update(
                            {"adversarial_mpd": gen_losses["adversarial"]}
                        )
                    if not self.config.TRAIN.ADVERSARIAL.ONLY_ADVERSARIAL_LOSS:
                        losses["generator"].update(
                            {"features_mpd": gen_losses["features"]}
                        )
                    losses["discriminator"].update({"mpd": disc_loss})
                if "msd" in self.config.TRAIN.ADVERSARIAL.DISCRIMINATORS:
                    gen_losses, disc_loss = self._get_msd_loss(wave_out, wave_target)
                    if not self.config.TRAIN.ADVERSARIAL.ONLY_FEATURE_LOSS:
                        losses["generator"].update(
                            {"adversarial_msd": gen_losses["adversarial"]}
                        )
                    if not self.config.TRAIN.ADVERSARIAL.ONLY_ADVERSARIAL_LOSS:
                        losses["generator"].update(
                            {"features_msd": gen_losses["features"]}
                        )
                    losses["discriminator"].update({"msd": disc_loss})

        return losses

    def _get_stft_loss(self, wave_out, wave_target):
        # Squeeze the channel dimension for the torch.stft
        sc_loss, mag_loss = self.multi_resolution_stft(
            wave_out.squeeze(1), wave_target.squeeze(1)
        )
        return sc_loss + mag_loss

    def _get_mpd_loss(self, wave_out, wave_target):
        # Discriminator loss
        # Detach the wave_out because we don't want to update the gradients of the generator
        y_real, y_gen, _, _ = self.models["mpd"](wave_target, wave_out.detach())
        disc_loss = self.higi_gan_loss.discriminator_loss(y_real, y_gen)
        if self.config.TRAIN.ADVERSARIAL.GAN_LOSS_TYPE == "wgan-gp":
            gp = self.higi_gan_loss.gradient_penalty(
                wave_target, wave_out.detach(), self.models["mpd"]
            )
            disc_loss += gp

        # Generator loss
        # We want to update the gradients of the generator, so we don't detach the wave_out
        y_real, y_gen, feature_map_real, feature_map_gen = self.models["mpd"](
            wave_target, wave_out
        )
        g_feat_loss = self.higi_gan_loss.feature_loss(feature_map_real, feature_map_gen)
        g_adv_loss = self.higi_gan_loss.generator_loss(y_gen)

        features_loss = self.config.TRAIN.ADVERSARIAL.FEATURE_LOSS_LAMBDA * g_feat_loss

        if self.config.TRAIN.ADVERSARIAL.ONLY_ADVERSARIAL_LOSS:
            return {"adversarial": g_adv_loss}, disc_loss

        if self.config.TRAIN.ADVERSARIAL.ONLY_FEATURE_LOSS:
            return {"features": features_loss}, disc_loss

        return {
            "adversarial": g_adv_loss,
            "features": features_loss,
        }, disc_loss

    def _get_msd_loss(self, wave_out, wave_target):
        # Discriminator loss
        # Detach the wave_out because we don't want to update the gradients of the generator
        y_real, y_gen, _, _ = self.models["msd"](wave_target, wave_out.detach())
        disc_loss = self.higi_gan_loss.discriminator_loss(y_real, y_gen)

        # Generator loss
        # We want to update the gradients of the generator, so we don't detach the wave_out
        y_real, y_gen, feature_map_real, feature_map_gen = self.models["msd"](
            wave_target, wave_out
        )
        g_feat_loss = self.higi_gan_loss.feature_loss(feature_map_real, feature_map_gen)
        g_adv_loss = self.higi_gan_loss.generator_loss(y_gen)

        features_loss = self.config.TRAIN.ADVERSARIAL.FEATURE_LOSS_LAMBDA * g_feat_loss

        if self.config.TRAIN.ADVERSARIAL.ONLY_ADVERSARIAL_LOSS:
            return {"adversarial": g_adv_loss}, disc_loss

        if self.config.TRAIN.ADVERSARIAL.ONLY_FEATURE_LOSS:
            return {"features": features_loss}, disc_loss

        return {
            "adversarial": g_adv_loss,
            "features": features_loss,
        }, disc_loss

    def _optimize(self, loss):
        self.optimizer_G.zero_grad()
        self.scaler_G.scale(loss).backward()
        self.scaler_G.step(self.optimizer_G)
        self.scaler_G.update()

    def _optimize_adversarial(self, loss):
        self.optimizer_D.zero_grad()
        self.scaler_D.scale(loss).backward()
        self.scaler_D.step(self.optimizer_D)
        self.scaler_D.update()

    def update_metrics(self, metrics_values, training=True):
        # Update the batch metrics
        for key, value in metrics_values.items():
            if training:
                self.train_metrics.update(key, value)
            else:
                self.valid_metrics.update(key, value)

    def log_outputs(self, wave_in, wave_out, wave_target):
        log_functions = {
            "audio": log_audio,
            "waveform": log_waveform,
            "spectogram": log_spectrogram,
        }

        for item, func in log_functions.items():
            if item in self.config.TENSORBOARD.LOG_ITEMS:
                func(self.writer, wave_in, wave_out, wave_target, self.config)

    @staticmethod
    def update_progress_bar(tepoch, metrics_values):
        # Update the progress bar
        # Filter out the necessary metrics to display in the progress bar
        progress_metrics = {
            k: v
            for k, v in metrics_values.items()
            if k
            in [
                "total_loss",
                "total_disc_loss",
                "snr",
                "lsd",
                "lsd-hf",
                "lsd-lf",
            ]
        }
        # Add the memory usage to the progress bar
        progress_metrics["mem"] = (
            f"{torch.cuda.max_memory_allocated() / (1024.0 * 1024.0):.0f} MB"
        )
        tepoch.set_postfix(progress_metrics)

    def _progress(self, batch_idx, training=True):
        base = "[{}/{} ({:.0f}%)]"
        if training:
            # Get the total number of samples in the training dataset
            total = len(self.data_loader.dataset)
        else:
            total = len(self.data_loader_val.dataset)

        if batch_idx == -1:
            current = total
        else:
            current = batch_idx * self.data_loader.batch_size

        return base.format(current, total, 100.0 * current / total)
