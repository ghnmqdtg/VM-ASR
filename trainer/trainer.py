import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from tqdm import tqdm
from data_loader import prepocessing, postpocessing


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
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
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker(
            'loss', 'waveform_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(
            'loss', 'waveform_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(tqdm(self.data_loader)):
            # Both data and target are in the shape of (batch_size, 2 (mag and phase), num_chunks, frequency_bins, frames)
            data, target = data.to(self.device), target.to(self.device)
            # Set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes
            self.optimizer.zero_grad()
            # Initialize the chunk ouptut and losses
            chunk_outputs = []
            total_loss = 0
            chunk_losses = []
            # Iterate through the chunks and calculate the loss for each chunk
            for chunk_idx in range(data.size(2)):
                # Get current chunk data and target (only focus on mag for now)
                chunk_data = data[:, 0, chunk_idx, :, :].unsqueeze(1)
                chunk_target = target[:, 0, chunk_idx, :, :].unsqueeze(1)
                # Print the shape of the chunk data and target
                # print(f'Chunk data shape: {chunk_data.shape}, Chunk target shape: {chunk_target.shape}')
                # Forward pass
                chunk_output = self.model(chunk_data)
                # Calculate the chunk loss
                chunk_loss = self.criterion(chunk_output, chunk_target)
                # Accumulate the chunk loss
                chunk_losses.append(chunk_loss)
                # Store the chunk output
                chunk_outputs.append(chunk_output)

            # Concatenate the outputs along the chunk dimension
            chunk_outputs = torch.cat(chunk_outputs, dim=1)
            # Reconstruct the waveform from the concatenated output and target
            output_waveform = postpocessing.reconstruct_from_stft_chunks(
                mag=chunk_outputs, phase=target[:, 1, ...].unsqueeze(1), batch_input=True)
            target_waveform = postpocessing.reconstruct_from_stft_chunks(
                mag=target[:, 0, ...].unsqueeze(1), phase=target[:, 1, ...].unsqueeze(1), batch_input=True, crop=True)
            # # Print the shape of the output and target waveforms
            # print(f'Output waveform shape: {output_waveform.shape}, Target waveform shape: {target_waveform.shape}')
            # Compute the STFT of the output and target waveforms
            output_mag, output_phase = prepocessing.get_mag_phase(
                output_waveform, chunk_wave=False, batch_input=True)
            target_mag, target_phase = prepocessing.get_mag_phase(
                target_waveform, chunk_wave=False, batch_input=True)
            # # Print the shape of the output and target mag
            # print(f'Output mag shape: {output_mag.shape}, Target mag shape: {target_mag.shape}')
            # # Print the shape of the output and target phase
            # print(f'Output phase shape: {output_phase.shape}, Target phase shape: {target_phase.shape}')
            
            waveform_loss = self.criterion(output_mag, target_mag)

            # Calculate the average chunk loss and add the waveform loss
            total_loss = torch.stack(chunk_losses).mean() + waveform_loss
            # Backward pass
            total_loss.backward()
            # Update the weights
            self.optimizer.step()
            
            # Update the train metrics
            self.train_metrics.update('loss', total_loss.item())
            self.train_metrics.update('waveform_loss', waveform_loss.item())

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            for met in self.metric_ftns:
                # Calculate the loss for the reconstructed waveform
                self.train_metrics.update(met.__name__, met(output_mag, target_mag))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    total_loss.item()))
                # self.writer.add_image('input', make_grid(`
                #     data.cpu(), nrow=8, normalize=True))`

            if batch_idx == self.len_epoch:
                break
        # Add histogram of model parameters to the tensorboard
        log = self.train_metrics.result()
        # Validate after each epoch
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})
        # Step the learning rate scheduler after each epoch
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
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
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(
                    data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
