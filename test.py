import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from data_loader import preprocessing, postprocessing


def main(config):
    logger = config.get_logger("test")

    # Setup data_loader instances
    data_loader = getattr(module_data, config["data_loader"]["type"])(
        config["data_loader"]["args"]["data_dir"],
        batch_size=4,
        shuffle=False,
        training=False,
        num_workers=4,
        validation_split=0.0,
        quantity=0.1,
        random_resample=config["data_loader"]["args"]["random_resample"],
        chunking_params=config["data_loader"]["args"]["chunking_params"],
    )

    # Build model architecture
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # Get function handles of loss and metrics
    criterion = {crit: getattr(module_loss, crit) for crit in config["loss"]}
    metric_ftns = [getattr(module_metric, met) for met in config["metrics"]]

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # Prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_ftns))

    # Set the progress bar for the epoch
    with tqdm(
        data_loader,
        desc=f"TESTING {progress(data_loader, 0)}",
        unit="batch",
    ) as tepoch:
        epoch_log_test = {
            "losses": {
                "total_loss": [],
                "global_mag_loss": [],
                "global_phase_loss": [],
                "local_loss": [],
                "local_mag_loss": [],
                "local_phase_loss": [],
            },
            "metrics": {m.__name__: [] for m in metric_ftns},
        }

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                # Set description for the progress bar
                tepoch.set_description(f"TESTING {progress(data_loader, batch_idx)}")
                data, target = data.to(device), target.to(device)
                # Initialize the chunk ouptut and losses
                chunk_inputs = {"mag": [], "phase": []}
                chunk_outputs = {"mag": [], "phase": []}
                batch_loss = 0
                chunk_losses = {"mag": [], "phase": []}

                # Iterate through the chunks and calculate the loss for each chunk
                for chunk_idx in range(data.size(2)):
                    # Get current chunk data (and unsqueeze the chunk dimension)
                    chunk_data = data[:, :, chunk_idx, :, :].unsqueeze(2)
                    # Get current chunk target (and unsqueeze the chunk dimension)
                    chunk_target = target[:, :, chunk_idx, :, :].unsqueeze(2)

                    # Save the chunk input if batch_idx == (len_epoch - 1) (the last batch of the epoch)
                    if batch_idx == (len(data_loader) - 1):
                        chunk_inputs["mag"].append(chunk_data[:, 0, ...])
                        chunk_inputs["phase"].append(chunk_data[:, 1, ...])

                    # Forward pass
                    chunk_mag, chunk_phase = model(chunk_data)
                    # Calculate the chunk loss
                    chunk_mag_loss = criterion["mse_loss"](
                        chunk_mag, chunk_target[:, 0, ...]
                    )
                    chunk_phase_loss = criterion["mse_loss"](
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
                    config_dataloader=config["data_loader"]["args"],
                )
                target_waveform = postprocessing.reconstruct_from_stft_chunks(
                    mag=target[:, 0, ...].unsqueeze(1),
                    phase=target[:, 1, ...].unsqueeze(1),
                    batch_input=True,
                    crop=True,
                    config_dataloader=config["data_loader"]["args"],
                )
                # Compute the STFT of the output and target waveforms
                output_mag, output_phase = preprocessing.get_mag_phase(
                    output_waveform,
                    chunk_wave=False,
                    batch_input=True,
                    stft_params=config["data_loader"]["args"]["stft_params"],
                )
                target_mag, target_phase = preprocessing.get_mag_phase(
                    target_waveform,
                    chunk_wave=False,
                    batch_input=True,
                    stft_params=config["data_loader"]["args"]["stft_params"],
                )

                # Calculate the mag and phase loss
                local_mag_loss = torch.stack(chunk_losses["mag"]).mean()
                local_phase_loss = torch.stack(chunk_losses["phase"]).mean()
                global_mag_loss = criterion["mse_loss"](output_mag, target_mag)
                global_phase_loss = criterion["mse_loss"](output_phase, target_phase)
                # Get global loss and local loss
                global_loss = global_mag_loss + global_phase_loss
                local_loss = local_mag_loss + local_phase_loss
                # Calculate total loss
                batch_loss = 0.3 * global_loss + 0.7 * local_loss

                # Save valid losses and metrics for this batch
                epoch_log_test["losses"]["total_loss"].append(batch_loss.item())
                epoch_log_test["losses"]["global_mag_loss"].append(
                    global_mag_loss.item()
                )
                epoch_log_test["losses"]["global_phase_loss"].append(
                    global_phase_loss.item()
                )
                epoch_log_test["losses"]["local_loss"].append(local_loss.item())
                epoch_log_test["losses"]["local_mag_loss"].append(local_mag_loss.item())
                epoch_log_test["losses"]["local_phase_loss"].append(
                    local_phase_loss.item()
                )
                for met in metric_ftns:
                    epoch_log_test["metrics"][met.__name__].append(
                        met(output_mag, target_mag)
                    )

                # Update the progress bar
                tepoch.set_postfix(
                    global_loss=global_loss.item(),
                    global_mag_loss=global_mag_loss.item(),
                    global_phase_loss=global_phase_loss.item(),
                )

                # Update the total loss and metrics
                total_loss += batch_loss.item() * data.size(0)
                for i, metric in enumerate(metric_ftns):
                    total_metrics[i] += metric(output_mag, target_mag) * data.size(0)

    n_samples = len(data_loader.sampler)
    log = {"loss": total_loss / n_samples}
    log.update(
        {
            met.__name__: total_metrics[i].item() / n_samples
            for i, met in enumerate(metric_ftns)
        }
    )
    logger.info(log)


def progress(data_loader, batch_idx):
    base = "[{}/{} ({:.0f}%)]"
    current = batch_idx * data_loader.batch_size
    total = data_loader.n_samples
    return base.format(current, total, 100.0 * current / total)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Mamba ASR")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    config = ConfigParser.from_args(args)
    main(config)
