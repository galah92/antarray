import json
import pickle
import subprocess as sp
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import typer
from matplotlib.ticker import FormatStrFormatter
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, random_split

import analyze
from generate_dataset import ff_from_phase_shifts, steering_repr

DEFAULT_DATASET_PATH: Path = Path.cwd() / "dataset" / "rand_bf_2d_only_40k.h5"
DEFAULT_EXPERIMENTS_PATH: Path = Path.cwd() / "experiments"
DEFAULT_MODEL_NAME = "model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


class RadiationPatternDataset(Dataset):
    def __init__(self, patterns, phase_shifts, transform=None):
        # Radiation patterns with shape (n_samples, n_phi, n_theta)
        self.patterns = torch.from_numpy(patterns)
        # Phase shift matrices with shape (n_samples, xn, yn)
        self.phase_shifts = torch.from_numpy(phase_shifts)
        # Optional transform to be applied to the patterns
        self.transform = transform

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        phase_shift = self.phase_shifts[idx]

        # Expand dims to create a channel dimension for the CNN
        pattern = pattern.unsqueeze(0)  # Shape becomes (1, n_phi, n_theta)

        if self.transform:
            pattern = self.transform(pattern)

        return pattern, phase_shift


class Hdf5Dataset(Dataset):
    """
    Dataset class for loading radiation patterns and phase shift matrices from an HDF5 file.
    """

    def __init__(self, dataset_file: Path, use_steering_info=False, add_fft=False):
        self.dataset_file = dataset_file
        self.h5f = None  # Lazy loading of the HDF5 file
        self.use_steering_info = use_steering_info
        self.add_fft = add_fft

    def get_dataset(self):
        if self.h5f is None:
            self.h5f = h5py.File(self.dataset_file, "r")
        return self.h5f

    def __len__(self):
        h5f = self.get_dataset()
        return len(h5f["patterns"])

    def __getitem__(self, idx):
        h5f = self.get_dataset()
        pattern = torch.from_numpy(h5f["patterns"][idx])
        phase_shift = torch.from_numpy(h5f["labels"][idx])

        pattern = pattern.unsqueeze(0)  # Add channel dimension for CNN
        pattern = pattern.clamp(min=0)  # Set negative values to 0
        pattern = pattern / 30  # Normalize

        if self.add_fft:
            fft = torch.fft.fft2(pattern, norm="ortho")
            pattern = torch.cat([pattern, torch.abs(fft), torch.angle(fft)], dim=0)

        if self.use_steering_info:
            steering_info = torch.from_numpy(h5f["steering_info"][idx])
            return pattern, phase_shift, steering_info

        return pattern, phase_shift


class ConvModel(nn.Module):
    def __init__(
        self,
        in_channels=1,  # Number of input channels (1 for single radiation pattern)
        in_shape=(180, 180),  # Shape of the input radiation pattern (n_phi, n_theta)
        out_shape=(16, 16),  # Size of the output phase shift matrix (xn, yn)
        # Number of channels in each convolutional layer
        conv_channels=[32, 64, 128, 256, 512],
        # Number of units in each fully connected layer
        fc_units=[2048, 1024],
        # Use global pooling instead ofusing max pooling to reduce feature size
        use_global_pool=False,
    ):
        super().__init__()

        self.use_global_pool = use_global_pool
        self.out_shape = out_shape

        self.conv = nn.Sequential()
        ch = [in_channels] + conv_channels
        for i in range(1, len(ch)):
            self.conv += nn.Sequential(
                nn.Conv2d(ch[i - 1], ch[i], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ch[i]),
                nn.ReLU(),
            )
            if not self.use_global_pool:  # Use max pooling
                self.conv += nn.Sequential(nn.MaxPool2d(2))

        if self.use_global_pool:
            # Reduces feature size while preserving information
            self.conv.append(nn.AdaptiveAvgPool2d(1))
            feature_size = 1
        else:
            # Calculate the size of the feature maps after encoding
            # After n channels MaxPool2d with stride=2
            feature_size = np.prod(np.array(in_shape) // (2 ** len(conv_channels)))

        self.fc = nn.Sequential(nn.Flatten())
        fcs = [conv_channels[-1] * feature_size] + fc_units
        for i in range(1, len(fcs)):
            self.fc += nn.Sequential(
                nn.Linear(fcs[i - 1], fcs[i]),
                nn.ReLU(),
                nn.Dropout(0.5),
            )

        self.fc.append(nn.Linear(fcs[-1], np.prod(out_shape)))

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        x = x.view(-1, *self.out_shape)
        return x


# --- Building Blocks (DoubleConv, Down, Up) ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# Helper function for cropping (if needed for skip connections)
def center_crop(layer, target_size):
    _, _, layer_height, layer_width = layer.size()
    diff_y = layer_height - target_size[0]
    diff_x = layer_width - target_size[1]
    return layer[
        :,
        :,
        diff_y // 2 : layer_height - (diff_y - diff_y // 2),
        diff_x // 2 : layer_width - (diff_x - diff_x // 2),
    ]


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        target_size = x1.size()[2:]
        x2_cropped = center_crop(x2, target_size)
        x = torch.cat([x2_cropped, x1], dim=1)
        return self.conv(x)


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        bilinear=True,
        final_stage="adaptive_pool",
    ):
        """
        U-Net architecture for predicting phase directly (linear output).
        Suitable for use with circular MSE/MAE loss functions.

        Args:
            in_channels (int): Number of input channels (usually 1 for DBi map).
            out_channels (int): Number of output channels (should be 1 for direct phase).
            bilinear (bool): If True, use bilinear upsampling. Otherwise, use ConvTranspose2d.
            final_stage (str): 'adaptive_pool' or 'large_kernel'. How to get to 16x16.
        """
        super().__init__()
        if out_channels != 1:
            print(f"{out_channels=}!=1 was not tested")

        self.in_channels = in_channels

        # --- Encoder ---
        self.inc = DoubleConv(in_channels, 32)  # -> (32, 180, 180)
        self.down1 = Down(32, 64)  # -> (64, 90, 90)
        self.down2 = Down(64, 128)  # -> (128, 45, 45)
        self.down3 = Down(128, 256)  # -> (256, 22, 22)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)  # -> (256, 11, 11) (or 512)

        # --- Decoder ---
        self.up1 = Up(512, 256 // factor, bilinear)  # -> (128, 22, 22)
        self.up2 = Up(256, 128 // factor, bilinear)  # -> (64, 44, 44)

        # --- Final Stage ---
        final_in_ch = 128 // factor  # Output channels of self.up2 (64)
        if final_stage == "adaptive_pool":
            self.final_conv = nn.Sequential(
                nn.AdaptiveAvgPool2d((16, 16)),  # -> (64, 16, 16)
                nn.Conv2d(final_in_ch, out_channels, kernel_size=1),  # -> (1, 16, 16)
            )
        elif final_stage == "large_kernel":
            self.final_conv = nn.Conv2d(final_in_ch, out_channels, kernel_size=29)
        else:
            raise ValueError(f"Unknown {final_stage=}")

    def forward(self, x):
        # --- Encoder ---
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # --- Decoder ---
        dx = self.up1(x5, x4)  # -> (128, 22, 22)
        dx = self.up2(dx, x3)  # -> (64, 44, 44)

        logits = self.final_conv(dx)  # -> (1, 16, 16)
        return logits.squeeze(1)  # -> (16, 16)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, out_shape=(16, 16)):
        super().__init__()
        self.out_shape = out_shape
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, np.prod(out_shape))

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = out.view(-1, *self.out_shape)
        return out


def resnet18(out_shape=(16, 16)):
    return ResNet(ResidualBlock, [2, 2, 2, 2], out_shape)


def resnet34(out_shape=(16, 16)):
    return ResNet(ResidualBlock, [3, 4, 6, 3], out_shape)


def circular_mse_loss_torch(pred: torch.Tensor, target: torch.Tensor):
    diff = torch.abs(pred - target)
    circular_diff = torch.min(diff, 2 * torch.pi - diff)
    return torch.mean(circular_diff**2)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_val_loss = float("inf")

    def early_stop(self, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


@app.command()
def run_model(
    experiment: str,
    overwrite: bool = False,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    exps_path: Path = DEFAULT_EXPERIMENTS_PATH,
    batch_size: int = 256,
    n_epochs: int = 200,
    lr: float = 2e-3,
    model_type: str = "unet",
    use_amp: bool = True,  # Enable Automatic Mixed Precision by default
    benchmark: bool = True,  # Enable CUDA benchmarking by default
):
    # Set benchmark mode for CUDA - improves performance when input sizes don't change
    if benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("CUDA benchmark mode enabled")

    exp_path = get_experiment_path(experiment, exps_path, overwrite)

    train_loader, val_loader, test_loader = create_dataloaders(dataset_path, batch_size)

    model = model_type_to_class(model_type)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{model_type=} ({n_params:,}), {batch_size=}, {n_epochs=}")

    criterion = circular_mse_loss_torch

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    # Train model
    model, history, interrupted = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        EarlyStopper(patience=10, min_delta=1e-4),
        n_epochs,
        use_amp=use_amp,
    )

    # Save model (including model type)
    model_path = exp_path / DEFAULT_MODEL_NAME
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "model_type": model_type,
        },
        model_path,
    )
    print(f"Model saved to {model_path}")

    plot_training(experiment, overwrite=overwrite, exps_path=exps_path)

    if not interrupted:
        pred_model(experiment, dataset_path, exps_path, n_examples=5)


def get_experiment_path(
    experiment: str, exps_path: Path = DEFAULT_EXPERIMENTS_PATH, overwrite: bool = False
):
    exp_path = exps_path / experiment
    exp_path.mkdir(exist_ok=overwrite, parents=True)
    print(f"experiment={exp_path}")
    return exp_path


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    early_stopper,
    n_epochs,
    log_interval=10,
    use_amp=True,
):
    interrupted = False
    since = time.time()
    history = {"train_loss": [], "val_loss": [], "lr": []}

    model = model.to(device)
    best_val_loss, best_model_wts = float("inf"), None

    scaler = GradScaler() if use_amp and torch.cuda.is_available() else None
    gpu_warmup(model, (model.in_channels, 180, 180))

    try:
        for epoch in range(n_epochs):
            lr = scheduler.get_last_lr()[0]
            print()
            print(f"Epoch {epoch:03}/{n_epochs:03} | {lr=:1.1e}")

            # Track timing statistics
            epoch_start_time = time.time()
            timing_stats = {
                "data_time": 0.0,
                "to_device_time": 0.0,
                "forward_time": 0.0,
                "loss_time": 0.0,
                "backward_time": 0.0,
                "optimizer_time": 0.0,
                "synchronization": 0.0,
                "batch_overhead": 0.0,
            }

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                phase_start_time = time.time()
                if phase == "train":
                    model.train()
                    dataloader = train_loader
                else:
                    model.eval()
                    dataloader = val_loader

                running_loss = 0.0
                n_batches, dataloader_iter = len(dataloader), iter(dataloader)

                for i in range(n_batches):
                    batch_start_time = time.time()
                    curr_time = time.time()

                    inputs, targets = next(dataloader_iter)
                    data_time, curr_time = time.time() - curr_time, time.time()
                    timing_stats["data_time"] += data_time

                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    to_device_time, curr_time = time.time() - curr_time, time.time()
                    timing_stats["to_device_time"] += to_device_time

                    optimizer.zero_grad()
                    if device == "cuda":
                        torch.cuda.synchronize()
                    optim_zero_time, curr_time = time.time() - curr_time, time.time()
                    timing_stats["optimizer_time"] += optim_zero_time

                    with torch.set_grad_enabled(phase == "train"):
                        enabled = use_amp and device == "cuda"
                        with autocast(device, enabled=enabled):
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)

                        if device == "cuda":
                            torch.cuda.synchronize()
                        forward_time, curr_time = time.time() - curr_time, time.time()
                        timing_stats["forward_time"] += forward_time

                        loss_time = 0
                        timing_stats["loss_time"] += loss_time

                        if phase == "train":
                            if use_amp and device == "cuda":
                                scaler.scale(loss).backward()
                                if device == "cuda":
                                    torch.cuda.synchronize()
                                bwd_time, curr_time = (
                                    time.time() - curr_time,
                                    time.time(),
                                )
                                timing_stats["backward_time"] += bwd_time

                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                loss.backward()
                                if device == "cuda":
                                    torch.cuda.synchronize()
                                bwd_time, curr_time = (
                                    time.time() - curr_time,
                                    time.time(),
                                )
                                timing_stats["backward_time"] += bwd_time

                                optimizer.step()

                            if device == "cuda":
                                torch.cuda.synchronize()
                            optim_step_time, curr_time = (
                                time.time() - curr_time,
                                time.time(),
                            )
                            timing_stats["optimizer_time"] += optim_step_time

                    if device == "cuda":
                        torch.cuda.synchronize()
                    sync_time, curr_time = time.time() - curr_time, time.time()
                    timing_stats["synchronization"] += sync_time

                    running_loss += loss.item() * inputs.size(0)

                    # Calculate batch overhead (everything not explicitly timed)
                    batch_end_time = time.time()
                    batch_total_time = batch_end_time - batch_start_time
                    explicitly_timed = (
                        data_time
                        + to_device_time
                        + forward_time
                        + loss_time
                        + sync_time
                        + optim_zero_time
                    )
                    if phase == "train":
                        explicitly_timed += bwd_time + optim_step_time

                    batch_overhead = batch_total_time - explicitly_timed
                    timing_stats["batch_overhead"] += batch_overhead

                    if phase == "train" and i % log_interval == 0:
                        gpu_memory = ""
                        if device == "cuda":
                            gpu_memory = f" | gpu_mem={torch.cuda.memory_allocated() / 1024**3:.2f}GB"

                        print(
                            f"Batch {i:03}/{n_batches:03} | "
                            f"loss={loss.item():.03f} | "
                            f"data={data_time * 1000:.1f}ms | "
                            f"to_dev={to_device_time * 1000:.1f}ms | "
                            f"fwd={forward_time * 1000:.1f}ms | "
                            f"loss={loss_time * 1000:.1f}ms | "
                            f"bwd={bwd_time * 1000:.1f}ms | "
                            f"optim={optim_zero_time * 1000 + optim_step_time * 1000:.1f}ms | "
                            f"sync={sync_time * 1000:.1f}ms | "
                            f"overhead={batch_overhead * 1000:.1f}ms{gpu_memory}"
                        )

                epoch_loss = running_loss / len(dataloader.dataset)
                phase_time = time.time() - phase_start_time

                loss_str = f"{phase}_loss={epoch_loss:.03f}"
                time_str = f"{phase}_time={phase_time:.2f}s"
                print(f"Epoch {epoch:03}/{n_epochs:03} | {loss_str} | {time_str}")

                if phase == "train":
                    history["train_loss"].append(epoch_loss)
                    if scheduler:
                        history["lr"].append(lr)
                else:
                    history["val_loss"].append(epoch_loss)

                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        best_model_wts = model.state_dict().copy()

            epoch_time = time.time() - epoch_start_time
            avg_data_time = timing_stats["data_time"] * 1000 / (n_batches * 2)
            avg_to_device = timing_stats["to_device_time"] * 1000 / (n_batches * 2)
            avg_forward_time = timing_stats["forward_time"] * 1000 / (n_batches * 2)
            avg_loss_time = timing_stats["loss_time"] * 1000 / (n_batches * 2)
            avg_backward_time = timing_stats["backward_time"] * 1000 / n_batches
            avg_optimizer_time = (
                timing_stats["optimizer_time"]
                * 1000
                / (n_batches * (1 if phase == "val" else 2))
            )
            avg_sync_time = timing_stats["synchronization"] * 1000 / (n_batches * 2)
            avg_overhead_time = timing_stats["batch_overhead"] * 1000 / (n_batches * 2)

            print("Epoch timing summary:")
            print(f"  Total epoch time: {epoch_time:.2f}s")

            gpu_memory = ""
            if torch.cuda.is_available():
                gpu_memory = (
                    f"  GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB / "
                    f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}GB (current/max)"
                )
                print(gpu_memory)

            print(
                f"  Avg times per batch: data={avg_data_time:.1f}ms, to_device={avg_to_device:.1f}ms, "
                f"forward={avg_forward_time:.1f}ms, loss={avg_loss_time:.1f}ms, backward={avg_backward_time:.1f}ms, "
                f"optimizer={avg_optimizer_time:.1f}ms, sync={avg_sync_time:.1f}ms, overhead={avg_overhead_time:.1f}ms"
            )

            # Identify bottleneck
            times = {
                "Data loading": avg_data_time,
                "To device": avg_to_device,
                "Forward pass": avg_forward_time,
                "Loss calculation": avg_loss_time,
                "Backward pass": avg_backward_time,
                "Optimizer operations": avg_optimizer_time,
                "CUDA synchronization": avg_sync_time,
                "Remaining overhead": avg_overhead_time,
            }
            bottleneck = max(times.items(), key=lambda x: x[1])
            print(f"  Bottleneck: {bottleneck[0]} ({bottleneck[1]:.1f}ms)")

            if early_stopper is not None:
                if early_stopper.early_stop(history["val_loss"][-1]):
                    print("Early stopping...")
                    break

            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(history["val_loss"][-1])  # Pass the validation loss
                else:
                    scheduler.step()

    except KeyboardInterrupt:
        print("Training interrupted by user...")
        interrupted = True
    finally:
        if best_model_wts:
            model.load_state_dict(best_model_wts)

        elapsed = time.time() - since
        print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
        print(f"Best val loss: {best_val_loss:.4f}")

    return model, history, interrupted


def gpu_warmup(model, in_shape, num_iterations=10):
    """
    Warm up the GPU by running a dummy input through the model.
    This is useful for ensuring that the GPU is ready for training.
    """
    if device == "cuda":
        print()
        print("GPU warm-up started")
        dummy_input = torch.randn(1, *in_shape, device=device)
        with torch.no_grad():
            for _ in range(num_iterations):
                model(dummy_input)
        torch.cuda.synchronize()
        print("GPU warm-up complete")


def create_dataloaders(dataset_path: Path, batch_size: int):
    ds = Hdf5Dataset(dataset_path)

    # Split data into train, validation, and test sets
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(ds, [0.9, 0.05, 0.05], generator=gen)

    train, val, test = len(train_ds), len(val_ds), len(test_ds)
    print(f"dataset={dataset_path}, {train=:,}, {val=:,}, {test=:,}")

    # Create dataloaders with pin_memory for faster transfers to GPU
    train_loader = DataLoader(
        train_ds,
        batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader


@app.command()
def pred_model(
    experiment: str,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    exps_path: Path = DEFAULT_EXPERIMENTS_PATH,
    batch_size: int = 128,
    n_examples: int = 5,
):
    exp_path = get_experiment_path(experiment, exps_path, overwrite=True)

    _, _, test_loader = create_dataloaders(dataset_path, batch_size)
    test_indices = test_loader.dataset.indices

    with h5py.File(dataset_path, "r") as h5f:
        theta = h5f["theta"][:]
        phi = h5f["phi"][:]
        steering_info = h5f["steering_info"][:]
        steering_info = steering_info[test_indices]

    checkpoint = torch.load(exps_path / experiment / DEFAULT_MODEL_NAME)
    model_type = checkpoint["model_type"]
    model = model_type_to_class(model_type).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    n_test = len(test_loader.dataset)
    all_preds = np.empty((n_test, 16, 16))
    all_targets = np.empty((n_test, 16, 16))

    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            j, k = i * batch_size, inputs.size(0)
            all_preds[j : j + k] = model(inputs.to(device)).cpu().numpy()
            all_targets[j : j + k] = targets.numpy()

    calc_metrics(exp_path, all_preds, all_targets)
    plot_steer_loss(exp_path, all_preds, all_targets, steering_info)

    rand_indices = np.random.choice(n_test, n_examples, replace=False)
    for idx in rand_indices:
        pred, target = all_preds[idx], all_targets[idx]
        steering = steering_info[idx]
        filepath = exp_path / f"prediction_example_{test_indices[idx]}.png"
        compare_phase_shifts(pred, target, theta, phi, steering, filepath)


def calc_metrics(exp_path: Path, preds, targets):
    mse = circular_mse_loss_np(preds, targets)
    mae = circular_mae_loss_np(preds, targets)
    rmse = np.sqrt(mse)
    metrics = {"mae": mae, "mse": mse, "rmse": rmse}

    metrics_json = json.dumps(metrics, indent=4)
    metrics_path = exp_path / "metrics.json"
    metrics_path.write_text(metrics_json)
    print(f"Metrics saved to {metrics_path}")


def plot_steer_loss(exp_path: Path, preds, targets, steering_info):
    """
    Plot the loss by steering angle in the test set.
    """
    losses = circular_mse_loss_np(preds, targets, axis=(1, 2))
    steering_info = steering_info[:, :, 0]  # Take only the first beamforming angle

    fig, ax = plt.subplots()
    ax.scatter(*steering_info.T, c=losses, s=16)
    ax.set_xlabel("Theta (degrees)")
    ax.set_ylabel("Phi (degrees)")
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.set_title(f"Loss by steering angle ({losses.size} samples)")
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Circular MSE Loss")
    fig.savefig(exp_path / "steer_loss.png", dpi=600, bbox_inches="tight")
    print(f"Steering loss plot saved to {exp_path / 'steer_loss.png'}")


def model_type_to_class(model_type: str):
    if model_type == "cnn":
        return ConvModel()
    elif model_type == "resnet":
        return resnet18()
    elif model_type == "unet":
        return UNetModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compare_phase_shifts(
    pred,
    label,
    theta,
    phi,
    steering_info,
    filepath: Path | None = None,
    clip_ff: bool = True,
):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    analyze.plot_phase_shifts(label, title="Ground Truth Phase Shifts", ax=axs[0, 0])
    analyze.plot_phase_shifts(pred, title="Predicted Phase Shifts", ax=axs[1, 0])

    label_ff, pred_ff = ff_from_phase_shifts(label), ff_from_phase_shifts(pred)

    if clip_ff:
        label_ff, pred_ff = label_ff.clip(min=0), pred_ff.clip(min=0)

    title = "Ground Truth 2D Radiation Pattern"
    analyze.plot_ff_2d(label_ff, theta, phi, title=title, ax=axs[0, 1])
    title = "Predicted 2D Radiation Pattern"
    analyze.plot_ff_2d(pred_ff, theta, phi, title=title, ax=axs[1, 1])

    axs[0, 2].remove()
    axs[0, 2] = fig.add_subplot(2, 3, 3, projection="3d")
    title = "Ground Truth 3D Radiation Pattern"
    analyze.plot_ff_3d(theta, phi, label_ff, title=title, ax=axs[0, 2])

    axs[1, 2].remove()
    axs[1, 2] = fig.add_subplot(2, 3, 6, projection="3d")
    title = "Predicted 3D Radiation Pattern"
    analyze.plot_ff_3d(theta, phi, pred_ff, title=title, ax=axs[1, 2])

    loss = circular_mse_loss_np(pred, label)
    steering_str = steering_repr(steering_info)
    fig.suptitle(f"Prediction Example | loss {loss:.4f} | steering {steering_str}")

    fig.set_tight_layout(True)

    if filepath:
        fig.savefig(filepath, dpi=600, bbox_inches="tight")
        print(f"Prediction example saved to {filepath}")


@app.command()
def plot_training(
    experiment: str,
    exps_path: Path = DEFAULT_EXPERIMENTS_PATH,
    overwrite: bool = False,
):
    exp_path = get_experiment_path(experiment, exps_path, overwrite)

    model_path = exp_path / DEFAULT_MODEL_NAME
    history = torch.load(model_path)["history"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    xticks = list(range(0, 200 + 1, 20))
    yticks = np.arange(0, 4 + 0.1, 0.1)

    axs[0].plot(history["train_loss"], "b-", label="Training Loss")
    axs[0].plot(history["val_loss"], "r-", label="Validation Loss")
    axs[0].set_xlim(xticks[0], xticks[-1])
    # axs[0].set_xticks(xticks)
    axs[0].set_ylim(yticks[0], yticks[-1])
    # axs[0].set_yticks(yticks)
    axs[0].set_title("Training and Validation Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True, which="major")
    axs[0].grid(True, which="minor", alpha=0.25)
    axs[0].minorticks_on()

    if "lr" in history and history["lr"]:
        axs[1].plot(history["lr"], "g-")
        axs[1].set_xlim(xticks[0], xticks[-1])
        axs[1].set_xticks(xticks)
        axs[1].set_yscale("log")
        axs[1].yaxis.set_major_formatter(FormatStrFormatter("%0.1e"))
        axs[1].set_title("Learning Rate")
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Learning Rate")
        axs[1].grid(True, which="major")
        axs[1].grid(True, which="minor", alpha=0.25)
        axs[1].minorticks_on()

    fig.set_tight_layout(True)
    save_path = exp_path / "training_history.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Training history plot saved to {save_path}")


def index_from_beamforming_angles(steering_info, theta: int, phi: int) -> int:
    """
    Find the index of the given steering angles: https://stackoverflow.com/a/25823710/5151909
    """
    idx = (steering_info == (theta, phi)).all(axis=1).nonzero()[0].min()
    return idx


@app.command()
def run_knn(
    experiment: str,
    overwrite: bool = False,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    exps_path: Path = DEFAULT_EXPERIMENTS_PATH,
    n_neighbors: int = 5,
):
    exp_path = get_experiment_path(experiment, exps_path, overwrite)

    with h5py.File(dataset_path, "r") as h5f:
        patterns, labels = h5f["patterns"][:], h5f["labels"][:]
        theta, phi = h5f["theta"][:], h5f["phi"][:]
        steering_info = h5f["steering_info"][:]

    # Flatten the features
    patterns = patterns.reshape(patterns.shape[0], -1)
    labels = labels.reshape(labels.shape[0], -1)

    patterns[patterns < 0] = 0  # Set negative values to 0
    patterns = patterns / 30  # Normalize

    # Print dataset info
    print(f"Dataset loaded: {len(patterns)} samples")
    print(f"Pattern shape: {patterns.shape}")
    print(f"Labels shape: {labels.shape}")

    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = (
        train_test_split(
            patterns,
            labels,
            np.arange(labels.shape[0]),
            test_size=0.2,
            random_state=42,
        )
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    y_pred = knn.fit(X_train, y_train).predict(X_test)

    loss = circular_mse_loss_np(y_pred, y_test)
    print(f"Val loss: {loss:.4f}")

    # save y_pred to h5 file
    with h5py.File(exp_path / "knn_pred.h5", "w") as h5f:
        h5f.create_dataset("y_pred", data=y_pred)
        h5f.create_dataset("y_test", data=y_test)
        h5f.create_dataset("loss", data=loss)
        h5f.create_dataset("steering_info", data=steering_info[idx_test])
        h5f.create_dataset("theta", data=theta)
        h5f.create_dataset("phi", data=phi)

    mode = "wb" if overwrite else "xb"
    with (exp_path / "knn_model.pkl").open(mode) as f:
        pickle.dump(knn, f)


@app.command()
def pred_knn(
    experiment: str,
    idx: int | None = None,
    exps_path: Path = DEFAULT_EXPERIMENTS_PATH,
):
    exp_path = exps_path / experiment

    with h5py.File(exp_path / "knn_pred.h5", "r") as h5f:
        y_pred = h5f["y_pred"][:].reshape(-1, 16, 16)
        y_test = h5f["y_test"][:].reshape(-1, 16, 16)
        steering_info = h5f["steering_info"][:]
        theta, phi = h5f["theta"][:], h5f["phi"][:]

    plot_steer_loss(exp_path, y_pred, y_test, steering_info)

    if idx is None:
        idx = np.random.choice(len(y_pred), 1)[0]

    pred, test, steering_info = y_pred[idx], y_test[idx], steering_info[idx]
    filepath = exp_path / f"knn_pred_example_{idx}.png"
    compare_phase_shifts(pred, test, theta, phi, steering_info, filepath)


def cosine_angular_loss_np(inputs: np.ndarray, targets: np.ndarray):
    return np.mean(1 - np.cos(inputs - targets))


def circular_mse_loss_np(pred: np.ndarray, target: np.ndarray, axis=None):
    diff = np.abs(pred - target)
    circular_diff = np.minimum(diff, 2 * np.pi - diff)
    return np.mean(circular_diff**2, axis=axis)


def circular_mae_loss_np(pred: np.ndarray, target: np.ndarray):
    diff = np.abs(pred - target)
    circular_diff = np.minimum(diff, 2 * np.pi - diff)
    return np.mean(circular_diff)


@app.command()
def simulate(sim_path: str = "antenna_array.py"):
    image_name = "openems-image"

    cmd = f"""
	docker run -it --rm \
		-e DISPLAY=host.docker.internal:0 \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v ./src:/app/ \
		-v /tmp:/tmp/ \
		{image_name} \
		python3 /app/{sim_path}
	"""
    sp.run(cmd, shell=True)


if __name__ == "__main__":
    app()
