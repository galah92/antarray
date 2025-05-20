import json
import math
import pickle
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import typer
from matplotlib.ticker import FormatStrFormatter
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

import analyze
from generate_dataset import ff_from_phase_shifts, steering_repr

DEFAULT_DATASET_PATH: Path = Path.cwd() / "dataset" / "rand_bf_4d_160k_prop.h5"
DEFAULT_EXPERIMENTS_PATH: Path = Path.cwd() / "experiments"
DEFAULT_MODEL_NAME = "model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


class Hdf5Dataset(Dataset):
    """
    Dataset class for loading radiation patterns and phase shift matrices from an HDF5 file.
    """

    def __init__(
        self,
        dataset_file: Path,
        indices,
        stats=None,
        add_fft=False,
        unet_depth: int | None = None,
    ):
        self.dataset_file = dataset_file
        self.h5f = None  # Lazy loading of the HDF5 file
        self.indices = indices
        self.stats = stats
        self.add_fft = add_fft

        self.padding_values = None
        if unet_depth:
            H, W = (180, 180)
            divisor = 2**unet_depth
            target_H = math.ceil(H / divisor) * divisor
            target_W = math.ceil(W / divisor) * divisor
            pad_height = target_H - H
            pad_width = target_W - W
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            self.padding_values = (pad_left, pad_right, pad_top, pad_bottom)

    def get_dataset(self):
        if self.h5f is None:
            self.h5f = h5py.File(self.dataset_file, "r")
        return self.h5f

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        h5f = self.get_dataset()
        pattern = torch.from_numpy(h5f["patterns"][actual_idx])
        phase_shift = torch.from_numpy(h5f["labels"][actual_idx])

        pattern = pattern.unsqueeze(0)  # Add channel dimension for CNN
        pattern = pattern.clamp(min=0)  # Set negative values to 0

        if self.stats is not None:
            pattern_mean = self.stats["pattern_mean"]
            pattern_std = self.stats["pattern_std"]
            pattern = (pattern - pattern_mean) / (pattern_std + 1e-6)
        else:
            pattern = pattern / 30  # Normalize

        if self.add_fft:
            fft = torch.fft.fft2(pattern, norm="ortho")
            fft_amp, fft_phase = torch.abs(fft), torch.angle(fft)
            if self.stats is not None:
                fft_amp_mean = self.stats["fft_amp_mean"]
                fft_amp_std = self.stats["fft_amp_std"]
                fft_phase_mean = self.stats["fft_phase_mean"]
                fft_phase_std = self.stats["fft_phase_std"]
                fft_amp = (fft_amp - fft_amp_mean) / (fft_amp_std + 1e-6)
                fft_phase = (fft_phase - fft_phase_mean) / (fft_phase_std + 1e-6)
            pattern = torch.cat([pattern, fft_amp, fft_phase], dim=0)

        if self.padding_values:
            pattern = F.pad(pattern, self.padding_values, mode="reflect")

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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_att = self.conv1(x_cat)
        return self.sigmoid(x_att)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, attention_type="none"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        self.attention = None
        if attention_type == "se":
            self.attention = SEBlock(out_channels)
        elif attention_type == "cbam":
            self.attention = CBAM(out_channels)
        elif attention_type is not None and attention_type != "none":
            raise ValueError(f"Unknown attention type: {attention_type}")

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
        x = self.block(x)
        if self.attention:
            x = self.attention(x)
        x += self.shortcut(residual)
        x = self.relu(x)
        return x


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal (e.g., from upsampled decoder) [B, F_g, H, W]
            x: Input signal (e.g., from encoder skip connection) [B, F_l, H', W']
               where potentially H' >= H and W' >= W
        Returns:
            Attention map psi [B, 1, H, W]
        """
        g1 = self.W_g(g)  # Shape: [B, F_int, H, W]
        x1 = self.W_x(x)  # Shape: [B, F_int, H', W']

        # Crop x1 spatially to match g1
        # Calculate padding differences (should be non-negative)
        diff_y = x1.size()[2] - g1.size()[2]
        diff_x = x1.size()[3] - g1.size()[3]

        if diff_y < 0 or diff_x < 0:
            # This implies g is spatially larger than x, which is unusual for this setup.
            # Could happen with odd input sizes and specific ConvTranspose settings.
            # Option 1: Pad g1 (less common)
            # Option 2: Resize g1 down to x1 using interpolation (potential info loss)
            # Option 3: Raise error, indicating an issue in network design / layer dims
            # Let's try resizing g1 down as a fallback, but print a warning.
            print(f"Warning: AttentionGate resizing g1 {g1.shape} to x1 {x1.shape}")
            g1 = F.interpolate(
                g1, size=x1.size()[2:], mode="bilinear", align_corners=False
            )
            # Set diffs to 0 as they are now aligned
            diff_y, diff_x = 0, 0
            # If using this fallback, x1 is not cropped below.
            x1_cropped = x1
        else:
            # Standard case: Crop x1
            x1_cropped = x1[
                :,
                :,
                diff_y // 2 : x1.size()[2] - (diff_y - diff_y // 2),
                diff_x // 2 : x1.size()[3] - (diff_x - diff_x // 2),
            ]

        # Add aligned tensors
        psi = self.relu(g1 + x1_cropped)
        psi = self.psi(psi)  # Calculate attention map [B, 1, H, W]

        return psi  # Return only the attention map


class ConvBlock(nn.Module):
    """A block consisting of two convolutions with BN and ReLU, optionally followed by attention."""

    def __init__(self, in_channels, out_channels, attention_type="none"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        self.attention = None
        if attention_type == "se":
            self.attention = SEBlock(out_channels)
        elif attention_type == "cbam":
            self.attention = CBAM(out_channels)
        elif attention_type is not None and attention_type != "none":
            raise ValueError(f"Unknown attention type: {attention_type}")

    def forward(self, x):
        x = self.block(x)
        if self.attention:
            x = self.attention(x)
        x = self.relu(x)
        return x


class DownBlock(nn.Module):
    """Downscaling with maxpool then ConvBlock"""

    def __init__(self, in_channels, out_channels, attention_type="none"):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, attention_type),
        )

    def forward(self, x):
        return self.pool_conv(x)


class UpBlock(nn.Module):
    """Upscaling then ConvBlock, handles optional skip connection."""

    def __init__(
        self,
        in_channels,
        skip_channels,  # 0 if no skip
        out_channels,
        use_attention_gate=False,
        attention_type="none",
    ):
        super().__init__()
        upsampled_channels = in_channels
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.att_gate = None
        if use_attention_gate and skip_channels > 0:
            F_g, F_l, F_int = upsampled_channels, skip_channels, skip_channels // 2
            self.att_gate = AttentionGate(F_g=F_g, F_l=F_l, F_int=F_int)

        concat_in_channels = upsampled_channels + skip_channels
        self.conv = ConvBlock(concat_in_channels, out_channels, attention_type)

    def forward(self, x, x_skip=None):
        x = self.up(x)

        if x_skip is not None:
            if x.size()[2:] != x_skip.size()[2:]:
                print(f"Align skip connection {x_skip.shape} to upsampled {x.shape}")
                x_skip = F.interpolate(x_skip, size=x.size()[2:], mode="bilinear")

            if self.att_gate is not None:
                psi = self.att_gate(g=x, x=x_skip)
                x_skip = x_skip * psi

            x = torch.cat([x, x_skip], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        down_depth=4,
        bottleneck_depth=1,
        bottleneck_type=ConvBlock,
        up_depth=4,
        attention_type="none",
        use_attention_gate=False,
        out_shape=(16, 16),
    ):
        super().__init__()
        if down_depth < 1 or up_depth < 1:
            raise ValueError("Depths must be at least 1.")
        self.in_channels = in_channels
        self.out_channels = out_channels

        in_ch, out_ch = in_channels, base_channels
        self.inc = ConvBlock(in_ch, out_ch, attention_type)

        self.downs = nn.ModuleList()
        for i in range(down_depth):
            in_ch, out_ch = out_ch, out_ch * 2  # Double the channels
            self.downs.append(DownBlock(in_ch, out_ch, attention_type))

        self.bottleneck = nn.Sequential()
        for i in range(bottleneck_depth):
            in_ch, out_ch = out_ch, out_ch  # Same channels in bottleneck
            block = bottleneck_type(in_ch, out_ch, attention_type=attention_type)
            self.bottleneck.append(block)

        self.ups = nn.ModuleList()
        for j in range(up_depth):
            in_ch = out_ch
            # Half the channels in the decoder up to a minimum of base_channels
            in_ch, out_ch = out_ch, max(out_ch // 2, base_channels)
            # Set skip channels to 0 if no skip connection exists
            # FIXME: a possible bug, skip_ch should be in_ch, not out_ch
            skip_ch = out_ch if j < down_depth + 1 else 0  # +1 for inc

            self.ups.append(
                UpBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    use_attention_gate=use_attention_gate,
                    attention_type=attention_type if not use_attention_gate else "none",
                )
            )

        self.final_conv = final_block(out_ch, out_channels, out_shape)

    def forward(self, x):
        x = self.inc(x)
        skip_connections = [x]

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

        x = self.bottleneck(x)

        for i, up in enumerate(self.ups):
            # FIXME: (i + 2) is probably due to the bug above in skip_ch calculation
            skip = skip_connections[-(i + 2)] if i < len(skip_connections) else None
            x = up(x, skip)

        logits = self.final_conv(x)
        if self.out_channels == 1:
            logits = logits.squeeze(1)

        return logits


def final_block(in_channels, out_channels, out_shape):
    """Final output block for the model."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(out_shape),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
    )


class DecoderBlock(nn.Module):
    """Upsamples then applies a ConvBlock."""

    def __init__(self, in_channels, out_channels, attention_type="none"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels, attention_type)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class ConvAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        down_depth=4,  # Depth of the encoder
        bottleneck_depth=2,
        bottleneck_type=ConvBlock,
        decoder_depth=3,
        attention_type="none",  # Attention in ConvBlocks ('none', 'se', 'cbam')
        out_shape=(16, 16),  # Target spatial size
    ):
        """
        Convolutional Autoencoder-style architecture without skip connections.

        Args:
            in_channels: Input channels.
            out_channels: Output channels (1 for phase).
            base_channels: Starting channel count for the encoder.
            down_depth: Number of downsampling stages in the encoder.
            bottleneck_depth: Number of ConvBlocks applied at the bottleneck resolution.
            decoder_depth: Number of upsampling stages in the decoder.
            attention_type: Type of attention to use within ConvBlocks.
            out_shape: Final spatial output dimensions (e.g., (16, 16)).
        """
        super().__init__()
        if down_depth < 1 or decoder_depth < 1:
            raise ValueError("Depths must be at least 1.")
        self.in_channels = in_channels
        self.out_channels = out_channels

        in_ch, out_ch = in_channels, base_channels
        self.inc = ConvBlock(in_ch, out_ch, attention_type)

        self.encoder = nn.Sequential()
        for _ in range(down_depth):
            in_ch, out_ch = out_ch, out_ch * 2  # Double the channels
            self.encoder.append(DownBlock(in_ch, out_ch, attention_type))

        self.bottleneck = nn.Sequential()
        for _ in range(bottleneck_depth):
            in_ch, out_ch = out_ch, out_ch  # Same channels in bottleneck
            block = bottleneck_type(out_ch, out_ch, attention_type=attention_type)
            self.bottleneck.append(block)

        self.decoder = nn.Sequential()
        for _ in range(decoder_depth):
            # Half the channels in the decoder up to a minimum of base_channels
            in_ch, out_ch = out_ch, max(out_ch // 2, base_channels)
            self.decoder.append(DecoderBlock(in_ch, out_ch, attention_type))

        self.final_conv = final_block(out_ch, out_channels, out_shape)

    def forward(self, x):
        x = self.inc(x)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        logits = self.final_conv(x)
        if self.out_channels == 1:
            logits = logits.squeeze(1)

        return logits


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


class FileIO:
    """
    This is a simple wrapper around the standard output stream.
    It captures all print statements and writes them to a file.
    Based on https://stackoverflow.com/a/14906787/5151909
    """

    def __init__(self, file_path: Path):
        self.stdout = sys.stdout
        self.file = open(file_path, "w+")

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


@app.command()
def run_model(
    experiment: str,
    overwrite: bool = False,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    exps_path: Path = DEFAULT_EXPERIMENTS_PATH,
    batch_size: int = 512,
    n_epochs: int = 200,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    model_type: str = "cae",
    use_fft: bool = True,
    use_stats: bool = True,  # Normalization stats
    use_amp: bool = True,  # Automatic Mixed Precision for training
    benchmark: bool = True,  # CUDA benchmarking
    # U-Net specific parameters
    base_channels: int = 16,
    down_depth: int = 4,
    up_depth: int = 2,
    bottleneck_depth: int = 2,
    attention_type: str = "none",
    use_attention_gate: bool = False,
):
    exp_path = get_experiment_path(experiment, exps_path, overwrite)
    sys.stdout = FileIO(exp_path / "stdout.log")

    train_loader, val_loader, _ = create_dataloaders(
        dataset_path,
        batch_size,
        use_fft,
        use_stats,
        unet_depth=down_depth,
    )

    in_channels = 3 if use_fft else 1
    unet_params = {
        "base_channels": base_channels,
        "down_depth": down_depth,
        "up_depth": up_depth,
        "bottleneck_depth": bottleneck_depth,
        "attention_type": attention_type,
        "use_attention_gate": use_attention_gate,
    }
    model = model_type_to_class(model_type, in_channels=in_channels, **unet_params)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{model_type=} ({n_params:,}), {use_fft=}")

    criterion = circular_mse_loss_torch
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    print(f"{batch_size=}, {n_epochs=}, {lr=}, {weight_decay=}")

    if benchmark and device == "cuda":
        torch.backends.cudnn.benchmark = True

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
        pred_model(
            experiment,
            dataset_path,
            exps_path,
            use_fft=use_fft,
            use_stats=use_stats,
            n_examples=5,
        )


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

    use_amp = use_amp and torch.cuda.is_available()
    scaler = torch.GradScaler(device, enabled=use_amp)
    gpu_warmup(model, (model.in_channels, 192, 192))

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
                        with torch.autocast(device, enabled=use_amp):
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)

                        if device == "cuda":
                            torch.cuda.synchronize()
                        forward_time, curr_time = time.time() - curr_time, time.time()
                        timing_stats["forward_time"] += forward_time

                        loss_time = 0
                        timing_stats["loss_time"] += loss_time

                        if phase == "train":
                            scaler.scale(loss).backward()
                            if device == "cuda":
                                torch.cuda.synchronize()
                            bwd_time, curr_time = (time.time() - curr_time, time.time())
                            timing_stats["backward_time"] += bwd_time

                            scaler.step(optimizer)
                            scaler.update()

                            if device == "cuda":
                                torch.cuda.synchronize()
                            opt_time, curr_time = (time.time() - curr_time, time.time())
                            timing_stats["optimizer_time"] += opt_time

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
                        explicitly_timed += bwd_time + opt_time

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
                            f"optim={optim_zero_time * 1000 + opt_time * 1000:.1f}ms | "
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

            print(
                f"  Avg times per batch: data={avg_data_time:.1f}ms, to_device={avg_to_device:.1f}ms, "
                f"forward={avg_forward_time:.1f}ms, loss={avg_loss_time:.1f}ms, backward={avg_backward_time:.1f}ms, "
                f"optimizer={avg_optimizer_time:.1f}ms, sync={avg_sync_time:.1f}ms, overhead={avg_overhead_time:.1f}ms"
            )

            if early_stopper is not None:
                if early_stopper.early_stop(history["val_loss"][-1]):
                    print("Early stopping...")
                    break

            if scheduler:
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


def create_dataloaders(
    dataset_path: Path,
    batch_size: int,
    use_fft: bool = False,
    use_stats: bool = False,
    unet_depth: int | None = None,
):
    with h5py.File(dataset_path, "r") as h5f:
        ds_size = h5f["patterns"].shape[0]

    train_idx, val_idx, test_idx = get_indices(ds_size, [0.9, 0.05, 0.05])
    train, val, test = len(train_idx), len(val_idx), len(test_idx)
    print(f"dataset={dataset_path}, {use_stats=}, {train=:,}, {val=:,}, {test=:,}")

    stats = calc_normalization_stats(dataset_path, train_idx) if use_stats else None

    ds_kwargs = {"stats": stats, "add_fft": use_fft, "unet_depth": unet_depth}
    train_ds = Hdf5Dataset(dataset_path, train_idx, **ds_kwargs)
    val_ds = Hdf5Dataset(dataset_path, val_idx, **ds_kwargs)
    test_ds = Hdf5Dataset(dataset_path, test_idx, **ds_kwargs)

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


def get_indices(n, split_ratios):
    rng = np.random.default_rng(seed=42)
    indices_range = rng.integers(n, size=n, endpoint=False)

    split_ratios = np.array(split_ratios)
    if not np.isclose(split_ratios.sum(), 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    split_indices = np.cumsum(split_ratios[:-1]) * n
    indices = np.split(indices_range, split_indices.astype(np.int32))
    return indices


def calc_normalization_stats(dataset_path: Path, train_indices: list | np.ndarray):
    """
    Calculates normalization statistics (mean, std) iteratively for original patterns,
    FFT amplitude, and FFT phase over the training set indices.
    """
    # Read patterns from HDF5 file if exists
    stats = {
        "pattern_mean": None,
        "pattern_std": None,
        "fft_amp_mean": None,
        "fft_amp_std": None,
        "fft_phase_mean": None,
        "fft_phase_std": None,
    }
    with h5py.File(dataset_path, "r") as h5f:
        for key in stats:
            stats[key] = h5f["patterns"].attrs.get(key)
        if all(val is not None for val in stats.values()):
            print("Normalization stats already exist in the dataset.")
            print(stats)
            return stats

    pattern_sum = torch.tensor(0.0, dtype=torch.float64)
    pattern_sum_sq = torch.tensor(0.0, dtype=torch.float64)

    with h5py.File(dataset_path, "r") as h5f:
        patterns = h5f["patterns"]
        pattern_count = np.prod(patterns.shape)

        for idx in train_indices:
            pattern = torch.from_numpy(patterns[idx]).to(torch.float64)
            pattern = pattern.clamp(min=0)  # Set negative values to 0
            pattern_sum += torch.sum(pattern)
            pattern_sum_sq += torch.sum(pattern**2)

    # Calculate final original stats
    pattern_mean = pattern_sum / pattern_count
    # Var = E[X^2] - (E[X])^2. Add small epsilon for stability
    pattern_var = (pattern_sum_sq / pattern_count) - (pattern_mean**2)
    # Clamp variance to avoid negative values due to floating point errors
    pattern_std = torch.sqrt(torch.clamp(pattern_var, min=1e-12))

    # Convert to float32 for use in standardization during pass 2
    pattern_mean_f32 = pattern_mean.float()
    pattern_std_f32 = pattern_std.float()
    std_epsilon = 1e-6  # Epsilon for division

    fft_amp_sum = torch.tensor(0.0, dtype=torch.float64)
    fft_amp_sum_sq = torch.tensor(0.0, dtype=torch.float64)
    fft_phase_sum = torch.tensor(0.0, dtype=torch.float64)
    fft_phase_sum_sq = torch.tensor(0.0, dtype=torch.float64)

    with h5py.File(dataset_path, "r") as h5f:
        patterns = h5f["patterns"]

        for idx in train_indices:
            pattern = torch.from_numpy(patterns[idx]).float()
            pattern = pattern.clamp(min=0)  # Set negative values to 0

            # Standardize original pattern using stats from Pass 1
            normalized_pattern = (pattern - pattern_mean_f32) / (
                pattern_std_f32 + std_epsilon
            )

            # Calculate FFT
            fft_input = normalized_pattern.unsqueeze(0)  # Add channel dim
            # Using float32 for FFT is usually fine and faster
            fft_val = torch.fft.fft2(fft_input, norm="ortho")

            # Extract Amp and Phase
            fft_amp = torch.abs(fft_val).to(torch.float64)  # Cast sums to float64
            fft_phase = torch.angle(fft_val).to(torch.float64)

            # Accumulate FFT stats
            fft_amp_sum += torch.sum(fft_amp)
            fft_amp_sum_sq += torch.sum(fft_amp**2)
            fft_phase_sum += torch.sum(fft_phase)
            fft_phase_sum_sq += torch.sum(fft_phase**2)

    # Calculate final FFT stats
    fft_amp_mean = fft_amp_sum / pattern_count
    fft_amp_var = (fft_amp_sum_sq / pattern_count) - (fft_amp_mean**2)
    fft_amp_std = torch.sqrt(torch.clamp(fft_amp_var, min=1e-12))

    fft_phase_mean = fft_phase_sum / pattern_count
    fft_phase_var = (fft_phase_sum_sq / pattern_count) - (fft_phase_mean**2)
    fft_phase_std = torch.sqrt(torch.clamp(fft_phase_var, min=1e-12))

    stats = {
        "pattern_mean": pattern_mean.item(),
        "pattern_std": pattern_std.item(),
        "fft_amp_mean": fft_amp_mean.item(),
        "fft_amp_std": fft_amp_std.item(),
        "fft_phase_mean": fft_phase_mean.item(),
        "fft_phase_std": fft_phase_std.item(),
    }
    print(stats)

    # Store as HDF5 attributes
    with h5py.File(dataset_path, "r+") as h5f:
        for key, value in stats.items():
            h5f["patterns"].attrs[key] = value

    return stats


@app.command()
def pred_model(
    experiment: str,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    exps_path: Path = DEFAULT_EXPERIMENTS_PATH,
    n_examples: int = 5,
    batch_size: int = 128,
    use_fft: bool = True,
    use_stats: bool = True,
    # U-Net specific parameters
    base_channels: int = 16,
    down_depth: int = 4,
    up_depth: int = 2,
    bottleneck_depth: int = 2,
    attention_type: str = "none",
    use_attention_gate: bool = False,
):
    exp_path = get_experiment_path(experiment, exps_path, overwrite=True)

    _, _, test_loader = create_dataloaders(
        dataset_path,
        batch_size,
        use_fft,
        use_stats,
        unet_depth=down_depth,
    )
    test_indices = test_loader.dataset.indices

    with h5py.File(dataset_path, "r") as h5f:
        theta = h5f["theta"][:]
        phi = h5f["phi"][:]
        steering_info = h5f["steering_info"][:]
        steering_info = steering_info[test_indices]

    checkpoint = torch.load(exps_path / experiment / DEFAULT_MODEL_NAME)
    model_type = checkpoint["model_type"]

    in_channels = 3 if use_fft else 1
    unet_params = {
        "base_channels": base_channels,
        "down_depth": down_depth,
        "up_depth": up_depth,
        "bottleneck_depth": bottleneck_depth,
        "attention_type": attention_type,
        "use_attention_gate": use_attention_gate,
    }

    model = model_type_to_class(model_type, in_channels=in_channels, **unet_params)
    model = model.to(device)
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
    loss = losses.mean()

    print(f"{steering_info.shape=}")
    indices_1_beam = np.argwhere(np.isnan(steering_info[:, 0, 1]))
    indices_2_beam = np.argwhere(~np.isnan(steering_info[:, 0, 1]))

    loss_1_beam = circular_mse_loss_np(preds[indices_1_beam], targets[indices_1_beam])
    loss_2_beam = circular_mse_loss_np(preds[indices_2_beam], targets[indices_2_beam])
    print(f"Loss for 1 beam: {loss_1_beam.mean():.4f}")
    print(f"Loss for 2 beams: {loss_2_beam.mean():.4f}")
    print(f"Overall loss: {loss:.4f}")

    steering_info = steering_info[:, :, 0]  # Take only the first beamforming angle

    fig, ax = plt.subplots()
    ax.scatter(*steering_info.T, c=losses, s=16)
    ax.set_xlabel("Theta (degrees)")
    ax.set_ylabel("Phi (degrees)")
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.set_title(f"Loss by steering angle ({losses.size} samples) | {loss:.4f} avg")
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Circular MSE Loss")
    fig.savefig(exp_path / "steer_loss.png", dpi=600, bbox_inches="tight")
    print(f"Steering loss plot saved to {exp_path / 'steer_loss.png'}")


def model_type_to_class(
    model_type: str,
    in_channels: int = 1,
    base_channels=32,
    down_depth=4,
    up_depth=3,
    bottleneck_depth=1,
    attention_type="none",
    use_attention_gate=False,
):
    print(locals())
    if model_type == "cnn":
        return ConvModel(in_channels=in_channels)
    elif model_type == "resnet":
        return resnet18()
    elif model_type == "unet":
        return UNet(
            in_channels=in_channels,
            base_channels=base_channels,
            down_depth=down_depth,
            up_depth=up_depth,
            bottleneck_depth=bottleneck_depth,
            attention_type=attention_type,
            use_attention_gate=use_attention_gate,
            bottleneck_type=ResidualBlock,
        )
    elif model_type == "cae":
        return ConvAutoencoder(
            in_channels=in_channels,
            base_channels=base_channels,
            down_depth=down_depth,
            bottleneck_depth=bottleneck_depth,
            decoder_depth=up_depth,
            attention_type=attention_type,
            bottleneck_type=ResidualBlock,
        )
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
    analyze.plot_ff_2d(theta, phi, label_ff, title=title, ax=axs[0, 1])
    title = "Predicted 2D Radiation Pattern"
    analyze.plot_ff_2d(theta, phi, pred_ff, title=title, ax=axs[1, 1])

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


if __name__ == "__main__":
    app()
