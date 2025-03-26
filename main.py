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
from torch.utils.data import DataLoader, Dataset, random_split

import analyze
from generate_dataset import ff_from_phase_shifts, steering_repr

DEFAULT_DATASET_PATH: Path = Path.cwd() / "dataset" / "rand_bf_2d_4k.h5"
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

    def __init__(self, dataset_file: Path, use_steering_info=False):
        self.dataset_file = dataset_file
        self.h5f = None  # Lazy loading of the HDF5 file
        self.use_steering_info = use_steering_info

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

        if self.use_steering_info:
            steering_info = torch.from_numpy(h5f["steering_info"][idx])
            return pattern, phase_shift, steering_info

        return pattern, phase_shift


class ConvModel(nn.Module):
    def __init__(
        self,
        input_channels=1,  # Number of input channels (1 for single radiation pattern)
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
        ch = [input_channels] + conv_channels
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
        x = torch.tanh(x) * torch.pi  # Scale to [-1, 1] and then to [-pi, pi]
        x = x.view(-1, *self.out_shape)
        return x


class InverseConvModel(nn.Module):
    def __init__(
        self,
        input_channels=3,  # Number of input channels (2 for radiation pattern fft)
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
        ch = [input_channels] + conv_channels
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
        x_fft = torch.fft.fft2(x.squeeze(1))
        x_fft_amplitude = torch.abs(x_fft).unsqueeze(1)
        x_fft_phase = torch.angle(x_fft).unsqueeze(1)
        x_fft_features = torch.cat([x, x_fft_amplitude, x_fft_phase], dim=1)

        x = self.conv(x_fft_features)
        x = self.fc(x)
        x = torch.tanh(x) * torch.pi  # Scale to [-1, 1] and then to [-pi, pi]
        x = x.view(-1, *self.out_shape)
        return x


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
        out = torch.tanh(out) * torch.pi  # Scale to [-1, 1] and then to [-pi, pi]
        out = out.view(-1, *self.out_shape)
        return out


def resnet18(out_shape=(16, 16)):
    return ResNet(ResidualBlock, [2, 2, 2, 2], out_shape)


def resnet34(out_shape=(16, 16)):
    return ResNet(ResidualBlock, [3, 4, 6, 3], out_shape)


class SpectralSpatialModel(nn.Module):
    def __init__(
        self,
        input_channels=1,
        in_shape=(180, 180),
        out_shape=(16, 16),
        spatial_channels=[32, 64, 128, 256, 512],
        spectral_channels=[32, 64, 128, 256, 512],
        fc_units=[2048, 1024],
    ):
        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

        # Spatial processing branch - standard convolutional layers
        self.spatial_branch = nn.Sequential()
        ch_spatial = [input_channels] + spatial_channels
        for i in range(1, len(ch_spatial)):
            self.spatial_branch.append(
                nn.Conv2d(
                    ch_spatial[i - 1], ch_spatial[i], kernel_size=3, stride=1, padding=1
                )
            )
            self.spatial_branch.append(nn.BatchNorm2d(ch_spatial[i]))
            self.spatial_branch.append(nn.ReLU())
            self.spatial_branch.append(nn.MaxPool2d(2))

        # Spectral processing branch (for FFT features)
        self.spectral_branch = nn.Sequential()
        # FFT produces complex output with real and imaginary parts, so double the channels
        ch_spectral = [input_channels * 2] + spectral_channels
        for i in range(1, len(ch_spectral)):
            self.spectral_branch.append(
                nn.Conv2d(
                    ch_spectral[i - 1],
                    ch_spectral[i],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            self.spectral_branch.append(nn.BatchNorm2d(ch_spectral[i]))
            self.spectral_branch.append(nn.ReLU())
            self.spectral_branch.append(nn.MaxPool2d(2))

        # Calculate feature sizes after convolutional layers
        spatial_feature_size = (
            spatial_channels[-1]
            * (in_shape[0] // (2 ** len(spatial_channels)))
            * (in_shape[1] // (2 ** len(spatial_channels)))
        )
        spectral_feature_size = (
            spectral_channels[-1]
            * (in_shape[0] // (2 ** len(spectral_channels)))
            * (in_shape[1] // (2 ** len(spectral_channels)))
        )

        # Combine features from both branches
        combined_size = spatial_feature_size + spectral_feature_size

        # Fully connected layers for final prediction
        self.fc_layers = nn.Sequential()
        fc_sizes = [combined_size] + fc_units + [np.prod(out_shape)]
        for i in range(1, len(fc_sizes)):
            self.fc_layers.append(nn.Linear(fc_sizes[i - 1], fc_sizes[i]))
            if i < len(fc_sizes) - 1:
                self.fc_layers.append(nn.ReLU())
                self.fc_layers.append(nn.Dropout(0.5))

    def forward(self, x):
        batch_size = x.size(0)

        # Process spatial features
        spatial_features = self.spatial_branch(x)
        spatial_features = spatial_features.view(batch_size, -1)

        # Compute FFT and process spectral features
        # Convert to complex tensor, compute 2D FFT
        x_complex = torch.complex(x.squeeze(1), torch.zeros_like(x.squeeze(1)))
        x_fft = torch.fft.fft2(x_complex)

        # Extract amplitude and phase information
        x_fft_amplitude = torch.abs(x_fft).unsqueeze(1)
        x_fft_phase = torch.angle(x_fft).unsqueeze(1)

        # Concatenate amplitude and phase as channels
        x_fft_features = torch.cat([x_fft_amplitude, x_fft_phase], dim=1)

        # Process spectral features
        spectral_features = self.spectral_branch(x_fft_features)
        spectral_features = spectral_features.view(batch_size, -1)

        # Combine features from both branches
        combined_features = torch.cat([spatial_features, spectral_features], dim=1)

        # Final fully connected layers
        outputs = self.fc_layers(combined_features)
        outputs = torch.tanh(outputs) * torch.pi  # Scale outputs to [-pi, pi]

        # Reshape to desired output shape
        outputs = outputs.view(batch_size, *self.out_shape)

        return outputs


def cosine_angular_loss_torch(inputs: torch.Tensor, targets: torch.Tensor):
    return torch.mean(1 - torch.cos(inputs - targets))


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
    n_epochs: int = 100,
    lr: float = 1e-4,
    model_type: str = "cnn",
):
    exp_path = get_experiment_path(experiment, exps_path, overwrite)

    train_loader, val_loader, test_loader = create_dataloaders(dataset_path, batch_size)

    model = model_type_to_class(model_type)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{model_type=} ({n_params:,}), {batch_size=}, {n_epochs=}")

    criterion = circular_mse_loss_torch
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

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
        pred_model(experiment, dataset_path, exps_path, num_examples=5)


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
):
    interrupted = False
    since = time.time()
    history = {"train_loss": [], "val_loss": [], "lr": []}

    model = model.to(device)
    best_val_loss, best_model_wts = float("inf"), None

    try:
        for epoch in range(n_epochs):
            lr = scheduler.get_last_lr()[0]
            print()
            print(f"Epoch {epoch:03}/{n_epochs:03} | {lr=:1.1e}")

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                    dataloader = train_loader
                else:
                    model.eval()  # Set model to evaluate mode
                    dataloader = val_loader

                running_loss = 0.0
                n_batches = len(dataloader)

                for i, (inputs, targets) in enumerate(dataloader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        # Backward + optimize only in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)

                    if phase == "train" and i % log_interval == 0:
                        print(f"Batch {i:03}/{n_batches:03} | loss={loss.item():.03f}")

                epoch_loss = running_loss / len(dataloader.dataset)
                loss_str = f"{phase}_loss={epoch_loss:.03f}"
                print(f"Epoch {epoch:03}/{n_epochs:03} | {loss_str}")

                if phase == "train":
                    history["train_loss"].append(epoch_loss)
                    if scheduler:
                        history["lr"].append(lr)
                else:
                    history["val_loss"].append(epoch_loss)

                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        best_model_wts = model.state_dict().copy()

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
        # Load best model weights
        if best_model_wts:
            model.load_state_dict(best_model_wts)

        elapsed = time.time() - since
        print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
        print(f"Best val loss: {best_val_loss:.4f}")

    return model, history, interrupted


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
    elif model_type == "spectral_spatial":
        return SpectralSpatialModel()
    elif model_type == "inv_cnn":
        return InverseConvModel()
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

    xticks = range(0, 100 + 1, 10)
    yticks = np.arange(0, 4 + 0.1, 0.1)

    axs[0].plot(history["train_loss"], "b-", label="Training Loss")
    axs[0].plot(history["val_loss"], "r-", label="Validation Loss")
    axs[0].set_xlim(xticks[0], xticks[-1])
    axs[0].set_xticks(xticks)
    axs[0].set_ylim(yticks[0], yticks[-1])
    # axs[0].set_yticks(yticks)
    axs[0].set_title("Training and Validation Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    if "lr" in history and history["lr"]:
        axs[1].plot(history["lr"], "g-")
        axs[1].set_xlim(xticks[0], xticks[-1])
        axs[1].set_xticks(xticks)
        axs[1].set_yscale("log")
        axs[1].yaxis.set_major_formatter(FormatStrFormatter("%0.1e"))
        axs[1].set_title("Learning Rate")
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Learning Rate")
        axs[1].grid(True)

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


def create_dataloaders(dataset_path: Path, batch_size: int):
    ds = Hdf5Dataset(dataset_path)

    # Split data into train, validation, and test sets
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(ds, [0.9, 0.05, 0.05], generator=gen)

    train, val, test = len(train_ds), len(val_ds), len(test_ds)
    print(f"dataset={dataset_path}, {train=:,}, {val=:,}, {test=:,}")

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


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
        patterns, labels = h5f["patterns"], h5f["labels"]
        theta, phi = h5f["theta"], h5f["phi"]
        steering_info = h5f["steering_info"]

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
