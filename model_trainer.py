import pickle
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import typer
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split

import analyze
import generate_dataset
from generate_dataset import load_dataset

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


class RadiationPatternDataset(Dataset):
    """
    PyTorch Dataset for radiation patterns and phase shifts.
    """

    def __init__(self, patterns, phase_shifts, transform=None):
        """
        Initialize dataset.

        Parameters:
        -----------
        patterns : numpy.ndarray
            Radiation patterns with shape (n_samples, n_phi, n_theta)
        phase_shifts : numpy.ndarray
            Phase shift matrices with shape (n_samples, xn, yn)
        transform : callable, optional
            Optional transform to be applied to the patterns
        """
        self.patterns = patterns
        self.phase_shifts = phase_shifts
        self.transform = transform

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        phase_shift = self.phase_shifts[idx]

        # Convert to PyTorch tensors
        pattern = torch.tensor(pattern, dtype=torch.float32)
        phase_shift = torch.tensor(phase_shift, dtype=torch.float32)

        # Expand dims to create a channel dimension for the CNN
        pattern = pattern.unsqueeze(0)  # Shape becomes (1, n_phi, n_theta)

        if self.transform:
            pattern = self.transform(pattern)

        return pattern, phase_shift


class PhaseShiftModel(nn.Module):
    def __init__(
        self,
        input_channels=1,  # Number of input channels (1 for single radiation pattern)
        input_height=180,  # Number of phi values
        input_width=180,  # Number of theta values
        output_size=(16, 16),  # Size of the output phase shift matrix (xn, yn)
    ):
        super().__init__()

        self.output_size = output_size

        self.encoder = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Fifth convolutional block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Calculate the size of the feature maps after encoding
        feature_size = input_height // 32  # After 5 MaxPool2d with stride=2
        feature_width = input_width // 32

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * feature_size * feature_width, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, output_size[0] * output_size[1]),
        )

    def forward(self, x):
        """Forward pass through the network."""
        x = self.encoder(x)
        x = self.fc(x)
        # Reshape to the desired output size (batch_size, xn, yn)
        x = x.view(-1, self.output_size[0], self.output_size[1])
        return x


import torch.nn.functional as F


class PhaseShiftPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.global_pool = nn.AdaptiveAvgPool2d(
            1
        )  # Reduces feature size while preserving information
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 16 * 16)  # Output phase shifts for 16x16 antenna array

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation since phase angles are continuous

        return x.view(-1, 16, 16)  # Output shape: (batch, 16, 16) for phase shifts


import torch
import torch.nn as nn
import torch.nn.functional as F


class LargePhaseShiftPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        # First block: Conv + BatchNorm + ReLU
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Second block: Conv + BatchNorm + ReLU
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Third block: Conv + BatchNorm + ReLU
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Fourth block: Conv + BatchNorm + ReLU
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Fifth block: Conv + BatchNorm + ReLU
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 16 * 16)  # Output for 16x16 antenna array

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))

        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))

        # Block 5
        x = F.relu(self.bn5(self.conv5(x)))

        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.view(-1, 16, 16)  # Output shape: (batch, 16, 16) for phase shifts


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (identity mapping)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        identity = self.skip(x)

        # First convolution block
        out = F.relu(self.bn1(self.conv1(x)))

        # Second convolution block
        out = self.bn2(self.conv2(out))

        # Adding residual connection (skip connection)
        out += identity
        out = F.relu(out)

        return out


class ResNetPhaseShiftPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet blocks
        self.block1 = ResNetBlock(64, 128)
        self.block2 = ResNetBlock(128, 256)
        self.block3 = ResNetBlock(256, 512)
        self.block4 = ResNetBlock(512, 1024)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers for output
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 16 * 16)  # Output for 16x16 antenna array

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Initial convolution + batchnorm
        x = F.relu(self.bn1(self.conv1(x)))

        # ResNet blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x.view(-1, 16, 16)  # Output shape: (batch, 16, 16) for phase shifts


def cosine_angular_loss_torch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return torch.mean(1 - torch.cos(inputs - targets))


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    n_epochs=25,
    device="cuda",
    log_interval=10,
    clip_grad=1.0,
):
    """
    Train the model.

    Parameters:
    -----------
    model : nn.Module
        The PyTorch model to train
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    criterion : nn.Module
        Loss function
    optimizer : torch.optim
        Optimizer
    scheduler : torch.optim.lr_scheduler, optional
        Learning rate scheduler
    n_epochs : int
        Number of epochs to train
    device : str
        Device to use for training ('cuda' or 'cpu')
    log_interval : int
        How often to log progress
    clip_grad : float, optional
        Maximum norm of the gradients for clipping

    Returns:
    --------
    model : nn.Module
        Trained model
    history : dict
        Training history
    """
    since = time.time()
    history = {"train_loss": [], "val_loss": [], "lr": []}

    # Move model to device
    model = model.to(device)
    best_val_loss = float("inf")
    best_model_wts = None

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs} lr={scheduler.get_last_lr()[0]}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0

            # Iterate over data
            for i, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only in train phase
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    # Backward + optimize only in training phase
                    if phase == "train":
                        loss.backward()
                        # Clip gradients
                        if clip_grad is not None:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), clip_grad
                            )
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)

                # Print progress
                if phase == "train" and i % log_interval == 0:
                    print(f"Batch {i}/{len(dataloader)}, Loss: {loss.item():.4f}")

            epoch_loss = running_loss / len(dataloader.dataset)

            # Print epoch results
            print(f"{phase} Loss: {epoch_loss:.4f}")

            # Store history
            if phase == "train":
                history["train_loss"].append(epoch_loss)
                if scheduler:
                    history["lr"].append(scheduler.get_last_lr()[0])
            else:
                history["val_loss"].append(epoch_loss)

                # Save best model
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = model.state_dict().copy()

        # Step the scheduler
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(history["val_loss"][-1])  # Pass the validation loss
            else:
                scheduler.step()

        print()

    # Load best model weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val loss: {best_val_loss:.4f}")

    return model, history


def evaluate_model(
    model,
    test_loader,
    theta,
    phi,
    device="cuda",
    num_examples=5,
    save_dir=None,
):
    """
    Evaluate the model and visualize results.

    Parameters:
    -----------
    model : nn.Module
        Trained PyTorch model
    test_loader : DataLoader
        DataLoader for test data
    device : str
        Device to use ('cuda' or 'cpu')
    num_examples : int
        Number of examples to visualize
    save_dir : str or Path, optional
        Directory to save visualization plots

    Returns:
    --------
    metrics : dict
        Evaluation metrics
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Move predictions to CPU for numpy conversion
            preds = outputs.cpu().numpy()
            targets = targets.numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    # Convert to numpy arrays
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Calculate metrics
    # For phase values, we need to handle the circular nature
    phase_diff = all_preds - all_targets
    # Wrap to [-pi, pi]
    phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

    # Calculate MSE and MAE
    mse = np.mean(phase_diff**2)
    mae = np.mean(np.abs(phase_diff))

    metrics = {
        "mse": mse,
        "mae": mae,
        "rmse": np.sqrt(mse),
    }

    print("Evaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")

    # Visualize a few examples
    if num_examples > 0:
        indices = np.random.choice(len(all_preds), num_examples, replace=False)

        for idx in indices:
            title = f"Prediction Example {idx}"
            filepath = save_dir / f"prediction_example_{idx}.png"
            pred, target = all_preds[idx], all_targets[idx]
            compare_phase_shifts(pred, target, theta, phi, title, filepath)

            print(f"Prediction example saved to {filepath}")

    return metrics


def compare_phase_shifts(
    output,
    label,
    theta,
    phi,
    title: str | None = None,
    filepath: Path | None = None,
):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    analyze.plot_phase_shifts(label, title="Ground Truth Phase Shifts", ax=axs[0, 0])
    analyze.plot_phase_shifts(output, title="Predicted Phase Shifts", ax=axs[0, 1])

    diff = output - label
    # diff = np.arctan2(np.sin(output - label), np.cos(output - label))
    analyze.plot_phase_shifts(diff, title="Phase Shift Error", ax=axs[0, 2])

    label_ff = generate_dataset.ff_from_phase_shifts(output)
    axs[1, 0].remove()
    axs[1, 0] = fig.add_subplot(2, 3, 4, projection="3d")
    title = "Ground Truth Far Field Pattern"
    analyze.plot_ff_3d(theta, phi, label_ff, title=title, ax=axs[1, 0])

    output_ff = generate_dataset.ff_from_phase_shifts(label)
    axs[1, 1].remove()
    axs[1, 1] = fig.add_subplot(2, 3, 5, projection="3d")
    title = "Predicted Far Field Pattern"
    analyze.plot_ff_3d(theta, phi, output_ff, title=title, ax=axs[1, 1])

    axs[1, 2].remove()

    if title is not None:
        fig.suptitle(title)
    fig.set_tight_layout(True)

    if filepath:
        fig.savefig(filepath, dpi=600, bbox_inches="tight")


def visualize_training_history(history, save_path=None):
    """
    Visualize training history and optionally save the plot.

    Parameters:
    -----------
    history : dict
        Dictionary containing training history
    save_path : str or Path, optional
        Path to save the plot. If None, plot will be shown.
    """
    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot training and validation loss
    epochs = range(1, len(history["train_loss"]) + 1)
    axs[0].plot(epochs, history["train_loss"], "b-", label="Training Loss")
    axs[0].plot(epochs, history["val_loss"], "r-", label="Validation Loss")
    axs[0].set_title("Training and Validation Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    # Plot learning rate
    if "lr" in history and history["lr"]:
        axs[1].plot(epochs, history["lr"], "g-")
        axs[1].set_title("Learning Rate")
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Learning Rate")
        axs[1].grid(True)

    fig.set_tight_layout(True)

    if save_path:
        # Ensure the directory exists
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to {save_path}")


DEFAULT_MODELS_DIR = Path.cwd() / "experiments"
DEFAULT_MODELS_DIR.mkdir(exist_ok=True, parents=True)
DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "phase_shift_prediction_model.pth"

DEFAULT_DATA_DIR = Path.cwd() / "dataset"
DEFAULT_DATA_DIR.mkdir(exist_ok=True, parents=True)
DEFAULT_DATASET_PATH = DEFAULT_DATA_DIR / "rand_bf_2d.h5"


@app.command()
def pred_beamforming(
    theta_steer: int = 0,
    phi_steer: int = 0,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
):
    dataset = load_dataset(dataset_path)
    patterns, labels = dataset["patterns"], dataset["labels"]
    theta, phi = dataset["theta"], dataset["phi"]

    # Find the index of the given steering angles: https://stackoverflow.com/a/25823710/5151909
    steer_angles = (theta_steer, phi_steer)
    idx = (dataset["steering_info"] == steer_angles).all(axis=1).nonzero()[0].min()

    pattern = patterns[idx][None, None, ...]  # Add batch and channel dimensions
    pattern[pattern < 0] = 0  # Set negative values to 0
    pattern = pattern / 20  # Normalize
    label = labels[idx]

    checkpoint = torch.load(model_path)
    model = PhaseShiftModel()
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(pattern)).numpy().squeeze()

    title = (f"Prediction Example {idx} (θ={theta_steer:.1f}°, φ={phi_steer:.1f}°)",)
    filepath = (f"prediction_example_{idx}_t{theta_steer}_p{phi_steer}.png",)
    compare_phase_shifts(output, label, theta, phi, title, filepath)


DEFAULT_DATASET_PATH: Path = Path.cwd() / "dataset" / "rand_bf_2d.h5"
DEFAULT_OUTPUT_DIR: Path = Path.cwd() / "experiments"


@app.command()
def run_cnn(
    experiment: str,
    overwrite: bool = False,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 128,
    n_epochs: int = 100,
    lr: float = 1e-4,
):
    output_path = output_dir / experiment
    output_path.mkdir(exist_ok=overwrite, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)

    patterns, labels = dataset["patterns"], dataset["labels"]

    patterns[patterns < 0] = 0  # Set negative values to 0
    patterns = patterns / 20  # Normalize

    # Split data into train, validation, and test sets
    ds = RadiationPatternDataset(patterns, labels)
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(ds, [0.8, 0.1, 0.1], generator=gen)

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=4)

    model = PhaseShiftModel()
    # model = PhaseShiftPredictor()
    # model = LargePhaseShiftPredictor()
    # model = ResNetPhaseShiftPredictor()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # Define loss function and optimizer
    criterion = cosine_angular_loss_torch
    # Use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Train model
    print("Starting training...")
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
        n_epochs=n_epochs,
        device=device,
    )

    # Save model
    model_save_path = output_path / "phase_shift_prediction_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        model_save_path,
    )
    print(f"Model saved to {model_save_path}")

    # Visualize and save training history
    visualize_training_history(history, save_path=output_path / "training_history.png")

    # Evaluate model and save prediction examples
    print("Evaluating model on test set...")
    metrics = evaluate_model(
        model,
        test_loader,
        theta=dataset["theta"],
        phi=dataset["phi"],
        device=device,
        num_examples=5,
        save_dir=output_path,
    )

    # Save metrics
    metrics_path = output_path / "metrics.txt"
    with open(metrics_path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Metrics saved to {metrics_path}")

    return model, history, metrics


@app.command()
def run_knn(
    experiment: str,
    overwrite: bool = False,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n_neighbors: int = 5,
):
    output_path = output_dir / experiment
    output_path.mkdir(exist_ok=overwrite, parents=True)

    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)

    patterns, labels = dataset["patterns"], dataset["labels"]
    theta, phi = dataset["theta"], dataset["phi"]
    steering_info = dataset["steering_info"]

    # Flatten the features
    patterns = patterns.reshape(patterns.shape[0], -1)
    labels = labels.reshape(labels.shape[0], -1)

    patterns[patterns < 0] = 0  # Set negative values to 0
    patterns = patterns / 20  # Normalize

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

    loss = cosine_angular_loss_np(y_pred, y_test)
    print(f"Val loss: {loss:.4f}")

    # save y_pred to h5 file
    with h5py.File(output_path / "knn_pred.h5", "w") as h5f:
        h5f.create_dataset("y_pred", data=y_pred)
        h5f.create_dataset("y_test", data=y_test)
        h5f.create_dataset("loss", data=loss)
        h5f.create_dataset("steering_info", data=steering_info[idx_test])
        h5f.create_dataset("theta", data=theta)
        h5f.create_dataset("phi", data=phi)

    mode = "wb" if overwrite else "xb"
    with (output_path / "knn_model.pkl").open(mode) as f:
        pickle.dump(knn, f)


@app.command()
def analyze_knn_pred(idx: int | None = None):
    output_dir = Path.cwd() / "experiments"

    with h5py.File(output_dir / "knn_pred.h5", "r") as h5f:
        y_pred = h5f["y_pred"][:].reshape(-1, 16, 16)
        y_test = h5f["y_test"][:].reshape(-1, 16, 16)
        steering_info = h5f["steering_info"][:]
        theta, phi = h5f["theta"][:], h5f["phi"][:]

    if idx is None:
        idx = np.random.choice(len(y_pred), 1)[0]

    pred, test = y_pred[idx], y_test[idx]
    thetas_s, phis_s = steering_info[idx]

    thetas_s, phis_s = thetas_s[~np.isnan(thetas_s)], phis_s[~np.isnan(phis_s)]
    thetas_s = np.array2string(thetas_s, precision=2, separator=", ")
    phis_s = np.array2string(phis_s, precision=2, separator=", ")

    loss = cosine_angular_loss_np(pred, test)
    title = f"Prediction Example {idx}: {loss:.4f} (θ={thetas_s}°, φ={phis_s}°)"
    filepath = output_dir / f"knn_pred_example_{idx}.png"
    compare_phase_shifts(pred, test, theta, phi, title, filepath)

    print(f"Prediction example saved to {filepath}")


@app.command()
def analyze_knn_beams():
    output_dir = Path.cwd() / "experiments"
    with h5py.File(output_dir / "knn_pred.h5", "r") as h5f:
        steer = h5f["steering_info"][:]

    theta_steer = steer[:, 0]
    theta_diff = theta_steer[:, 0] - theta_steer[:, 1]
    theta_diff = (theta_diff + 180) % 360 - 180

    phi_steer = steer[:, 1]
    phi_diff = phi_steer[:, 0] - phi_steer[:, 1]
    phi_diff = (phi_diff + 180) % 360 - 180

    diff = phi_diff
    diff = diff[~np.isnan(diff)]
    y = np.argmin(diff)
    print(y)
    print(steer[y])


def cosine_angular_loss_np(inputs: np.ndarray, targets: np.ndarray):
    return np.mean(1 - np.cos(inputs - targets))


if __name__ == "__main__":
    app()
