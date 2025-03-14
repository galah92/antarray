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


def cosine_angular_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Convert angles to unit vectors
    inputs_x, inputs_y = torch.cos(inputs), torch.sin(inputs)
    targets_x, targets_y = torch.cos(targets), torch.sin(targets)

    # Cosine similarity between the vectors
    cos_sim = inputs_x * targets_x + inputs_y * targets_y

    # Convert to distance (1 - similarity)
    return torch.mean(1 - cos_sim)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=25,
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
    num_epochs : int
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

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} lr={scheduler.get_last_lr()[0]}")
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


def evaluate_model(model, test_loader, device="cuda", num_examples=5, save_dir=None):
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

        for i, idx in enumerate(indices):
            pred = all_preds[idx]
            target = all_targets[idx]

            fig, axs = plt.subplots(1, 3, figsize=(18, 5))

            # Plot target phase shifts
            im0 = axs[0].imshow(
                np.rad2deg(target), cmap="viridis", origin="lower", vmin=-180, vmax=180
            )
            axs[0].set_title("Ground Truth Phase Shifts")
            axs[0].set_xlabel("Element Y index")
            axs[0].set_ylabel("Element X index")
            plt.colorbar(im0, ax=axs[0])

            # Plot predicted phase shifts
            im1 = axs[1].imshow(
                np.rad2deg(pred), cmap="viridis", origin="lower", vmin=-180, vmax=180
            )
            axs[1].set_title("Predicted Phase Shifts")
            axs[1].set_xlabel("Element Y index")
            axs[1].set_ylabel("Element X index")
            plt.colorbar(im1, ax=axs[1])

            # Plot error
            diff = np.rad2deg(np.arctan2(np.sin(pred - target), np.cos(pred - target)))
            im2 = axs[2].imshow(
                diff, cmap="coolwarm", origin="lower", vmin=-180, vmax=180
            )
            axs[2].set_title("Phase Error (degrees)")
            axs[2].set_xlabel("Element Y index")
            axs[2].set_ylabel("Element X index")
            plt.colorbar(im2, ax=axs[2])

            plt.tight_layout()

            if save_dir:
                save_dir.mkdir(exist_ok=True, parents=True)
                save_path = Path(save_dir) / f"prediction_example_{i}.png"
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Prediction example saved to {save_path}")
            else:
                plt.show()

            plt.close()

    return metrics


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

    plt.tight_layout()

    if save_path:
        # Ensure the directory exists
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


DEFAULT_MODELS_DIR = Path.cwd() / "model_results"
DEFAULT_MODELS_DIR.mkdir(exist_ok=True, parents=True)
DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "phase_shift_prediction_model.pth"

DEFAULT_DATA_DIR = Path.cwd() / "dataset"
DEFAULT_DATA_DIR.mkdir(exist_ok=True, parents=True)
DEFAULT_DATASET_PATH = DEFAULT_DATA_DIR / "ff_beamforming.h5"


@app.command()
def pred_beamforming(
    theta_steer: int = 0,
    phi_steer: int = 0,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    savefig: bool = True,
):
    dataset = load_dataset(dataset_path)
    patterns, labels = dataset["patterns"], dataset["labels"]
    theta, phi = dataset["theta"], dataset["phi"]
    freq = dataset["frequency"]

    # Find the index of the given steering angles: https://stackoverflow.com/a/25823710/5151909
    steer_angles = (theta_steer, phi_steer)
    idx = (dataset["steering_info"] == steer_angles).all(axis=1).nonzero()[0].min()

    pattern = patterns[idx][None, None, ...]  # Add batch and channel dimensions
    pattern[pattern < 0] = 0  # Set negative values to 0
    pattern = pattern / 20  # Normalize
    label = labels[idx]

    input_height, input_width = patterns.shape[1], patterns.shape[2]
    output_size = (labels.shape[1], labels.shape[2])  # Typically (16, 16) for our array

    checkpoint = torch.load(model_path)
    model = PhaseShiftModel()
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        outputs = model(torch.from_numpy(pattern)).numpy()

    output = outputs.squeeze()

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    analyze.plot_phase_shifts(label, title="Ground Truth Phase Shifts", ax=axs[0, 0])
    analyze.plot_phase_shifts(output, title="Predicted Phase Shifts", ax=axs[0, 1])

    # TODO: should we use diff or output - label?
    diff = np.arctan2(np.sin(output - label), np.cos(output - label))
    analyze.plot_phase_shifts(diff, title="Phase Shift Error", ax=axs[0, 2])
    # plot_phase_shifts(output - label, title="Phase Shift Error", ax=axs[0, 2])

    label_ff = generate_dataset.ff_from_phase_shifts(output)
    axs[1, 0].remove()
    axs[1, 0] = fig.add_subplot(2, 3, 4, projection="3d")
    title = "Ground Truth Far Field Pattern"
    analyze.plot_ff_3d(theta, phi, label_ff, freq=freq, title=title, ax=axs[1, 0])

    output_ff = generate_dataset.ff_from_phase_shifts(label)
    axs[1, 1].remove()
    axs[1, 1] = fig.add_subplot(2, 3, 5, projection="3d")
    title = "Predicted Far Field Pattern"
    analyze.plot_ff_3d(theta, phi, output_ff, freq=freq, title=title, ax=axs[1, 1])

    axs[1, 2].remove()

    fig.suptitle(f"Prediction Example {idx} (θ={theta_steer:.1f}°, φ={phi_steer:.1f}°)")
    fig.set_tight_layout(True)

    if savefig:
        filename = f"prediction_example_{idx}_t{theta_steer}_p{phi_steer}.png"
        save_path = model_path.parent / filename
        plt.savefig(save_path, dpi=600, bbox_inches="tight")


@app.command()
def pred_beamforming_all(
    theta_steer: int = 0,
    phi_steer: int = 0,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    savefig: bool = True,
):
    # Get indices of test set
    n_samples = 12321
    indices = np.arange(n_samples)
    _, indices_test = train_test_split(indices, test_size=0.2, random_state=42)

    with h5py.File(dataset_path, "r") as h5f:
        steering_info = h5f["steering_info"]

        for idx in indices_test:
            theta_steer, phi_steer = steering_info[idx]
            pred_beamforming(
                theta_steer=theta_steer,
                phi_steer=phi_steer,
                dataset_path=dataset_path,
                model_path=model_path,
                savefig=savefig,
            )


DEFAULT_DATASET_PATH: Path = Path.cwd() / "dataset" / "ff_beamforming.h5"
DEFAULT_OUTPUT_DIR: Path = Path.cwd() / "model_results"


@app.command()
def run_cnn(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 128,
    num_epochs: int = 100,
):
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create subdirectories for plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # Define loss function and optimizer
    criterion = cosine_angular_loss
    # Use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
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
        num_epochs=num_epochs,
        device=device,
    )

    # Save model
    model_save_path = output_dir / "phase_shift_prediction_model.pth"
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
    history_plot_path = plots_dir / "training_history.png"
    visualize_training_history(history, save_path=history_plot_path)

    # Evaluate model and save prediction examples
    print("Evaluating model on test set...")
    metrics = evaluate_model(
        model,
        test_loader,
        device,
        num_examples=5,
        save_dir=plots_dir / "prediction_examples",
    )

    # Save metrics
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Metrics saved to {metrics_path}")

    return model, history, metrics


@app.command()
def run_knn(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
):
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create subdirectories for plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)

    patterns, labels = dataset["patterns"], dataset["labels"]

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
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        patterns, labels, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    knn = neighbors.KNeighborsRegressor(n_neighbors=5, weights="distance")
    y_pred = knn.fit(X_train, y_train).predict(X_test)

    # loss = circular_loss(y_pred, y_test, axis=(0, 1))
    loss = circular_loss(y_pred, y_test)
    print(f"Val loss: {loss:.4f}")

    idx_test = (idx_test).nonzero()[0]

    # save y_pred to h5 file
    with h5py.File(output_dir / "knn_pred.h5", "w") as h5f:
        h5f.create_dataset("y_pred", data=y_pred)
        h5f.create_dataset("y_test", data=y_test)
        h5f.create_dataset("loss", data=loss)
        h5f.create_dataset("idx_test", data=idx_test)

    mode = "wb" if overwrite else "xb"
    with (output_dir / "knn_model.pkl").open(mode) as f:
        pickle.dump(knn, f)


@app.command()
def analyze_knn_pred():
    output_dir = Path.cwd() / "model_results"
    with h5py.File(output_dir / "knn_pred.h5", "r") as h5f:
        y_pred = h5f["y_pred"][:].reshape(-1, 16, 16)
        y_test = h5f["y_test"][:].reshape(-1, 16, 16)
        idx_test = h5f["idx_test"][:]

    for idx in range(len(y_pred)):
        # idx = np.random.choice(len(y_pred), 1)[0]
        print(idx_test[idx], y_pred.shape, y_test.shape)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        title = "Ground Truth Phase Shifts"
        analyze.plot_phase_shifts(y_test[idx], title=title, ax=axs[0])
        title = "Predicted Phase Shifts"
        analyze.plot_phase_shifts(y_pred[idx], title=title, ax=axs[1])

        fig.suptitle(f"Prediction Example {idx_test[idx]}")
        fig.set_tight_layout(True)
        filename = f"knn_pred_example_{idx_test[idx]}.png"
        fig.savefig(output_dir / filename, dpi=600, bbox_inches="tight")


def circular_loss(predictions: np.ndarray, targets: np.ndarray, axis=None):
    """
    Compute the circular mean squared error.

    Parameters:
    -----------
    predictions : numpy.ndarray
        Predicted phase shifts in radians
    targets : numpy.ndarray
        Target phase shifts in radians

    Returns:
    --------
    loss : numpy.ndarray
        Mean squared error considering the circular nature of phases
    """
    # Calculate the difference and wrap to [-pi, pi]
    diff = predictions - targets
    diff = np.arctan2(np.sin(diff), np.cos(diff))

    # Calculate MSE on the wrapped differences
    return np.mean(diff**2, axis=axis)


if __name__ == "__main__":
    app()
