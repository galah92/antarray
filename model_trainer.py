import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import typer

import analyze
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


class PhaseShiftPredictionModel(nn.Module):
    """
    CNN model to predict phase shifts from radiation patterns.
    Designed to handle smaller input dimensions (e.g., input_height=2)
    """

    def __init__(
        self, input_channels=1, input_height=2, input_width=360, output_size=(16, 16)
    ):
        """
        Initialize the model with architecture appropriate for small input dimensions.

        Parameters:
        -----------
        input_channels : int
            Number of input channels (1 for single radiation pattern)
        input_height : int
            Height of the input (number of phi values)
        input_width : int
            Width of the input (number of theta values)
        output_size : tuple
            Size of the output phase shift matrix (xn, yn)
        """
        super(PhaseShiftPredictionModel, self).__init__()

        self.output_size = output_size

        # For very small height (e.g., 2), we don't use pooling in that dimension
        # Instead we'll use a different architecture optimized for this shape
        if input_height <= 4:
            # Special case for extremely small height (e.g., 2)
            self.encoder = nn.Sequential(
                # First layer processes each phi slice
                nn.Conv2d(
                    input_channels, 64, kernel_size=(1, 5), stride=1, padding=(0, 2)
                ),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # Second layer continues processing with no height reduction
                nn.Conv2d(64, 128, kernel_size=(1, 5), stride=1, padding=(0, 2)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(
                    kernel_size=(1, 2), stride=(1, 2)
                ),  # Only pool along width
                # Third layer
                nn.Conv2d(128, 256, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(
                    kernel_size=(1, 2), stride=(1, 2)
                ),  # Only pool along width
                # Fourth layer to extract higher-level features
                nn.Conv2d(
                    256, 512, kernel_size=(input_height, 3), stride=1, padding=(0, 1)
                ),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(
                    kernel_size=(1, 2), stride=(1, 2)
                ),  # Only pool along width
            )

            # Calculate output size after convolutions
            width_after_conv = input_width // 8  # After 3 width pooling operations

            # Fully connected layers
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    512 * 1 * width_after_conv, 1024
                ),  # Height is now 1 due to kernel_size=(input_height, 3)
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, output_size[0] * output_size[1]),
            )
        else:
            # Standard architecture for larger inputs
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


class CircularLoss(nn.Module):
    """
    Custom loss function for circular/phase values.
    Handles the cyclic nature of phase angles.
    """

    def __init__(self):
        super(CircularLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Compute the circular mean squared error.

        Parameters:
        -----------
        predictions : torch.Tensor
            Predicted phase shifts in radians
        targets : torch.Tensor
            Target phase shifts in radians

        Returns:
        --------
        loss : torch.Tensor
            Mean squared error considering the circular nature of phases
        """
        # Calculate the difference and wrap to [-pi, pi]
        diff = predictions - targets
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))

        # Calculate MSE on the wrapped differences
        return torch.mean(diff**2)


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
        print(f"Epoch {epoch + 1}/{num_epochs}")
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
def pred(
    idx: int = 0,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    savefig: bool = True,
):
    dataset = load_dataset(dataset_path)
    patterns, labels = dataset["patterns"], dataset["labels"]

    pattern = patterns[idx][None, None, ...]  # Add batch and channel dimensions
    pattern[pattern < 0] = 0  # Set negative values to 0
    pattern = pattern / 20  # Normalize
    label = labels[idx]

    input_height, input_width = patterns.shape[1], patterns.shape[2]
    output_size = (labels.shape[1], labels.shape[2])  # Typically (16, 16) for our array

    checkpoint = torch.load(model_path)
    model = PhaseShiftPredictionModel(
        input_channels=1,
        input_height=input_height,
        input_width=input_width,
        output_size=output_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        outputs = model(torch.from_numpy(pattern)).numpy()

    output = outputs.squeeze()

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    analyze.plot_phase_shifts(label, title="Ground Truth Phase Shifts", ax=axs[0])
    analyze.plot_phase_shifts(output, title="Predicted Phase Shifts", ax=axs[1])

    # TODO: should we use diff or output - label?
    diff = np.arctan2(np.sin(output - label), np.cos(output - label))
    analyze.plot_phase_shifts(diff, title="Phase Shift Error", ax=axs[2])
    # plot_phase_shifts(output - label, title="Phase Shift Error", ax=axs[2])

    fig.set_tight_layout(True)

    if savefig:
        save_path = model_path.parent / f"prediction_example_{idx}.png"
        plt.savefig(save_path, dpi=600, bbox_inches="tight")


def main(dataset_path, output_dir=None, batch_size=32, num_epochs=50, device="cuda"):
    """
    Main training function.

    Parameters:
    -----------
    dataset_path : str or Path
        Path to the dataset H5 file
    output_dir : str or Path, optional
        Directory to save results
    batch_size : int
        Batch size for training
    num_epochs : int
        Number of training epochs
    device : str
        Device to use for training ('cuda' or 'cpu')
    """
    if output_dir is None:
        output_dir = Path.cwd() / "model_results"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    # Create subdirectories for plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Set device
    device = torch.device(
        device if torch.cuda.is_available() and device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)

    patterns = dataset["patterns"]
    labels = dataset["labels"]

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

    # Create datasets
    train_dataset = RadiationPatternDataset(X_train, y_train)
    val_dataset = RadiationPatternDataset(X_val, y_val)
    test_dataset = RadiationPatternDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Initialize model
    input_height, input_width = patterns.shape[1], patterns.shape[2]
    output_size = (labels.shape[1], labels.shape[2])  # Typically (16, 16) for our array

    # Print input shape info for debugging
    print(
        f"Model input dimensions: channels=1, height={input_height}, width={input_width}"
    )

    # Initialize model with appropriate architecture
    model = PhaseShiftPredictionModel(
        input_channels=1,
        input_height=input_height,
        input_width=input_width,
        output_size=output_size,
    )

    # Print model summary
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # Define loss function and optimizer
    criterion = CircularLoss()
    # Use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
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
def old_main():
    # Example usage
    dataset_path = Path.cwd() / "dataset" / "farfield_dataset.h5"
    dataset_path = Path.cwd() / "dataset" / "ff_beamforming.h5"
    output_dir = Path.cwd() / "model_results"

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    main(
        dataset_path=dataset_path,
        output_dir=output_dir,
        batch_size=128,
        num_epochs=100,
        device=device,
    )


if __name__ == "__main__":
    app()
