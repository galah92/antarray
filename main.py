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
import torch.nn.functional as F
import torch.optim as optim
import typer
from matplotlib.ticker import FormatStrFormatter
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split

import analyze
import generate_dataset
from generate_dataset import load_dataset

DEFAULT_DATASET_PATH: Path = Path.cwd() / "dataset" / "rand_bf_2d.h5"
DEFAULT_EXPERIMENTS_PATH: Path = Path.cwd() / "experiments"
DEFAULT_MODEL_NAME = "model.pth"


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

        # Normalize to [-1, 1]
        phase_shift = phase_shift / torch.pi

        if self.transform:
            pattern = self.transform(pattern)

        return pattern, phase_shift


class PhaseShiftModel(nn.Module):
    def __init__(
        self,
        input_channels=1,  # Number of input channels (1 for single radiation pattern)
        in_shape=(180, 180),  # Shape of the input radiation pattern (n_phi, n_theta)
        out_shape=(16, 16),  # Size of the output phase shift matrix (xn, yn)
        # Number of channels in each convolutional layer
        conv_channels=[32, 64, 128, 256, 512],
        # Number of units in each fully connected layer
        fc_units=[2048, 1024],
    ):
        super().__init__()

        self.out_shape = out_shape

        self.conv = nn.Sequential()
        ch = [input_channels] + conv_channels
        for i in range(1, len(ch)):
            self.conv += nn.Sequential(
                nn.Conv2d(ch[i - 1], ch[i], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(ch[i]),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

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
        x = torch.tanh(x)  # For output range [-1, 1]
        x = x * torch.pi  # Scale to [-pi, pi]
        x = x.view(-1, *self.out_shape)
        return x


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


def cosine_angular_loss_torch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return torch.mean(1 - torch.cos(inputs - targets))


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


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    early_stopper: EarlyStopper | None = None,
    n_epochs=25,
    log_interval=10,
    clip_grad=1.0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    since = time.time()
    history = {"train_loss": [], "val_loss": [], "lr": []}

    # Move model to device
    model = model.to(device)
    best_val_loss = float("inf")
    best_model_wts = None

    try:
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

            if early_stopper is not None:
                if early_stopper.early_stop(history["val_loss"][-1]):
                    print("Early stopping...")
                    break

            # Step the scheduler
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(history["val_loss"][-1])  # Pass the validation loss
                else:
                    scheduler.step()

            print()

    except KeyboardInterrupt:
        print("Training interrupted by user...")
    finally:
        # Load best model weights
        if best_model_wts:
            model.load_state_dict(best_model_wts)

        elapsed = time.time() - since
        print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
        print(f"Best val loss: {best_val_loss:.4f}")

    return model, history


def evaluate_model(
    model,
    test_loader,
    dataset_path: Path,
    num_examples=5,
    save_dir=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        "mae": mae.item(),
        "mse": mse.item(),
        "rmse": np.sqrt(mse).item(),
    }

    with h5py.File(dataset_path, "r") as h5f:
        theta = h5f["theta"][:]
        phi = h5f["phi"][:]

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


@app.command()
def pred_cnn(
    experiment: str,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    exps_path: Path = DEFAULT_EXPERIMENTS_PATH,
    num_examples: int = 5,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_path = exps_path / experiment

    _, _, test_loader = create_dataloaders(dataset_path, batch_size=128)
    test_indices = test_loader.dataset.indices

    with h5py.File(dataset_path, "r") as h5f:
        theta = h5f["theta"][:]
        phi = h5f["phi"][:]
        steering_info = h5f["steering_info"][:]
        labels = h5f["labels"][:]

    labels = labels[test_indices]

    checkpoint = torch.load(exps_path / experiment / DEFAULT_MODEL_NAME)
    model = PhaseShiftModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_preds = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Move predictions to CPU for numpy conversion
            preds = outputs.cpu().numpy()

            all_preds.append(preds)

    # Convert to numpy arrays
    all_preds = np.vstack(all_preds)
    all_targets = labels

    rand_indices = np.random.choice(len(all_preds), num_examples, replace=False)

    for idx in rand_indices:
        pred, target = all_preds[idx], all_targets[idx]

        thetas_s, phis_s = steering_info[test_indices[idx]]
        thetas_s, phis_s = thetas_s[~np.isnan(thetas_s)], phis_s[~np.isnan(phis_s)]
        thetas_s = np.array2string(thetas_s, precision=2, separator=", ")
        phis_s = np.array2string(phis_s, precision=2, separator=", ")

        print(pred.min(), pred.max(), target.min(), target.max())
        loss = cosine_angular_loss_np(pred, target)
        title = f"Prediction Example {test_indices[idx]}: {loss:.4f} (θ={thetas_s}°, φ={phis_s}°)"
        filepath = exp_path / f"prediction_example_{test_indices[idx]}.png"
        compare_phase_shifts(pred, target, theta, phi, title, filepath)

        print(f"Prediction example saved to {filepath}")


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

    label_ff = generate_dataset.ff_from_phase_shifts(label)
    axs[1, 0].remove()
    axs[1, 0] = fig.add_subplot(2, 3, 4, projection="3d")
    title_gt = "Ground Truth Far Field Pattern"
    analyze.plot_ff_3d(theta, phi, label_ff, title=title_gt, ax=axs[1, 0])

    output_ff = generate_dataset.ff_from_phase_shifts(output)
    axs[1, 1].remove()
    axs[1, 1] = fig.add_subplot(2, 3, 5, projection="3d")
    title_pred = "Predicted Far Field Pattern"
    analyze.plot_ff_3d(theta, phi, output_ff, title=title_pred, ax=axs[1, 1])

    axs[1, 2].remove()

    if title is not None:
        fig.suptitle(title)
    fig.set_tight_layout(True)

    if filepath:
        fig.savefig(filepath, dpi=600, bbox_inches="tight")


@app.command()
def plot_training(
    experiment: str,
    exps_path: Path = DEFAULT_EXPERIMENTS_PATH,
    overwrite: bool = False,
):
    exp_path = exps_path / experiment

    save_path = exp_path / "training_history.png"
    if save_path.exists() and not overwrite:
        raise Exception(f"Training history plot already exists at {save_path}")

    model_path = exp_path / DEFAULT_MODEL_NAME
    history = torch.load(model_path)["history"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    xticks = range(0, 100 + 1, 10)
    yticks = np.arange(0, 1 + 0.1, 0.1)

    axs[0].plot(history["train_loss"], "b-", label="Training Loss")
    axs[0].plot(history["val_loss"], "r-", label="Validation Loss")
    axs[0].set_xlim(xticks[0], xticks[-1])
    axs[0].set_xticks(xticks)
    axs[0].set_ylim(yticks[0], yticks[-1])
    axs[0].set_yticks(yticks)
    axs[0].set_title("Training and Validation Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    if "lr" in history and history["lr"]:
        axs[1].plot(history["lr"], "g-")
        axs[1].set_xlim(xticks[0], xticks[-1])
        axs[1].set_xticks(xticks)
        axs[1].yaxis.set_major_formatter(FormatStrFormatter("%.1e"))
        axs[1].set_title("Learning Rate")
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Learning Rate")
        axs[1].grid(True)

    fig.set_tight_layout(True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Training history plot saved to {save_path}")


@app.command()
def pred_beamforming(
    experiment: str,
    theta_steer: int = 0,
    phi_steer: int = 0,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    exps_path: Path = DEFAULT_EXPERIMENTS_PATH,
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

    checkpoint = torch.load(exps_path / experiment / DEFAULT_MODEL_NAME)
    model = PhaseShiftModel()
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(pattern)).numpy().squeeze()

    title = (f"Prediction Example {idx} (θ={theta_steer:.1f}°, φ={phi_steer:.1f}°)",)
    filepath = (f"prediction_example_{idx}_t{theta_steer}_p{phi_steer}.png",)
    compare_phase_shifts(output, label, theta, phi, title, filepath)


@app.command()
def run_cnn(
    experiment: str,
    overwrite: bool = False,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    exps_path: Path = DEFAULT_EXPERIMENTS_PATH,
    batch_size: int = 128,
    n_epochs: int = 100,
    lr: float = 1e-4,
):
    exp_path = exps_path / experiment
    exp_path.mkdir(exist_ok=overwrite, parents=True)

    train_loader, val_loader, test_loader = create_dataloaders(dataset_path, batch_size)

    model = PhaseShiftModel()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # Define loss function and optimizer
    criterion = cosine_angular_loss_torch
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=5
    # )

    # Train model
    print("Starting training...")
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
        early_stopper=EarlyStopper(patience=10, min_delta=1e-4),
        n_epochs=n_epochs,
    )

    # Save model
    model_path = exp_path / DEFAULT_MODEL_NAME
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        model_path,
    )
    print(f"Model saved to {model_path}")

    plot_training(experiment, overwrite=overwrite, exps_path=exps_path)

    # Evaluate model and save prediction examples
    print("Evaluating model on test set...")
    metrics = evaluate_model(
        model,
        test_loader,
        dataset_path,
        num_examples=5,
        save_dir=exp_path,
    )

    # Save metrics
    metrics_json = json.dumps(metrics, indent=4)
    metrics_path = exp_path / "metrics.json"
    metrics_path.write_text(metrics_json)
    print(f"Metrics saved to {metrics_path}")


def create_dataloaders(dataset_path: Path, batch_size: int):
    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)

    patterns, labels = dataset["patterns"], dataset["labels"]

    patterns[patterns < 0] = 0  # Set negative values to 0
    patterns = patterns / 30  # Normalize

    # Split data into train, validation, and test sets
    ds = RadiationPatternDataset(patterns, labels)
    gen = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(ds, [0.8, 0.1, 0.1], generator=gen)

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
    exp_path = exps_path / experiment
    exp_path.mkdir(exist_ok=overwrite, parents=True)

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

    loss = cosine_angular_loss_np(y_pred, y_test)
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
def analyze_knn_pred(idx: int | None = None):
    exps_path = Path.cwd() / "experiments"

    with h5py.File(exps_path / "knn_pred.h5", "r") as h5f:
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
    filepath = exps_path / f"knn_pred_example_{idx}.png"
    compare_phase_shifts(pred, test, theta, phi, title, filepath)

    print(f"Prediction example saved to {filepath}")


@app.command()
def analyze_knn_beams():
    exps_path = Path.cwd() / "experiments"
    with h5py.File(exps_path / "knn_pred.h5", "r") as h5f:
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
