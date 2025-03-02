import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


class ArrayPatternDataset(Dataset):
    """Dataset for loading array pattern data in batches"""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

        # Load metadata
        metadata = torch.load(self.data_dir / "metadata.pt")
        self.total_samples = metadata["total_samples"]
        self.theta = metadata["theta"]
        self.xn_range = metadata["xn_range"]
        self.dx_range = metadata["dx_range"]

        # Get list of batch files
        self.batch_files = sorted(
            [f for f in os.listdir(self.data_dir) if f.startswith("array_batch_")]
        )

        # Load first batch to get dimensions
        first_batch = torch.load(self.data_dir / self.batch_files[0])
        self.input_dim = first_batch["patterns"].shape[1]

        # Create index mapping for efficient loading
        self.index_map = []
        samples_so_far = 0
        for i, batch_file in enumerate(self.batch_files):
            batch = torch.load(self.data_dir / batch_file)
            batch_size = batch["patterns"].shape[0]
            for j in range(batch_size):
                self.index_map.append((i, j))
            samples_so_far += batch_size

            # Free memory
            del batch

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        batch_idx, sample_idx = self.index_map[idx]
        batch = torch.load(self.data_dir / self.batch_files[batch_idx])

        pattern = batch["patterns"][sample_idx]
        label = batch["labels"][sample_idx]

        # Normalize labels for better training
        normalized_label = torch.tensor(
            [
                (label[0] - self.xn_range[0]) / (self.xn_range[1] - self.xn_range[0]),
                (label[1] - self.dx_range[0]) / (self.dx_range[1] - self.dx_range[0]),
            ],
            dtype=torch.float32,
        )

        return (
            pattern,
            normalized_label,
            label,
        )  # Return both normalized and original labels


class ArrayPatternCNN(nn.Module):
    """CNN model for predicting array parameters from patterns"""

    def __init__(self, input_dim):
        super().__init__()

        # 1D CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Calculate the size after convolutions and pooling
        self.flat_features = 128 * (input_dim // 8)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.flat_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Output: [xn, dx] in normalized form
        )

    def forward(self, x):
        # Add channel dimension
        x = x.unsqueeze(1)  # Shape: [batch, 1, pattern_length]

        # Pass through CNN
        x = self.cnn(x)

        # Flatten
        x = x.view(-1, self.flat_features)

        # Pass through fully connected layers
        x = self.fc(x)

        return x


def train_model(
    data_dir="array_dataset", batch_size=32, epochs=50, lr=0.001, val_split=0.2
):
    """Train the CNN model"""

    # Create dataset and split into train/val
    dataset = ArrayPatternDataset(data_dir)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = ArrayPatternCNN(dataset.input_dim)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for patterns, normalized_labels, _ in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"
        ):
            patterns = patterns.to(device)
            normalized_labels = normalized_labels.to(device)

            # Forward pass
            outputs = model(patterns)
            loss = criterion(outputs, normalized_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * patterns.size(0)

        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        xn_errors = []
        dx_errors = []

        with torch.no_grad():
            for patterns, normalized_labels, original_labels in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"
            ):
                patterns = patterns.to(device)
                normalized_labels = normalized_labels.to(device)

                # Forward pass
                outputs = model(patterns)
                loss = criterion(outputs, normalized_labels)

                val_loss += loss.item() * patterns.size(0)

                # Denormalize predictions for error calculation
                pred_normalized = outputs.cpu().numpy()
                pred_xn = (
                    pred_normalized[:, 0] * (dataset.xn_range[1] - dataset.xn_range[0])
                    + dataset.xn_range[0]
                )
                pred_dx = (
                    pred_normalized[:, 1] * (dataset.dx_range[1] - dataset.dx_range[0])
                    + dataset.dx_range[0]
                )

                # Calculate errors
                xn_error = np.abs(pred_xn - original_labels[:, 0].numpy())
                dx_error = np.abs(pred_dx - original_labels[:, 1].numpy())

                xn_errors.extend(xn_error)
                dx_errors.extend(dx_error)

        val_loss /= len(val_dataset)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)

        # Print statistics
        mean_xn_error = np.mean(xn_errors)
        mean_dx_error = np.mean(dx_errors)
        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(
            f"  Mean xn error: {mean_xn_error:.2f}, Mean dx error: {mean_dx_error:.2f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                "array_parameter_model_best.pt",
            )
            print("  Saved new best model")

    # Save final model
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        "array_parameter_model_final.pt",
    )

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("training_curve.png")
    plt.close()


if __name__ == "__main__":
    train_model(epochs=50, batch_size=64)
