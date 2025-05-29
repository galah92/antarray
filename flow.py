import logging
import sys
import time
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import typer
from flax import nnx

import analyze
import data

logger = logging.getLogger(__name__)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

DEFAULT_ARRAY_SIZE = (16, 16)
DEFAULT_SPACING_MM = (60, 60)
DEFAULT_THETA_END = 65.0
DEFAULT_MAX_N_BEAMS = 3


class DataSample(NamedTuple):
    radiation_pattern: jnp.ndarray  # Shape: (n_theta, n_phi)
    phase_shifts: jnp.ndarray  # Shape: (array_x, array_y)
    steering_angles: jnp.ndarray  # Shape: (n_beams, 2) - theta, phi in radians


class Dataset:
    def __init__(
        self,
        array_size: tuple[int, int] = DEFAULT_ARRAY_SIZE,
        spacing_mm: tuple[float, float] = DEFAULT_SPACING_MM,
        theta_end: float = DEFAULT_THETA_END,
        max_n_beams: int = DEFAULT_MAX_N_BEAMS,
        batch_size: int = 64,
        sim_dir_path: Path = data.DEFAULT_SIM_DIR,
        key: jax.Array = None,
        prefetch: bool = True,
        normalize: bool = True,
        db_range: float = 60.0,
    ):
        self.theta_end = jnp.radians(theta_end)
        self.sim_dir_path = sim_dir_path
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.normalize = normalize
        self.db_range = db_range

        if key is None:
            key = jax.random.PRNGKey(0)
        self.key = key

        # Load and prepare array parameters
        array_params = analyze.calc_array_params2(
            array_size=array_size,
            spacing_mm=spacing_mm,
            theta_rad=jnp.radians(jnp.arange(180)),
            phi_rad=jnp.radians(jnp.arange(360)),
            sim_path=sim_dir_path / data.DEFAULT_SINGLE_ANT_FILENAME,
        )

        # Convert to JAX arrays and make them static
        self.array_params = [jnp.asarray(param) for param in array_params]

        # Beam probability distribution
        self.n_beams_prob = data.get_beams_prob(max_n_beams)

        self.vmapped_generate_sample = jax.vmap(self.generate_sample)

        self._prefetched_batch = None
        if self.prefetch:
            self._prefetched_batch = self.generate_batch()

    def generate_sample(self, key: jax.Array) -> DataSample:
        key1, key2 = jax.random.split(key)

        # Generate random steering angles for multiple beams
        n_beams = 4
        theta_steering = jax.random.uniform(key1, (n_beams,)) * self.theta_end
        phi_steering = jax.random.uniform(key2, (n_beams,)) * (2 * jnp.pi)
        steering_angles = jnp.stack((theta_steering, phi_steering), axis=-1)

        radiation_pattern, excitations = analyze.rad_pattern_from_geo(
            *self.array_params,
            steering_angles,
        )
        phase_shifts = jnp.angle(excitations)

        if self.normalize:  # Clip and normalize dB values to [0, 1]
            rp_max = jnp.max(radiation_pattern)
            rp_clipped = jnp.clip(radiation_pattern, rp_max - self.db_range, rp_max)
            radiation_pattern = (rp_clipped - (rp_max - self.db_range)) / self.db_range

        return DataSample(radiation_pattern, phase_shifts, steering_angles)

    def generate_batch(self) -> dict[str, jax.Array]:
        self.key, batch_key = jax.random.split(self.key)
        sample_keys = jax.random.split(batch_key, self.batch_size)
        samples = self.vmapped_generate_sample(sample_keys)

        return {
            "radiation_patterns": samples.radiation_pattern,
            "phase_shifts": samples.phase_shifts,
            "steering_angles": samples.steering_angles,
        }

    def __next__(self) -> dict[str, jax.Array]:
        if self.prefetch:
            current_batch = self._prefetched_batch
            self._prefetched_batch = self.generate_batch()
            return current_batch
        else:
            return self.generate_batch()

    def __iter__(self):
        return self


class SimplePhaseShiftPredictor(nnx.Module):
    def __init__(self, array_size: tuple[int, int], *, rngs: nnx.Rngs):
        self.array_size = array_size

        self.conv1 = nnx.Conv(1, 32, kernel_size=(7, 7), strides=(2, 4), rngs=rngs)
        self.norm1 = nnx.BatchNorm(32, rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(5, 5), strides=(2, 2), rngs=rngs)
        self.norm2 = nnx.BatchNorm(64, rngs=rngs)
        self.conv3 = nnx.Conv(64, 1, kernel_size=(3, 3), rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = x.reshape(x.shape[0], 180, 360, 1)

        x = self.conv1(x)
        x = self.norm1(x, use_running_average=not training)
        x = jax.nn.relu(x)

        x = self.conv2(x)
        x = self.norm2(x, use_running_average=not training)
        x = jax.nn.relu(x)

        x = self.conv3(x)

        target_h, target_w = self.array_size
        x = jax.image.resize(x, (x.shape[0], target_h, target_w, 1), method="bilinear")

        return x.squeeze(-1)


class CircularConv(nnx.Module):
    """Convolution with circular padding in phi dimension (width)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        *,
        rngs: nnx.Rngs,
    ):
        self.conv = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            padding="VALID",  # We handle padding manually
            rngs=rngs,
        )
        self.kernel_size = kernel_size

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x shape: (batch, theta, phi, channels)
        pad_size = self.kernel_size // 2

        # Circular padding in phi dimension (axis=2, width)
        phi_padded = jnp.concatenate(
            [
                x[:, :, -pad_size:, :],  # Wrap φ=359° to beginning
                x,
                x[:, :, :pad_size, :],  # Wrap φ=0° to end
            ],
            axis=2,
        )

        # Regular zero-padding in theta dimension (axis=1, height)
        theta_phi_padded = jnp.pad(
            phi_padded,
            ((0, 0), (pad_size, pad_size), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        return self.conv(theta_phi_padded)


class CircularConvBlock(nnx.Module):
    """ConvBlock using CircularConv for phi wrapping"""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.conv = CircularConv(in_channels, out_channels, 3, rngs=rngs)
        self.norm = nnx.BatchNorm(out_channels, rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = self.conv(x)
        x = self.norm(x, use_running_average=not training)
        return nnx.relu(x)


class CircularDoubleConv(nnx.Module):
    """DoubleConv using CircularConv for phi wrapping"""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.conv1 = CircularConvBlock(in_channels, out_channels, rngs=rngs)
        self.conv2 = CircularConvBlock(out_channels, out_channels, rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class ConvBlock(nnx.Module):
    """Basic convolutional block with conv + batchnorm + activation"""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_channels, out_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs
        )
        self.norm = nnx.BatchNorm(out_channels, rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = self.conv(x)
        x = self.norm(x, use_running_average=not training)
        return nnx.relu(x)


class DoubleConv(nnx.Module):
    """Two consecutive conv blocks (like in U-Net)"""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.conv1 = ConvBlock(in_channels, out_channels, rngs=rngs)
        self.conv2 = ConvBlock(out_channels, out_channels, rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class AsymmetricDown(nnx.Module):
    """Asymmetric downsampling with circular phi convolutions"""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        # 1x2 pooling: keeps theta, halves phi
        self.maxpool = partial(nnx.max_pool, window_shape=(1, 2), strides=(1, 2))
        self.conv = CircularDoubleConv(in_channels, out_channels, rngs=rngs)  # Changed!

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = self.maxpool(x)
        return self.conv(x, training=training)


class Down(nnx.Module):
    """Standard downsampling with circular phi convolutions"""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.maxpool = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))
        self.conv = CircularDoubleConv(in_channels, out_channels, rngs=rngs)  # Changed!

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = self.maxpool(x)
        return self.conv(x, training=training)


class Up(nnx.Module):
    """Standard upsampling with circular phi convolutions"""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.conv = CircularDoubleConv(in_channels, out_channels, rngs=rngs)  # Changed!

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        h, w = x.shape[1] * 2, x.shape[2] * 2
        x = jax.image.resize(x, (x.shape[0], h, w, x.shape[3]), method="bilinear")
        return self.conv(x, training=training)


class ConvAutoencoder(nnx.Module):
    """Aspect-ratio aware autoencoder with circular phi convolutions"""

    def __init__(
        self,
        array_size: tuple[int, int],
        base_channels: int = 16,
        use_circular: bool = True,  # New parameter
        *,
        rngs: nnx.Rngs,
    ):
        self.array_size = array_size
        self.use_circular = use_circular

        # Choose conv type based on use_circular
        ConvType = CircularDoubleConv if use_circular else DoubleConv

        # Initial convolution
        self.inc = ConvType(1, base_channels, rngs=rngs)

        # Encoder - same structure, but with circular convs
        self.down1 = AsymmetricDown(base_channels, base_channels * 2, rngs=rngs)
        self.down2 = Down(base_channels * 2, base_channels * 4, rngs=rngs)
        self.down3 = Down(base_channels * 4, base_channels * 8, rngs=rngs)
        self.down4 = Down(base_channels * 8, base_channels * 16, rngs=rngs)

        # Bottleneck
        self.bottleneck1 = ConvType(base_channels * 16, base_channels * 16, rngs=rngs)
        self.bottleneck2 = ConvType(base_channels * 16, base_channels * 16, rngs=rngs)

        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8, rngs=rngs)
        self.up2 = Up(base_channels * 8, base_channels * 4, rngs=rngs)

        # Final output layer - use regular conv since we're at low resolution
        self.final_conv = nnx.Conv(base_channels * 4, 1, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Same forward pass as before
        x = x.reshape(x.shape[0], 180, 360, 1)

        x = self.inc(x, training=training)
        x = self.down1(x, training=training)
        x = self.down2(x, training=training)
        x = self.down3(x, training=training)
        x = self.down4(x, training=training)

        x = self.bottleneck1(x, training=training)
        x = self.bottleneck2(x, training=training)

        x = self.up1(x, training=training)
        x = self.up2(x, training=training)

        logits = self.final_conv(x)

        target_h, target_w = self.array_size
        logits = jax.image.resize(
            logits, (logits.shape[0], target_h, target_w, 1), method="bilinear"
        )

        return logits.squeeze(-1)


@nnx.jit
def train_step(
    optimizer: nnx.Optimizer,
    batch: dict[str, jnp.ndarray],
) -> tuple[nnx.Optimizer, dict[str, float]]:
    def loss_fn(model: nnx.Module):
        radiation_patterns = batch["radiation_patterns"]
        phase_shifts = batch["phase_shifts"]
        predictions = model(radiation_patterns, training=True)

        # Circular MSE loss
        phase_diff = jnp.abs(predictions - phase_shifts)
        circular_diff = jnp.minimum(phase_diff, 2 * jnp.pi - phase_diff)
        loss = jnp.mean(circular_diff**2)

        phase_rmse = jnp.sqrt(loss)
        return loss, {"loss": loss, "phase_rmse": phase_rmse}

    model = optimizer.model
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads)

    return optimizer, metrics


def warmup_step(dataset: Dataset, optimizer: nnx.Optimizer):
    logger.info("Warming up GPU kernels")
    warmup_batch = next(dataset)
    _ = train_step(optimizer, warmup_batch)
    logger.info("Warmup completed")


@app.command()
def dev(
    array_size: tuple[int, int] = DEFAULT_ARRAY_SIZE,
    spacing_mm: tuple[float, float] = DEFAULT_SPACING_MM,
    theta_end: float = DEFAULT_THETA_END,
    max_n_beams: int = DEFAULT_MAX_N_BEAMS,
    n_steps: int = 10_000,
    batch_size: int = 64,
    lr: float = 2e-3,
    seed: int = 42,
    prefetch: bool = True,
):
    key = jax.random.key(seed)

    key, dataset_key, model_key = jax.random.split(key, num=3)
    dataset = Dataset(
        array_size,
        spacing_mm,
        theta_end,
        max_n_beams,
        batch_size=batch_size,
        key=dataset_key,
        prefetch=prefetch,
    )

    logger.info("Initializing model and optimizer")
    model = ConvAutoencoder(array_size, rngs=nnx.Rngs(model_key))
    # model = SimplePhaseShiftPredictor(array_size, rngs=nnx.Rngs(model_key))
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_steps)
    optimizer = nnx.Optimizer(model, optax.adam(schedule))

    warmup_step(dataset, optimizer)

    logger.info("Starting development run")
    start_time = time.perf_counter()

    for step in range(n_steps):
        batch = next(dataset)
        optimizer, train_metrics = train_step(optimizer, batch)

        if (step + 1) % 10 == 0:
            elapsed = time.perf_counter() - start_time
            avg_time = elapsed / (step + 1)

            loss = train_metrics["loss"]
            logger.info(
                f"step {step + 1:04d}, "
                f"avg={avg_time:.3f}s/step, "
                f"total={elapsed:.1f}s, "
                f"loss={loss:.3f}"
            )


@app.command()
def pred():
    pass


@app.command()
def visualize_dataset(
    array_size: tuple[int, int] = DEFAULT_ARRAY_SIZE,
    spacing_mm: tuple[float, float] = DEFAULT_SPACING_MM,
    theta_end: float = DEFAULT_THETA_END,
    max_n_beams: int = DEFAULT_MAX_N_BEAMS,
    batch_size: int = 4,
    seed: int = 42,
):
    key = jax.random.key(seed)

    dataset_args = array_size, spacing_mm, theta_end, max_n_beams, batch_size
    dataset = Dataset(*dataset_args, key=key, normalize=True)
    batch = next(dataset)
    patterns, phase_shifts = batch["radiation_patterns"], batch["phase_shifts"]

    theta_rad, phi_rad = np.radians(np.arange(180)), np.radians(np.arange(360))

    fig, axes = plt.subplots(batch_size, 2, figsize=(12, 3 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)

    for i in range(batch_size):
        title = f"Sample {i + 1}: Radiation Pattern"
        analyze.plot_ff_2d(theta_rad, phi_rad, patterns[i], title=title, ax=axes[i, 0])
        title = f"Sample {i + 1}: Phase Shifts"
        analyze.plot_phase_shifts(phase_shifts[i], title=title, ax=axes[i, 1])

    fig.suptitle("Batch Overview: Model Inputs")
    fig.set_tight_layout(True)

    plot_path = "batch_overview.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved batch overview {plot_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} {levelname} {filename}:{lineno} {message}",
        style="{",
        handlers=[
            logging.FileHandler(Path("app.log"), mode="w+"),  # Overwrite log
            logging.StreamHandler(),
        ],
    )
    logger.info(f"uv run {' '.join(sys.argv)}")  # Log command line args
    app()
