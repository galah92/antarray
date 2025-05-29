import logging
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
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
        sim_dir_path: Path = data.DEFAULT_SIM_DIR,
    ):
        self.theta_end = theta_end
        self.sim_dir_path = sim_dir_path

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

    def generate_sample(self, key: jax.random.PRNGKey) -> DataSample:
        key1, key2 = jax.random.split(key, 2)

        # For now, let's simplify to single beam to avoid vmap issues
        # Generate random steering angles for a single beam
        n_beams = 4
        theta_steering = jax.random.uniform(key1, n_beams) * jnp.radians(self.theta_end)
        phi_steering = jax.random.uniform(key2, n_beams) * (2 * jnp.pi)
        steering_angles = jnp.stack((theta_steering, phi_steering), axis=-1)

        radiation_pattern, excitations = analyze.rad_pattern_from_geo(
            *self.array_params,
            steering_angles,
        )
        phase_shifts = jnp.angle(excitations)

        return DataSample(radiation_pattern, phase_shifts, steering_angles)

    def generate_batch(
        self, key: jax.random.PRNGKey, batch_size: int
    ) -> dict[str, jnp.ndarray]:
        keys = jax.random.split(key, batch_size)
        samples = jax.vmap(self.generate_sample)(keys)
        return {
            "radiation_patterns": samples.radiation_pattern,
            "phase_shifts": samples.phase_shifts,
            "steering_angles": samples.steering_angles,
        }


class AttentionBlock(nnx.Module):
    def __init__(self, channels: int, *, rngs: nnx.Rngs):
        self.channels = channels
        self.query = nnx.Conv(channels, channels // 8, kernel_size=(1, 1), rngs=rngs)
        self.key = nnx.Conv(channels, channels // 8, kernel_size=(1, 1), rngs=rngs)
        self.value = nnx.Conv(channels, channels, kernel_size=(1, 1), rngs=rngs)
        self.gamma = nnx.Param(jnp.zeros(1))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, height, width, channels = x.shape

        # Compute attention
        q = self.query(x).reshape(batch_size, -1, channels // 8)
        k = self.key(x).reshape(batch_size, -1, channels // 8)
        v = self.value(x).reshape(batch_size, -1, channels)

        # Attention weights
        attention = jax.nn.softmax(q @ jnp.transpose(k, (0, 2, 1)), axis=-1)

        # Apply attention
        out = (attention @ v).reshape(batch_size, height, width, channels)

        # Residual connection with learnable weight
        return self.gamma * out + x


class ResidualBlock(nnx.Module):
    def __init__(self, channels: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs
        )
        self.conv2 = nnx.Conv(
            channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs
        )
        self.norm1 = nnx.BatchNorm(channels, rngs=rngs)
        self.norm2 = nnx.BatchNorm(channels, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x

        out = nnx.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))

        return nnx.relu(out + residual)


class PhaseShiftPredictor(nnx.Module):
    def __init__(self, array_size: tuple[int, int], *, rngs: nnx.Rngs):
        self.array_size = array_size

        # Feature extraction backbone
        self.conv1 = nnx.Conv(
            1, 32, kernel_size=(7, 7), strides=(2, 2), padding="SAME", rngs=rngs
        )
        self.norm1 = nnx.BatchNorm(32, rngs=rngs)

        self.conv2 = nnx.Conv(
            32, 64, kernel_size=(5, 5), strides=(2, 2), padding="SAME", rngs=rngs
        )
        self.norm2 = nnx.BatchNorm(64, rngs=rngs)

        self.conv3 = nnx.Conv(
            64, 128, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs
        )
        self.norm3 = nnx.BatchNorm(128, rngs=rngs)

        # Residual blocks for better feature learning
        self.res_block1 = ResidualBlock(128, rngs=rngs)
        self.res_block2 = ResidualBlock(128, rngs=rngs)

        # Attention mechanism to focus on beam peaks
        self.attention = AttentionBlock(128, rngs=rngs)

        # Spatial upsampling to match array dimensions - avoiding checkerboard artifacts
        # Using resize + conv instead of transposed convolutions
        self.upsample_conv1 = nnx.Conv(
            128, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs
        )
        self.norm_up1 = nnx.BatchNorm(64, rngs=rngs)

        self.upsample_conv2 = nnx.Conv(
            64, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs
        )
        self.norm_up2 = nnx.BatchNorm(32, rngs=rngs)

        # Adaptive pooling to exact array size
        self.spatial_adapt = nnx.Conv(
            32, 16, kernel_size=(3, 3), padding="SAME", rngs=rngs
        )
        self.norm_adapt = nnx.BatchNorm(16, rngs=rngs)

        # Final phase prediction layer - outputs 1 phase per array element
        self.phase_conv = nnx.Conv(16, 1, kernel_size=(1, 1), rngs=rngs)

        # Phase normalization parameters
        self.phase_scale = nnx.Param(jnp.ones(1))
        self.phase_bias = nnx.Param(jnp.zeros(1))

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Input shape: (batch_size, 180, 360)
        x = x.reshape(x.shape[0], 180, 360, 1)

        # Feature extraction with progressive downsampling
        x = nnx.relu(self.norm1(self.conv1(x)))  # -> (batch, 90, 180, 32)
        x = nnx.relu(self.norm2(self.conv2(x)))  # -> (batch, 45, 90, 64)
        x = nnx.relu(self.norm3(self.conv3(x)))  # -> (batch, 23, 45, 128)

        # Residual blocks for deeper feature learning
        x = self.res_block1(x)
        x = self.res_block2(x)

        # Attention mechanism to focus on important regions
        x = self.attention(x)

        # Spatial upsampling using resize + conv to avoid checkerboard artifacts
        # Upsample 1: (23, 45) -> (46, 90)
        x = jax.image.resize(x, (x.shape[0], 46, 90, x.shape[3]), method="bilinear")
        x = nnx.relu(self.norm_up1(self.upsample_conv1(x)))  # -> (batch, 46, 90, 64)

        # Upsample 2: (46, 90) -> (92, 180)
        x = jax.image.resize(x, (x.shape[0], 92, 180, x.shape[3]), method="bilinear")
        x = nnx.relu(self.norm_up2(self.upsample_conv2(x)))  # -> (batch, 92, 180, 32)

        # Adaptive spatial processing
        x = nnx.relu(self.norm_adapt(self.spatial_adapt(x)))  # -> (batch, 92, 180, 16)

        # Resize to exact array dimensions using adaptive average pooling
        # This preserves spatial structure while matching target size
        target_h, target_w = self.array_size
        current_h, current_w = x.shape[1], x.shape[2]

        # Use stride and kernel size to downsample to target size
        stride_h = current_h // target_h
        stride_w = current_w // target_w
        kernel_h = current_h - (target_h - 1) * stride_h
        kernel_w = current_w - (target_w - 1) * stride_w

        x = nnx.avg_pool(
            x, window_shape=(kernel_h, kernel_w), strides=(stride_h, stride_w)
        )

        # Ensure exact target dimensions (crop or pad if needed)
        if x.shape[1] != target_h or x.shape[2] != target_w:
            x = jax.image.resize(
                x, (x.shape[0], target_h, target_w, x.shape[3]), method="bilinear"
            )

        # Final phase prediction
        x = self.phase_conv(x)  # -> (batch, array_x, array_y, 1)

        # Apply phase normalization and wrap to [-π, π]
        x = self.phase_scale * x + self.phase_bias
        x = jnp.arctan2(jnp.sin(x), jnp.cos(x))  # Wrap phases to [-π, π]

        # Remove channel dimension: (batch, array_x, array_y, 1) -> (batch, array_x, array_y)
        predictions = x.squeeze(-1)
        return predictions


@nnx.jit
def train_step(
    optimizer: nnx.Optimizer,
    batch: dict[str, jnp.ndarray],
) -> tuple[nnx.Optimizer, dict[str, float]]:
    def loss_fn(model: nnx.Module):
        radiation_patterns = batch["radiation_patterns"]
        phase_shifts = batch["phase_shifts"]
        predictions = model(radiation_patterns, training=True)

        # Phase-aware loss: account for phase wrapping
        phase_diff = predictions - phase_shifts
        phase_diff_wrapped = jnp.arctan2(jnp.sin(phase_diff), jnp.cos(phase_diff))
        loss = jnp.mean(phase_diff_wrapped**2)

        phase_rmse = jnp.sqrt(jnp.mean(phase_diff_wrapped**2))
        return loss, {"loss": loss, "phase_rmse": phase_rmse}

    model = optimizer.model
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads)

    return optimizer, metrics


@app.command()
def dev(
    array_size: tuple[int, int] = DEFAULT_ARRAY_SIZE,
    spacing_mm: tuple[float, float] = DEFAULT_SPACING_MM,
    theta_end: float = DEFAULT_THETA_END,
    max_n_beams: int = DEFAULT_MAX_N_BEAMS,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
):
    key = jax.random.PRNGKey(0)

    dataset = Dataset(array_size, spacing_mm, theta_end, max_n_beams)

    rngs = nnx.Rngs(seed)
    model = PhaseShiftPredictor(array_size, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))

    for step in range(100):
        step_start_time = time.time()
        batch = dataset.generate_batch(key, batch_size=batch_size)
        optimizer, train_metrics = train_step(optimizer, batch)

        loss = train_metrics["loss"]
        duration = time.time() - step_start_time
        logger.info(f"{step=:02}, {duration=:.2f}s, {loss=:.3f}")

    logger.info("Development run completed successfully")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} {levelname} {filename}:{lineno} {message}",
        style="{",
        handlers=[
            logging.FileHandler(
                Path("app.log"), mode="w+"
            ),  # Overwrite log on each run
            logging.StreamHandler(),
        ],
    )
    app()
