import logging
import sys
import time
from functools import partial
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


class SimplePhaseShiftPredictor(nnx.Module):
    def __init__(self, array_size: tuple[int, int], *, rngs: nnx.Rngs):
        self.array_size = array_size

        self.conv1 = nnx.Conv(1, 32, kernel_size=(7, 7), strides=(2, 2), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(5, 5), strides=(2, 2), rngs=rngs)
        self.conv3 = nnx.Conv(64, 1, kernel_size=(3, 3), rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = x.reshape(x.shape[0], 180, 360, 1)
        x = jax.nn.relu(self.conv1(x))
        x = jax.nn.relu(self.conv2(x))
        x = self.conv3(x)

        target_h, target_w = self.array_size
        x = jax.image.resize(x, (x.shape[0], target_h, target_w, 1), method="bilinear")

        return jnp.arctan2(jnp.sin(x.squeeze(-1)), jnp.cos(x.squeeze(-1)))


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


class Down(nnx.Module):
    """Downsampling block: maxpool + double conv"""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.maxpool = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))
        self.conv = DoubleConv(in_channels, out_channels, rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = self.maxpool(x)
        return self.conv(x, training=training)


class Up(nnx.Module):
    """Upsampling block: upsample + double conv"""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.conv = DoubleConv(in_channels, out_channels, rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Upsample by 2x using bilinear interpolation
        h, w = x.shape[1] * 2, x.shape[2] * 2
        x = jax.image.resize(x, (x.shape[0], h, w, x.shape[3]), method="bilinear")
        return self.conv(x, training=training)


class ConvAutoencoder(nnx.Module):
    def __init__(
        self,
        array_size: tuple[int, int],
        base_channels: int = 16,
        down_depth: int = 4,
        up_depth: int = 2,
        bottleneck_depth: int = 2,
        *,
        rngs: nnx.Rngs,
    ):
        self.array_size = array_size

        # Initial convolution (like 'inc' in PyTorch version)
        self.inc = DoubleConv(1, base_channels, rngs=rngs)

        # Encoder (downsampling path)
        self.encoder = []
        in_ch = base_channels
        for i in range(down_depth):
            out_ch = base_channels * (2 ** (i + 1))
            self.encoder.append(Down(in_ch, out_ch, rngs=rngs))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = []
        for _ in range(bottleneck_depth):
            self.bottleneck.append(DoubleConv(in_ch, in_ch, rngs=rngs))

        # Decoder (upsampling path)
        self.decoder = []
        for i in range(up_depth):
            out_ch = in_ch // 2
            self.decoder.append(Up(in_ch, out_ch, rngs=rngs))
            in_ch = out_ch

        # Final output layer
        self.final_conv = nnx.Conv(in_ch, 1, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        # Input shape: (batch_size, 180, 360) -> (batch_size, 180, 360, 1)
        x = x.reshape(x.shape[0], 180, 360, 1)

        # Initial convolution
        x = self.inc(x, training=training)

        # Encoder
        for down_block in self.encoder:
            x = down_block(x, training=training)

        # Bottleneck
        for bottleneck_block in self.bottleneck:
            x = bottleneck_block(x, training=training)

        # Decoder
        for up_block in self.decoder:
            x = up_block(x, training=training)

        # Final convolution
        logits = self.final_conv(x)

        # Resize to exact array dimensions
        target_h, target_w = self.array_size
        if x.shape[1] != target_h or x.shape[2] != target_w:
            logits = jax.image.resize(
                logits, (logits.shape[0], target_h, target_w, 1), method="bilinear"
            )

        # Remove channel dimension
        predictions = logits.squeeze(-1)

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
    learning_rate: float = 1e-3,
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
    # model = PhaseShiftPredictor(array_size, rngs=nnx.Rngs(model_key))
    # model = SimplePhaseShiftPredictor(array_size, rngs=nnx.Rngs(model_key))
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))

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
def inspect_data(
    array_size: tuple[int, int] = DEFAULT_ARRAY_SIZE,
    seed: int = 42,
):
    key = jax.random.PRNGKey(seed)
    dataset = Dataset(array_size, batch_size=8, key=key)
    for i in range(3):
        batch = next(dataset)
        rp = batch["radiation_patterns"]
        ps = batch["phase_shifts"]

        logger.info(f"Batch {i + 1}:")
        for x, label in zip(
            [rp, ps],
            ["Radiation Patterns", "Phase Shifts"],
        ):
            logger.info(
                f"{label} statistics:, "
                f"Min: {jnp.min(x):.6f}, "
                f"Max: {jnp.max(x):.6f}, "
                f"Mean: {jnp.mean(x):.6f}, "
                f"Std: {jnp.std(x):.6f}, "
                f"Has NaN: {jnp.any(jnp.isnan(x))}, "
                f"Has Inf: {jnp.any(jnp.isinf(x))}, "
            )


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
    logger.info(f"uv run {' '.join(sys.argv)}")
    app()
