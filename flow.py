import logging
import sys
import time
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
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
    radiation_pattern: jax.Array  # (n_theta, n_phi, 3) - pattern & trig encoding
    phase_shifts: jax.Array  # (array_x, array_y)
    steering_angles: jax.Array  # (n_beams, 2) - theta, phi in radians


class Dataset:
    def __init__(
        self,
        array_size: tuple[int, int] = DEFAULT_ARRAY_SIZE,
        spacing_mm: tuple[float, float] = DEFAULT_SPACING_MM,
        theta_end: float = DEFAULT_THETA_END,
        max_n_beams: int = DEFAULT_MAX_N_BEAMS,
        batch_size: int = 128,
        sim_dir_path: Path = data.DEFAULT_SIM_DIR,
        key: jax.Array = None,
        prefetch: bool = True,
        normalize: bool = True,
        radiation_pattern_max=30,  # Maximum radiation pattern value in dB observed
    ):
        self.theta_end = jnp.radians(theta_end)
        self.sim_dir_path = sim_dir_path
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.normalize = normalize
        self.radiation_pattern_max = radiation_pattern_max

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
        n_beams = 2
        theta_steering = jax.random.uniform(key1, (n_beams,)) * self.theta_end
        phi_steering = jax.random.uniform(key2, (n_beams,)) * (2 * jnp.pi)
        steering_angles = jnp.stack((theta_steering, phi_steering), axis=-1)

        radiation_pattern, excitations = analyze.rad_pattern_from_geo(
            *self.array_params,
            steering_angles,
        )
        radiation_pattern = radiation_pattern[:90]  # Clip to front hemisphere
        phase_shifts = jnp.angle(excitations)

        # Add clamping to set negative values to 0 (equivalent to main.py)
        radiation_pattern = jnp.clip(radiation_pattern, a_min=0.0)

        if self.normalize:
            radiation_pattern = radiation_pattern / self.radiation_pattern_max

        # Add trigonometric encoding channels
        phi_rad = jnp.arange(360) * jnp.pi / 180
        sin_phi = jnp.sin(phi_rad)[None, :] * jnp.ones((90, 1))
        cos_phi = jnp.cos(phi_rad)[None, :] * jnp.ones((90, 1))
        radiation_pattern = jnp.stack([radiation_pattern, sin_phi, cos_phi], axis=-1)

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

        self.conv1 = nnx.Conv(3, 32, kernel_size=(7, 7), strides=(2, 4), rngs=rngs)
        self.norm1 = nnx.BatchNorm(32, rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(5, 5), strides=(2, 2), rngs=rngs)
        self.norm2 = nnx.BatchNorm(64, rngs=rngs)
        self.conv3 = nnx.Conv(64, 1, kernel_size=(3, 3), rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = x.reshape(x.shape[0], 90, 360, 3)

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
        # x shape: (B, theta, phi, channels)
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
    """Standard upsampling with transposed convolution and circular phi convolutions"""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        # Use transposed convolution for proper upsampling
        self.upsample = nnx.ConvTranspose(
            in_channels, in_channels, kernel_size=(2, 2), strides=(2, 2), rngs=rngs
        )
        self.conv = CircularDoubleConv(in_channels, out_channels, rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = self.upsample(x)
        return self.conv(x, training=training)


class ConvAutoencoder(nnx.Module):
    def __init__(
        self,
        array_size: tuple[int, int],
        base_channels: int = 16,
        unet_depth: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        self.array_size = array_size
        self.unet_depth = unet_depth

        # Calculate padding needed for exact divisibility to target array size
        H, W = 90, 360  # Input radiation pattern size
        target_h, target_w = array_size  # (16, 16)

        # We want final output to be exactly divisible by target size
        # After 2 up layers from bottleneck, we multiply by 4
        # So we need input dimensions such that after processing we get multiples of target size

        # Working backwards: 96 comes from 24 after 2 up layers (each 2x)
        # 24 comes from 96 after 2 down layers (each 2x)
        # 96 comes from 96 after 2 asymmetric down layers (1x2 each)
        # So we need input to be (96, 384) to get (96, 96) final output

        target_H = 96
        target_W = 384

        pad_height = target_H - H  # 96 - 90 = 6
        pad_width = target_W - W  # 384 - 360 = 24
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Store padding values
        self.input_padding = (
            (0, 0),
            (pad_top, pad_bottom),
            (pad_left, pad_right),
            (0, 0),
        )

        ch = base_channels

        self.inc = DoubleConv(3, ch, rngs=rngs)

        # Encoder
        self.down1 = AsymmetricDown(ch, ch * 2, rngs=rngs)  # (96,384) → (96,192)
        self.down2 = AsymmetricDown(ch * 2, ch * 4, rngs=rngs)  # (96,192) → (96,96)
        self.down3 = Down(ch * 4, ch * 8, rngs=rngs)  # (96,96) → (48,48)
        self.down4 = Down(ch * 8, ch * 16, rngs=rngs)  # (48,48) → (24,24)

        # Bottleneck
        self.bottleneck1 = DoubleConv(ch * 16, ch * 16, rngs=rngs)
        self.bottleneck2 = DoubleConv(ch * 16, ch * 16, rngs=rngs)

        # Decoder
        self.up1 = Up(ch * 16, ch * 8, rngs=rngs)  # (24,24) → (48,48)
        self.up2 = Up(ch * 8, ch * 4, rngs=rngs)  # (48,48) → (96,96)

        self.final_conv = nnx.Conv(ch * 4, 1, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        x = x.reshape(x.shape[0], 90, 360, 3)

        # Pad input to make it divisible by 2^unet_depth
        x = jnp.pad(x, self.input_padding, mode="reflect")

        # Encoder
        x = self.inc(x, training=training)
        x = self.down1(x, training=training)
        x = self.down2(x, training=training)
        x = self.down3(x, training=training)
        x = self.down4(x, training=training)

        # Bottleneck
        x = self.bottleneck1(x, training=training)
        x = self.bottleneck2(x, training=training)

        # Decoder
        x = self.up1(x, training=training)
        x = self.up2(x, training=training)

        target_h, target_w = self.array_size  # (16, 16)
        _, current_h, current_w, _ = x.shape  # (96, 96)

        # (B, 96, 96, ch*4) -> (B, 16, 16, ch*4)
        strides = current_h // target_h, current_w // target_w

        x = nnx.avg_pool(x, window_shape=strides, strides=strides)  # (B, 16, 16, ch*4)

        x = self.final_conv(x)  # (B, 16, 16, 1)

        return x.squeeze(-1)  # (B, 16, 16)


def save_checkpoint(
    mngr: ocp.CheckpointManager,
    optimizer: nnx.Optimizer,
    step: int,
    overwrite: bool = False,
):
    if not overwrite:
        return
    state = nnx.state(optimizer)
    mngr.save(step, args=ocp.args.Composite(state=ocp.args.StandardSave(state)))


def restore_checkpoint(
    mngr: ocp.CheckpointManager,
    optimizer: nnx.Optimizer,
    step: int | None,
) -> int:
    if step is None:
        step = mngr.latest_step()

    state = nnx.state(optimizer)
    restored = mngr.restore(
        step,
        args=ocp.args.Composite(state=ocp.args.StandardRestore(item=state)),
    )
    logger.info(f"Restored checkpoint at step {step}")
    nnx.update(optimizer, restored.state)
    return step


@nnx.jit
def train_step(
    optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
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
def train(
    array_size: tuple[int, int] = DEFAULT_ARRAY_SIZE,
    spacing_mm: tuple[float, float] = DEFAULT_SPACING_MM,
    theta_end: float = DEFAULT_THETA_END,
    max_n_beams: int = DEFAULT_MAX_N_BEAMS,
    n_steps: int = 100_000,
    batch_size: int = 128,
    lr: float = 2e-3,
    seed: int = 42,
    prefetch: bool = True,
    restore: bool = True,
    overwrite: bool = False,
):
    key = jax.random.key(seed)
    key, dataset_key, model_key = jax.random.split(key, num=3)

    dataset = Dataset(
        array_size=array_size,
        spacing_mm=spacing_mm,
        theta_end=theta_end,
        max_n_beams=max_n_beams,
        batch_size=batch_size,
        prefetch=prefetch,
        key=dataset_key,
    )

    ckpt_path = Path.cwd() / "checkpoints"
    ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=1, save_interval_steps=10)
    ckpt_mngr = ocp.CheckpointManager(ckpt_path, options=ckpt_options)

    logger.info("Initializing model and optimizer")
    model = ConvAutoencoder(array_size, rngs=nnx.Rngs(model_key))
    # model = SimplePhaseShiftPredictor(array_size, rngs=nnx.Rngs(model_key))
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_steps)
    optimizer = nnx.Optimizer(model, optax.adam(schedule))

    start_step = 0
    if restore:
        start_step = restore_checkpoint(ckpt_mngr, optimizer, step=None)

    warmup_step(dataset, optimizer)

    logger.info("Starting development run")
    start_time = time.perf_counter()

    try:
        for step in range(start_step, n_steps):
            batch = next(dataset)
            optimizer, train_metrics = train_step(optimizer, batch)
            save_checkpoint(ckpt_mngr, optimizer, step, overwrite)

            if (step + 1) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                avg_time = elapsed / (step + 1)

                elapsed = datetime.min + timedelta(seconds=elapsed)
                logger.info(
                    f"Step: {step + 1:04d} ({elapsed.strftime('%H:%M:%S')}) | "
                    f"Time: {avg_time * 1000:.1f}ms/step | "
                    f"Loss: {train_metrics['loss']:.3f}"
                )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        raise
    finally:
        ckpt_mngr.wait_until_finished()


@app.command()
def pred(
    checkpoint_dir: Path = Path.cwd() / "checkpoints",
    theta_deg: list[float] = None,
    phi_deg: list[float] = None,
    array_size: tuple[int, int] = DEFAULT_ARRAY_SIZE,
    spacing_mm: tuple[float, float] = DEFAULT_SPACING_MM,
    theta_end: float = DEFAULT_THETA_END,
    max_n_beams: int = DEFAULT_MAX_N_BEAMS,
    seed: int = 42,
):
    """Predict phase shifts for given or random steering angles (CPU only)"""

    with jax.default_device(jax.devices("cpu")[0]):
        logger.info("Running prediction on CPU")

        # Create model
        key = jax.random.key(seed)
        model_key, angles_key = jax.random.split(key)
        model = ConvAutoencoder(array_size, rngs=nnx.Rngs(model_key))

        # Get and load latest checkpoint
        restored = ocp.CheckpointManager(checkpoint_dir).restore(
            step=None,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(item=nnx.state(model))
            ),
        )
        step = restored.state["step"]["value"]
        logger.info(f"Restored checkpoint at step {step}")
        # logger.info(f"{'model' in restored.state=}")
        nnx.update(model, restored.state["model"])

        # Generate or use provided steering angles
        if theta_deg is None or phi_deg is None:
            steering_angles = analyze.generate_steering_angles(
                key=angles_key, theta_end=theta_end, max_n_beams=max_n_beams
            )
            logger.info(f"Generated {len(steering_angles)} random steering angles")
        else:
            if len(theta_deg) != len(phi_deg):
                raise typer.BadParameter("theta_deg and phi_deg must have same length")
            steering_angles = jnp.array(
                [[jnp.radians(t), jnp.radians(p)] for t, p in zip(theta_deg, phi_deg)]
            )
            logger.info(f"Using {len(steering_angles)} provided steering angles")

        # Load array parameters and generate radiation pattern
        array_params = analyze.calc_array_params2(
            array_size=array_size,
            spacing_mm=spacing_mm,
            theta_rad=jnp.radians(jnp.arange(180)),
            phi_rad=jnp.radians(jnp.arange(360)),
            sim_path=data.DEFAULT_SIM_DIR / data.DEFAULT_SINGLE_ANT_FILENAME,
        )
        array_params = [jnp.asarray(param) for param in array_params]

        radiation_pattern, _ = analyze.rad_pattern_from_geo(
            *array_params, steering_angles
        )
        radiation_pattern = jnp.clip(radiation_pattern[:90], a_min=0.0) / 30.0
        predicted_phases = model(radiation_pattern[None, :], training=False)[0]

        steering_str = analyze.steering_repr(jnp.degrees(steering_angles.T))
        logger.info(f"Steering: {steering_str}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        theta_rad, phi_rad = jnp.radians(jnp.arange(90)), jnp.radians(jnp.arange(360))
        analyze.plot_ff_2d(
            theta_rad,
            phi_rad,
            radiation_pattern,
            title=f"Input Pattern\n{steering_str}",
            ax=ax1,
        )
        analyze.plot_phase_shifts(
            predicted_phases, title=f"Predicted Phases\n{steering_str}", ax=ax2
        )

        plt.tight_layout()

        plot_path = "prediction.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {plot_path}")
        plt.close()


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
    steering_angles = batch["steering_angles"]

    theta_rad, phi_rad = np.radians(np.arange(90)), np.radians(np.arange(360))

    fig, axes = plt.subplots(batch_size, 2, figsize=(12, 3 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)

    for i in range(batch_size):
        steering_str = analyze.steering_repr(np.degrees(steering_angles[i].T))
        title = f"Radiation Pattern\n{steering_str}"
        analyze.plot_ff_2d(theta_rad, phi_rad, patterns[i], title=title, ax=axes[i, 0])
        title = "Phase Shifts\n"
        analyze.plot_phase_shifts(phase_shifts[i], title=title, ax=axes[i, 1])

    fig.suptitle("Batch Overview: Model Inputs")
    fig.set_tight_layout(True)

    plot_path = "batch_overview.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved batch overview {plot_path}")


@app.command()
def inspect_data(
    array_size: tuple[int, int] = DEFAULT_ARRAY_SIZE,
    spacing_mm: tuple[float, float] = DEFAULT_SPACING_MM,
    theta_end: float = DEFAULT_THETA_END,
    max_n_beams: int = DEFAULT_MAX_N_BEAMS,
    seed: int = 42,
):
    key = jax.random.key(seed)

    dataset_args = array_size, spacing_mm, theta_end, max_n_beams
    dataset = Dataset(*dataset_args, key=key, batch_size=8, normalize=False)
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
        force=True,  # https://github.com/google/orbax/issues/1248
    )
    logging.getLogger("absl").setLevel(logging.CRITICAL)  # Suppress absl logging

    logger.info(f"uv run {' '.join(sys.argv)}")  # Log command line args
    app()
