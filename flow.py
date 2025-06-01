import logging
import sys
import time
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
import typer
from flax import nnx
from jax.experimental.compilation_cache import compilation_cache as cc

import analyze
import data

logger = logging.getLogger(__name__)

# Persistent Jax compilation cache: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
cc.set_cache_dir("/tmp/jax_cache")

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
        batch_size: int = 512,
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

        # Precompute trigonometric encoding channels
        phi_rad = jnp.arange(360) * jnp.pi / 180
        self.sin_phi = jnp.sin(phi_rad)[None, :] * jnp.ones((90, 1))
        self.cos_phi = jnp.cos(phi_rad)[None, :] * jnp.ones((90, 1))

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
        radiation_pattern = jnp.dstack([radiation_pattern, self.sin_phi, self.cos_phi])

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


class ConvBlock(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | Sequence[int],
        padding: str = "SAME",
        *,
        rngs: nnx.Rngs,
    ):
        self.conv = nnx.Conv(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=False,
            rngs=rngs,
        )
        self.norm = nnx.BatchNorm(out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)
        x = self.norm(x)
        x = nnx.relu(x)
        return x


def resize_batch(image, shape: Sequence[int], method: str | jax.image.ResizeMethod):
    shape = (image.shape[0], *shape)  # Add batch dimension
    return jax.image.resize(image, shape=shape, method=method)


class ExactConvNet(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        padding = ((0, 0), (3, 3), (12, 12), (0, 0))  # (batch, height, width, channels)
        self.pad = partial(jnp.pad, pad_width=padding, mode="wrap")  # (96, 384, 3)

        self.encoder = nnx.Sequential(
            ConvBlock(3, 32, (3, 3), padding="CIRCULAR", rngs=rngs),  # (96, 384, 32)
            partial(nnx.avg_pool, window_shape=(3, 6), strides=(3, 6)),  # (32, 64, 32)
            ConvBlock(32, 64, (3, 3), padding="CIRCULAR", rngs=rngs),  # (32, 64, 64)
            partial(nnx.avg_pool, window_shape=(2, 4), strides=(2, 4)),  # (16, 16, 64)
            ConvBlock(64, 128, (3, 3), padding="CIRCULAR", rngs=rngs),  # (16, 16, 128)
            partial(nnx.avg_pool, window_shape=(2, 4), strides=(2, 4)),  # (8, 8, 128)
        )
        self.bottleneck = nnx.Sequential(
            ConvBlock(128, 256, (3, 3), rngs=rngs),  # (8, 8, 256)
            ConvBlock(256, 256, (3, 3), rngs=rngs),  # (8, 8, 256)
            ConvBlock(256, 128, (3, 3), rngs=rngs),  # (8, 8, 128)
        )
        self.decoder = nnx.Sequential(
            ConvBlock(128, 64, (3, 3), rngs=rngs),  # (8, 8, 64)
            partial(resize_batch, shape=(16, 16, 64), method="bilinear"),
            ConvBlock(64, 32, (3, 3), rngs=rngs),  # (16, 16, 32)
            ConvBlock(32, 1, (3, 3), rngs=rngs),  # (16, 16, 1)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.pad(x)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = x.squeeze(-1)  # Remove the channel dimension
        return x


def save_checkpoint(
    mngr: ocp.CheckpointManager,
    optimizer: nnx.Optimizer,
    step: int,
    overwrite: bool = False,
):
    if not overwrite:
        return

    handler = ocp.args.StandardSave(nnx.state(optimizer))
    mngr.save(step, args=ocp.args.Composite(state=handler))


def restore_checkpoint(
    mngr: ocp.CheckpointManager,
    optimizer: nnx.Optimizer,
    step: int | None,
) -> int:
    if step is None:
        step = mngr.latest_step()

    handler = ocp.args.StandardRestore(nnx.state(optimizer))
    restored = mngr.restore(step, args=ocp.args.Composite(state=handler))
    nnx.update(optimizer, restored.state)

    logger.info(f"Restored checkpoint at step {step}")
    return step


@nnx.jit
def train_step(optimizer: nnx.Optimizer, batch: DataSample) -> dict[str, float]:
    def loss_fn(model: nnx.Module):
        radiation_patterns = batch["radiation_patterns"]
        phase_shifts = batch["phase_shifts"]
        predictions = model(radiation_patterns)

        # Circular MSE loss
        phase_diff = jnp.abs(predictions - phase_shifts)
        circular_diff = jnp.minimum(phase_diff, 2 * jnp.pi - phase_diff)
        loss = jnp.mean(circular_diff**2)

        phase_rmse = jnp.sqrt(loss)
        return loss, {"loss": loss, "phase_rmse": phase_rmse}

    model = optimizer.model
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads)

    return metrics


def warmup_step(dataset: Dataset, optimizer: nnx.Optimizer):
    logger.info("Warming up GPU kernels")
    warmup_batch = next(dataset)
    _ = train_step(optimizer, warmup_batch)
    logger.info("Warmup completed")


def create_trainables(n_steps: int, lr: float, *, key: jax.Array) -> nnx.Optimizer:
    model = ExactConvNet(rngs=nnx.Rngs(key))
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_steps)
    optimizer = nnx.Optimizer(model, optax.adamw(schedule, weight_decay=1e-4))
    return optimizer


@app.command()
def train(
    n_steps: int = 100_000,
    batch_size: int = 512,
    lr: float = 2e-3,
    seed: int = 42,
    restore: bool = True,
    overwrite: bool = False,
):
    key = jax.random.key(seed)
    key, dataset_key, model_key = jax.random.split(key, num=3)

    dataset = Dataset(batch_size=batch_size, key=dataset_key)
    optimizer = create_trainables(n_steps, lr, key=model_key)

    ckpt_path = Path.cwd() / "checkpoints"
    ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=1, save_interval_steps=10)
    ckpt_mngr = ocp.CheckpointManager(ckpt_path, options=ckpt_options)

    start_step = 0
    if restore:
        start_step = restore_checkpoint(ckpt_mngr, optimizer, step=None)

    warmup_step(dataset, optimizer)

    logger.info("Starting development run")
    start_time = time.perf_counter()

    try:
        for step, batch in zip(range(start_step, n_steps), dataset):
            metrics = train_step(optimizer, batch)
            save_checkpoint(ckpt_mngr, optimizer, step, overwrite)

            if (step + 1) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                avg_time = elapsed / (step + 1 - start_step)

                elapsed = datetime.min + timedelta(seconds=elapsed)
                logger.info(
                    f"Step: {step + 1:04d} ({elapsed.strftime('%H:%M:%S')}) | "
                    f"Time: {avg_time * 1000:.1f}ms/step | "
                    f"Loss: {metrics['loss']:.3f}"
                )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        raise
    finally:
        ckpt_mngr.wait_until_finished()


@app.command()
def pred(
    theta_deg: list[float] = None,
    phi_deg: list[float] = None,
    seed: int = 42,
):
    key = jax.random.key(seed)

    with jax.default_device(jax.devices("cpu")[0]):
        logger.info("Running prediction on CPU")

        # Create model
        model_key, angles_key = jax.random.split(key)
        optimizer = create_trainables(
            n_steps=1,  # Dummy value, we won't train
            lr=0.0,  # No learning rate needed for prediction
            key=model_key,
        )

        # Get and load latest checkpoint
        ckpt_path = Path.cwd() / "checkpoints"
        ckpt_options = ocp.CheckpointManagerOptions(read_only=True)
        ckpt_mngr = ocp.CheckpointManager(ckpt_path, options=ckpt_options)
        _ = restore_checkpoint(ckpt_mngr, optimizer, step=None)

        optimizer.model.eval()

        steering_angles = jnp.array(
            [[jnp.radians(t), jnp.radians(p)] for t, p in zip(theta_deg, phi_deg)]
        )
        logger.info(f"Using {len(steering_angles)} provided steering angles")

        # Load array parameters and generate radiation pattern
        array_params = analyze.calc_array_params2(
            array_size=DEFAULT_ARRAY_SIZE,
            spacing_mm=DEFAULT_SPACING_MM,
            theta_rad=jnp.radians(jnp.arange(180)),
            phi_rad=jnp.radians(jnp.arange(360)),
            sim_path=data.DEFAULT_SIM_DIR / data.DEFAULT_SINGLE_ANT_FILENAME,
        )
        array_params = [jnp.asarray(param) for param in array_params]

        radiation_pattern, _ = analyze.rad_pattern_from_geo(
            *array_params, steering_angles
        )
        transformed = jnp.clip(radiation_pattern[:90], a_min=0.0) / 30.0
        phi_rad = jnp.arange(360) * jnp.pi / 180
        sin_phi = jnp.sin(phi_rad)[None, :] * jnp.ones((90, 1))
        cos_phi = jnp.cos(phi_rad)[None, :] * jnp.ones((90, 1))
        transformed = jnp.dstack([transformed, sin_phi, cos_phi])
        predicted_phases = optimizer.model(transformed[None, ...])[0]

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
def visualize_dataset(batch_size: int = 4, seed: int = 42):
    key = jax.random.key(seed)

    dataset = Dataset(batch_size=batch_size, key=key)
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
