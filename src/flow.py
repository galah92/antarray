import logging
import sys
import time
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Sequence

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

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    add_completion=False,
)


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
        self.conv1 = nnx.Conv(
            in_features,
            in_features,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=False,
            rngs=rngs,
        )
        self.norm1 = nnx.BatchNorm(in_features, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=False,
            rngs=rngs,
        )
        self.norm2 = nnx.BatchNorm(out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv1(x)
        x = self.norm1(x)
        x = nnx.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = nnx.relu(x)
        return x


def resize_batch(image, shape: Sequence[int], method: str | jax.image.ResizeMethod):
    shape = (image.shape[0], *shape)  # Add batch dimension
    return jax.image.resize(image, shape=shape, method=method)


class ResidualBlock(nnx.Module):
    def __init__(
        self,
        features: int,
        kernel_size: int | Sequence[int],
        padding: str = "SAME",
        *,
        rngs: nnx.Rngs,
    ):
        self.conv1 = nnx.Conv(
            features,
            features,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=False,
            rngs=rngs,
        )
        self.norm1 = nnx.BatchNorm(features, rngs=rngs)
        self.conv2 = nnx.Conv(
            features,
            features,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=False,
            rngs=rngs,
        )
        self.norm2 = nnx.BatchNorm(features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = nnx.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual  # Residual connection
        x = nnx.relu(x)
        return x


class AttentionBlock(nnx.Module):
    def __init__(self, features: int, *, rngs: nnx.Rngs):
        self.query_conv = nnx.Conv(
            features, features // 8, kernel_size=(1, 1), rngs=rngs
        )
        self.key_conv = nnx.Conv(features, features // 8, kernel_size=(1, 1), rngs=rngs)
        self.value_conv = nnx.Conv(features, features, kernel_size=(1, 1), rngs=rngs)
        self.output_conv = nnx.Conv(features, features, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch_size, height, width, channels = x.shape

        # Compute attention
        query = self.query_conv(x).reshape(batch_size, height * width, -1)
        key = self.key_conv(x).reshape(batch_size, height * width, -1)
        value = self.value_conv(x).reshape(batch_size, height * width, -1)

        # Attention weights
        attention = jax.nn.softmax(
            jnp.matmul(query, key.transpose(0, 2, 1)) / jnp.sqrt(key.shape[-1]), axis=-1
        )
        attended = jnp.matmul(attention, value)
        attended = attended.reshape(batch_size, height, width, channels)

        # Output projection
        output = self.output_conv(attended)
        return x + output  # Residual connection


def pad_batch(
    image: jax.Array,
    pad_width: Sequence[int | Sequence[int]],
    mode: str = "constant",
) -> jax.Array:
    pad_width = np.asarray(pad_width, dtype=np.int32)
    if pad_width.shape[0] == 3:  # Add batch dimension
        pad_width = np.pad(pad_width, ((1, 0), (0, 0)))
    return jnp.pad(image, pad_width=pad_width, mode=mode)


class SEBlock(nnx.Module):
    """Squeeze-and-Excitation block"""

    def __init__(self, channels: int, reduction: int = 2, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(channels, channels // reduction, rngs=rngs)
        self.fc2 = nnx.Linear(channels // reduction, channels, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Global average pooling
        se = jnp.mean(x, axis=(1, 2), keepdims=True)  # (batch, 1, 1, channels)
        se = se.squeeze((1, 2))  # (batch, channels)

        # Squeeze
        se = self.fc1(se)
        se = nnx.relu(se)

        # Excitation
        se = self.fc2(se)
        se = nnx.sigmoid(se)

        # Scale
        se = se[:, None, None, :]  # (batch, 1, 1, channels)
        return x * se


class SEConvBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = (3, 3),
        padding: str = "SAME",
        *,
        rngs: nnx.Rngs,
    ):
        self.conv = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=False,
            rngs=rngs,
        )
        self.norm = nnx.BatchNorm(out_channels, rngs=rngs)
        self.activation = nnx.gelu
        self.se = SEBlock(out_channels, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.se(x)
        return x


class SEResidualBlock(nnx.Module):
    def __init__(self, features: int, kernel_size, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            features,
            features,
            kernel_size=kernel_size,
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.norm1 = nnx.BatchNorm(features, rngs=rngs)
        self.conv2 = nnx.Conv(
            features,
            features,
            kernel_size=kernel_size,
            padding="SAME",
            use_bias=False,
            rngs=rngs,
        )
        self.norm2 = nnx.BatchNorm(features, rngs=rngs)
        self.se = SEBlock(features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = nnx.gelu(x)  # Changed from relu to match SEConvBlock
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.se(x)
        x = x + residual
        x = nnx.gelu(x)
        return x


class ConvNet(nnx.Module):
    def __init__(
        self,
        enc_pad="SAME",
        *,
        rngs: nnx.Rngs,
    ):
        self.pad = nnx.Sequential(
            # Simmetry around the zenith: (93, 360, 3)
            partial(pad_batch, pad_width=((3, 0), (0, 0), (0, 0)), mode="reflect"),
            # Ignore the horizon: (96, 360, 3)
            partial(pad_batch, pad_width=((0, 3), (0, 0), (0, 0)), mode="constant"),
            # Wrap around the azimuth: (96, 384, 3)
            partial(pad_batch, pad_width=((0, 0), (12, 12), (0, 0)), mode="wrap"),
        )
        self.encoder = nnx.Sequential(
            SEConvBlock(3, 16, (3, 3), padding=enc_pad, rngs=rngs),  # (96, 384, 16)
            partial(nnx.max_pool, window_shape=(3, 6), strides=(3, 6)),  # (32, 64, 16)
            SEConvBlock(16, 32, (3, 3), padding=enc_pad, rngs=rngs),  # (32, 64, 32)
            partial(nnx.max_pool, window_shape=(2, 4), strides=(2, 4)),  # (16, 16, 32)
            SEConvBlock(32, 64, (3, 3), padding=enc_pad, rngs=rngs),  # (16, 16, 64)
            partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2)),  # (8, 8, 64)
            SEConvBlock(64, 128, (3, 3), padding=enc_pad, rngs=rngs),  # (8, 8, 128)
            partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2)),  # (4, 4, 128)
        )
        self.bottleneck = nnx.Sequential(
            SEResidualBlock(128, (3, 3), rngs=rngs),  # (4, 4, 128)
            SEResidualBlock(128, (3, 3), rngs=rngs),  # (4, 4, 128)
            SEResidualBlock(128, (3, 3), rngs=rngs),  # (4, 4, 128)
        )
        self.decoder = nnx.Sequential(
            SEConvBlock(128, 64, (3, 3), rngs=rngs),  # (4, 4, 64)
            partial(resize_batch, shape=(8, 8, 64), method="bilinear"),
            SEConvBlock(64, 32, (3, 3), rngs=rngs),  # (8, 8, 32)
            partial(resize_batch, shape=(16, 16, 32), method="bilinear"),
            SEConvBlock(32, 16, (3, 3), rngs=rngs),  # (16, 16, 16)
            nnx.Conv(16, 1, (1, 1), padding="SAME", rngs=rngs),  # (16, 16, 1)
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


def circular_mse_fn(
    batch: data.DataBatch,
    pred_phase_shifts: jax.Array,
) -> jax.Array:
    phase_shifts = batch.phase_shifts

    # Circular MSE loss
    phase_diff = jnp.abs(pred_phase_shifts - phase_shifts)
    circular_diff = jnp.minimum(phase_diff, 2 * jnp.pi - phase_diff)
    circular_mse = jnp.mean(circular_diff**2)

    return circular_mse


def create_physics_loss_fn(dataset: data.Dataset):
    array_params = dataset.array_params
    f = partial(analyze.rad_pattern_from_geo_and_phase_shifts, *array_params[2:])
    pattern_from_phase_shifts = jax.vmap(f)
    transform_fn = jax.vmap(dataset.transform_fn)

    def physics_loss_fn(
        batch: data.DataBatch,
        pred_phase_shifts: jax.Array,
    ):
        radiation_patterns = batch.radiation_patterns

        pred_patterns = pattern_from_phase_shifts(pred_phase_shifts)
        pred_patterns = transform_fn(pred_patterns)

        # Remove trig encoding dimensions
        radiation_patterns = radiation_patterns[:, :, :, 0]
        pred_patterns = pred_patterns[:, :, :, 0]

        loss = jnp.mean((pred_patterns - radiation_patterns) ** 2)
        loss = loss * (30.0**2)  # Scale by max_dB^2

        return loss

    return physics_loss_fn


def loss_fn(
    model: ConvNet,
    batch: data.DataBatch,
    dataset: data.Dataset,
    *,
    use_physics_loss: bool = False,
    use_circular_mse: bool = True,
) -> tuple[jax.Array, dict]:
    pred_phase_shifts = model(batch.radiation_patterns)

    metrics = {}

    loss = jnp.zeros(())

    if use_physics_loss:
        physics_loss_fn = create_physics_loss_fn(dataset)
        physics_loss = physics_loss_fn(batch, pred_phase_shifts)
        loss += physics_loss
        metrics["physics_loss"] = physics_loss

    if use_circular_mse:
        circular_mse = circular_mse_fn(batch, pred_phase_shifts)
        alpha = 10 if use_physics_loss else 1
        loss += circular_mse * alpha
        metrics["circular_mse"] = circular_mse

    metrics["loss"] = loss
    return loss, metrics


def create_train_step(loss_fn):
    @nnx.jit
    def train_step(optimizer: nnx.Optimizer, batch: data.DataBatch) -> dict[str, float]:
        model = optimizer.model
        (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
        optimizer.update(grads)
        metrics["grad_norm"] = optax.global_norm(grads)

        return metrics

    return train_step


def warmup_step(dataset: data.Dataset, optimizer: nnx.Optimizer, train_step):
    logger.info("Warming up GPU kernels")
    warmup_batch = next(dataset)
    _ = train_step(optimizer, warmup_batch)
    logger.info("Warmup completed")


def create_trainables(n_steps: int, lr: float, *, key: jax.Array) -> nnx.Optimizer:
    model = ConvNet(rngs=nnx.Rngs(key))
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_steps)
    optimizer = nnx.Optimizer(model, optax.adamw(schedule, weight_decay=1e-4))
    return optimizer


@app.command()
def train(
    n_steps: int = 100_000,
    batch_size: int = 1024,
    lr: float = 2e-3,
    seed: int = 42,
    restore: bool = True,
    overwrite: bool = False,
):
    key = jax.random.key(seed)
    key, dataset_key, model_key = jax.random.split(key, num=3)

    optimizer = create_trainables(n_steps, lr, key=model_key)

    ckpt_path = Path.cwd() / "checkpoints"
    ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=1, save_interval_steps=10)
    ckpt_mngr = ocp.CheckpointManager(ckpt_path, options=ckpt_options)

    start_step = 0
    if restore:
        start_step = restore_checkpoint(ckpt_mngr, optimizer, step=None)
        logger.info(f"Resuming from step {start_step}")

    n_steps -= start_step  # Adjust n_steps based on the restored step
    dataset = data.Dataset(batch_size=batch_size, limit=n_steps, key=dataset_key)

    train_step = create_train_step(loss_fn=partial(loss_fn, dataset=dataset))
    warmup_step(dataset, optimizer, train_step)

    logger.info("Starting development run")
    start_time = time.perf_counter()

    try:
        for step, batch in enumerate(dataset, start=start_step):
            metrics = train_step(optimizer, batch)
            save_checkpoint(ckpt_mngr, optimizer, step, overwrite)

            if (step + 1) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                avg_time = elapsed / (step + 1 - start_step)

                elapsed = datetime.min + timedelta(seconds=elapsed)
                logger.info(
                    f"Step: {step + 1:04d} ({elapsed.strftime('%H:%M:%S')}) | "
                    f"Time: {avg_time * 1000:.1f}ms/step | "
                    f"Loss: {metrics['loss']:.3f} | "
                    f"Grad Norm: {metrics['grad_norm']:.3f} | "
                    f"Circular MSE: {metrics.get('circular_mse', 0):.3f} | "
                    f"Physics Loss: {metrics.get('physics_loss', 0):.3f} | "
                )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        raise
    finally:
        ckpt_mngr.wait_until_finished()


@app.command()
def eval(seed: int = 42):
    key = jax.random.key(seed)
    key, dataset_key, model_key = jax.random.split(key, num=3)
    batch_size = 1024

    optimizer = create_trainables(n_steps=batch_size, lr=0.0, key=model_key)

    ckpt_path = Path.cwd() / "checkpoints"
    ckpt_options = ocp.CheckpointManagerOptions(read_only=True)
    ckpt_mngr = ocp.CheckpointManager(ckpt_path, options=ckpt_options)

    start_step = restore_checkpoint(ckpt_mngr, optimizer, step=None)
    logger.info(f"Resuming from step {start_step}")

    optimizer.model.eval()

    dataset = data.Dataset(batch_size=batch_size, limit=batch_size, key=dataset_key)
    _batch = next(dataset)
    # metrics = train_step(optimizer, batch)
    # logger.info(
    #     f"Evaluation at step {start_step}: "
    #     f"Loss: {metrics['loss']:.3f} | "
    #     f"Phase RMSE: {metrics['phase_rmse']:.3f} | "
    #     f"Grad Norm: {metrics['grad_norm']:.3f}"
    # )


@app.command()
def pred(
    theta_deg: list[float] = [],
    phi_deg: list[float] = [],
    filepath: str = "prediction.png",
    seed: int = 42,
):
    key = jax.random.key(seed)
    key, dataset_key, model_key = jax.random.split(key, num=3)

    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        logger.info("Running prediction on CPU")

        optimizer = create_trainables(n_steps=1, lr=0.0, key=model_key)

        # Get and load latest checkpoint
        ckpt_path = Path.cwd() / "checkpoints"
        ckpt_options = ocp.CheckpointManagerOptions(read_only=True)
        with ocp.CheckpointManager(ckpt_path, options=ckpt_options) as ckpt_mngr:
            _ = restore_checkpoint(ckpt_mngr, optimizer, step=None)

        optimizer.model.eval()

        # Use provided steering angles
        theta_d, phi_d = jnp.asarray(theta_deg), jnp.asarray(phi_deg)
        steering_deg = jnp.stack([theta_d, phi_d], axis=-1)
        steering_angles = jnp.radians(steering_deg)

        dataset = data.Dataset(batch_size=1, limit=1, key=dataset_key)
        radiation_pattern, true_excitations = analyze.rad_pattern_from_geo(
            *dataset.array_params, steering_angles
        )
        true_ps = np.asarray(np.angle(true_excitations))

        transformed = dataset.transform_fn(radiation_pattern)
        pred_ps = optimizer.model(transformed[None, ...])[0]

        true_rp = data.ff_from_phase_shifts(true_ps)
        pred_rp = data.ff_from_phase_shifts(pred_ps)
        true_rp, pred_rp = true_rp[:90], pred_rp[:90]  # Limit to zenith angles

    fig, axs = plt.subplots(2, 4, figsize=(18, 8))

    analyze.plot_phase_shifts(true_ps, title="Ground Truth Phase Shifts", ax=axs[0, 0])
    analyze.plot_phase_shifts(pred_ps, title="Predicted Phase Shifts", ax=axs[1, 0])

    clip_rp = True  # Clip ratiation patterns to non-negative values
    if clip_rp:
        true_rp, pred_rp = true_rp.clip(min=0), pred_rp.clip(min=0)

    theta, phi = np.asarray(dataset.theta_rad), np.asarray(dataset.phi_rad)

    title = "Ground Truth 2D Radiation Pattern"
    analyze.plot_ff_2d(theta, phi, true_rp, title=title, ax=axs[0, 1])
    title = "Predicted 2D Radiation Pattern"
    analyze.plot_ff_2d(theta, phi, pred_rp, title=title, ax=axs[1, 1])

    title = "Ground Truth Sine-Space Radiation Pattern"
    analyze.plot_sine_space(theta, phi, true_rp, title=title, ax=axs[0, 2])
    title = "Predicted Sine-Space Radiation Pattern"
    analyze.plot_sine_space(theta, phi, pred_rp, title=title, ax=axs[1, 2])

    axs[0, 3].remove()
    axs[0, 3] = fig.add_subplot(2, 4, 4, projection="3d")
    title = "Ground Truth 3D Radiation Pattern"
    analyze.plot_ff_3d(theta, phi, true_rp, title=title, ax=axs[0, 3])

    axs[1, 3].remove()
    axs[1, 3] = fig.add_subplot(2, 4, 8, projection="3d")
    title = "Predicted 3D Radiation Pattern"
    analyze.plot_ff_3d(theta, phi, pred_rp, title=title, ax=axs[1, 3])

    steering_str = analyze.steering_repr(np.degrees(steering_angles.T))
    fig.suptitle(f"Prediction with {steering_str}")

    fig.set_layout_engine("tight")

    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Prediction example saved to {filepath}")


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
