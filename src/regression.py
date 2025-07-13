import logging
from functools import partial
from pathlib import Path
from typing import NamedTuple, Sequence

import cyclopts
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx
from jax.typing import ArrayLike

import data
import physics
from physics import (
    ArrayConfig,
    compute_geps,
    load_aeps,
    synthesize_pattern,
)
from training import (
    circular_mse_fn,
    create_progress_logger,
    pad_batch,
    resize_batch,
    restore_checkpoint,
    save_checkpoint,
)
from utils import setup_logging

logger = logging.getLogger(__name__)

app = cyclopts.App()


class SEBlock(nnx.Module):
    """Squeeze-and-Excitation block"""

    def __init__(self, channels: int, reduction: int = 2, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(channels, channels // reduction, rngs=rngs)
        self.fc2 = nnx.Linear(channels // reduction, channels, rngs=rngs)

    def __call__(self, x: ArrayLike) -> jax.Array:
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

    def __call__(self, x: ArrayLike) -> jax.Array:
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

    def __call__(self, x: ArrayLike) -> jax.Array:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = nnx.gelu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.se(x)
        x = x + residual
        x = nnx.gelu(x)
        return x


class RegressionNet(nnx.Module):
    """ConvNet for direct regression from radiation patterns to phase shifts."""

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

    def __call__(self, x: ArrayLike) -> jax.Array:
        x = self.pad(x)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = x.squeeze(-1)  # Remove the channel dimension
        return x


synthesize_patterns = jax.vmap(synthesize_pattern, in_axes=(None, 0))  # Batch weights


class ArrayParams(NamedTuple):
    element_fields: jax.Array


@nnx.jit
def train_step(
    optimizer: nnx.Optimizer,
    batch: data.DataBatch,
    params: ArrayParams,
) -> dict[str, float]:
    model = optimizer.model

    def loss_fn(model: RegressionNet, batch: data.DataBatch) -> tuple[jax.Array, dict]:
        pred_phase_shifts = model(batch.radiation_patterns)

        circular_mse = circular_mse_fn(batch.phase_shifts, pred_phase_shifts)

        pred_weights = jnp.exp(1j * pred_phase_shifts)
        pred_patterns = synthesize_patterns(params.element_fields, pred_weights)
        # FIXME: also need to denormalize the patterns
        radiation_patterns = batch.radiation_patterns[..., 0]
        pattern_loss = ((pred_patterns - radiation_patterns) ** 2).mean()

        loss = circular_mse + pattern_loss
        metrics = dict(
            loss=loss,
            circular_mse=circular_mse,
            circular_rmse=jnp.sqrt(circular_mse),
            pattern_loss=pattern_loss,
        )
        return loss, metrics

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
    optimizer.update(grads)
    metrics["grad_norm"] = optax.global_norm(grads)

    return metrics


def warmup_step(
    dataset: data.Dataset,
    optimizer: nnx.Optimizer,
    params: ArrayParams,
):
    logger.info("Warming up GPU kernels")
    optimizer.model.eval()
    warmup_batch = next(dataset)
    _ = train_step(optimizer, warmup_batch, params)
    optimizer.model.train()
    logger.info("Warmup completed")


def create_trainables(n_steps: int, lr: float, *, key: jax.Array) -> nnx.Optimizer:
    model = RegressionNet(rngs=nnx.Rngs(key))
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
    kind: physics.Kind = "cst",
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

    n_steps -= start_step
    dataset = data.Dataset(batch_size=batch_size, limit=n_steps, key=dataset_key)

    config = ArrayConfig()
    element_data = load_aeps(config, kind=kind)
    aeps = element_data.aeps
    config = element_data.config
    geps = compute_geps(aeps, config)
    array_params = ArrayParams(element_fields=jnp.asarray(geps))

    warmup_step(dataset, optimizer, array_params)

    logger.info("Starting training")
    log_progress = create_progress_logger(
        total_steps=n_steps, log_every=10, start_step=start_step
    )

    try:
        for step, batch in enumerate(dataset, start=start_step):
            metrics = train_step(optimizer, batch, array_params)
            save_checkpoint(ckpt_mngr, optimizer, step, overwrite)
            log_progress(step, metrics)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        raise
    finally:
        ckpt_mngr.wait_until_finished()


if __name__ == "__main__":
    setup_logging()
    app()
