"""Shared utilities, neural network components, and model architectures."""

import logging
import time
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.typing import ArrayLike

logger = logging.getLogger(__name__)

# Persistent Jax compilation cache
cc.set_cache_dir("/tmp/jax_cache")


# =============================================================================
# Array and Image Utilities
# =============================================================================


def pad_batch(
    image: jax.Array,
    pad_width: Sequence[int | Sequence[int]],
    mode: str = "constant",
) -> jax.Array:
    """Pad batch of images with proper dimension handling."""
    pad_width = np.asarray(pad_width, dtype=np.int32)
    if pad_width.shape[0] == 3:  # Add batch dimension
        pad_width = np.pad(pad_width, ((1, 0), (0, 0)))
    return jnp.pad(image, pad_width=pad_width, mode=mode)


def resize_batch(image, shape: Sequence[int], method: str | jax.image.ResizeMethod):
    """Resize batch of images maintaining batch dimension."""
    shape = (image.shape[0], *shape)  # Add batch dimension
    return jax.image.resize(image, shape=shape, method=method)


def circular_mse_fn(target: ArrayLike, pred: ArrayLike) -> jax.Array:
    phase_diff = jnp.abs(pred - target)
    circular_diff = jnp.minimum(phase_diff, 2 * jnp.pi - phase_diff)
    circular_mse = (circular_diff**2).mean()

    return circular_mse


# =============================================================================
# Basic Neural Network Components
# =============================================================================


class ConvBlock(nnx.Module):
    """Standard convolutional block with BatchNorm and ReLU."""

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
        self.out_features = out_features

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv(x)
        x = self.norm(x)
        x = nnx.relu(x)
        return x


# =============================================================================
# Model Components and Architectures
# =============================================================================


class PatternEncoder(nnx.Module):
    """Shared pattern encoder for conditioning on target radiation patterns."""

    def __init__(self, base_channels: int = 64, *, rngs: nnx.Rngs):
        self.pattern_pad = nnx.Sequential(
            partial(pad_batch, pad_width=((6, 6), (0, 0), (0, 0)), mode="reflect"),
            partial(pad_batch, pad_width=((0, 0), (12, 12), (0, 0)), mode="wrap"),
        )
        self.encoder = nnx.Sequential(
            ConvBlock(1, base_channels // 4, (3, 3), rngs=rngs),
            partial(nnx.max_pool, window_shape=(3, 6), strides=(3, 6)),
            ConvBlock(base_channels // 4, base_channels // 2, (3, 3), rngs=rngs),
            partial(nnx.max_pool, window_shape=(4, 4), strides=(4, 4)),
            ConvBlock(base_channels // 2, base_channels, (3, 3), rngs=rngs),
        )

    def __call__(self, pattern: ArrayLike) -> jax.Array:
        x = pattern[..., None]  # Add channel dimension
        x = self.pattern_pad(x)
        return self.encoder(x)


class WeightsProcessor(nnx.Module):
    """Shared weights processor for complex antenna weights."""

    def __init__(self, base_channels: int = 64, *, rngs: nnx.Rngs):
        self.processor = ConvBlock(2, base_channels, (3, 3), rngs=rngs)

    def __call__(self, weights: ArrayLike) -> jax.Array:
        # Convert complex weights to real/imag channels
        weights_real = jnp.real(weights)[..., None]
        weights_imag = jnp.imag(weights)[..., None]
        weights_input = jnp.concatenate([weights_real, weights_imag], axis=-1)
        return self.processor(weights_input)


class UNetCore(nnx.Module):
    """Core UNet architecture with encoder-decoder structure."""

    def __init__(self, base_channels: int = 64, *, rngs: nnx.Rngs):
        # Encoder
        self.down1 = ConvBlock(base_channels * 2, base_channels * 2, (3, 3), rngs=rngs)
        self.down2 = ConvBlock(base_channels * 2, base_channels * 4, (3, 3), rngs=rngs)
        self.down3 = ConvBlock(base_channels * 4, base_channels * 8, (3, 3), rngs=rngs)

        # Bottleneck
        self.bottleneck = ConvBlock(
            base_channels * 8, base_channels * 8, (3, 3), rngs=rngs
        )

        # Decoder
        self.up3 = ConvBlock(base_channels * 16, base_channels * 4, (3, 3), rngs=rngs)
        self.up2 = ConvBlock(base_channels * 8, base_channels * 2, (3, 3), rngs=rngs)
        self.up1 = ConvBlock(base_channels * 4, base_channels, (3, 3), rngs=rngs)

    def __call__(self, x: ArrayLike) -> jax.Array:
        # Encoder path
        x1 = self.down1(x)
        x2 = self.down2(nnx.max_pool(x1, (2, 2), (2, 2)))
        x3 = self.down3(nnx.max_pool(x2, (2, 2), (2, 2)))

        # Bottleneck
        bottleneck = self.bottleneck(nnx.max_pool(x3, (2, 2), (2, 2)))

        # Decoder path with skip connections
        up3 = resize_batch(bottleneck, (4, 4, bottleneck.shape[-1]), "bilinear")
        up3 = jnp.concatenate([up3, x3], axis=-1)
        up3 = self.up3(up3)

        up2 = resize_batch(up3, (8, 8, up3.shape[-1]), "bilinear")
        up2 = jnp.concatenate([up2, x2], axis=-1)
        up2 = self.up2(up2)

        up1 = resize_batch(up2, (16, 16, up2.shape[-1]), "bilinear")
        up1 = jnp.concatenate([up1, x1], axis=-1)
        up1 = self.up1(up1)

        return up1


class DenoisingUNet(nnx.Module):
    """UNet for denoising complex antenna weights with pattern conditioning."""

    def __init__(self, base_channels: int = 64, *, rngs: nnx.Rngs):
        self.pattern_encoder = PatternEncoder(base_channels, rngs=rngs)
        self.weights_processor = WeightsProcessor(base_channels, rngs=rngs)
        self.unet_core = UNetCore(base_channels, rngs=rngs)

        # Time embedding for diffusion timestep
        self.time_mlp = nnx.Sequential(
            nnx.Linear(1, base_channels, rngs=rngs),
            nnx.relu,
            nnx.Linear(base_channels, base_channels * 2, rngs=rngs),
        )

        # Output
        self.output = nnx.Conv(base_channels, 2, (1, 1), rngs=rngs)

    def __call__(
        self, noisy_weights: ArrayLike, target_pattern: ArrayLike, timestep: ArrayLike
    ) -> jax.Array:
        # Encode inputs
        pattern_features = self.pattern_encoder(target_pattern)
        weights_features = self.weights_processor(noisy_weights)

        # Time embedding
        time_emb = self.time_mlp(timestep[..., None])
        time_emb = time_emb[:, None, None, :]

        # Combine features
        x = jnp.concatenate([pattern_features, weights_features], axis=-1)
        x = x + time_emb

        # UNet processing
        x = self.unet_core(x)

        # Output
        output = self.output(x)
        return output[..., 0] + 1j * output[..., 1]


class InterferenceCorrector(nnx.Module):
    """UNet for interference correction with skip connections."""

    def __init__(self, *, enc_pad="SAME", rngs: nnx.Rngs):
        self.pad = nnx.Sequential(
            partial(pad_batch, pad_width=((6, 6), (0, 0), (0, 0)), mode="reflect"),
            partial(pad_batch, pad_width=((0, 0), (12, 12), (0, 0)), mode="wrap"),
        )

        # Encoder
        self.enc_conv0 = ConvBlock(1, 16, (3, 3), padding=enc_pad, rngs=rngs)
        self.pool0 = partial(nnx.max_pool, window_shape=(3, 6), strides=(3, 6))

        self.enc_conv1 = ConvBlock(16, 32, (3, 3), padding=enc_pad, rngs=rngs)
        self.pool1 = partial(nnx.max_pool, window_shape=(4, 4), strides=(4, 4))

        self.enc_conv2 = ConvBlock(32, 64, (3, 3), padding=enc_pad, rngs=rngs)
        self.pool2 = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))

        self.enc_conv3 = ConvBlock(64, 128, (3, 3), padding=enc_pad, rngs=rngs)
        self.pool3 = partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2))

        # Bottleneck
        self.bottleneck = nnx.Sequential(
            ConvBlock(128, 128, (3, 3), rngs=rngs),
            ConvBlock(128, 128, (3, 3), rngs=rngs),
            ConvBlock(128, 128, (3, 3), rngs=rngs),
        )

        # Decoder
        self.dec_conv0 = ConvBlock(128, 64, (3, 3), rngs=rngs)
        self.upsample0 = partial(resize_batch, shape=(8, 8, 64), method="bilinear")
        self.dec_conv1 = ConvBlock(192, 64, (3, 3), rngs=rngs)

        self.upsample1 = partial(resize_batch, shape=(16, 16, 64), method="bilinear")
        self.dec_conv2 = ConvBlock(128, 32, (3, 3), rngs=rngs)
        self.dec_conv3 = ConvBlock(32, 16, (3, 3), rngs=rngs)

        self.final_conv = nnx.Conv(16, 1, (1, 1), padding="SAME", rngs=rngs)

    def __call__(self, x: ArrayLike) -> tuple[jax.Array, jax.Array]:
        x = x[..., None]  # Add channel dimension
        x = self.pad(x)

        # Encoder path
        e0 = self.enc_conv0(x)
        p0 = self.pool0(e0)

        s1 = self.enc_conv1(p0)
        p1 = self.pool1(s1)

        s2 = self.enc_conv2(p1)
        p2 = self.pool2(s2)

        s3 = self.enc_conv3(p2)
        p3 = self.pool3(s3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder path
        d0 = self.dec_conv0(b)
        u0 = self.upsample0(d0)

        # Skip connections
        c0 = jnp.concatenate([u0, s3], axis=-1)
        d1 = self.dec_conv1(c0)

        u1 = self.upsample1(d1)
        c1 = jnp.concatenate([u1, s2], axis=-1)
        d2 = self.dec_conv2(c1)

        d3 = self.dec_conv3(d2)
        phase_shifts = self.final_conv(d3)

        phase_shifts = phase_shifts.squeeze(-1)  # Remove channel dimension
        complex_weights = jnp.exp(-1j * phase_shifts)
        return complex_weights, phase_shifts


# =============================================================================
# Checkpoint Management
# =============================================================================


def save_checkpoint(
    mngr: ocp.CheckpointManager,
    item: nnx.Optimizer | nnx.Module,
    step: int,
    overwrite: bool = False,
):
    """Save checkpoint using orbax."""
    if not overwrite:
        return

    if isinstance(item, nnx.Optimizer):
        handler = ocp.args.StandardSave(nnx.state(item))
        mngr.save(step, args=ocp.args.Composite(state=handler))
    else:  # Module
        state = nnx.state(item)
        mngr.save(step, args=ocp.args.StandardSave(state))
        logger.info(f"Saved checkpoint at step {step}")


def restore_checkpoint(
    mngr: ocp.CheckpointManager,
    item: nnx.Optimizer | nnx.Module,
    step: int | None = None,
) -> int:
    """Restore checkpoint using orbax."""
    if step is None:
        step = mngr.latest_step()

    if step is None:
        return 0

    if isinstance(item, nnx.Optimizer):
        handler = ocp.args.StandardRestore(nnx.state(item))
        restored = mngr.restore(step, args=ocp.args.Composite(state=handler))
        nnx.update(item, restored.state)  # ty: ignore[possibly-unbound-attribute]
    else:
        state = nnx.state(item)
        restored = mngr.restore(step, args=ocp.args.StandardRestore(state))
        nnx.update(item, restored)

    logger.info(f"Restored checkpoint at step {step}")
    return step


# =============================================================================
# Training Utilities
# =============================================================================


def steering_angles_sampler(
    key: jax.Array,
    batch_size: int,
    limit: int,
    theta_end: float = 60,
    device: str | None = None,
) -> Iterator[jax.Array]:
    """Creates a generator that yields batches of random steering angles."""
    device = jax.devices(device)[0]
    minval = jax.device_put(0.0, device=device)
    theta_end = jax.device_put(jnp.radians(theta_end), device=device)
    phi_end = jax.device_put(2 * jnp.pi, device=device)
    uniform = jax.random.uniform
    for _ in range(limit):
        key, theta_key, phi_key = jax.random.split(key, num=3)
        theta = uniform(theta_key, shape=(batch_size,), minval=minval, maxval=theta_end)
        phi = uniform(phi_key, shape=(batch_size,), minval=minval, maxval=phi_end)
        yield jnp.stack([theta, phi], axis=-1)


def create_progress_logger(
    total_steps: int,
    log_every: int = 100,
    start_step: int = 0,
):
    """Factory function that creates a specialized progress logger."""

    start_time = time.perf_counter()

    def log_progress(step: int, metrics: dict):
        """Log training progress with timing and metrics."""
        if (step + 1) % log_every != 0:
            return

        # Calculate timing based on steps since start
        steps_elapsed = step - start_step + 1
        elapsed = time.perf_counter() - start_time
        avg_time = elapsed / steps_elapsed
        elapsed_str = (datetime.min + timedelta(seconds=elapsed)).strftime("%H:%M:%S")

        # Base progress info
        progress_str = f"Step: {step + 1:04d}/{total_steps + start_step} ({elapsed_str}) | Time: {avg_time * 1000:.1f}ms/step"

        # Format metrics
        metric_strs = []
        for key, value in metrics.items():
            if hasattr(value, "item"):  # JAX array
                value = value.item()
            metric_strs.append(f"{key}: {value:.3f}")

        full_message = progress_str + " | " + " | ".join(metric_strs)

        logger.info(full_message)

    return log_progress
