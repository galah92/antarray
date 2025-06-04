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
from jax.experimental.compilation_cache import compilation_cache as cc

import analyze
import data

logger = logging.getLogger(__name__)

# Persistent Jax compilation cache: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
cc.set_cache_dir("/tmp/jax_cache")

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
            ConvBlock(3, 16, (3, 3), padding=enc_pad, rngs=rngs),  # (96, 384, 16)
            partial(nnx.max_pool, window_shape=(3, 6), strides=(3, 6)),  # (32, 64, 16)
            ConvBlock(16, 32, (3, 3), padding=enc_pad, rngs=rngs),  # (32, 64, 32)
            partial(nnx.max_pool, window_shape=(2, 4), strides=(2, 4)),  # (16, 16, 32)
            ConvBlock(32, 64, (3, 3), padding=enc_pad, rngs=rngs),  # (16, 16, 64)
            partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2)),  # (8, 8, 64)
            ConvBlock(64, 128, (3, 3), padding=enc_pad, rngs=rngs),  # (8, 8, 128)
            partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2)),  # (4, 4, 128)
        )
        self.bottleneck = nnx.Sequential(
            ResidualBlock(128, (3, 3), rngs=rngs),  # (4, 4, 128)
            ResidualBlock(128, (3, 3), rngs=rngs),  # (4, 4, 128)
            ResidualBlock(128, (3, 3), rngs=rngs),  # (4, 4, 128)
        )
        self.decoder = nnx.Sequential(
            ConvBlock(128, 64, (3, 3), rngs=rngs),  # (4, 4, 64)
            partial(resize_batch, shape=(8, 8, 64), method="bilinear"),
            ConvBlock(64, 32, (3, 3), rngs=rngs),  # (8, 8, 32)
            partial(resize_batch, shape=(16, 16, 32), method="bilinear"),
            ConvBlock(32, 16, (3, 3), rngs=rngs),  # (16, 16, 16)
            ConvBlock(16, 1, (1, 1), rngs=rngs),  # (16, 16, 1)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.pad(x)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = x.squeeze(-1)  # Remove the channel dimension
        return x


class UNet(nnx.Module):
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

        enc_conv = partial(ConvBlock, padding=enc_pad, rngs=rngs)
        max_pool = lambda window: partial(
            nnx.max_pool(window_shape=window, strides=window)
        )

        self.enc_conv1 = enc_conv(3, 16, (3, 3))  # (96, 384, 16)
        self.pool1 = max_pool((3, 6))  # (32, 64, 16)
        self.enc_conv2 = enc_conv(16, 32, (3, 3))  # (32, 64, 32)
        self.pool2 = max_pool((2, 4))  # (16, 16, 32)
        self.enc_conv3 = enc_conv(32, 64, (3, 3))  # (16, 16, 64)
        self.pool3 = max_pool((2, 2))  # (8, 8, 64)
        self.enc_conv4 = enc_conv(64, 128, (3, 3))  # (8, 8, 128)
        self.pool4 = max_pool((2, 2))  # (4, 4, 128)

        self.bottleneck = nnx.Sequential(
            ResidualBlock(128, (3, 3), rngs=rngs),  # (4, 4, 128)
            ResidualBlock(128, (3, 3), rngs=rngs),  # (4, 4, 128)
            ResidualBlock(128, (3, 3), rngs=rngs),  # (4, 4, 128)
        )

        resize_bilinear = partial(resize_batch, method="bilinear")

        self.dec_conv1 = ConvBlock(128, 64, (3, 3), rngs=rngs)  # (4, 4, 64)
        self.up1 = partial(resize_bilinear, shape=(8, 8, 64))  # (8, 8, 64)
        self.dec_conv2 = ConvBlock(64 + 128, 32, (3, 3), rngs=rngs)  # (8, 8, 32)
        self.up2 = partial(resize_bilinear, shape=(16, 16, 32))  # (16, 16, 32)
        self.dec_conv3 = ConvBlock(32 + 64, 16, (3, 3), rngs=rngs)  # (16, 16, 16)

        self.final_conv = ConvBlock(16, 1, (1, 1), rngs=rngs)  # (16, 16, 1)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.pad(x)

        s1 = self.enc_conv1(x)
        p1 = self.pool1(s1)
        s2 = self.enc_conv2(p1)
        p2 = self.pool2(s2)
        s3 = self.enc_conv3(p2)
        p3 = self.pool3(s3)
        s4 = self.enc_conv4(p3)
        p4 = self.pool4(s4)

        b = self.bottleneck(p4)

        d1 = self.dec_conv1(b)
        u1 = self.up1(d1)
        c1 = jnp.concatenate([u1, s4], axis=-1)  # Skip connection from enc_conv4 output
        d2 = self.dec_conv2(c1)
        u2 = self.up2(d2)
        c2 = jnp.concatenate([u2, s3], axis=-1)  # Skip connection from enc_conv3 output
        d3 = self.dec_conv3(c2)

        out = self.final_conv(d3)

        out = out.squeeze(-1)  # Remove the channel dimension
        return out


class ConvNeXtBlock(nnx.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        expansion_ratio: int = 4,
        layer_scale_init_value: float = 1e-6,
        *,
        rngs: nnx.Rngs,
    ):
        self.dwconv = nnx.Conv(
            dim,
            dim,
            kernel_size=(kernel_size, kernel_size),
            padding="SAME",
            feature_group_count=dim,
            use_bias=True,
            rngs=rngs,
        )
        self.norm = nnx.LayerNorm(dim, rngs=rngs)

        expanded_dim = dim * expansion_ratio
        self.pwconv1 = nnx.Linear(dim, expanded_dim, rngs=rngs)
        self.pwconv2 = nnx.Linear(expanded_dim, dim, rngs=rngs)

        # Add LayerScale
        if layer_scale_init_value > 0:
            self.gamma = nnx.Param(layer_scale_init_value * jnp.ones((dim,)))
        else:
            self.gamma = None

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = nnx.gelu(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma.value * x

        return residual + x


class ConvNeXt(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        # Same padding as other models
        self.pad = nnx.Sequential(
            partial(pad_batch, pad_width=((3, 0), (0, 0), (0, 0)), mode="reflect"),
            partial(pad_batch, pad_width=((0, 3), (0, 0), (0, 0)), mode="constant"),
            partial(pad_batch, pad_width=((0, 0), (12, 12), (0, 0)), mode="wrap"),
        )

        # Match ConvNet's progression more closely
        self.stem = nnx.Sequential(
            nnx.Conv(3, 16, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.LayerNorm(16, rngs=rngs),
        )  # (96, 384, 16)

        # Reduce depth significantly to match ConvNet's complexity
        dims = [16, 32, 64, 128]
        depths = [1, 1, 2, 1]  # Much smaller - only 5 blocks total

        # Use smaller expansion ratio to save memory
        expansion_ratios = [2, 2, 3, 3]  # Instead of 4x everywhere

        # Use smaller kernels to save memory
        kernel_sizes = [3, 5, 7, 7]  # Progressive increase, starting smaller

        self.stages = []
        for i, (dim, depth, exp_ratio, kernel_size) in enumerate(
            zip(dims, depths, expansion_ratios, kernel_sizes)
        ):
            # Downsampling layer (except for first stage)
            if i > 0:
                # Use the same aggressive downsampling as ConvNet
                if i == 1:
                    downsample = partial(
                        nnx.max_pool, window_shape=(3, 6), strides=(3, 6)
                    )
                elif i == 2:
                    downsample = partial(
                        nnx.max_pool, window_shape=(2, 4), strides=(2, 4)
                    )
                else:
                    downsample = partial(
                        nnx.max_pool, window_shape=(2, 2), strides=(2, 2)
                    )
                self.stages.append(downsample)

                # Add a conv to change dimensions after pooling
                self.stages.append(
                    nnx.Conv(dims[i - 1], dim, kernel_size=(1, 1), rngs=rngs)
                )

            # ConvNeXt blocks with reduced expansion
            stage_blocks = []
            for _ in range(depth):
                stage_blocks.append(
                    ConvNeXtBlock(
                        dim,
                        kernel_size=kernel_size,
                        expansion_ratio=exp_ratio,  # Pass custom expansion ratio
                        rngs=rngs,
                    )
                )
            if stage_blocks:  # Only add if there are blocks
                self.stages.append(nnx.Sequential(*stage_blocks))

        self.stages = nnx.Sequential(*self.stages)

        # Simpler head to match ConvNet
        self.head = nnx.Sequential(
            partial(resize_batch, shape=(8, 8, 128), method="bilinear"),
            nnx.Conv(128, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.gelu,
            partial(resize_batch, shape=(16, 16, 64), method="bilinear"),
            nnx.Conv(64, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.gelu,
            nnx.Conv(32, 1, kernel_size=(1, 1), padding="SAME", rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.pad(x)  # (96, 384, 3)
        x = self.stem(x)  # (96, 384, 16)
        x = self.stages(x)  # (4, 4, 128) after all downsampling
        x = self.head(x)  # (16, 16, 1)
        x = x.squeeze(-1)  # (16, 16)
        return x


class FourierLayer(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, int],
        *,
        rngs: nnx.Rngs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1, self.modes2 = modes

        # Much smaller initialization for better stability
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nnx.Param(
            jax.random.normal(
                rngs(), (self.modes1, self.modes2, in_channels, out_channels, 2)
            )
            * scale
        )
        self.weights2 = nnx.Param(
            jax.random.normal(
                rngs(), (self.modes1, self.modes2, in_channels, out_channels, 2)
            )
            * scale
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        batch_size, size_x, size_y, channels = x.shape

        # Only keep low-frequency modes to save memory
        effective_modes1 = min(self.modes1, size_x // 2)
        effective_modes2 = min(
            self.modes2, size_y // 4
        )  # rfft2 has size_y//2+1 frequencies

        if effective_modes1 == 0 or effective_modes2 == 0:
            # If no modes to process, return zero tensor
            return jnp.zeros((batch_size, size_x, size_y, self.out_channels))

        # Compute 2D FFT
        x_ft = jnp.fft.rfft2(x, axes=[1, 2])
        x_ft_real = jnp.stack([x_ft.real, x_ft.imag], axis=-1)

        # Initialize output - only process the modes we actually use
        out_ft = jnp.zeros((batch_size, size_x, size_y // 2 + 1, self.out_channels, 2))

        # Process positive frequencies only (memory efficient)
        if effective_modes1 <= size_x and effective_modes2 <= size_y // 2 + 1:
            x_ft_slice = x_ft_real[:, :effective_modes1, :effective_modes2]
            weights_slice = self.weights1[:effective_modes1, :effective_modes2]
            out_slice = jnp.einsum("bxyic,xyior->bxyor", x_ft_slice, weights_slice)
            out_ft = out_ft.at[:, :effective_modes1, :effective_modes2].set(out_slice)

        # Convert back to complex and inverse FFT
        out_ft_complex = out_ft[..., 0] + 1j * out_ft[..., 1]
        x = jnp.fft.irfft2(out_ft_complex, s=(size_x, size_y), axes=[1, 2])

        return x


class LightFNO(nnx.Module):
    def __init__(
        self,
        modes: tuple[int, int] = (6, 6),  # Much smaller modes
        width: int = 32,  # Much smaller width
        *,
        rngs: nnx.Rngs,
    ):
        self.modes = modes
        self.width = width

        # Same padding as other models
        self.pad = nnx.Sequential(
            partial(pad_batch, pad_width=((3, 0), (0, 0), (0, 0)), mode="reflect"),
            partial(pad_batch, pad_width=((0, 3), (0, 0), (0, 0)), mode="constant"),
            partial(pad_batch, pad_width=((0, 0), (12, 12), (0, 0)), mode="wrap"),
        )

        # Adjust downsampling to naturally reach (16, 16)
        self.stem = nnx.Sequential(
            nnx.Linear(3, self.width, rngs=rngs),
            nnx.Conv(
                self.width, self.width, kernel_size=(3, 3), strides=(6, 24), rngs=rngs
            ),  # (16, 16)
        )

        # Work directly at target resolution
        self.fourier1 = FourierLayer(self.width, self.width, modes, rngs=rngs)
        self.conv1 = nnx.Conv(self.width, self.width, kernel_size=(1, 1), rngs=rngs)

        self.fourier2 = FourierLayer(self.width, self.width, modes, rngs=rngs)
        self.conv2 = nnx.Conv(self.width, self.width, kernel_size=(1, 1), rngs=rngs)

        # Direct output without resize
        self.head = nnx.Sequential(
            nnx.Conv(self.width, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.gelu,
            nnx.Conv(64, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs),
            nnx.gelu,
            nnx.Conv(32, 1, kernel_size=(1, 1), padding="SAME", rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.pad(x)  # (96, 384, 3)
        x = self.stem(x)  # (16, 16, width)

        # Fourier blocks
        x1 = self.fourier1(x)
        x2 = self.conv1(x)
        x = x1 + x2
        x = nnx.gelu(x)

        x1 = self.fourier2(x)
        x2 = self.conv2(x)
        x = x1 + x2
        x = nnx.gelu(x)

        x = self.head(x)  # (16, 16, 1)
        return x.squeeze(-1)  # (16, 16)


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
def train_step(optimizer: nnx.Optimizer, batch: data.DataBatch) -> dict[str, float]:
    def loss_fn(model: nnx.Module):
        radiation_patterns, phase_shifts = batch.radiation_patterns, batch.phase_shifts
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
    metrics["grad_norm"] = optax.global_norm(grads)

    return metrics


def warmup_step(dataset: data.Dataset, optimizer: nnx.Optimizer):
    logger.info("Warming up GPU kernels")
    warmup_batch = next(dataset)
    _ = train_step(optimizer, warmup_batch)
    logger.info("Warmup completed")


def create_trainables(n_steps: int, lr: float, *, key: jax.Array) -> nnx.Optimizer:
    # model = ConvNet(rngs=nnx.Rngs(key))
    # model = UNet(rngs=nnx.Rngs(key))
    # model = ConvNeXt(rngs=nnx.Rngs(key))
    model = LightFNO(rngs=nnx.Rngs(key))
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
    warmup_step(dataset, optimizer)

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
                )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        raise
    finally:
        ckpt_mngr.wait_until_finished()


@app.command()
def pred(theta_deg: list[float] = [], phi_deg: list[float] = [], seed: int = 42):
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
        theta_deg, phi_deg = jnp.asarray(theta_deg), jnp.asarray(phi_deg)
        steering_deg = jnp.stack([theta_deg, phi_deg], axis=-1)
        steering_angles = jnp.radians(steering_deg)

        dataset = data.Dataset(batch_size=1, limit=1, key=dataset_key)
        radiation_pattern, true_excitations = analyze.rad_pattern_from_geo(
            *dataset.array_params, steering_angles
        )
        true_ps = jnp.angle(true_excitations)

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

    theta, phi = dataset.theta_rad, dataset.phi_rad

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

    steering_str = analyze.steering_repr(jnp.degrees(steering_angles.T))
    fig.suptitle(f"Prediction with {steering_str}")

    fig.set_tight_layout(True)

    filepath = "prediction.png"
    if filepath:
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
