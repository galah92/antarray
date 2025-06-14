from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from shared import (
    ArrayConfig,
    ConvBlock,
    calculate_pattern_loss,
    convert_to_db,
    create_analytical_weight_calculator,
    create_element_patterns,
    create_pattern_synthesizer,
    normalize_patterns,
    pad_batch,
    resize_batch,
    steering_angles_sampler,
)


class InterferenceCorrector(nnx.Module):
    def __init__(
        self,
        *,
        enc_pad="SAME",
        rngs: nnx.Rngs,
    ):
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

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = x[..., None]  # Add channel dimension
        x = self.pad(x)

        # Encoder Path
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

        # Decoder Path
        d0 = self.dec_conv0(b)
        u0 = self.upsample0(d0)

        # Concatenate with skip connection s3
        c0 = jnp.concatenate([u0, s3], axis=-1)
        d1 = self.dec_conv1(c0)

        u1 = self.upsample1(d1)
        # Concatenate with skip connection s2
        c1 = jnp.concatenate([u1, s2], axis=-1)
        d2 = self.dec_conv2(c1)

        d3 = self.dec_conv3(d2)

        phase_shifts = self.final_conv(d3)

        phase_shifts = phase_shifts.squeeze(-1)  # Remove channel dimension
        complex_weights = jnp.exp(-1j * phase_shifts)
        return complex_weights, phase_shifts


def create_train_step_fn(
    synthesize_ideal_pattern: Callable,
    synthesize_embedded_pattern: Callable,
    compute_analytical_weights: Callable,
):
    """Factory that creates the jitted training step function."""
    vmapped_analytical_weights = jax.vmap(compute_analytical_weights)
    vmapped_ideal_synthesizer = jax.vmap(synthesize_ideal_pattern)
    vmapped_embedded_synthesizer = jax.vmap(synthesize_embedded_pattern)

    def loss_fn(model: InterferenceCorrector, batch_of_angles_rad: jax.Array):
        analytical_weights, analytical_phase_shifts = vmapped_analytical_weights(
            batch_of_angles_rad
        )

        ideal_patterns = vmapped_ideal_synthesizer(analytical_weights)
        normalized_ideal_patterns = normalize_patterns(ideal_patterns)

        corrective_weights, corrective_phase_shifts = model(normalized_ideal_patterns)

        embedded_patterns = vmapped_embedded_synthesizer(corrective_weights)
        normalized_embedded_patterns = normalize_patterns(embedded_patterns)

        ideal_patterns_db = convert_to_db(normalized_ideal_patterns)
        embedded_patterns_db = convert_to_db(normalized_embedded_patterns)

        loss, metrics = calculate_pattern_loss(embedded_patterns_db, ideal_patterns_db)

        phase_shifts_mse = optax.losses.squared_error(
            corrective_phase_shifts, analytical_phase_shifts
        ).mean()
        metrics["phase_shifts_mse"] = phase_shifts_mse
        metrics["phase_shifts_rmse"] = jnp.sqrt(phase_shifts_mse)
        metrics["phase_shifts_std"] = jnp.std(corrective_phase_shifts)

        loss = loss + phase_shifts_mse  # Combine losses

        return loss, metrics

    @nnx.jit
    def train_step_fn(optimizer: nnx.Optimizer, batch: jax.Array):
        model = optimizer.model
        (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
        optimizer.update(grads)
        metrics["grad_norm"] = optax.global_norm(grads)
        return metrics

    return train_step_fn


def train_pipeline(
    n_steps: int = 10_000,
    batch_size: int = 512,
    lr: float = 5e-4,
    seed: int = 42,
):
    """Main function to set up and run the training pipeline."""
    config = ArrayConfig()
    key = jax.random.key(seed)

    print("Performing one-time precomputation")

    key, ideal_key, embedded_key = jax.random.split(key, 3)
    ideal_patterns = create_element_patterns(config, ideal_key, is_embedded=False)
    _embedded_patterns = create_element_patterns(config, embedded_key, is_embedded=True)

    train_step = create_train_step_fn(
        create_pattern_synthesizer(ideal_patterns, config),
        create_pattern_synthesizer(ideal_patterns, config),
        create_analytical_weight_calculator(config),
    )

    key, model_key = jax.random.split(key)
    model = InterferenceCorrector(rngs=nnx.Rngs(model_key))
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=lr))

    key, data_key = jax.random.split(key)
    sampler = steering_angles_sampler(data_key, batch_size, limit=n_steps)

    print("Starting training")
    try:
        for step, batch in enumerate(sampler):
            metrics = train_step(optimizer, batch)

            if (step + 1) % 100 == 0:
                print(
                    f"step {step + 1}/{n_steps}, "
                    f"grad_norm: {metrics['grad_norm'].item():.3f}, "
                    f"MSE: {metrics['mse'].item():.3f}, "
                    f"RMSE: {metrics['rmse'].item():.3f}, "
                    f"phase_shifts_std: {metrics['phase_shifts_std'].item():.3f}, "
                    f"phase_shifts_mse: {metrics['phase_shifts_mse'].item():.3f}, "
                    f"phase_shifts_rmse: {metrics['phase_shifts_rmse'].item():.3f}"
                )
    except KeyboardInterrupt:
        print("Training interrupted by user")

    print("Training completed")


if __name__ == "__main__":
    train_pipeline()
