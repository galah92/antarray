from collections.abc import Iterable, Sequence
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.experimental.compilation_cache import compilation_cache as cc

# Persistent Jax compilation cache: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
cc.set_cache_dir("/tmp/jax_cache")


class ArrayConfig:
    ARRAY_SIZE: tuple[int, int] = (16, 16)
    SPACING_MM: tuple[float, float] = (60.0, 60.0)
    FREQUENCY_HZ: float = 2.45e9
    PATTERN_SHAPE: tuple[int, int] = (180, 360)  # (theta, phi)


def pad_batch(
    image: jax.Array,
    pad_width: Sequence[int | Sequence[int]],
    mode: str = "constant",
) -> jax.Array:
    pad_width = np.asarray(pad_width, dtype=np.int32)
    if pad_width.shape[0] == 3:  # Add batch dimension
        pad_width = np.pad(pad_width, ((1, 0), (0, 0)))
    return jnp.pad(image, pad_width=pad_width, mode=mode)


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


class InterferenceCorrector(nnx.Module):
    def __init__(
        self,
        *,
        enc_pad="SAME",
        rngs: nnx.Rngs,
    ):
        self.pad = nnx.Sequential(
            # Simmetry around the zenith and nadir: (192, 360, 1)
            partial(pad_batch, pad_width=((6, 6), (0, 0), (0, 0)), mode="reflect"),
            # Wrap around the azimuth: (192, 384, 1)
            partial(pad_batch, pad_width=((0, 0), (12, 12), (0, 0)), mode="wrap"),
        )
        self.encoder = nnx.Sequential(
            ConvBlock(1, 16, (3, 3), padding=enc_pad, rngs=rngs),  # (192, 384, 16)
            partial(nnx.max_pool, window_shape=(3, 6), strides=(3, 6)),  # (64, 64, 16)
            ConvBlock(16, 32, (3, 3), padding=enc_pad, rngs=rngs),  # (64, 64, 32)
            partial(nnx.max_pool, window_shape=(4, 4), strides=(4, 4)),  # (16, 16, 32)
            ConvBlock(32, 64, (3, 3), padding=enc_pad, rngs=rngs),  # (16, 16, 64)
            partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2)),  # (8, 8, 64)
            ConvBlock(64, 128, (3, 3), padding=enc_pad, rngs=rngs),  # (8, 8, 128)
            partial(nnx.max_pool, window_shape=(2, 2), strides=(2, 2)),  # (4, 4, 128)
        )
        self.bottleneck = nnx.Sequential(
            ConvBlock(128, 128, (3, 3), rngs=rngs),  # (4, 4, 128)
            ConvBlock(128, 128, (3, 3), rngs=rngs),  # (4, 4, 128)
            ConvBlock(128, 128, (3, 3), rngs=rngs),  # (4, 4, 128)
        )
        self.decoder = nnx.Sequential(
            ConvBlock(128, 64, (3, 3), rngs=rngs),  # (4, 4, 64)
            partial(resize_batch, shape=(8, 8, 64), method="bilinear"),
            ConvBlock(64, 32, (3, 3), rngs=rngs),  # (8, 8, 32)
            partial(resize_batch, shape=(16, 16, 32), method="bilinear"),
            ConvBlock(32, 16, (3, 3), rngs=rngs),  # (16, 16, 16)
            nnx.Conv(16, 1, (1, 1), padding="SAME", rngs=rngs),  # (16, 16, 1)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x[..., None]  # Add channel dimension
        x = self.pad(x)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = x.squeeze(-1)  # Remove channel dimension
        x = jnp.exp(-1j * x)  # Convert phase shifts to complex weights
        return x


def get_wavenumber(freq_hz: float) -> float:
    """Calculate the wavenumber for a given frequency in Hz."""
    c = 299792458
    wavelength = c / freq_hz
    return 2 * np.pi / wavelength


def get_element_positions(
    array_size: tuple[int, int], spacing_mm: tuple[float, float]
) -> tuple[jax.Array, jax.Array]:
    """Calculates element positions in meters, centered at the origin."""
    xn, yn = array_size
    dx_m, dy_m = spacing_mm[0] / 1000, spacing_mm[1] / 1000
    x_pos = (jnp.arange(xn) - (xn - 1) / 2) * dx_m
    y_pos = (jnp.arange(yn) - (yn - 1) / 2) * dy_m
    return x_pos, y_pos


def create_analytical_weight_calculator(config: ArrayConfig) -> callable:
    """Factory to create a specialized function for calculating analytical weights."""
    k = get_wavenumber(config.FREQUENCY_HZ)
    x_pos, y_pos = get_element_positions(config.ARRAY_SIZE, config.SPACING_MM)
    k_pos_x, k_pos_y = k * x_pos, k * y_pos

    @jax.jit
    def calculate(steering_angle_rad: jax.Array) -> jax.Array:
        """Calculates ideal, analytical weights for a given steering angle."""
        theta_steer, phi_steer = steering_angle_rad[0], steering_angle_rad[1]

        ux = jnp.sin(theta_steer) * jnp.cos(phi_steer)
        uy = jnp.sin(theta_steer) * jnp.sin(phi_steer)

        x_phase, y_phase = k_pos_x * ux, k_pos_y * uy

        phase_shifts = jnp.add.outer(x_phase, y_phase)
        return jnp.exp(-1j * phase_shifts)

    return calculate


def create_pattern_synthesizer(
    element_patterns: jax.Array, config: ArrayConfig
) -> callable:
    """Factory to create a specialized pattern synthesis function."""

    @jax.jit
    def precompute_basis(raw_patterns):
        k = get_wavenumber(config.FREQUENCY_HZ)
        x_pos, y_pos = get_element_positions(config.ARRAY_SIZE, config.SPACING_MM)
        k_pos_x, k_pos_y = k * x_pos, k * y_pos

        theta_size, phi_size = config.PATTERN_SHAPE
        theta_rad = jnp.radians(jnp.arange(theta_size))
        phi_rad = jnp.radians(jnp.arange(phi_size))

        sin_theta = jnp.sin(theta_rad)
        ux = sin_theta[:, None] * jnp.cos(phi_rad)[None, :]
        uy = sin_theta[:, None] * jnp.sin(phi_rad)[None, :]

        phase_x, phase_y = ux[..., None] * k_pos_x, uy[..., None] * k_pos_y
        geo_phase = phase_x[..., None] + phase_y[:, :, None, :]
        geo_factor = jnp.exp(1j * geo_phase)

        return jnp.einsum("xytpz,tpxy->tpzxy", raw_patterns, geo_factor)

    element_field_basis = precompute_basis(element_patterns)

    @jax.jit
    def synthesize(weights: jax.Array) -> jax.Array:
        """Synthesizes a pattern from weights using the precomputed basis."""
        total_field = jnp.einsum("xy,tpzxy->tpz", weights, element_field_basis)
        power_pattern = jnp.sum(jnp.abs(total_field) ** 2, axis=-1)
        return power_pattern

    return synthesize


def create_element_patterns(
    config: ArrayConfig, key: jax.Array, is_embedded: bool
) -> jax.Array:
    """Simulates the element patterns for either an ideal or embedded array."""
    theta_size, phi_size = config.PATTERN_SHAPE
    theta = jnp.radians(jnp.arange(theta_size))

    # --- MINIMAL CHANGE IS HERE ---
    # Start with a base cosine model for the field amplitude.
    base_field_amp = jnp.cos(theta)
    # A realistic element on a ground plane has no back-lobe.
    # Set field to zero for theta > 90 degrees.
    base_field_amp = base_field_amp.at[theta > np.pi / 2].set(0)

    # Expand to a full 2D pattern (omnidirectional in phi)
    base_field_amp = base_field_amp[:, None] * jnp.ones((theta_size, phi_size))

    if not is_embedded:
        ideal_field = base_field_amp[None, None, :, :, None]
        ideal_field = jnp.tile(ideal_field, (*config.ARRAY_SIZE, 1, 1, 1))
        return ideal_field.astype(jnp.complex64)

    # --- Embedded Case: Simulate distortion ---
    num_pols = 2
    final_shape = (*config.ARRAY_SIZE, *config.PATTERN_SHAPE, num_pols)
    low_res_shape = (*config.ARRAY_SIZE, 10, 20, num_pols)

    key, amp_key, phase_key = jax.random.split(key, 3)

    amp_dist_low_res = jax.random.uniform(
        amp_key, low_res_shape, minval=0.5, maxval=1.5
    )
    amp_distortion = jax.image.resize(amp_dist_low_res, final_shape, method="bicubic")

    phase_dist_low_res = jax.random.uniform(phase_key, low_res_shape, maxval=2 * np.pi)
    phase_distortion = jax.image.resize(
        phase_dist_low_res, final_shape, method="bicubic"
    )

    distorted_amplitude = base_field_amp[None, None, ..., None] * amp_distortion
    distorted_field = distorted_amplitude * jnp.exp(1j * phase_distortion)

    return distorted_field.astype(jnp.complex64)


def steering_angles_sampler(
    key: jax.Array,
    batch_size: int,
    theta_end: float = np.radians(60),
    limit: int | None = None,
) -> Iterable[jax.Array]:
    """
    Creates a Python generator that yields batches of random steering angles.
    """
    if limit is None:
        limit = float("inf")
    i = 0
    while i < limit:
        key, theta_key, phi_key = jax.random.split(key, num=3)
        thetas = jax.random.uniform(theta_key, shape=(batch_size,), maxval=theta_end)
        phis = jax.random.uniform(phi_key, shape=(batch_size,), maxval=2 * jnp.pi)
        yield jnp.stack([thetas, phis], axis=-1)
        i += 1


@jax.jit
def normalize_patterns(patterns: jax.Array) -> jax.Array:
    """Performs peak normalization on a batch of radiation patterns."""
    max_vals = jnp.max(patterns, axis=(1, 2), keepdims=True)
    return patterns / (max_vals + 1e-8)


@jax.jit
def convert_to_db(patterns: jax.Array, floor_db: float = -60) -> jax.Array:
    """Converts a batch of normalized linear power patterns to a dB scale."""
    linear_floor = 10.0 ** (floor_db / 10.0)
    clipped_patterns = jnp.maximum(patterns, linear_floor)
    return 10.0 * jnp.log10(clipped_patterns)


@jax.jit
def calculate_pattern_loss(
    predicted_patterns: jax.Array, target_patterns: jax.Array
) -> tuple[jax.Array, dict]:
    """Calculates the loss and metrics between predicted and target patterns."""
    mse = optax.losses.squared_error(predicted_patterns, target_patterns).mean()
    rmse = jnp.sqrt(mse)
    return mse, {"mse": mse, "rmse": rmse}


def create_train_step_fn(
    synthesize_ideal_pattern: callable,
    synthesize_embedded_pattern: callable,
    compute_analytical_weights: callable,
):
    """Factory that creates the jitted training step function."""
    vmapped_analytical_weights = jax.vmap(compute_analytical_weights)
    vmapped_ideal_synthesizer = jax.vmap(synthesize_ideal_pattern)
    vmapped_embedded_synthesizer = jax.vmap(synthesize_embedded_pattern)

    def loss_fn(model: nnx.Module, batch_of_angles_rad: jax.Array):
        analytical_weights = vmapped_analytical_weights(batch_of_angles_rad)
        ideal_patterns = vmapped_ideal_synthesizer(analytical_weights)
        normalized_ideal_patterns = normalize_patterns(ideal_patterns)

        corrective_weights = model(normalized_ideal_patterns)

        embedded_patterns = vmapped_embedded_synthesizer(corrective_weights)
        normalized_embedded_patterns = normalize_patterns(embedded_patterns)

        ideal_patterns_db = convert_to_db(normalized_ideal_patterns)
        embedded_patterns_db = convert_to_db(normalized_embedded_patterns)

        return calculate_pattern_loss(embedded_patterns_db, ideal_patterns_db)

    @nnx.jit
    def train_step_fn(optimizer: nnx.Optimizer, batch: jax.Array):
        model = optimizer.model
        (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
        optimizer.update(grads)
        return metrics

    return train_step_fn


def train_pipeline(
    n_steps: int = 10_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 42,
):
    """Main function to set up and run the training pipeline."""
    config = ArrayConfig()
    key = jax.random.key(seed)

    print("Performing one-time precomputation")

    key, ideal_key, embedded_key = jax.random.split(key, 3)
    ideal_patterns = create_element_patterns(config, ideal_key, is_embedded=False)
    embedded_patterns = create_element_patterns(config, embedded_key, is_embedded=True)

    train_step = create_train_step_fn(
        create_pattern_synthesizer(ideal_patterns, config),
        create_pattern_synthesizer(embedded_patterns, config),
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
                    f"MSE: {metrics['mse'].item():.3f}, "
                    f"RMSE: {metrics['rmse'].item():.3f}"
                )
    except KeyboardInterrupt:
        print("Training interrupted by user")

    print("Training completed")


if __name__ == "__main__":
    train_pipeline()
