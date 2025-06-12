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
        return jnp.exp(-1j * phase_shifts), phase_shifts

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
