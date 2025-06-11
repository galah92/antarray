import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


class ArrayConfig:
    ARRAY_SIZE: tuple[int, int] = (8, 8)
    SPACING_MM: tuple[float, float] = (60.0, 60.0)
    FREQUENCY_HZ: float = 2.45e9
    PATTERN_SHAPE: tuple[int, int] = (180, 360)  # (theta, phi)


class InterferenceCorrector(nnx.Module):
    """
    Predicts corrective weights to counteract material interference effects,
    mapping a desired ideal pattern to the weights needed to produce it on a
    physically embedded array.
    """

    def __init__(self, config: ArrayConfig, *, rngs: nnx.Rngs):
        input_size = np.prod(config.PATTERN_SHAPE)
        self.output_shape = (*config.ARRAY_SIZE, 2)

        self.dense1 = nnx.Linear(input_size, 256, rngs=rngs)
        self.dense2 = nnx.Linear(256, np.prod(self.output_shape), rngs=rngs)

    def __call__(self, target_ideal_pattern: jax.Array) -> jax.Array:
        """Takes a batch of ideal patterns and returns a batch of corrective weights."""
        batch_size = target_ideal_pattern.shape[0]

        x = target_ideal_pattern.reshape((batch_size, -1))
        x = nnx.relu(self.dense1(x))
        predicted_weights_flat = self.dense2(x)

        real, imag = jnp.split(predicted_weights_flat, 2, axis=-1)
        return (real + 1j * imag).reshape((batch_size, *self.output_shape))


def get_wavenumber(freq_hz: float) -> float:
    """
    Calculate the wavenumber for a given frequency in Hz.
    The wavenumber is defined as k = 2π/λ, where λ is the wavelength.
    The wavelength is calculated as λ = c/f, where c is the speed of light.
    """
    c = 299792458  # Speed of light in m/s
    wavelength = c / freq_hz  # Wavelength in meters
    k = 2 * np.pi / wavelength  # Wavenumber in radians/meter
    return k


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
    """
    Factory to create a specialized function for calculating analytical weights.

    This function precomputes the element position vectors (k_pos) and returns a
    lightweight, jitted function for calculating steering weights.
    """
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
        phase_shifts = x_phase[:, None] + y_phase[None, :]
        return jnp.exp(-1j * phase_shifts)

    return calculate


def create_pattern_synthesizer(
    element_patterns: jax.Array, config: ArrayConfig
) -> callable:
    """
    Factory to create a specialized pattern synthesis function.

    This function performs a one-time, expensive precomputation of the element
    field basis and returns a lightweight, jitted function for synthesizing
    patterns from that basis.
    """

    @jax.jit
    def precompute_basis(raw_patterns):
        k = get_wavenumber(config.FREQUENCY_HZ)
        x_pos, y_pos = get_element_positions(config.ARRAY_SIZE, config.SPACING_MM)
        k_pos_x, k_pos_y = k * x_pos, k * y_pos

        theta_size, phi_size = raw_patterns.shape[2:4]
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


def steering_angles_sampler(
    key: jax.Array,
    batch_size: int,
    limit: int | None = None,
) -> iter[jax.Array]:
    """
    Creates a Python generator that yields batches of random steering angles.
    """
    if limit is None:
        limit = float("inf")
    for _ in range(limit):
        key, theta_key, phi_key = jax.random.split(key, num=3)
        thetas = jax.random.uniform(theta_key, shape=(batch_size,), maxval=jnp.pi / 2)
        phis = jax.random.uniform(phi_key, shape=(batch_size,), maxval=2 * jnp.pi)
        yield jnp.stack([thetas, phis], axis=-1)


@jax.jit
def calculate_pattern_loss(
    predicted_patterns: jax.Array, target_patterns: jax.Array
) -> jax.Array:
    """Calculates the loss between batches of predicted and target radiation patterns."""
    return optax.losses.squared_error(predicted_patterns, target_patterns).mean()


def create_train_step_fn(
    synthesize_ideal_pattern: callable,
    synthesize_embedded_pattern: callable,
    compute_analytical_weights: callable,
):
    """Factory that creates the jitted training step function."""
    vmapped_analytical_weights = jax.vmap(compute_analytical_weights)
    vmapped_ideal_synthesizer = jax.vmap(synthesize_ideal_pattern)
    vmapped_embedded_synthesizer = jax.vmap(synthesize_embedded_pattern)

    def loss_fn(model: InterferenceCorrector, batch_of_angles_rad: jax.Array):
        analytical_weights = vmapped_analytical_weights(batch_of_angles_rad)
        ideal_patterns = vmapped_ideal_synthesizer(analytical_weights)

        corrective_weights = model(ideal_patterns)
        embedded_patterns = vmapped_embedded_synthesizer(corrective_weights)

        return calculate_pattern_loss(embedded_patterns, ideal_patterns)

    @nnx.jit
    def train_step_fn(optimizer: nnx.Optimizer, batch: jax.Array):
        loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model, batch)
        optimizer.update(grads)
        return loss

    return train_step_fn


def train_pipeline(
    n_steps: int = 10_000,
    batch_size: int = 32,
    lr: float = 1e-4,
    seed: int = 42,
) -> InterferenceCorrector:
    """Main function to set up and run the training pipeline."""
    config = ArrayConfig()
    key = jax.random.key(seed)

    print("Performing one-time precomputation...")

    ideal_shape = (*config.ARRAY_SIZE, *config.PATTERN_SHAPE, 1)
    ideal_patterns = jnp.ones(ideal_shape, dtype=jnp.complex64)

    key, subkey = jax.random.split(key)
    embedded_shape = (*config.ARRAY_SIZE, *config.PATTERN_SHAPE, 2)
    embedded_patterns = (
        jax.random.uniform(subkey, embedded_shape, dtype=jnp.complex64) + 0.5
    )

    synthesize_ideal_pattern = create_pattern_synthesizer(ideal_patterns, config)
    synthesize_embedded_pattern = create_pattern_synthesizer(embedded_patterns, config)
    compute_analytical_weights = create_analytical_weight_calculator(config)

    train_step = create_train_step_fn(
        synthesize_ideal_pattern,
        synthesize_embedded_pattern,
        compute_analytical_weights,
    )

    key, model_key = jax.random.split(key)
    model = InterferenceCorrector(config, rngs=nnx.Rngs(model_key))
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=lr))

    key, data_key = jax.random.split(key)
    sampler = steering_angles_sampler(data_key, batch_size, limit=n_steps)

    print("Starting training...")
    for step, batch in enumerate(sampler):
        loss = train_step(optimizer, batch)

        if (step + 1) % 100 == 0:
            print(f"step {step + 1}/{n_steps}, Loss: {loss.item():.6f}")

    print("Training complete.")
    return optimizer.model


if __name__ == "__main__":
    trained_model = train_pipeline()
