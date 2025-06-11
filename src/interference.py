from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import nnx


class Config:
    ARRAY_SIZE: tuple[int, int] = (8, 8)
    SPACING_MM: tuple[float, float] = (60.0, 60.0)
    FREQUENCY: float = 2.45e9
    C: float = 299792458.0
    WAVELENGTH: float = C / FREQUENCY
    K_WAVENUMBER: float = 2 * jnp.pi / WAVELENGTH
    THETA_POINTS: int = 180
    PHI_POINTS: int = 360


class InterferenceCorrector(nnx.Module):
    """
    Predicts corrective weights to counteract material interference effects,
    mapping a desired ideal pattern to the weights needed to produce it on a
    physically embedded array.
    """

    def __init__(self, config: Config, *, rngs: nnx.Rngs):
        self.config = config
        xn, yn = self.config.ARRAY_SIZE
        output_size = xn * yn * 2
        input_size = config.THETA_POINTS * config.PHI_POINTS

        self.dense1 = nnx.Linear(input_size, 256, rngs=rngs)
        self.dense2 = nnx.Linear(256, output_size, rngs=rngs)

    def __call__(self, target_ideal_pattern: jax.Array) -> jax.Array:
        """Takes a batch of ideal patterns and returns a batch of corrective weights."""
        batch_size = target_ideal_pattern.shape[0]

        x = target_ideal_pattern.reshape((batch_size, -1))
        x = nnx.relu(self.dense1(x))
        predicted_weights_flat = self.dense2(x)

        real_part, imag_part = jnp.split(predicted_weights_flat, 2, axis=-1)
        return (real_part + 1j * imag_part).reshape(
            (batch_size, *self.config.ARRAY_SIZE)
        )


def get_element_positions(
    array_size: tuple[int, int], spacing_mm: tuple[float, float]
) -> tuple[jax.Array, jax.Array]:
    """Calculates element positions in meters, centered at the origin."""
    xn, yn = array_size
    dx_m, dy_m = spacing_mm[0] / 1000, spacing_mm[1] / 1000
    x_pos = (jnp.arange(xn) - (xn - 1) / 2) * dx_m
    y_pos = (jnp.arange(yn) - (yn - 1) / 2) * dy_m
    return x_pos, y_pos


@jax.jit
def calculate_array_factor_weights(
    k_pos_x: jax.Array, k_pos_y: jax.Array, steering_angle_rad: jax.Array
) -> jax.Array:
    """Calculates ideal, analytical weights for a given steering angle."""
    theta_steer, phi_steer = steering_angle_rad[0], steering_angle_rad[1]

    ux = jnp.sin(theta_steer) * jnp.cos(phi_steer)
    uy = jnp.sin(theta_steer) * jnp.sin(phi_steer)

    x_phase, y_phase = k_pos_x * ux, k_pos_y * uy
    phase_shifts = x_phase[:, None] + y_phase[None, :]
    return jnp.exp(-1j * phase_shifts)


@jax.jit
def precompute_element_field_basis(
    raw_element_patterns: jax.Array, config: Config
) -> jax.Array:
    """
    Precomputes the field basis for each element, combining its intrinsic
    pattern with its geometric phase position.
    """
    k = config.K_WAVENUMBER
    theta_rad = jnp.radians(jnp.arange(config.THETA_POINTS))
    phi_rad = jnp.radians(jnp.arange(config.PHI_POINTS))
    x_pos, y_pos = get_element_positions(config.ARRAY_SIZE, config.SPACING_MM)
    k_pos_x = k * x_pos
    k_pos_y = k * y_pos

    sin_theta = jnp.sin(theta_rad)
    ux = sin_theta[:, None] * jnp.cos(phi_rad)[None, :]
    uy = sin_theta[:, None] * jnp.sin(phi_rad)[None, :]

    phase_x = ux[..., None] * k_pos_x
    phase_y = uy[..., None] * k_pos_y

    geo_phase = phase_x[..., None] + phase_y[:, :, None, :]
    geo_factor = jnp.exp(1j * geo_phase)

    return jnp.einsum("xytpz,tpxy->tpzxy", raw_element_patterns, geo_factor)


@jax.jit
def synthesize_pattern(weights: jax.Array, element_field_basis: jax.Array) -> jax.Array:
    """Synthesizes a total power pattern from weights and a precomputed field basis."""
    total_field = jnp.einsum("xy,tpzxy->tpz", weights, element_field_basis)
    power_pattern = jnp.sum(jnp.abs(total_field) ** 2, axis=-1)
    return power_pattern


def generate_random_angles(key: jax.Array, batch_size: int) -> jax.Array:
    """Generates a batch of random (theta, phi) angles in radians."""
    key_theta, key_phi = jax.random.split(key)
    thetas = jax.random.uniform(key_theta, shape=(batch_size,), maxval=jnp.pi / 2)
    phis = jax.random.uniform(key_phi, shape=(batch_size,), maxval=2 * jnp.pi)
    return jnp.stack([thetas, phis], axis=-1)


@jax.jit
def calculate_pattern_loss(
    predicted_patterns: jax.Array,
    target_patterns: jax.Array,
) -> jax.Array:
    """Calculates the loss between batches of predicted and target radiation patterns."""
    return optax.losses.squared_error(predicted_patterns, target_patterns).mean()


def create_train_step_fn(
    ideal_element_basis: jax.Array,
    embedded_element_basis: jax.Array,
    k_pos_x: jax.Array,
    k_pos_y: jax.Array,
):
    """Factory that creates the jitted training step function."""
    compute_analytical_weights = jax.vmap(
        partial(calculate_array_factor_weights, k_pos_x, k_pos_y)
    )
    synthesize_ideal_pattern = jax.vmap(
        partial(synthesize_pattern, element_field_basis=ideal_element_basis)
    )
    synthesize_embedded_pattern = jax.vmap(
        partial(synthesize_pattern, element_field_basis=embedded_element_basis)
    )

    def loss_fn(model: InterferenceCorrector, batch_of_angles_rad: jax.Array):
        # 1. Generate the TARGET: the ideal pattern in a perfect, free-space environment.
        analytical_weights = compute_analytical_weights(batch_of_angles_rad)
        ideal_patterns = synthesize_ideal_pattern(analytical_weights)

        # 2. PREDICT: Give the model the ideal pattern and ask it to predict
        #    the corrective weights needed to counteract material interference.
        corrective_weights = model(ideal_patterns)

        # 3. SIMULATE: Synthesize the pattern that would actually be produced
        #    by applying the corrective weights to the embedded (with interference) array.
        embedded_patterns = synthesize_embedded_pattern(corrective_weights)

        # 4. COMPARE: The loss is the difference between the desired ideal
        #    pattern and the actual pattern produced by the compensated embedded array.
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
    config = Config()
    key = jax.random.key(seed)

    print("Performing one-time precomputation...")
    ideal_shape = (*config.ARRAY_SIZE, config.THETA_POINTS, config.PHI_POINTS, 1)
    embedded_shape = (*config.ARRAY_SIZE, config.THETA_POINTS, config.PHI_POINTS, 2)

    ideal_patterns = jnp.ones(ideal_shape, dtype=jnp.complex64)

    key, subkey = jax.random.split(key)
    embedded_patterns = (
        jax.random.uniform(subkey, embedded_shape, dtype=jnp.complex64) + 0.5
    )

    ideal_element_basis = precompute_element_field_basis(ideal_patterns, config)
    embedded_element_basis = precompute_element_field_basis(embedded_patterns, config)

    x_pos, y_pos = get_element_positions(config.ARRAY_SIZE, config.SPACING_MM)
    k_pos_x, k_pos_y = config.K_WAVENUMBER * x_pos, config.K_WAVENUMBER * y_pos

    key, model_key = jax.random.split(key)
    model = InterferenceCorrector(config, rngs=nnx.Rngs(model_key))
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=lr))

    train_step = create_train_step_fn(
        ideal_element_basis, embedded_element_basis, k_pos_x, k_pos_y
    )

    print("Starting training...")
    for step in range(n_steps):
        key, step_key = jax.random.split(key)
        batch_of_angles = generate_random_angles(step_key, batch_size)
        loss = train_step(optimizer, batch_of_angles)

        if (step + 1) % 100 == 0:
            print(f"step {step + 1}/{n_steps}, Loss: {loss.item():.3f}")

    print("Training complete.")
    return optimizer.model


if __name__ == "__main__":
    trained_model = train_pipeline()
