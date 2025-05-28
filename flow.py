from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import typer
from flax import nnx

import analyze
import data

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

DEFAULT_ARRAY_SIZE = (16, 16)
DEFAULT_SPACING_MM = (60, 60)
DEFAULT_THETA_END = 65.0
DEFAULT_MAX_N_BEAMS = 3
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_NUM_EPOCHS = 100
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")


class TrainingConfig(NamedTuple):
    array_size: tuple[int, int] = DEFAULT_ARRAY_SIZE
    spacing_mm: tuple[float, float] = DEFAULT_SPACING_MM
    theta_end: float = DEFAULT_THETA_END
    max_n_beams: int = DEFAULT_MAX_N_BEAMS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    num_epochs: int = DEFAULT_NUM_EPOCHS
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    eval_every: int = 1000
    checkpoint_every: int = 5000
    seed: int = 42


class DataSample(NamedTuple):
    radiation_pattern: jnp.ndarray  # Shape: (n_theta, n_phi)
    phase_shifts: jnp.ndarray  # Shape: (array_x, array_y)
    steering_angles: jnp.ndarray  # Shape: (n_beams, 2) - theta, phi in radians


class DataGenerator:
    def __init__(
        self, config: TrainingConfig, sim_dir_path: Path = data.DEFAULT_SIM_DIR
    ):
        self.theta_end = config.theta_end
        self.sim_dir_path = sim_dir_path

        # Load and prepare array parameters
        array_params = analyze.calc_array_params2(
            array_size=config.array_size,
            spacing_mm=config.spacing_mm,
            theta_rad=jnp.radians(jnp.arange(180)),
            phi_rad=jnp.radians(jnp.arange(360)),
            sim_path=sim_dir_path / data.DEFAULT_SINGLE_ANT_FILENAME,
        )

        # Convert to JAX arrays and make them static
        self.array_params = [jnp.asarray(param) for param in array_params]

        # Beam probability distribution
        self.n_beams_prob = data.get_beams_prob(config.max_n_beams)

    def generate_sample(self, key: jax.random.PRNGKey) -> DataSample:
        key1, key2 = jax.random.split(key, 2)

        # For now, let's simplify to single beam to avoid vmap issues
        # Generate random steering angles for a single beam
        theta_steering = jax.random.uniform(key1) * jnp.radians(self.theta_end)
        phi_steering = jax.random.uniform(key2) * (2 * jnp.pi)
        steering_angles = jnp.array([[theta_steering, phi_steering]])

        radiation_pattern, excitations = analyze.rad_pattern_from_geo(
            *self.array_params,
            steering_angles,
        )
        phase_shifts = jnp.angle(excitations)

        return DataSample(radiation_pattern, phase_shifts, steering_angles)

    def generate_batch(
        self, key: jax.random.PRNGKey, batch_size: int
    ) -> dict[str, jnp.ndarray]:
        keys = jax.random.split(key, batch_size)
        samples = jax.vmap(self.generate_sample)(keys)
        return {
            "radiation_patterns": samples.radiation_pattern,
            "phase_shifts": samples.phase_shifts,
            "steering_angles": samples.steering_angles,
        }


class PhaseShiftPredictor(nnx.Module):
    def __init__(self, array_size: tuple[int, int], *, rngs: nnx.Rngs):
        self.array_size = array_size

        self.conv1 = nnx.Conv(1, 4, kernel_size=(5, 5), strides=(8, 8), rngs=rngs)
        self.conv2 = nnx.Conv(4, 8, kernel_size=(3, 3), strides=(4, 4), rngs=rngs)

        # Calculate the actual output size after conv layers
        dummy_input = jnp.ones((1, 180, 360, 1))
        h1 = self.conv1(dummy_input)
        h2 = self.conv2(h1)
        conv_output_size = h2.shape[1] * h2.shape[2] * h2.shape[3]

        self.compress1 = nnx.Linear(conv_output_size, 64, rngs=rngs)
        self.output_layer = nnx.Linear(64, np.prod(array_size), rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_reshaped = x.reshape(x.shape[0], 180, 360, 1)

        h1 = nnx.relu(self.conv1(x_reshaped))
        h2 = nnx.relu(self.conv2(h1))

        h_flat = h2.reshape(h2.shape[0], -1)

        h3 = nnx.relu(self.compress1(h_flat))
        predictions_flat = self.output_layer(h3)

        predictions = predictions_flat.reshape(x.shape[0], *self.array_size)
        return predictions


def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch: dict[str, jnp.ndarray],
) -> tuple[nnx.Optimizer, dict[str, float]]:
    def loss_fn(model: nnx.Module):
        radiation_patterns = batch["radiation_patterns"]
        phase_shifts = batch["phase_shifts"]

        predictions = model(radiation_patterns)
        loss = jnp.mean((predictions - phase_shifts) ** 2)

        return loss, {"loss": loss}

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads)

    return optimizer, metrics


@app.command()
def dev():
    config = TrainingConfig()
    key = jax.random.PRNGKey(0)
    generator = DataGenerator(config)

    rngs = nnx.Rngs(config.seed)
    model = PhaseShiftPredictor(array_size=config.array_size, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(config.learning_rate))

    print("Starting development run")
    for i in range(10):
        batch = generator.generate_batch(key, batch_size=1)
        print(f"Generated batch {i}")
        optimizer, train_metrics = train_step(model, optimizer, batch)
        print(f"Step {i}, Loss: {train_metrics['loss']:.4f}")

    print("Development run completed successfully")


if __name__ == "__main__":
    app()
