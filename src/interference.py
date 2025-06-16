import logging
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from physics import convert_to_db, create_physics_setup, normalize_patterns
from training import (
    InterferenceCorrector,
    calculate_pattern_loss,
    create_progress_logger,
    steering_angles_sampler,
)

logger = logging.getLogger(__name__)


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
    openems_path: Path | None = None,
):
    """Main function to set up and run the training pipeline."""
    key = jax.random.key(seed)

    logger.info("Performing one-time precomputation")

    # Create physics setup with optional OpenEMS support
    key, physics_key = jax.random.split(key)
    synthesize_ideal, synthesize_embedded, compute_analytical = create_physics_setup(
        physics_key, openems_path=openems_path
    )

    train_step = create_train_step_fn(
        synthesize_ideal,
        synthesize_embedded,
        compute_analytical,
    )

    key, model_key = jax.random.split(key)
    model = InterferenceCorrector(rngs=nnx.Rngs(model_key))
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=lr))

    key, data_key = jax.random.split(key)
    sampler = steering_angles_sampler(data_key, batch_size, limit=n_steps)

    logger.info("Starting training")
    log_progress = create_progress_logger(n_steps, log_every=100)

    try:
        for step, batch in enumerate(sampler):
            metrics = train_step(optimizer, batch)

            log_progress(step, metrics)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    logger.info("Training completed")


if __name__ == "__main__":
    train_pipeline()
