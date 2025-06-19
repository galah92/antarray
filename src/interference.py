import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from physics import ArrayConfig, convert_to_db, create_physics_setup
from training import (
    InterferenceCorrector,
    circular_mse_fn,
    create_progress_logger,
    steering_angles_sampler,
)
from utils import setup_logging

logger = logging.getLogger(__name__)


@partial(nnx.jit, static_argnames=("synth_pattern", "compute_weights"))
def train_step(
    optimizer: nnx.Optimizer,
    batch: jax.Array,
    synth_pattern: Callable,
    compute_weights: Callable,
) -> dict[str, float]:
    model = optimizer.model

    convert_to_db_vm = jax.vmap(convert_to_db)

    target_weights, target_phase_shifts = compute_weights(batch)
    target_phase_shifts = jnp.squeeze(target_phase_shifts)

    target_patterns = synth_pattern(target_weights)

    def loss_fn(model: InterferenceCorrector, target_patterns: jax.Array):
        pred_weights, pred_phase_shifts = model(target_patterns)

        pred_patterns = synth_pattern(pred_weights)

        target_patterns_db = convert_to_db_vm(target_patterns)
        pred_patterns_db = convert_to_db_vm(pred_patterns)

        patterns_mse = ((target_patterns_db - pred_patterns_db) ** 2).mean()
        phase_shifts_mse = circular_mse_fn(target_phase_shifts, pred_phase_shifts)

        loss = patterns_mse + phase_shifts_mse

        metrics = {
            "loss": loss,
            "patterns_mse": patterns_mse,
            "phase_shifts_mse": phase_shifts_mse,
            "phase_shifts_rmse": jnp.sqrt(phase_shifts_mse),
        }
        return loss, metrics

    (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model, target_patterns
    )
    optimizer.update(grads)
    metrics["grad_norm"] = optax.global_norm(grads)
    return metrics


def train_pipeline(
    n_steps: int = 10_000,
    batch_size: int = 512,
    lr: float = 5e-4,
    seed: int = 42,
    openems_path: Path | None = None,
):
    """Main function to set up and run the training pipeline."""
    key = jax.random.key(seed)
    key, model_key, data_key = jax.random.split(key, 3)

    logger.info("Performing one-time precomputation")
    config = ArrayConfig()
    synth_pattern, compute_weights = create_physics_setup(
        config, openems_path=openems_path
    )
    synth_pattern, compute_weights = jax.vmap(synth_pattern), jax.vmap(compute_weights)

    model = InterferenceCorrector(rngs=nnx.Rngs(model_key))
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=lr))

    sampler = steering_angles_sampler(data_key, batch_size, limit=n_steps)

    logger.info("Starting training")
    log_progress = create_progress_logger(n_steps, log_every=100)

    try:
        for step, batch in enumerate(sampler):
            metrics = train_step(optimizer, batch, synth_pattern, compute_weights)
            log_progress(step, metrics)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    logger.info("Training completed")


if __name__ == "__main__":
    setup_logging()
    train_pipeline()
