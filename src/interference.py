import logging
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from physics import (
    ArrayConfig,
    calculate_weights,
    compute_element_fields,
    compute_spatial_phase_coeffs,
    convert_to_db,
    load_aeps,
    synthesize_pattern,
)
from training import (
    InterferenceCorrector,
    circular_mse_fn,
    create_progress_logger,
    steering_angles_sampler,
)
from utils import setup_logging

logger = logging.getLogger(__name__)


def preprocess_pattern(pattern: jax.Array) -> jax.Array:
    """Preprocess the pattern by converting it to dB scale, clipping and normalizing."""
    floor_db = -30.0  # dB threshold for clipping
    pattern_db = convert_to_db(pattern, floor_db=floor_db)
    normalized = (pattern_db - floor_db) / (0 - floor_db)  # Normalize to [0, 1]
    return normalized


preprocess_pattern_vm = jax.vmap(preprocess_pattern)
calculate_weights_vm = jax.vmap(calculate_weights, in_axes=(None, None, 0))
synthesize_pattern_vm = jax.vmap(synthesize_pattern, in_axes=(None, 0))


class PhysicsParams(NamedTuple):
    element_fields: jax.Array
    kx: jax.Array
    ky: jax.Array


@nnx.jit
def train_step(
    optimizer: nnx.Optimizer,
    batch: jax.Array,
    params: PhysicsParams,
) -> dict[str, float]:
    model = optimizer.model

    target_weights, target_phase_shifts = calculate_weights_vm(
        params.kx, params.ky, batch
    )
    target_phase_shifts = jnp.squeeze(target_phase_shifts)
    target_patterns = synthesize_pattern_vm(params.element_fields, target_weights)

    def loss_fn(model: InterferenceCorrector, target_patterns: jax.Array):
        preprocessed_patterns = preprocess_pattern_vm(target_patterns)
        pred_weights, pred_phase_shifts = model(preprocessed_patterns)
        pred_patterns = synthesize_pattern_vm(params.element_fields, pred_weights)

        target_patterns_db = preprocess_pattern_vm(target_patterns)
        pred_patterns_db = preprocess_pattern_vm(pred_patterns)
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
    batch_size: int = 32,
    lr: float = 5e-4,
    seed: int = 42,
):
    """Main function to set up and run the training pipeline."""
    key = jax.random.key(seed)
    key, model_key, data_key = jax.random.split(key, 3)

    logger.info("Performing one-time precomputation")
    config = ArrayConfig()
    element_data = load_aeps(config, kind="cst")
    aeps = element_data.aeps
    config = element_data.config
    kx, ky = compute_spatial_phase_coeffs(config)
    element_fields = compute_element_fields(aeps, config)

    physics_params = PhysicsParams(
        element_fields=jnp.asarray(element_fields),
        kx=jnp.asarray(kx),
        ky=jnp.asarray(ky),
    )

    model = InterferenceCorrector(rngs=nnx.Rngs(model_key))
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=lr))

    sampler = steering_angles_sampler(data_key, batch_size, limit=n_steps)

    logger.info("Warming up GPU kernels")
    model.eval()
    train_step(optimizer, next(sampler), physics_params)
    model.train()

    log_progress = create_progress_logger(n_steps, log_every=10)
    logger.info("Starting training")
    try:
        for step, batch in enumerate(sampler):
            metrics = train_step(optimizer, batch, physics_params)
            log_progress(step, metrics)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    logger.info("Training completed")


if __name__ == "__main__":
    setup_logging()
    try:
        train_pipeline()
    except Exception as e:
        logger.error("An error occurred during training", exc_info=e)
        raise
