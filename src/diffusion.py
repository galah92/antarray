import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import NamedTuple

import cyclopts
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from physics import (
    ArrayConfig,
    calculate_weights,
    compute_geps,
    compute_spatial_phase_coeffs,
    load_aeps,
    normalize_patterns,
    synthesize_pattern,
)
from training import (
    DenoisingUNet,
    create_progress_logger,
    restore_checkpoint,
    save_checkpoint,
    steering_angles_sampler,
)
from utils import setup_logging

logger = logging.getLogger(__name__)

app = cyclopts.App()


class DiffusionParams(NamedTuple):
    element_fields: jax.Array
    kx: jax.Array
    ky: jax.Array


vmapped_synthesize = jax.vmap(synthesize_pattern, in_axes=(None, 0))
vmapped_calculate_weights = jax.vmap(calculate_weights, in_axes=(None, None, 0))


class DDPMScheduler:
    """Denoising Diffusion Probabilistic Model scheduler."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        self.num_train_timesteps = num_train_timesteps

        # Linear beta schedule
        self.betas = jnp.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)

        # Precompute values for sampling
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(
        self, original_samples: jax.Array, noise: jax.Array, timesteps: jax.Array
    ) -> jax.Array:
        """Add noise to samples according to the noise schedule."""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod[:, None, None]
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[:, None, None]

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def step(
        self, model_output: jax.Array, timestep: int, sample: jax.Array, key: jax.Array
    ) -> jax.Array:
        """Perform one denoising step."""
        alpha = self.alphas[timestep]
        alpha_cumprod = self.alphas_cumprod[timestep]
        beta = self.betas[timestep]

        # Predict original sample
        pred_original_sample = (
            sample - jnp.sqrt(1 - alpha_cumprod) * model_output
        ) / jnp.sqrt(alpha_cumprod)

        # Compute coefficients
        pred_sample_coeff = (
            jnp.sqrt(alpha)
            * (1 - self.alphas_cumprod[timestep - 1])
            / (1 - alpha_cumprod)
        )
        current_sample_coeff = (
            jnp.sqrt(self.alphas_cumprod[timestep - 1]) * beta / (1 - alpha_cumprod)
        )

        # Compute previous sample
        pred_prev_sample = (
            pred_sample_coeff * sample + current_sample_coeff * pred_original_sample
        )

        # Add noise if not the last timestep
        if timestep > 0:
            noise = jax.random.normal(key, sample.shape, dtype=sample.dtype)
            variance = (
                beta * (1 - self.alphas_cumprod[timestep - 1]) / (1 - alpha_cumprod)
            )
            pred_prev_sample = pred_prev_sample + jnp.sqrt(variance) * noise

        return pred_prev_sample


@nnx.jit(static_argnames="scheduler")
def train_step(
    optimizer: nnx.Optimizer,
    batch_of_angles_rad: jax.Array,
    scheduler: DDPMScheduler,
    params: DiffusionParams,
    key: jax.Array,
):
    """Jitted training step for diffusion."""
    model = optimizer.model

    def loss_fn(model: DenoisingUNet, batch_of_angles_rad: jax.Array, key: jax.Array):
        kx, ky, element_fields = params.kx, params.ky, params.element_fields
        batch_size = batch_of_angles_rad.shape[0]

        # Generate element weights and target patterns
        element_weights, _ = vmapped_calculate_weights(kx, ky, batch_of_angles_rad)

        # Create target patterns (ideal case)
        ideal_patterns = vmapped_synthesize(element_fields, element_weights)
        target_patterns = normalize_patterns(ideal_patterns)

        # Sample random timesteps
        key, timestep_key, noise_key = jax.random.split(key, 3)
        timesteps = jax.random.randint(
            timestep_key, (batch_size,), 0, scheduler.num_train_timesteps
        )

        # Add noise to element weights
        noise = jax.random.normal(
            noise_key, element_weights.shape, dtype=element_weights.dtype
        )
        noisy_weights = scheduler.add_noise(element_weights, noise, timesteps)

        # Model predicts the noise
        predicted_noise = model(
            noisy_weights, target_patterns, timesteps.astype(jnp.float32)
        )

        # Physics-based guidance: evaluate predicted clean weights
        predicted_clean_weights = (
            noisy_weights
            - scheduler.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None]
            * predicted_noise
        ) / scheduler.sqrt_alphas_cumprod[timesteps][:, None, None]

        # Synthesize patterns with predicted weights
        predicted_patterns = vmapped_synthesize(element_fields, predicted_clean_weights)

        # Scale the predicted patterns by the peak of the ideal patterns
        # to ensure they are compared on the same scale for the physics loss.
        ideal_max_vals = jnp.max(ideal_patterns, axis=(1, 2), keepdims=True)
        scaled_predicted_patterns = predicted_patterns / (ideal_max_vals + 1e-8)

        denoising_loss = jnp.mean(jnp.abs(predicted_noise - noise) ** 2)
        physics_loss = jnp.mean((scaled_predicted_patterns - target_patterns) ** 2)

        total_loss = denoising_loss + 0.05 * physics_loss

        metrics = {
            "denoising_loss": denoising_loss,
            "physics_loss": physics_loss,
            "total_loss": total_loss,
        }

        return total_loss, metrics

    (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model, batch_of_angles_rad, key
    )
    optimizer.update(grads)
    metrics["grad_norm"] = optax.global_norm(grads)
    return metrics


def solve_with_diffusion(
    model: DenoisingUNet,
    target_pattern: jax.Array,
    scheduler: DDPMScheduler,
    synthesize_embedded_pattern: Callable,
    num_inference_steps: int = 200,
    guidance_scale: float = 1.0,
    key: jax.Array | None = None,
) -> jax.Array:
    """Phase 2: Use trained model to solve for corrective weights."""

    # Start with pure noise
    sample_shape = (16, 16)  # Array size
    sample = jax.random.normal(key, sample_shape, dtype=jnp.complex64)

    # Create inference timesteps
    inference_timesteps = jnp.linspace(
        scheduler.num_train_timesteps - 1, 0, num_inference_steps, dtype=jnp.int32
    )

    target_pattern_batch = target_pattern[None, ...]  # Add batch dimension

    for i, timestep in enumerate(inference_timesteps):
        key, step_key = jax.random.split(key)

        # Model prediction
        timestep_batch = jnp.array([timestep], dtype=jnp.float32)
        sample_batch = sample[None, ...]

        model_output = model(sample_batch, target_pattern_batch, timestep_batch)[0]

        # Physics guidance
        if guidance_scale > 1.0:
            # Predict clean sample
            alpha_cumprod = scheduler.alphas_cumprod[timestep]
            predicted_clean = (
                sample_batch - jnp.sqrt(1 - alpha_cumprod) * model_output
            ) / jnp.sqrt(alpha_cumprod)

            # Physics gradient
            def physics_loss_fn(weights):
                pattern = synthesize_embedded_pattern(weights)
                pattern = normalize_patterns(pattern[None, ...])[0]
                return jnp.mean((pattern - target_pattern) ** 2)

            physics_grad = jax.grad(physics_loss_fn)(predicted_clean[0])

            # Apply guidance
            model_output = model_output - guidance_scale * physics_grad[None, ...]

        # Denoising step
        sample = scheduler.step(model_output[0], timestep, sample, step_key)

    return sample


@app.command()
def train(
    n_steps: int = 10_000,
    batch_size: int = 256,
    lr: float = 1e-4,
    seed: int = 42,
    restore: bool = True,
    overwrite: bool = False,
    openems_path: Path | None = None,
):
    """Main training function for the diffusion model."""
    key = jax.random.key(seed)

    logger.info("Setting up diffusion training pipeline")

    # Create physics setup with optional OpenEMS support
    config = ArrayConfig()
    if openems_path is not None:
        element_data = load_aeps(config, kind="openems", path=openems_path)
    else:
        element_data = load_aeps(config, kind="synthetic")
    aeps = element_data.aeps
    config = element_data.config
    geps = compute_geps(aeps, config)
    kx, ky = compute_spatial_phase_coeffs(config)

    # Create scheduler and model
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    params = DiffusionParams(
        element_fields=jnp.asarray(geps),
        kx=jnp.asarray(kx),
        ky=jnp.asarray(ky),
    )

    key, model_key = jax.random.split(key)
    model = DenoisingUNet(base_channels=64, rngs=nnx.Rngs(model_key))

    # Use a learning rate schedule for better stability
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1,  # Start with 10% of target LR
        peak_value=lr,
        warmup_steps=500,
        decay_steps=n_steps - 500,
        end_value=lr * 0.01,  # End with 1% of target LR
    )

    optimizer = nnx.Optimizer(
        model, optax.adamw(learning_rate=lr_schedule, weight_decay=1e-6)
    )

    ckpt_path = Path.cwd() / "checkpoints_diffusion"
    ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=1, save_interval_steps=100)
    ckpt_mngr = ocp.CheckpointManager(ckpt_path, options=ckpt_options)

    start_step = 0
    if restore:
        start_step = restore_checkpoint(ckpt_mngr, optimizer, step=None)
        logger.info(f"Resuming from step {start_step}")

    n_steps -= start_step

    # Training data generator
    key, data_key = jax.random.split(key)
    sampler = steering_angles_sampler(data_key, batch_size, limit=n_steps)

    logger.info("Starting diffusion training")
    log_progress = create_progress_logger(n_steps, log_every=100, start_step=start_step)

    try:
        for step, batch in enumerate(sampler, start=start_step):
            key, step_key = jax.random.split(key)
            metrics = train_step(
                optimizer,
                batch,
                scheduler,
                params,
                step_key,
            )
            save_checkpoint(ckpt_mngr, optimizer, step, overwrite)
            log_progress(step, metrics)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        raise
    finally:
        ckpt_mngr.wait_until_finished()

    logger.info("Diffusion training completed")


@app.command()
def evaluate(
    n_eval_samples: int = 50,
    seed: int = 123,
    openems_path: Path | None = None,
):
    """Evaluate the trained diffusion model on test steering angles."""
    key = jax.random.key(seed)

    # Create physics setup for evaluation with optional OpenEMS support
    config = ArrayConfig()
    if openems_path is not None:
        element_data = load_aeps(config, kind="openems", path=openems_path)
    else:
        element_data = load_aeps(config, kind="synthetic")
    aeps = element_data.aeps
    config = element_data.config
    geps = compute_geps(aeps, config)
    kx, ky = compute_spatial_phase_coeffs(config)

    # Create scheduler and model
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    params = DiffusionParams(
        element_fields=jnp.asarray(geps),
        kx=jnp.asarray(kx),
        ky=jnp.asarray(ky),
    )

    key, model_key, data_key = jax.random.split(key, 3)
    model = DenoisingUNet(base_channels=64, rngs=nnx.Rngs(model_key))

    # Create a dummy optimizer to load the checkpoint
    optimizer = nnx.Optimizer(model, optax.adamw(1e-4))

    # Load the trained model
    ckpt_path = Path.cwd() / "checkpoints_diffusion"
    ckpt_mngr = ocp.CheckpointManager(ckpt_path)
    step = restore_checkpoint(ckpt_mngr, optimizer)
    if step == 0:
        logger.error("No checkpoint found, cannot evaluate.")
        return
    logger.info(f"Loaded checkpoint from step {step}")
    model = optimizer.model

    # Create test data
    sampler = steering_angles_sampler(data_key, n_eval_samples, limit=1)

    logger.info(f"Evaluating diffusion model on {n_eval_samples} samples")

    for i, steering_angles in enumerate(sampler):
        logger.info(f"Sample {i + 1}/{n_eval_samples}")

        # Create target pattern
        element_weights, _ = vmapped_calculate_weights(
            params.kx,
            params.ky,
            steering_angles,
        )
        target_pattern = vmapped_synthesize(params.element_fields, element_weights)
        target_pattern = normalize_patterns(target_pattern)

        # Solve for corrective weights
        key, solve_key = jax.random.split(key)
        synthesize_embedded = partial(synthesize_pattern, params.element_fields)
        corrective_weights = solve_with_diffusion(
            model,
            target_pattern[0],
            scheduler,
            synthesize_embedded,
            key=solve_key,
        )

        # Evaluate performance
        corrected_weights = element_weights * corrective_weights[None, ...]
        corrected_pattern = vmapped_synthesize(params.element_fields, corrected_weights)
        corrected_pattern = normalize_patterns(corrected_pattern)

        # ... plotting and logging ...


if __name__ == "__main__":
    setup_logging()
    app()
