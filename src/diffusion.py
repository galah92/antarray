import logging
import sys
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from physics import ArrayConfig, create_physics_setup
from shared import (
    DenoisingUNet,
    normalize_patterns,
    steering_angles_sampler,
)

logger = logging.getLogger(__name__)


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


def create_train_step_fn(
    synthesize_ideal_pattern: Callable,
    synthesize_embedded_pattern: Callable,
    compute_analytical_weights: Callable,
    scheduler: DDPMScheduler,
):
    """Factory that creates the jitted training step function for diffusion."""
    vmapped_analytical_weights = jax.vmap(compute_analytical_weights)
    vmapped_embedded_synthesizer = jax.vmap(synthesize_embedded_pattern)

    def loss_fn(model: DenoisingUNet, batch_of_angles_rad: jax.Array, key: jax.Array):
        batch_size = batch_of_angles_rad.shape[0]

        # Generate analytical weights and target patterns
        analytical_weights, _ = vmapped_analytical_weights(batch_of_angles_rad)

        # Create target patterns (ideal case)
        ideal_patterns = jax.vmap(synthesize_ideal_pattern)(analytical_weights)
        target_patterns = normalize_patterns(ideal_patterns)

        # Sample random timesteps
        key, timestep_key, noise_key = jax.random.split(key, 3)
        timesteps = jax.random.randint(
            timestep_key, (batch_size,), 0, scheduler.num_train_timesteps
        )

        # Add noise to analytical weights
        noise = jax.random.normal(
            noise_key, analytical_weights.shape, dtype=analytical_weights.dtype
        )
        noisy_weights = scheduler.add_noise(analytical_weights, noise, timesteps)

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
        predicted_patterns = vmapped_embedded_synthesizer(predicted_clean_weights)
        predicted_patterns = normalize_patterns(predicted_patterns)

        denoising_loss = jnp.mean(jnp.abs(predicted_noise - noise) ** 2)
        physics_loss = jnp.mean((predicted_patterns - target_patterns) ** 2)

        physics_weight = 0.05
        total_loss = denoising_loss + physics_weight * physics_loss

        metrics = {
            "denoising_loss": denoising_loss,
            "physics_loss": physics_loss,
            "total_loss": total_loss,
        }

        return total_loss, metrics

    @nnx.jit
    def train_step_fn(optimizer: nnx.Optimizer, batch: jax.Array, key: jax.Array):
        model = optimizer.model
        (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
            model, batch, key
        )
        optimizer.update(grads)
        metrics["grad_norm"] = optax.global_norm(grads)
        return metrics

    return train_step_fn


def solve_with_diffusion(
    model: DenoisingUNet,
    target_pattern: jax.Array,
    scheduler: DDPMScheduler,
    num_inference_steps: int = 200,
    guidance_scale: float = 1.0,
    synthesize_embedded_pattern: Callable | None = None,
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

        # Optional physics guidance
        if synthesize_embedded_pattern is not None and guidance_scale > 1.0:
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


def train_diffusion_pipeline(
    n_steps: int = 10_000,
    batch_size: int = 256,
    lr: float = 1e-4,
    seed: int = 42,
):
    """Main training function for the diffusion model."""
    config = ArrayConfig()
    key = jax.random.key(seed)

    logger.info("Setting up diffusion training pipeline")

    # Create physics setup
    key, physics_key = jax.random.split(key)
    synthesize_ideal, synthesize_embedded, compute_analytical = create_physics_setup(
        config, physics_key
    )

    # Create scheduler and model
    scheduler = DDPMScheduler(num_train_timesteps=1000)

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

    # Create training step
    train_step = create_train_step_fn(
        synthesize_ideal,
        synthesize_embedded,
        compute_analytical,
        scheduler,
    )

    # Training data generator
    key, data_key = jax.random.split(key)
    sampler = steering_angles_sampler(data_key, batch_size, limit=n_steps)

    logger.info("Starting diffusion training")
    try:
        for step, batch in enumerate(sampler):
            key, step_key = jax.random.split(key)
            metrics = train_step(optimizer, batch, step_key)

            if (step + 1) % 100 == 0:
                logger.info(
                    f"step {step + 1}/{n_steps}, "
                    f"grad_norm: {metrics['grad_norm'].item():.3f}, "
                    f"total_loss: {metrics['total_loss'].item():.3f}, "
                    f"denoising_loss: {metrics['denoising_loss'].item():.3f}, "
                    f"physics_loss: {metrics['physics_loss'].item():.3f}"
                )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    logger.info("Diffusion training completed")

    return model, scheduler, synthesize_embedded


def evaluate_diffusion_model(
    model: DenoisingUNet,
    scheduler: DDPMScheduler,
    synthesize_embedded: Callable,
    config: ArrayConfig,
    n_eval_samples: int = 50,
    seed: int = 123,
):
    """Evaluate the trained diffusion model on test steering angles."""
    key = jax.random.key(seed)

    # Generate test steering angles
    key, angle_key = jax.random.split(key)
    test_angles = steering_angles_sampler(angle_key, batch_size=n_eval_samples, limit=1)
    test_batch = next(test_angles)

    # Create physics setup for evaluation
    key, physics_key = jax.random.split(key)
    synthesize_ideal, _, compute_analytical = create_physics_setup(config, physics_key)

    # Compute analytical weights and target patterns
    vmapped_analytical_weights = jax.vmap(compute_analytical)
    analytical_weights, _ = vmapped_analytical_weights(test_batch)

    # Create ideal target patterns
    ideal_target_patterns = jax.vmap(synthesize_ideal)(analytical_weights)
    ideal_target_patterns = normalize_patterns(ideal_target_patterns)

    # Solve using diffusion for each target pattern
    key, solve_key = jax.random.split(key)
    solve_keys = jax.random.split(solve_key, n_eval_samples)

    def solve_single(target_pattern, solve_key):
        return solve_with_diffusion(
            model,
            target_pattern,
            scheduler,
            num_inference_steps=200,
            guidance_scale=1.5,
            synthesize_embedded_pattern=synthesize_embedded,
            key=solve_key,
        )

    logger.info(f"Evaluating on {n_eval_samples} test samples...")
    predicted_weights = jax.vmap(solve_single)(ideal_target_patterns, solve_keys)

    # Compute evaluation metrics
    weight_mse = jnp.mean(jnp.abs(predicted_weights - analytical_weights) ** 2)

    # Pattern quality metrics
    predicted_patterns = jax.vmap(synthesize_embedded)(predicted_weights)
    predicted_patterns = normalize_patterns(predicted_patterns)
    pattern_mse = jnp.mean((predicted_patterns - ideal_target_patterns) ** 2)

    logger.info("Evaluation Results:")
    logger.info(f"  Weight MSE: {weight_mse:.6f}")
    logger.info(f"  Pattern MSE: {pattern_mse:.6f}")

    return {
        "weight_mse": weight_mse,
        "pattern_mse": pattern_mse,
        "test_angles": test_batch,
        "predicted_weights": predicted_weights,
        "analytical_weights": analytical_weights,
        "predicted_patterns": predicted_patterns,
        "target_patterns": ideal_target_patterns,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} {levelname} {filename}:{lineno} {message}",
        style="{",
        handlers=[
            logging.FileHandler(Path("app.log"), mode="w+"),  # Overwrite log
            logging.StreamHandler(),
        ],
        force=True,  # https://github.com/google/orbax/issues/1248
    )
    logging.getLogger("absl").setLevel(logging.CRITICAL)  # Suppress absl logging

    logger.info(f"uv run {' '.join(sys.argv)}")  # Log command line args

    # Train the model
    model, scheduler, synthesize_embedded = train_diffusion_pipeline()

    # Evaluate the trained model
    config = ArrayConfig()
    eval_results = evaluate_diffusion_model(
        model, scheduler, synthesize_embedded, config
    )

    logger.info("Training and evaluation completed!")
