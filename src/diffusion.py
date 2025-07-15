import logging
from pathlib import Path
from typing import NamedTuple

import cyclopts
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from jax.typing import ArrayLike

from physics import (
    calculate_weights,
    compute_spatial_phase_coeffs,
    convert_to_db,
    load_element_patterns,
    synthesize_pattern,
)
from training import (
    ConvBlock,
    create_progress_logger,
    restore_checkpoint,
    save_checkpoint,
    steering_angles_sampler,
)
from utils import setup_logging

logger = logging.getLogger(__name__)

app = cyclopts.App()


class Denoiser(nnx.Module):
    def __init__(self, base_channels: int = 64, *, rngs: nnx.Rngs):
        # Process complex weights (2 channels: real + imag)
        # Input: (batch, x_n, y_n, 2) -> Output: (batch, x_n, y_n, base_channels)
        self.weights_conv = nnx.Conv(
            2, base_channels, (3, 3), padding="SAME", rngs=rngs
        )

        # Steering angle encoder - processes target steering angles (theta, phi)
        # Input: (batch, 2) -> Output: (batch, base_channels)
        self.angle_encoder = nnx.Sequential(
            nnx.Linear(2, base_channels // 2, rngs=rngs),
            nnx.relu,
            nnx.Linear(base_channels // 2, base_channels, rngs=rngs),
            nnx.relu,
            nnx.Linear(base_channels, base_channels, rngs=rngs),
        )

        # Time embedding
        self.time_mlp = nnx.Sequential(
            nnx.Linear(1, base_channels, rngs=rngs),
            nnx.relu,
            nnx.Linear(base_channels, base_channels, rngs=rngs),
        )

        # Main processing network
        self.net = nnx.Sequential(
            ConvBlock(base_channels, base_channels, (3, 3), rngs=rngs),
            ConvBlock(base_channels, base_channels, (3, 3), rngs=rngs),
            ConvBlock(base_channels, base_channels, (3, 3), rngs=rngs),
            nnx.Conv(base_channels, 2, (1, 1), rngs=rngs),
        )

    def __call__(
        self,
        weights: ArrayLike,
        steering_angles: ArrayLike,
        timestep: ArrayLike,
    ) -> jax.Array:
        # Convert complex weights to real/imag channels
        weights = weights.view(jnp.float32).reshape(weights.shape + (2,))

        # Process weights
        x = self.weights_conv(weights)  # (batch, x_n, y_n, base_channels)

        # Steering angles embedding
        angle_features = self.angle_encoder(steering_angles)  # (batch, base_channels)
        angle_emb = angle_features[:, None, None, :]  # (batch, 1, 1, channels)
        angle_emb = jnp.broadcast_to(angle_emb, x.shape)  # (batch, x_n, y_n, channels)

        # Time embedding
        time_emb = self.time_mlp(timestep[..., None])
        time_emb = time_emb[:, None, None, :]  # Broadcast to spatial dims
        time_emb = jnp.broadcast_to(time_emb, x.shape)

        # Combine all features via addition
        x = x + angle_emb + time_emb
        output = self.net(x)

        # Convert back to complex
        output = output.view(jnp.complex64).reshape(output.shape[:-1])
        return output


class DDPMScheduler:
    """Denoising Diffusion Probabilistic Model scheduler."""

    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        self.T = T

        # Linear beta schedule
        self.betas = jnp.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)

        # Precompute values for sampling
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(
        self,
        samples: jax.Array,
        noise: jax.Array,
        timesteps: jax.Array,
    ) -> jax.Array:
        """Add noise to samples according to the noise schedule."""
        alpha_t = self.sqrt_alphas_cumprod[timesteps]
        beta_t = self.sqrt_one_minus_alphas_cumprod[timesteps]
        noisy_samples = alpha_t[:, None, None] * samples + beta_t[:, None, None] * noise
        return noisy_samples

    def step(
        self,
        model_output: jax.Array,
        timestep: int,
        sample: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Perform one complex-valued denoising step.

        Adapted for complex-valued data following PhaseGen approach.
        """
        alpha = self.alphas[timestep]
        alpha_cumprod = self.alphas_cumprod[timestep]
        beta = self.betas[timestep]

        # Predict original sample using complex-aware operations
        sqrt_alpha_cumprod = jnp.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha_cumprod = jnp.sqrt(1 - alpha_cumprod)

        pred_original_sample = (
            sample - sqrt_one_minus_alpha_cumprod * model_output
        ) / sqrt_alpha_cumprod

        # Compute coefficients for complex diffusion
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

        # Add complex noise if not the last timestep
        if timestep > 0:
            # Generate complex-valued noise for the step
            complex_noise = generate_complex_noise(key, sample.shape)
            variance = (
                beta * (1 - self.alphas_cumprod[timestep - 1]) / (1 - alpha_cumprod)
            )
            pred_prev_sample = pred_prev_sample + jnp.sqrt(variance) * complex_noise

        return pred_prev_sample


class DiffusionParams(NamedTuple):
    geps: jax.Array
    kx: jax.Array
    ky: jax.Array


def generate_complex_noise(key: jax.Array, shape: tuple) -> jax.Array:
    """Generate complex-valued noise suitable for diffusion.

    Based on torch.randn: https://docs.pytorch.org/docs/stable/generated/torch.randn.html#torch.randn)
    """
    real_key, imag_key = jax.random.split(key)
    real = np.sqrt(0.5) * jax.random.normal(real_key, shape)
    imag = np.sqrt(0.5) * jax.random.normal(imag_key, shape)
    noise = jnp.exp(1j * jnp.angle(real + 1j * imag))
    return noise


vmapped_synthesize = jax.vmap(synthesize_pattern, in_axes=(None, 0))
vmapped_convert_to_db = jax.vmap(convert_to_db)
vmapped_calculate_weights = jax.vmap(calculate_weights, in_axes=(None, None, 0))


@nnx.jit(static_argnames="scheduler")
def train_step(
    optimizer: nnx.Optimizer,
    scheduler: DDPMScheduler,
    params: DiffusionParams,
    batch: jax.Array,
    key: jax.Array,
):
    # Generate original weights (not normalized)
    weights, _ = vmapped_calculate_weights(params.kx, params.ky, batch)

    # Create target patterns from original weights for physics loss
    target_patterns = vmapped_synthesize(params.geps, weights)
    target_patterns = vmapped_convert_to_db(target_patterns)

    # Sample random timesteps and generate proper complex noise
    key, timestep_key, noise_key = jax.random.split(key, 3)
    timesteps = jax.random.randint(timestep_key, (batch.shape[0],), 0, scheduler.T)

    noise = generate_complex_noise(noise_key, weights.shape)
    # Normalize noise to match weights' scale
    noise = noise * jnp.mean(jnp.abs(weights)) / jnp.mean(jnp.abs(noise))
    noisy_weights = scheduler.add_noise(weights, noise, timesteps)

    def loss_fn(model: Denoiser, batch: jax.Array):
        predicted_noise = model(noisy_weights, batch, timesteps.astype(jnp.float32))

        # Physics-based guidance: evaluate predicted clean weights
        k0 = scheduler.sqrt_alphas_cumprod[timesteps][:, None, None]
        k1 = scheduler.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None]
        predicted_weights = (noisy_weights - k1 * predicted_noise) / k0

        # Synthesize patterns with predicted weights
        predicted_patterns = vmapped_synthesize(params.geps, predicted_weights)
        predicted_patterns = vmapped_convert_to_db(predicted_patterns)

        # Compute losses
        noise_diff = predicted_noise - noise
        denoising_loss = jnp.mean(jnp.real(noise_diff) ** 2 + jnp.imag(noise_diff) ** 2)
        physics_loss = jnp.mean((predicted_patterns - target_patterns) ** 2)

        total_loss = denoising_loss + 1e-3 * physics_loss

        metrics = {
            "denoising_loss": denoising_loss,
            "physics_loss": physics_loss,
            "total_loss": total_loss,
        }

        return total_loss, metrics

    model = optimizer.model
    (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
    optimizer.update(grads)
    metrics["grad_norm"] = optax.global_norm(grads)
    return metrics


def solve_with_diffusion(
    model: Denoiser,
    steering_angles: jax.Array,
    scheduler: DDPMScheduler,
    array_size: tuple[int, int] = (4, 4),
    n_steps: int = 200,
    key: jax.Array | None = None,
) -> jax.Array:
    if key is None:
        key = jax.random.key(0)
    sample = generate_complex_noise(key, array_size)

    # Create inference timesteps
    inference_timesteps = jnp.linspace(scheduler.T - 1, 0, n_steps, dtype=jnp.int32)

    for timestep in inference_timesteps:
        key, step_key = jax.random.split(key)

        # Model prediction with steering angles
        timestep_batch = jnp.array([timestep], dtype=jnp.float32)
        sample_batch = sample[None, ...]
        angles_batch = steering_angles[None, ...]  # (1, 2)

        model_output = model(sample_batch, angles_batch, timestep_batch)[0]

        # Denoising step
        sample = scheduler.step(model_output, timestep, sample, step_key)

    return sample


@app.command()
def train(
    n_steps: int = 100_000,
    batch_size: int = 256,
    lr: float = 5e-4,
    seed: int = 42,
    restore: bool = False,
):
    key = jax.random.key(seed)

    logger.info("Setting up diffusion training pipeline")
    element_data = load_element_patterns(kind="cst")
    kx, ky = compute_spatial_phase_coeffs(element_data.config)
    params = DiffusionParams(
        geps=jnp.asarray(element_data.geps),
        kx=jnp.asarray(kx),
        ky=jnp.asarray(ky),
    )

    key, model_key = jax.random.split(key)
    model = Denoiser(base_channels=32, rngs=nnx.Rngs(model_key))
    scheduler = DDPMScheduler(T=1000)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=lr))

    ckpt_path = Path.cwd() / "checkpoints_diffusion"
    ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=2, save_interval_steps=100)
    ckpt_mngr = ocp.CheckpointManager(ckpt_path, options=ckpt_options)
    start_step = 0
    if restore:
        start_step = restore_checkpoint(ckpt_mngr, optimizer, step=None)
        logger.info(f"Resuming from step {start_step}")

    key, data_key = jax.random.split(key)
    sampler = steering_angles_sampler(data_key, batch_size, limit=n_steps)

    logger.info(f"Starting diffusion training for {n_steps} steps")
    log_progress = create_progress_logger(n_steps, log_every=50, start_step=start_step)

    try:
        for step, batch in enumerate(sampler, start=start_step):
            key, step_key = jax.random.split(key)
            metrics = train_step(optimizer, scheduler, params, batch, step_key)
            log_progress(step, metrics)
            if step % 100 == 0:
                save_checkpoint(ckpt_mngr, optimizer, step, True)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        ckpt_mngr.wait_until_finished()

    logger.info("Diffusion training completed")


@app.command()
def pred(
    theta: float = 0.0,
    phi: float = 0.0,
    seed: int = 42,
    array_size: tuple[int, int] = (4, 4),
):
    """Generate antenna array weights for given steering angles using trained diffusion model."""
    key = jax.random.key(seed)

    logger.info(
        f"Setting up diffusion prediction for steering angles: theta={theta}°, phi={phi}°"
    )

    # Load physics parameters
    element_data = load_element_patterns(kind="cst")
    kx, ky = compute_spatial_phase_coeffs(element_data.config)
    params = DiffusionParams(
        geps=jnp.asarray(element_data.geps),
        kx=jnp.asarray(kx),
        ky=jnp.asarray(ky),
    )

    key, model_key = jax.random.split(key)
    model = Denoiser(base_channels=32, rngs=nnx.Rngs(model_key))
    scheduler = DDPMScheduler(T=1000)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=5e-4))

    # Load trained checkpoint
    ckpt_path = Path.cwd() / "checkpoints_diffusion"
    ckpt_mngr = ocp.CheckpointManager(ckpt_path)
    restore_checkpoint(ckpt_mngr, optimizer, step=None)
    logger.info("Loaded trained model from checkpoint")

    # Convert steering angles to radians
    steering_angles = jnp.array([jnp.radians(theta), jnp.radians(phi)])

    # Generate weights using diffusion
    logger.info(f"Running diffusion sampling with {scheduler.T} steps...")
    key, sample_key = jax.random.split(key)
    generated_weights = solve_with_diffusion(
        model=optimizer.model,
        steering_angles=steering_angles,
        scheduler=scheduler,
        array_size=array_size,
        n_steps=scheduler.T,
        key=sample_key,
    )

    # Generate target weights for comparison
    target_weights, _ = calculate_weights(kx, ky, steering_angles)

    # Synthesize and compare patterns
    generated_pattern = synthesize_pattern(params.geps, generated_weights)
    target_pattern = synthesize_pattern(params.geps, target_weights)
    generated_pattern_db = convert_to_db(generated_pattern)
    target_pattern_db = convert_to_db(target_pattern)

    # Calculate metrics
    pattern_mse = float(jnp.mean((generated_pattern_db - target_pattern_db) ** 2))
    weights_mse = float(jnp.mean(jnp.abs(generated_weights - target_weights) ** 2))
    logger.info(generated_weights)
    logger.info(f"Weights MSE: {weights_mse:.6f}")
    logger.info(f"Pattern MSE (dB): {pattern_mse:.2f}")


if __name__ == "__main__":
    setup_logging()
    app()
