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

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.T = T
        self.betas = jnp.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = jnp.cumprod(self.alphas)
        self.sqrt_alphas_bar = jnp.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = jnp.sqrt(1.0 - self.alphas_bar)

    def add_noise(
        self,
        samples: jax.Array,
        noise: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        """Add noise to samples according to the noise schedule."""
        alpha_t = self.sqrt_alphas_bar[t]
        beta_t = self.sqrt_one_minus_alphas_bar[t]
        noisy_samples = alpha_t[:, None, None] * samples + beta_t[:, None, None] * noise
        return noisy_samples

    def predict_x0(
        self,
        xt: jax.Array,
        t: jax.Array,
        noise_pred: jax.Array,
    ) -> jax.Array:
        """Predict x0 (original sample) from xt (noisy sample) and predicted noise."""
        k0 = self.sqrt_alphas_bar[t][:, None, None]
        k1 = self.sqrt_one_minus_alphas_bar[t][:, None, None]
        x0_pred = (xt - k1 * noise_pred) / k0
        return x0_pred

    def step(
        self,
        pred_noise: jax.Array,
        t: int,
        sample: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        """Perform one denoising step."""
        k0 = jnp.sqrt(self.alphas_bar[t])
        k1 = (1 - self.alphas[t]) / jnp.sqrt(1 - self.alphas_bar[t])
        x_t = k0 * sample - k1 * pred_noise

        if t > 0:
            noise = generate_complex_noise(key, sample.shape)
            noise = noise / jnp.sqrt(np.prod(sample.shape))  # Normalize noise
            x_t = x_t + jnp.sqrt(self.betas[t]) * noise

        return x_t


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


def synth_power_db(geps: jax.Array, weights: jax.Array) -> jax.Array:
    return vmapped_convert_to_db(vmapped_synthesize(geps, weights))


@nnx.jit(static_argnames="scheduler")
def train_step(
    optimizer: nnx.Optimizer,
    scheduler: DDPMScheduler,
    params: DiffusionParams,
    batch: jax.Array,
    key: jax.Array,
):
    # Create target patterns for physics loss
    weights, _ = vmapped_calculate_weights(params.kx, params.ky, batch)
    target_patterns = synth_power_db(params.geps, weights)

    # Sample random timesteps and generate proper complex noise
    key, timestep_key, noise_key = jax.random.split(key, 3)
    batch_size, *array_size = weights.shape
    timesteps = jax.random.randint(timestep_key, (batch_size,), 0, scheduler.T)

    noise = generate_complex_noise(noise_key, weights.shape)
    noise = noise / np.sqrt(np.prod(array_size))  # Normalize to realistic amplitude
    noisy_weights = scheduler.add_noise(weights, noise, timesteps)

    def loss_fn(model: Denoiser, batch: jax.Array):
        pred_noise = model(noisy_weights, batch, timesteps.astype(jnp.float32))

        # Physics-based guidance: evaluate predicted weights
        pred_weights = scheduler.predict_x0(noisy_weights, timesteps, pred_noise)
        # Re-normalize predicted weights to have unit power, consistent with target
        pred_weights = pred_weights / jnp.sqrt(
            jnp.sum(jnp.abs(pred_weights) ** 2, axis=(-2, -1), keepdims=True)
        )
        pred_patterns = synth_power_db(params.geps, pred_weights)

        # Compute losses
        noise_diff = pred_noise - noise
        denoising_loss = jnp.mean(jnp.real(noise_diff) ** 2 + jnp.imag(noise_diff) ** 2)
        physics_loss = jnp.mean((pred_patterns - target_patterns) ** 2)

        total_loss = denoising_loss + 1e-3 * physics_loss

        metrics = {
            "t_weights": jnp.mean(jnp.sum(jnp.abs(weights), axis=0)),
            "p_weights": jnp.mean(jnp.sum(jnp.abs(pred_weights), axis=0)),
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
    key: jax.Array | None = None,
) -> jax.Array:
    if key is None:
        key = jax.random.key(0)
    x_t = generate_complex_noise(key, array_size)
    x_t = x_t / np.sqrt(np.prod(array_size))  # TODO: not sure if needed

    for t in reversed(range(scheduler.T)):
        t_batch = jnp.array([t], dtype=jnp.float32)
        pred_noise = model(x_t[None, ...], steering_angles[None, ...], t_batch)[0]

        # Denoising step
        key, step_key = jax.random.split(key)
        x_t = scheduler.step(pred_noise, t, x_t, step_key)

    x_t = x_t / jnp.sqrt(jnp.sum(jnp.abs(x_t) ** 2, keepdims=True))
    return x_t


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
    logger.info(f"{np.sum(np.abs(target_weights))=:.3f}")
    logger.info(f"{np.sum(np.abs(generated_weights))=:.3f}")
    logger.info(f"Weights MSE: {weights_mse:.4f}")
    logger.info(f"Pattern MSE (dB): {pattern_mse:.2f}")


if __name__ == "__main__":
    setup_logging()
    app()
