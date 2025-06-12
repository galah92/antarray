from collections.abc import Iterable, Sequence
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.experimental.compilation_cache import compilation_cache as cc

# Persistent Jax compilation cache: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
cc.set_cache_dir("/tmp/jax_cache")


class ArrayConfig:
    ARRAY_SIZE: tuple[int, int] = (16, 16)
    SPACING_MM: tuple[float, float] = (60.0, 60.0)
    FREQUENCY_HZ: float = 2.45e9
    PATTERN_SHAPE: tuple[int, int] = (180, 360)  # (theta, phi)


def pad_batch(
    image: jax.Array,
    pad_width: Sequence[int | Sequence[int]],
    mode: str = "constant",
) -> jax.Array:
    pad_width = np.asarray(pad_width, dtype=np.int32)
    if pad_width.shape[0] == 3:  # Add batch dimension
        pad_width = np.pad(pad_width, ((1, 0), (0, 0)))
    return jnp.pad(image, pad_width=pad_width, mode=mode)


class ConvBlock(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | Sequence[int],
        padding: str = "SAME",
        *,
        rngs: nnx.Rngs,
    ):
        self.conv1 = nnx.Conv(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=False,
            rngs=rngs,
        )
        self.norm1 = nnx.BatchNorm(out_features, rngs=rngs)
        self.conv2 = nnx.Conv(
            out_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=False,
            rngs=rngs,
        )
        self.norm2 = nnx.BatchNorm(out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.conv1(x)
        x = self.norm1(x)
        x = nnx.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = nnx.relu(x)
        return x


def resize_batch(image, shape: Sequence[int], method: str | jax.image.ResizeMethod):
    shape = (image.shape[0], *shape)  # Add batch dimension
    return jax.image.resize(image, shape=shape, method=method)


class DenoisingUNet(nnx.Module):
    """UNet for denoising complex antenna weights with pattern conditioning."""

    def __init__(
        self,
        base_channels: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        # Pattern encoder - encodes target pattern to conditioning features
        self.pattern_pad = nnx.Sequential(
            partial(pad_batch, pad_width=((6, 6), (0, 0), (0, 0)), mode="reflect"),
            partial(pad_batch, pad_width=((0, 0), (12, 12), (0, 0)), mode="wrap"),
        )
        self.pattern_encoder = nnx.Sequential(
            ConvBlock(1, base_channels // 4, (3, 3), rngs=rngs),  # (192, 384, 16)
            partial(nnx.max_pool, window_shape=(3, 6), strides=(3, 6)),  # (64, 64, 16)
            ConvBlock(
                base_channels // 4, base_channels // 2, (3, 3), rngs=rngs
            ),  # (64, 64, 32)
            partial(nnx.max_pool, window_shape=(4, 4), strides=(4, 4)),  # (16, 16, 32)
            ConvBlock(
                base_channels // 2, base_channels, (3, 3), rngs=rngs
            ),  # (16, 16, 64)
        )

        # Time embedding for diffusion timestep - match concatenated feature size
        self.time_mlp = nnx.Sequential(
            nnx.Linear(1, base_channels, rngs=rngs),
            nnx.relu,
            nnx.Linear(
                base_channels, base_channels * 2, rngs=rngs
            ),  # Output 128 channels
        )

        # UNet encoder
        self.weights_input = ConvBlock(
            2, base_channels, (3, 3), rngs=rngs
        )  # 2 channels for real/imag

        self.down1 = ConvBlock(
            base_channels * 2, base_channels * 2, (3, 3), rngs=rngs
        )  # Pattern + weights
        self.down2 = ConvBlock(base_channels * 2, base_channels * 4, (3, 3), rngs=rngs)
        self.down3 = ConvBlock(base_channels * 4, base_channels * 8, (3, 3), rngs=rngs)

        # Bottleneck
        self.bottleneck = ConvBlock(
            base_channels * 8, base_channels * 8, (3, 3), rngs=rngs
        )

        # UNet decoder
        self.up3 = ConvBlock(
            base_channels * 16, base_channels * 4, (3, 3), rngs=rngs
        )  # Skip connection
        self.up2 = ConvBlock(base_channels * 8, base_channels * 2, (3, 3), rngs=rngs)
        self.up1 = ConvBlock(base_channels * 4, base_channels, (3, 3), rngs=rngs)

        # Output
        self.output = nnx.Conv(base_channels, 2, (1, 1), rngs=rngs)  # Real/imag output

    def __call__(
        self, noisy_weights: jax.Array, target_pattern: jax.Array, timestep: jax.Array
    ) -> jax.Array:
        # Encode target pattern
        pattern_input = target_pattern[..., None]  # Add channel dim
        pattern_features = self.pattern_pad(pattern_input)
        pattern_features = self.pattern_encoder(
            pattern_features
        )  # (B, 16, 16, base_channels)

        # Time embedding
        time_emb = self.time_mlp(timestep[..., None])  # (B, base_channels)
        time_emb = time_emb[:, None, None, :]  # (B, 1, 1, base_channels)

        # Process noisy weights (convert complex to real/imag channels)
        weights_real = jnp.real(noisy_weights)[..., None]
        weights_imag = jnp.imag(noisy_weights)[..., None]
        weights_input = jnp.concatenate([weights_real, weights_imag], axis=-1)
        weights_features = self.weights_input(
            weights_input
        )  # (B, 16, 16, base_channels)

        # Combine pattern and weights features
        x = jnp.concatenate(
            [pattern_features, weights_features], axis=-1
        )  # (B, 16, 16, base_channels*2)

        # Add time embedding
        x = x + time_emb

        # Encoder path
        x1 = self.down1(x)  # (B, 16, 16, base_channels*2)
        x2 = self.down2(nnx.max_pool(x1, (2, 2), (2, 2)))  # (B, 8, 8, base_channels*4)
        x3 = self.down3(nnx.max_pool(x2, (2, 2), (2, 2)))  # (B, 4, 4, base_channels*8)

        # Bottleneck
        bottleneck = self.bottleneck(
            nnx.max_pool(x3, (2, 2), (2, 2))
        )  # (B, 2, 2, base_channels*8)

        # Decoder path with skip connections
        up3 = resize_batch(bottleneck, (4, 4, bottleneck.shape[-1]), "bilinear")
        up3 = jnp.concatenate([up3, x3], axis=-1)
        up3 = self.up3(up3)

        up2 = resize_batch(up3, (8, 8, up3.shape[-1]), "bilinear")
        up2 = jnp.concatenate([up2, x2], axis=-1)
        up2 = self.up2(up2)

        up1 = resize_batch(up2, (16, 16, up2.shape[-1]), "bilinear")
        up1 = jnp.concatenate([up1, x1], axis=-1)
        up1 = self.up1(up1)

        # Output (real/imag channels)
        output = self.output(up1)  # (B, 16, 16, 2)

        # Convert back to complex
        output_complex = output[..., 0] + 1j * output[..., 1]
        return output_complex


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


def get_wavenumber(freq_hz: float) -> float:
    """Calculate the wavenumber for a given frequency in Hz."""
    c = 299792458
    wavelength = c / freq_hz
    return 2 * np.pi / wavelength


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
    """Factory to create a specialized function for calculating analytical weights."""
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

        phase_shifts = jnp.add.outer(x_phase, y_phase)
        return jnp.exp(-1j * phase_shifts), phase_shifts

    return calculate


def create_pattern_synthesizer(
    element_patterns: jax.Array, config: ArrayConfig
) -> callable:
    """Factory to create a specialized pattern synthesis function."""

    @jax.jit
    def precompute_basis(raw_patterns):
        k = get_wavenumber(config.FREQUENCY_HZ)
        x_pos, y_pos = get_element_positions(config.ARRAY_SIZE, config.SPACING_MM)
        k_pos_x, k_pos_y = k * x_pos, k * y_pos

        theta_size, phi_size = config.PATTERN_SHAPE
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


def create_element_patterns(
    config: ArrayConfig, key: jax.Array, is_embedded: bool
) -> jax.Array:
    """Simulates the element patterns for either an ideal or embedded array."""
    theta_size, phi_size = config.PATTERN_SHAPE
    theta = jnp.radians(jnp.arange(theta_size))

    # --- MINIMAL CHANGE IS HERE ---
    # Start with a base cosine model for the field amplitude.
    base_field_amp = jnp.cos(theta)
    # A realistic element on a ground plane has no back-lobe.
    # Set field to zero for theta > 90 degrees.
    base_field_amp = base_field_amp.at[theta > np.pi / 2].set(0)

    # Expand to a full 2D pattern (omnidirectional in phi)
    base_field_amp = base_field_amp[:, None] * jnp.ones((theta_size, phi_size))

    if not is_embedded:
        ideal_field = base_field_amp[None, None, :, :, None]
        ideal_field = jnp.tile(ideal_field, (*config.ARRAY_SIZE, 1, 1, 1))
        return ideal_field.astype(jnp.complex64)

    # --- Embedded Case: Simulate distortion ---
    num_pols = 2
    final_shape = (*config.ARRAY_SIZE, *config.PATTERN_SHAPE, num_pols)
    low_res_shape = (*config.ARRAY_SIZE, 10, 20, num_pols)

    key, amp_key, phase_key = jax.random.split(key, 3)

    amp_dist_low_res = jax.random.uniform(
        amp_key, low_res_shape, minval=0.5, maxval=1.5
    )
    amp_distortion = jax.image.resize(amp_dist_low_res, final_shape, method="bicubic")

    phase_dist_low_res = jax.random.uniform(phase_key, low_res_shape, maxval=2 * np.pi)
    phase_distortion = jax.image.resize(
        phase_dist_low_res, final_shape, method="bicubic"
    )

    distorted_amplitude = base_field_amp[None, None, ..., None] * amp_distortion
    distorted_field = distorted_amplitude * jnp.exp(1j * phase_distortion)

    return distorted_field.astype(jnp.complex64)


def steering_angles_sampler(
    key: jax.Array,
    batch_size: int,
    theta_end: float = np.radians(60),
    limit: int | None = None,
) -> Iterable[jax.Array]:
    """
    Creates a Python generator that yields batches of random steering angles.
    """
    if limit is None:
        limit = float("inf")
    i = 0
    while i < limit:
        key, theta_key, phi_key = jax.random.split(key, num=3)
        thetas = jax.random.uniform(theta_key, shape=(batch_size,), maxval=theta_end)
        phis = jax.random.uniform(phi_key, shape=(batch_size,), maxval=2 * jnp.pi)
        yield jnp.stack([thetas, phis], axis=-1)
        i += 1


@jax.jit
def normalize_patterns(patterns: jax.Array) -> jax.Array:
    """Performs peak normalization on a batch of radiation patterns."""
    max_vals = jnp.max(patterns, axis=(1, 2), keepdims=True)
    return patterns / (max_vals + 1e-8)


@jax.jit
def convert_to_db(patterns: jax.Array, floor_db: float = -60) -> jax.Array:
    """Converts a batch of normalized linear power patterns to a dB scale."""
    linear_floor = 10.0 ** (floor_db / 10.0)
    clipped_patterns = jnp.maximum(patterns, linear_floor)
    return 10.0 * jnp.log10(clipped_patterns)


@jax.jit
def calculate_pattern_loss(
    predicted_patterns: jax.Array, target_patterns: jax.Array
) -> tuple[jax.Array, dict]:
    """Calculates the loss and metrics between predicted and target patterns."""
    mse = optax.losses.squared_error(predicted_patterns, target_patterns).mean()
    rmse = jnp.sqrt(mse)
    return mse, {"mse": mse, "rmse": rmse}


def create_train_step_fn(
    synthesize_ideal_pattern: callable,
    synthesize_embedded_pattern: callable,
    compute_analytical_weights: callable,
    scheduler: DDPMScheduler,
):
    """Factory that creates the jitted training step function for diffusion."""
    vmapped_analytical_weights = jax.vmap(compute_analytical_weights)
    vmapped_embedded_synthesizer = jax.vmap(synthesize_embedded_pattern)

    def loss_fn(model: nnx.Module, batch_of_angles_rad: jax.Array, key: jax.Array):
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

        # Combined loss: denoising + physics (ensure real values)
        denoising_loss = jnp.mean(
            jnp.abs(predicted_noise - noise) ** 2
        )  # Use abs for complex
        physics_loss = jnp.mean(
            (predicted_patterns - target_patterns) ** 2
        )  # Already real

        total_loss = denoising_loss + 0.1 * physics_loss  # Weight physics loss

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
    model: nnx.Module,
    target_pattern: jax.Array,
    scheduler: DDPMScheduler,
    num_inference_steps: int = 200,
    guidance_scale: float = 1.0,
    synthesize_embedded_pattern: callable = None,
    key: jax.Array = None,
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

    print("Setting up diffusion training pipeline")

    # Create element patterns and synthesizers
    key, ideal_key, embedded_key = jax.random.split(key, 3)
    ideal_patterns = create_element_patterns(config, ideal_key, is_embedded=False)
    embedded_patterns = create_element_patterns(config, embedded_key, is_embedded=True)

    synthesize_ideal = create_pattern_synthesizer(ideal_patterns, config)
    synthesize_embedded = create_pattern_synthesizer(embedded_patterns, config)
    compute_analytical = create_analytical_weight_calculator(config)

    # Create scheduler and model
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    key, model_key = jax.random.split(key)
    model = DenoisingUNet(base_channels=64, rngs=nnx.Rngs(model_key))
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=lr, weight_decay=1e-6))

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

    print("Starting diffusion training")
    try:
        for step, batch in enumerate(sampler):
            key, step_key = jax.random.split(key)
            metrics = train_step(optimizer, batch, step_key)

            if (step + 1) % 100 == 0:
                print(
                    f"step {step + 1}/{n_steps}, "
                    f"grad_norm: {metrics['grad_norm'].item():.3f}, "
                    f"total_loss: {metrics['total_loss'].item():.3f}, "
                    f"denoising_loss: {metrics['denoising_loss'].item():.3f}, "
                    f"physics_loss: {metrics['physics_loss'].item():.3f}"
                )
    except KeyboardInterrupt:
        print("Training interrupted by user")

    print("Diffusion training completed")

    return model, scheduler, synthesize_embedded


if __name__ == "__main__":
    train_diffusion_pipeline()
