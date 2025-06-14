import logging
import sys
import time
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax import nnx

from shared import (
    ArrayConfig,
    ConvBlock,
    create_analytical_weight_calculator,
    create_element_patterns,
    create_pattern_synthesizer,
    normalize_patterns,
    pad_batch,
    resize_batch,
    steering_angles_sampler,
)

logger = logging.getLogger(__name__)


class VelocityNet(nnx.Module):
    """Neural network that predicts the velocity field for flow matching."""

    def __init__(
        self,
        base_channels: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        # Pattern encoder - same as diffusion model
        self.pattern_pad = nnx.Sequential(
            partial(pad_batch, pad_width=((6, 6), (0, 0), (0, 0)), mode="reflect"),
            partial(pad_batch, pad_width=((0, 0), (12, 12), (0, 0)), mode="wrap"),
        )
        self.pattern_encoder = nnx.Sequential(
            ConvBlock(1, base_channels // 4, (3, 3), rngs=rngs),
            partial(nnx.max_pool, window_shape=(3, 6), strides=(3, 6)),
            ConvBlock(base_channels // 4, base_channels // 2, (3, 3), rngs=rngs),
            partial(nnx.max_pool, window_shape=(4, 4), strides=(4, 4)),
            ConvBlock(base_channels // 2, base_channels, (3, 3), rngs=rngs),
        )

        # Time embedding for flow time t ∈ [0, 1]
        self.time_mlp = nnx.Sequential(
            nnx.Linear(1, base_channels, rngs=rngs),
            nnx.relu,
            nnx.Linear(base_channels, base_channels * 2, rngs=rngs),
        )

        # UNet for processing current state
        self.weights_input = ConvBlock(2, base_channels, (3, 3), rngs=rngs)

        self.down1 = ConvBlock(base_channels * 2, base_channels * 2, (3, 3), rngs=rngs)
        self.down2 = ConvBlock(base_channels * 2, base_channels * 4, (3, 3), rngs=rngs)
        self.down3 = ConvBlock(base_channels * 4, base_channels * 8, (3, 3), rngs=rngs)

        self.bottleneck = ConvBlock(
            base_channels * 8, base_channels * 8, (3, 3), rngs=rngs
        )

        self.up3 = ConvBlock(base_channels * 16, base_channels * 4, (3, 3), rngs=rngs)
        self.up2 = ConvBlock(base_channels * 8, base_channels * 2, (3, 3), rngs=rngs)
        self.up1 = ConvBlock(base_channels * 4, base_channels, (3, 3), rngs=rngs)

        # Output velocity field (real/imag channels)
        self.output = nnx.Conv(base_channels, 2, (1, 1), rngs=rngs)

    def __call__(
        self, weights: jax.Array, target_pattern: jax.Array, time: jax.Array
    ) -> jax.Array:
        # Encode target pattern
        pattern_input = target_pattern[..., None]
        pattern_features = self.pattern_pad(pattern_input)
        pattern_features = self.pattern_encoder(pattern_features)

        # Time embedding
        time_emb = self.time_mlp(time[..., None])
        time_emb = time_emb[:, None, None, :]

        # Process current weights
        weights_real = jnp.real(weights)[..., None]
        weights_imag = jnp.imag(weights)[..., None]
        weights_input = jnp.concatenate([weights_real, weights_imag], axis=-1)
        weights_features = self.weights_input(weights_input)

        # Combine features
        x = jnp.concatenate([pattern_features, weights_features], axis=-1)
        x = x + time_emb

        # UNet forward pass
        x1 = self.down1(x)
        x2 = self.down2(nnx.max_pool(x1, (2, 2), (2, 2)))
        x3 = self.down3(nnx.max_pool(x2, (2, 2), (2, 2)))

        bottleneck = self.bottleneck(nnx.max_pool(x3, (2, 2), (2, 2)))

        up3 = resize_batch(bottleneck, (4, 4, bottleneck.shape[-1]), "bilinear")
        up3 = jnp.concatenate([up3, x3], axis=-1)
        up3 = self.up3(up3)

        up2 = resize_batch(up3, (8, 8, up3.shape[-1]), "bilinear")
        up2 = jnp.concatenate([up2, x2], axis=-1)
        up2 = self.up2(up2)

        up1 = resize_batch(up2, (16, 16, up2.shape[-1]), "bilinear")
        up1 = jnp.concatenate([up1, x1], axis=-1)
        up1 = self.up1(up1)

        output = self.output(up1)

        # Convert to complex velocity
        velocity_complex = output[..., 0] + 1j * output[..., 1]
        return velocity_complex


class FlowMatcher:
    """Flow matching using optimal transport paths."""

    def __init__(self, sigma_min: float = 1e-4):
        self.sigma_min = sigma_min

    def sample_time(self, key: jax.Array, batch_size: int) -> jax.Array:
        """Sample random times t ∈ [0, 1]."""
        return jax.random.uniform(key, (batch_size,))

    def optimal_transport_path(
        self, x0: jax.Array, x1: jax.Array, t: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """
        Compute optimal transport path and velocity.

        For flow matching, we use the simple linear interpolation:
        x_t = (1 - t) * x_0 + t * x_1 + σ_min * ε
        v_t = x_1 - x_0
        """
        # Reshape t for broadcasting
        t = t[:, None, None]

        # Add small noise for numerical stability
        noise = jax.random.normal(jax.random.key(0), x0.shape) * self.sigma_min

        # Linear interpolation path
        x_t = (1 - t) * x0 + t * x1 + noise

        # Constant velocity (target - source)
        v_t = x1 - x0

        return x_t, v_t

    def compute_loss(
        self,
        velocity_net: nnx.Module,
        x0: jax.Array,
        x1: jax.Array,
        target_pattern: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, dict]:
        """Compute flow matching loss."""
        batch_size = x0.shape[0]

        # Sample random times
        key, time_key = jax.random.split(key)
        t = self.sample_time(time_key, batch_size)

        # Get optimal transport path and true velocity
        x_t, v_t_true = self.optimal_transport_path(x0, x1, t)

        # Predict velocity
        v_t_pred = velocity_net(x_t, target_pattern, t)

        # Flow matching loss (MSE between predicted and true velocity)
        fm_loss = jnp.mean(jnp.abs(v_t_pred - v_t_true) ** 2)

        return fm_loss, {"flow_matching_loss": fm_loss}


class ODESolver:
    """Euler method ODE solver for flow generation."""

    def __init__(self, num_steps: int = 100):
        self.num_steps = num_steps
        self.dt = 1.0 / num_steps

    def solve(
        self, velocity_net: nnx.Module, x0: jax.Array, target_pattern: jax.Array
    ) -> jax.Array:
        """Solve ODE from t=0 to t=1 using Euler method."""
        x = x0

        for i in range(self.num_steps):
            t = jnp.array([i * self.dt])
            t_batch = jnp.full((x.shape[0],), t[0])

            # Predict velocity at current time and state
            v = velocity_net(x, target_pattern, t_batch)

            # Euler step
            x = x + self.dt * v

        return x


def create_train_step_fn(
    synthesize_ideal_pattern: callable,
    synthesize_embedded_pattern: callable,
    compute_analytical_weights: callable,
    flow_matcher: FlowMatcher,
):
    """Factory that creates the jitted training step function for flow matching."""
    vmapped_analytical_weights = jax.vmap(compute_analytical_weights)
    vmapped_embedded_synthesizer = jax.vmap(synthesize_embedded_pattern)

    def loss_fn(model: nnx.Module, batch_of_angles_rad: jax.Array, key: jax.Array):
        batch_size = batch_of_angles_rad.shape[0]

        # Generate analytical weights and target patterns
        analytical_weights, _ = vmapped_analytical_weights(batch_of_angles_rad)
        ideal_patterns = jax.vmap(synthesize_ideal_pattern)(analytical_weights)
        target_patterns = normalize_patterns(ideal_patterns)

        # Sample noise as starting point (x0)
        key, noise_key = jax.random.split(key)
        x0 = jax.random.normal(
            noise_key, analytical_weights.shape, dtype=analytical_weights.dtype
        )

        # Use analytical weights as target (x1)
        x1 = analytical_weights

        # Compute flow matching loss
        fm_loss, fm_metrics = flow_matcher.compute_loss(
            model, x0, x1, target_patterns, key
        )

        # Optional: Add physics-based regularization
        key, eval_key = jax.random.split(key)
        t_eval = jax.random.uniform(eval_key, (batch_size,))
        x_t, _ = flow_matcher.optimal_transport_path(x0, x1, t_eval)

        # Evaluate physics at intermediate points
        predicted_patterns = vmapped_embedded_synthesizer(x_t)
        predicted_patterns = normalize_patterns(predicted_patterns)
        physics_loss = jnp.mean((predicted_patterns - target_patterns) ** 2)

        total_loss = fm_loss + 10.0 * physics_loss  # Increased from 0.01 to 10.0

        metrics = {
            "flow_matching_loss": fm_loss,
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


def solve_with_flow_matching(
    model: nnx.Module,
    target_pattern: jax.Array,
    solver: ODESolver,
    key: jax.Array,
) -> jax.Array:
    """Phase 2: Use trained model to solve for corrective weights via flow generation."""

    # Ensure model is in evaluation mode
    model.eval()

    # Start with random noise
    sample_shape = (1, 16, 16)  # Add batch dimension
    x0 = jax.random.normal(key, sample_shape, dtype=jnp.complex64)

    target_pattern_batch = target_pattern[None, ...]  # Add batch dimension

    # Solve ODE from noise to solution
    solution = solver.solve(model, x0, target_pattern_batch)

    return solution[0]  # Remove batch dimension


def save_checkpoint(model: nnx.Module, step: int, checkpoint_dir: Path):
    """Save model checkpoint using orbax."""
    checkpoint_dir = checkpoint_dir.resolve()  # Ensure absolute path
    checkpoint_dir.mkdir(exist_ok=True)

    options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=1)
    with ocp.CheckpointManager(checkpoint_dir, options=options) as mngr:
        state = nnx.state(model)
        mngr.save(step, args=ocp.args.StandardSave(state))
        logger.info(f"Saved checkpoint at step {step}")


def load_checkpoint(model: nnx.Module, checkpoint_dir: Path, step: int = None) -> int:
    """Load model checkpoint using orbax."""
    checkpoint_dir = checkpoint_dir.resolve()  # Ensure absolute path
    if not checkpoint_dir.exists():
        logger.info("No checkpoint directory found, starting from scratch")
        return 0

    options = ocp.CheckpointManagerOptions(read_only=True)
    with ocp.CheckpointManager(checkpoint_dir, options=options) as mngr:
        if step is None:
            step = mngr.latest_step()

        if step is None:
            logger.info("No checkpoints found, starting from scratch")
            return 0

        state = nnx.state(model)
        restored = mngr.restore(step, args=ocp.args.StandardRestore(state))
        nnx.update(model, restored)
        logger.info(f"Restored checkpoint from step {step}")
        return step


def train_flow_matching_pipeline(
    n_steps: int = 10_000,
    batch_size: int = 256,
    lr: float = 1e-4,
    seed: int = 42,
    checkpoint_dir: str = "flow_matching_checkpoints",
    save_every: int = 1000,
    restore: bool = True,
):
    """Main training function for the flow matching model."""
    config = ArrayConfig()
    key = jax.random.key(seed)
    checkpoint_path = Path(checkpoint_dir).resolve()  # Make path absolute

    logger.info("Setting up flow matching training pipeline")

    # Create element patterns and synthesizers
    key, ideal_key, embedded_key = jax.random.split(key, 3)
    ideal_patterns = create_element_patterns(config, ideal_key, is_embedded=False)
    embedded_patterns = create_element_patterns(config, embedded_key, is_embedded=True)

    synthesize_ideal = create_pattern_synthesizer(ideal_patterns, config)
    synthesize_embedded = create_pattern_synthesizer(embedded_patterns, config)
    compute_analytical = create_analytical_weight_calculator(config)

    # Create flow matcher and model
    flow_matcher = FlowMatcher()

    key, model_key = jax.random.split(key)
    model = VelocityNet(base_channels=64, rngs=nnx.Rngs(model_key))

    # Restore from checkpoint if requested
    start_step = 0
    if restore:
        start_step = load_checkpoint(model, checkpoint_path)

    # Learning rate schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.1,
        peak_value=lr,
        warmup_steps=500,
        decay_steps=n_steps - 500,
        end_value=lr * 0.01,
    )

    optimizer = nnx.Optimizer(
        model, optax.adamw(learning_rate=lr_schedule, weight_decay=1e-6)
    )

    # Create training step
    train_step = create_train_step_fn(
        synthesize_ideal,
        synthesize_embedded,
        compute_analytical,
        flow_matcher,
    )

    # Training data generator
    key, data_key = jax.random.split(key)
    remaining_steps = n_steps - start_step
    sampler = steering_angles_sampler(data_key, batch_size, limit=remaining_steps)

    logger.info(f"Starting flow matching training from step {start_step}")
    start_time = time.perf_counter()
    try:
        for step_offset, batch in enumerate(sampler):
            step = start_step + step_offset
            key, step_key = jax.random.split(key)
            metrics = train_step(optimizer, batch, step_key)

            if (step + 1) % 100 == 0:
                elapsed = time.perf_counter() - start_time
                avg_time = elapsed / (step + 1 - start_step)
                elapsed = datetime.min + timedelta(seconds=elapsed)
                logger.info(
                    f"step {step + 1}/{n_steps}, "
                    f"time: {avg_time * 1000:.1f}ms/step, "
                    f"grad_norm: {metrics['grad_norm'].item():.3f}, "
                    f"total_loss: {metrics['total_loss'].item():.3f}, "
                    f"flow_matching_loss: {metrics['flow_matching_loss'].item():.3f}, "
                    f"physics_loss: {metrics['physics_loss'].item():.3f}"
                )

            # Save checkpoint
            if (step + 1) % save_every == 0:
                save_checkpoint(model, step + 1, checkpoint_path)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    # Save final checkpoint
    save_checkpoint(model, start_step + remaining_steps, checkpoint_path)
    logger.info("Flow matching training completed")

    return model, synthesize_embedded


def evaluate_flow_matching_model(
    model: nnx.Module,
    synthesize_embedded: callable,
    config: ArrayConfig,
    n_eval_samples: int = 50,
    seed: int = 123,
):
    """Evaluate the trained flow matching model on test steering angles."""

    # Set model to evaluation mode to disable batch norm updates
    model.eval()

    key = jax.random.key(seed)
    solver = ODESolver(num_steps=100)

    # Generate test steering angles
    key, angle_key = jax.random.split(key)
    test_angles = steering_angles_sampler(angle_key, batch_size=n_eval_samples, limit=1)
    test_batch = next(test_angles)

    # Compute analytical weights and target patterns
    compute_analytical = create_analytical_weight_calculator(config)
    vmapped_analytical_weights = jax.vmap(compute_analytical)
    analytical_weights, _ = vmapped_analytical_weights(test_batch)

    # Create ideal target patterns
    key, ideal_key = jax.random.split(key)
    ideal_patterns = create_element_patterns(config, ideal_key, is_embedded=False)
    synthesize_ideal = create_pattern_synthesizer(ideal_patterns, config)
    ideal_target_patterns = jax.vmap(synthesize_ideal)(analytical_weights)
    ideal_target_patterns = normalize_patterns(ideal_target_patterns)

    # Solve using flow matching for each target pattern - process one by one to avoid vmap issues
    logger.info(f"Evaluating flow matching on {n_eval_samples} test samples...")

    predicted_weights_list = []
    key, solve_key = jax.random.split(key)

    for i in range(n_eval_samples):
        solve_key, single_key = jax.random.split(solve_key)
        target_pattern = ideal_target_patterns[i]

        predicted_weight = solve_with_flow_matching(
            model, target_pattern, solver, single_key
        )
        predicted_weights_list.append(predicted_weight)

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{n_eval_samples} samples")

    predicted_weights = jnp.stack(predicted_weights_list)

    # Compute evaluation metrics
    weight_mse = jnp.mean(jnp.abs(predicted_weights - analytical_weights) ** 2)

    # Pattern quality metrics
    predicted_patterns = jax.vmap(synthesize_embedded)(predicted_weights)
    predicted_patterns = normalize_patterns(predicted_patterns)
    pattern_mse = jnp.mean((predicted_patterns - ideal_target_patterns) ** 2)

    logger.info("Flow Matching Evaluation Results:")
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
            logging.FileHandler(Path("app.log"), mode="w+"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.getLogger("absl").setLevel(logging.CRITICAL)

    logger.info(f"uv run {' '.join(sys.argv)}")

    # Train the model
    model, synthesize_embedded = train_flow_matching_pipeline()

    # Evaluate the trained model
    config = ArrayConfig()
    eval_results = evaluate_flow_matching_model(model, synthesize_embedded, config)

    logger.info("Flow matching training and evaluation completed!")
