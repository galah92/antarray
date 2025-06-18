import logging
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from physics import create_physics_setup, normalize_patterns
from training import (
    VelocityNet,
    create_checkpoint_manager,
    create_progress_logger,
    create_standard_optimizer,
    restore_checkpoint,
    save_checkpoint,
    steering_angles_sampler,
)

logger = logging.getLogger(__name__)


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
        velocity_net: VelocityNet,
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
        self, velocity_net: VelocityNet, x0: jax.Array, target_pattern: jax.Array
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


def solve_with_flow_matching(
    model: VelocityNet,
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


def create_train_step_fn(
    synthesize_ideal_pattern: Callable,
    synthesize_embedded_pattern: Callable,
    compute_analytical_weights: Callable,
    flow_matcher: FlowMatcher,
):
    """Factory that creates the jitted training step function for flow matching."""
    vmapped_analytical_weights = jax.vmap(compute_analytical_weights)
    vmapped_embedded_synthesizer = jax.vmap(synthesize_embedded_pattern)

    def loss_fn(model: VelocityNet, batch_of_angles_rad: jax.Array, key: jax.Array):
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

        total_loss = fm_loss + 10.0 * physics_loss

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


def train_flow_matching_pipeline(
    n_steps: int = 10_000,
    batch_size: int = 256,
    lr: float = 1e-4,
    seed: int = 42,
    checkpoint_dir: str = "flow_matching_checkpoints",
    save_every: int = 1000,
    restore: bool = True,
    openems_path: Path | None = None,
):
    """Main training function for the flow matching model."""
    key = jax.random.key(seed)
    checkpoint_path = Path(checkpoint_dir).resolve()

    logger.info("Setting up flow matching training pipeline")

    # Create physics setup with optional OpenEMS support
    synthesize_ideal, compute_analytical = create_physics_setup(
        key, openems_path=openems_path
    )

    # Create flow matcher and model
    flow_matcher = FlowMatcher()
    key, model_key = jax.random.split(key)
    model = VelocityNet(base_channels=64, rngs=nnx.Rngs(model_key))

    # Create checkpoint manager
    ckpt_mngr = create_checkpoint_manager(checkpoint_path)

    # Restore from checkpoint if requested
    start_step = 0
    if restore:
        start_step = restore_checkpoint(ckpt_mngr, model, step=None)

    # Create optimizer
    optimizer = create_standard_optimizer(model, lr, n_steps)

    # Create training step
    train_step = create_train_step_fn(
        synthesize_ideal, synthesize_ideal, compute_analytical, flow_matcher
    )

    # Training data generator
    key, data_key = jax.random.split(key)
    remaining_steps = n_steps - start_step
    sampler = steering_angles_sampler(data_key, batch_size, limit=remaining_steps)

    logger.info(f"Starting flow matching training from step {start_step}")
    log_progress = create_progress_logger(
        total_steps=n_steps, log_every=100, start_step=start_step
    )

    try:
        for step_offset, batch in enumerate(sampler):
            step = start_step + step_offset
            key, step_key = jax.random.split(key)
            metrics = train_step(optimizer, batch, step_key)

            log_progress(step, metrics)

            # Save checkpoint
            if (step + 1) % save_every == 0:
                save_checkpoint(ckpt_mngr, model, step + 1, overwrite=True)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    # Save final checkpoint
    save_checkpoint(ckpt_mngr, model, start_step + remaining_steps, overwrite=True)
    logger.info("Flow matching training completed")

    return model, synthesize_embedded


def evaluate_flow_matching_model(
    model: VelocityNet,
    synthesize_embedded: Callable,
    n_eval_samples: int = 50,
    seed: int = 123,
    openems_path: Path | None = None,
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

    # Create physics setup for evaluation with optional OpenEMS support
    key, physics_key = jax.random.split(key)
    synthesize_ideal, compute_analytical = create_physics_setup(
        physics_key, openems_path=openems_path
    )

    # Compute analytical weights and target patterns
    vmapped_analytical_weights = jax.vmap(compute_analytical)
    analytical_weights, _ = vmapped_analytical_weights(test_batch)

    # Create ideal target patterns
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
    # Train the model
    model, synthesize_embedded = train_flow_matching_pipeline()

    # Evaluate the trained model
    eval_results = evaluate_flow_matching_model(model, synthesize_embedded)

    logger.info("Flow matching training and evaluation completed!")
