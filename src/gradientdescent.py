import logging

import cyclopts
import jax
import jax.numpy as jnp
import numpy as np

import physics as py
from diffusion import load_cst_params
from utils import setup_logging

logger = logging.getLogger(__name__)

app = cyclopts.App()


@jax.jit
def power_pattern_loss(
    weights: jax.Array, target_pattern_db: jax.Array, geps: jax.Array
) -> jax.Array:
    pred_pattern = py.synthesize_pattern(geps, weights, power=True)
    pred_pattern_db = py.convert_to_db(pred_pattern)
    mse = ((pred_pattern_db - target_pattern_db) ** 2).mean()
    return mse


@jax.jit
def opt_step(
    weights: jax.Array,
    target_pattern_db: jax.Array,
    geps: jax.Array,
    lr: float,
) -> tuple[jax.Array, jax.Array]:
    loss, grads = jax.value_and_grad(power_pattern_loss)(
        weights, target_pattern_db, geps
    )
    new_weights = weights - lr * grads
    # Normalize weights to maintain unit power
    new_weights = new_weights / jnp.sqrt(jnp.sum(jnp.abs(new_weights) ** 2))
    jax.debug.print(
        "loss={:.3}, grads_mean={:.3f}, w_abs_mean={:.3f}",
        loss,
        jnp.abs(grads).mean(),
        jnp.abs(new_weights).mean(),
    )
    return new_weights, loss


@app.command()
def optimize(
    theta: float = 0.0,
    phi: float = 0.0,
    lr: float = 1e-4,
    n_steps: int = 10,
):
    params = load_cst_params()
    steering_angles = jnp.array([jnp.radians(theta), jnp.radians(phi)])

    # Generate target from reference array
    ref_weights, _ = py.calculate_weights(params.ref.kx, params.ref.ky, steering_angles)
    target_pattern = py.synthesize_pattern(params.ref.geps, ref_weights, power=True)
    target_pattern_db = py.convert_to_db(target_pattern)

    logger.info(f"Running optimization for {theta=:.1f}°, {phi=:.1f}°")

    array_size = params.dis.geps.shape[:2]
    weights = jnp.ones(array_size, dtype=jnp.complex64)
    loss_history = []

    for i in range(n_steps):
        weights, loss = opt_step(weights, target_pattern_db, params.dis.geps, lr)
        loss_history.append(loss.item())

        if i % 100 == 0 or i == n_steps - 1:
            logger.info(f"Step {i:4d}: loss = {loss:.6f}")

    pred_weights = weights

    # Compute final metrics
    pred_pattern = py.synthesize_pattern(params.dis.geps, pred_weights, power=True)
    pred_pattern_db = py.convert_to_db(pred_pattern)

    # Calculate metrics
    final_loss = jnp.mean((pred_pattern_db - target_pattern_db) ** 2)
    pattern_mse = jnp.mean((pred_pattern_db - target_pattern_db) ** 2)

    # Find mainlobe indices and powers
    pattern_shape = target_pattern_db.shape
    target_mainlobe_idx = np.unravel_index(np.argmax(target_pattern_db), pattern_shape)
    target_mainlobe_power_db = target_pattern_db[target_mainlobe_idx]
    pred_mainlobe_idx = np.unravel_index(np.argmax(pred_pattern_db), pattern_shape)
    pred_mainlobe_power_db = pred_pattern_db[pred_mainlobe_idx]

    logger.info(f"Optimization completed with final loss: {final_loss:.6f}")
    logger.info(f"Pattern MSE: {pattern_mse:.2f} dB²")
    logger.info(f"{target_mainlobe_idx=}, {target_mainlobe_power_db=:.2f} dB")
    logger.info(f"{pred_mainlobe_idx=}, {pred_mainlobe_power_db=:.2f} dB")
    logger.info(f"Optimized weights sum: {np.sum(np.abs(pred_weights)):.3f}")
    logger.info(f"Loss decreased from {loss_history[0]:.6f} to {loss_history[-1]:.6f}")


if __name__ == "__main__":
    setup_logging()
    app()
