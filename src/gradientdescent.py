import logging
from functools import partial

import cyclopts
import jax
import jax.numpy as jnp
import numpy as np
import optax
from matplotlib import pyplot as plt

import physics as py
from diffusion import load_cst_params
from utils import setup_logging

logger = logging.getLogger(__name__)

app = cyclopts.App()


@jax.jit
def complex_field_loss(
    weights: jax.Array, target_field: jax.Array, geps: jax.Array
) -> jax.Array:
    """Loss function based on complex field matching (similar to least squares)."""
    pred_field = py.synthesize_pattern(geps, weights, power=False)
    field_diff = pred_field - target_field
    # MSE loss on real and imaginary parts
    loss = jnp.mean(jnp.real(field_diff) ** 2 + jnp.imag(field_diff) ** 2)
    return loss


@jax.jit
def power_pattern_loss(
    weights: jax.Array, target_pattern_db: jax.Array, geps: jax.Array
) -> jax.Array:
    pred_pattern = py.synthesize_pattern(geps, weights, power=True)
    pred_pattern_db = py.convert_to_db(pred_pattern)
    mse = ((pred_pattern_db - target_pattern_db) ** 2).mean()
    return mse


@jax.jit
def linear_power_loss(
    weights: jax.Array, target_pattern: jax.Array, geps: jax.Array
) -> jax.Array:
    """Loss function based on linear power pattern matching (no dB conversion)."""
    pred_pattern = py.synthesize_pattern(geps, weights, power=True)
    mse = ((pred_pattern - target_pattern) ** 2).mean()
    return mse


@app.command()
def optimize(
    theta: float = 0.0,
    phi: float = 0.0,
    loss_type: str = "linear_power",  # "complex", "power", or "linear_power"
    lr: float = 1e-3,
    n_steps: int = 1000,
):
    params = load_cst_params()
    steering_angles = jnp.array([jnp.radians(theta), jnp.radians(phi)])

    # Generate target from reference array
    ref_weights, _ = py.calculate_weights(params.ref.kx, params.ref.ky, steering_angles)
    target_field = py.synthesize_pattern(params.ref.geps, ref_weights, power=False)
    target_pattern = py.synthesize_pattern(params.ref.geps, ref_weights, power=True)
    target_pattern_db = py.convert_to_db(target_pattern)

    # Generate least-squares solution for comparison
    lstsq_weights = py.solve_weights(target_field, params.dis.geps, alpha=1e-3)
    lstsq_pattern_db = py.convert_to_db(
        py.synthesize_pattern(params.dis.geps, lstsq_weights)
    )
    lstsq_mse = jnp.mean((lstsq_pattern_db - target_pattern_db) ** 2)

    logger.info(f"Least squares baseline MSE: {lstsq_mse:.2f} dB²")

    logger.info(f"Running {loss_type} optimization for {theta=:.1f}°, {phi=:.1f}°")

    # Initialize with least squares solution + small perturbation for gradient descent
    # Start from least squares solution as initial guess
    init_weights = lstsq_weights.copy()
    init_weights = init_weights / jnp.sqrt(jnp.sum(jnp.abs(init_weights) ** 2))

    logger.info(f"Initializing with LSQ solution (MSE: {lstsq_mse:.2f} dB²)")

    weights = init_weights

    # Choose loss function and target
    if loss_type == "complex":
        loss_fn = lambda w: complex_field_loss(w, target_field, params.dis.geps)
        target_name = "complex field"
    elif loss_type == "power":
        loss_fn = lambda w: power_pattern_loss(w, target_pattern_db, params.dis.geps)
        target_name = "power pattern (dB)"
    elif loss_type == "linear_power":
        loss_fn = lambda w: linear_power_loss(w, target_pattern, params.dis.geps)
        target_name = "linear power pattern"
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Setup Adam optimizer
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(weights)

    @jax.jit
    def step(weights, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(weights)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_weights = optax.apply_updates(weights, updates)
        # Normalize weights to maintain unit power
        new_weights = new_weights / jnp.sqrt(jnp.sum(jnp.abs(new_weights) ** 2))
        return new_weights, opt_state, loss, jnp.sqrt(jnp.sum(jnp.abs(grads) ** 2))

    loss_history = []

    for i in range(n_steps):
        weights, opt_state, loss, grad_norm = step(weights, opt_state)
        loss_history.append(float(loss))

        if i % 100 == 0 or i == n_steps - 1:
            logger.info(f"Step {i:4d}: loss = {loss:.6f}, grad_norm = {grad_norm:.3f}")

    pred_weights = weights

    # Compute final metrics
    pred_pattern = py.synthesize_pattern(params.dis.geps, pred_weights, power=True)
    pred_pattern_db = py.convert_to_db(pred_pattern)

    # Calculate pattern MSE (always use this for comparison)
    pattern_mse = jnp.mean((pred_pattern_db - target_pattern_db) ** 2)

    # Find mainlobe indices and powers
    pattern_shape = target_pattern_db.shape
    target_mainlobe_idx = np.unravel_index(np.argmax(target_pattern_db), pattern_shape)
    target_mainlobe_power_db = target_pattern_db[target_mainlobe_idx]
    pred_mainlobe_idx = np.unravel_index(np.argmax(pred_pattern_db), pattern_shape)
    pred_mainlobe_power_db = pred_pattern_db[pred_mainlobe_idx]

    logger.info(f"Optimization completed ({target_name} loss)")
    logger.info(f"Pattern MSE: {pattern_mse:.2f} dB² (vs LSQ: {lstsq_mse:.2f} dB²)")
    logger.info(f"{target_mainlobe_idx=}, {target_mainlobe_power_db=:.2f} dB")
    logger.info(f"{pred_mainlobe_idx=}, {pred_mainlobe_power_db=:.2f} dB")
    logger.info(f"Reference weights sum: {np.sum(np.abs(ref_weights)):.3f}")
    logger.info(f"Optimized weights sum: {np.sum(np.abs(pred_weights)):.3f}")
    logger.info(f"Loss decreased from {loss_history[0]:.6f} to {loss_history[-1]:.6f}")


@app.command()
def dev(theta: float = 0.0, phi: float = 0.0):
    params = load_cst_params()
    steering_angles = np.array([np.radians(theta), np.radians(phi)])

    synth_db = lambda geps, w: py.convert_to_db(py.synthesize_pattern(geps, w))

    w_ref, _ = py.calculate_weights(params.ref.kx, params.ref.ky, steering_angles)
    target_field = py.synthesize_pattern(params.ref.geps, w_ref, power=False)
    target_power_db = synth_db(params.ref.geps, w_ref)
    target_peak = np.max(target_power_db)
    peak_idx = np.unravel_index(np.argmax(target_power_db), target_power_db.shape)
    w_ref_power = np.abs(w_ref).sum()

    w_lstsq = py.solve_weights(target_field, params.dis.geps, alpha=None)
    power_lstsq_db = synth_db(params.dis.geps, w_lstsq)
    lstsq_mse = np.mean(np.square(power_lstsq_db - target_power_db))
    lstsq_peak = np.max(power_lstsq_db[peak_idx])
    w_lstsq_power = np.abs(w_lstsq).sum()

    theta_u32, phi_u32 = np.uint32(theta), np.uint32(phi)
    w_mf = np.conj(params.dis.geps[:, :, theta_u32, phi_u32, 1])
    w_mf /= np.linalg.norm(w_mf)
    power_mf_db = synth_db(params.dis.geps, w_mf)
    mf_mse = np.mean(np.square(power_mf_db - target_power_db))
    mf_peak = np.max(power_mf_db[peak_idx])
    w_mf_power = np.abs(w_mf).sum()

    kw = dict(subplot_kw=dict(projection="polar"), layout="compressed")
    fig, ax = plt.subplots(figsize=(8, 8), **kw)
    plot = partial(py.plot_E_plane, phi_idx=0, ax=ax)

    label = f"Target (Peak: {target_peak:.1f}dB, Power: {w_ref_power:.3f})"
    plot(target_power_db, fmt="r-", label=label)

    label = f"LstSq (Peak: {lstsq_peak:.1f}dB, MSE: {lstsq_mse:.3f}, Power: {w_lstsq_power:.3f})"
    plot(power_lstsq_db, fmt="g-", label=label)

    label = f"MatchedFilter (Peak: {mf_peak:.1f}dB, MSE: {mf_mse:.3f}, Power: {w_mf_power:.3f})"
    plot(power_mf_db, fmt="b-", label=label)

    ax.legend(loc="lower center")
    title = f"Dev Pattern ({theta=:.1f}, {phi=:.1f})"
    fig.suptitle(title, fontweight="bold")
    filename = "gd_dev.png"
    fig.savefig(filename, dpi=250, bbox_inches="tight")
    logger.info(f"Saved random pattern plot to {filename}")


if __name__ == "__main__":
    setup_logging()
    app()
