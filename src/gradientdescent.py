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


def normalize_pattern(pattern: jax.Array) -> jax.Array:
    """Normalize the input pattern to the range [0, 1]."""
    min_val = pattern.min()
    max_val = pattern.max()
    return (pattern - min_val) / (max_val - min_val + 1e-10)


@jax.jit
def power_loss(weights: jax.Array, target: jax.Array, geps: jax.Array) -> jax.Array:
    pred_pattern = py.synthesize_pattern(geps, weights)
    pred_pattern_db = py.convert_to_db(pred_pattern)

    pred_normalized = normalize_pattern(pred_pattern_db)

    mse = ((pred_normalized - target) ** 2).mean()
    return mse


def create_ideal_beam_pattern(
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    steering_theta: float,
    steering_phi: float,
    beam_width_deg: float = 10.0,
) -> np.ndarray:
    """Create an idealized beam pattern with sharp 0/1 transitions.

    Args:
        theta_rad: Theta angles in radians
        phi_rad: Phi angles in radians
        steering_theta: Steering angle theta in radians
        steering_phi: Steering angle phi in radians
        beam_width_deg: Main beam width in degrees (3dB beamwidth)

    Returns:
        Binary pattern with 1s in main beam, 0s elsewhere
    """
    theta_deg = np.degrees(theta_rad)
    phi_deg = np.degrees(phi_rad)
    steering_theta_deg = np.degrees(steering_theta)
    steering_phi_deg = np.degrees(steering_phi)

    # Create 2D grids
    theta_grid, phi_grid = np.meshgrid(theta_deg, phi_deg, indexing="ij")

    # Calculate angular distance from steering direction
    # For simplicity, use a rectangular window in theta-phi space
    half_beam_width = beam_width_deg / 2.0

    theta_in_beam = np.abs(theta_grid - steering_theta_deg) <= half_beam_width
    phi_in_beam = np.abs(phi_grid - steering_phi_deg) <= half_beam_width

    # Handle phi wraparound at 0°/360°
    phi_diff = np.abs(phi_grid - steering_phi_deg)
    phi_dist = np.minimum(phi_diff, 360 - phi_diff)
    phi_in_beam = phi_dist <= half_beam_width

    # Combine conditions - beam is 1 where both theta and phi are within limits
    beam_pattern = (theta_in_beam & phi_in_beam).astype(np.float32)

    return beam_pattern


@app.command()
def optimize(
    theta: float = 0.0,
    phi: float = 0.0,
    lr: float = 1e-4,
    n_steps: int = 1_000,
    beam_width: float = 10.0,
    ideal_target: bool = True,
):
    params = load_cst_params()
    steering_angles = np.array([np.radians(theta), np.radians(phi)])

    if ideal_target:
        # Generate idealized window pattern
        theta_rad, phi_rad = np.radians(np.arange(180)), np.radians(np.arange(360))
        target_pattern = create_ideal_beam_pattern(
            theta_rad,
            phi_rad,
            steering_angles[0],
            steering_angles[1],
            beam_width,
        )
        target_pattern_db = target_pattern
        target_pattern_norm = target_pattern
        target_type = f"Ideal window (beam_width={beam_width}°)"
    else:
        # Generate target from reference array
        ref_weights, _ = py.calculate_weights(
            params.ref.kx, params.ref.ky, steering_angles
        )
        target_pattern = py.synthesize_pattern(params.ref.geps, ref_weights, power=True)
        target_pattern_db = py.convert_to_db(target_pattern)
        target_pattern_norm = normalize_pattern(target_pattern_db)
        target_type = "Reference array"

    # Generate matched filter solution for comparison
    mf_weights = py.matched_filter_weights(np.asarray(params.dis.geps), steering_angles)
    mf_pattern_db = py.convert_to_db(py.synthesize_pattern(params.dis.geps, mf_weights))
    mf_pattern_norm = normalize_pattern(mf_pattern_db)
    mf_mse = jnp.mean((mf_pattern_db - target_pattern_db) ** 2)

    logger.info(f"Matched filter baseline MSE: {mf_mse:.2f} dB²")

    logger.info(
        f"Running optimization for {theta=:.1f}°, {phi=:.1f}° with target: {target_type}"
    )

    # Initialize with matched filter solution + small perturbation for gradient descent
    # Start from matched filter solution as initial guess
    weights = mf_weights.copy()
    weights = weights / jnp.sqrt(jnp.sum(jnp.abs(weights) ** 2))

    logger.info(f"Initializing with MF solution (MSE: {mf_mse:.2f} dB²)")

    loss_fn = lambda w: power_loss(w, target_pattern_norm, params.dis.geps)

    optimizer = optax.sgd(learning_rate=lr)
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
    pred_pattern_norm = normalize_pattern(pred_pattern_db)

    # Calculate pattern MSE (always use this for comparison)
    pattern_mse = jnp.mean((pred_pattern_db - target_pattern_db) ** 2)

    # Find mainlobe indices and powers
    pattern_shape = target_pattern_db.shape
    target_mainlobe_idx = np.unravel_index(np.argmax(target_pattern_db), pattern_shape)
    target_mainlobe_power_db = target_pattern_db[target_mainlobe_idx]
    pred_mainlobe_idx = np.unravel_index(np.argmax(pred_pattern_db), pattern_shape)
    pred_mainlobe_power_db = pred_pattern_db[pred_mainlobe_idx]

    logger.info(f"Pattern MSE: {pattern_mse:.2f} dB² (vs MF: {mf_mse:.2f} dB²)")
    logger.info(f"{target_mainlobe_idx=}, {target_mainlobe_power_db=:.2f} dB")
    logger.info(f"{pred_mainlobe_idx=}, {pred_mainlobe_power_db=:.2f} dB")

    if not ideal_target:
        logger.info(f"Reference weights sum: {np.sum(np.abs(ref_weights)):.3f}")

    logger.info(f"Optimized weights sum: {np.sum(np.abs(pred_weights)):.3f}")
    logger.info(f"Loss decreased from {loss_history[0]:.6f} to {loss_history[-1]:.6f}")

    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    phi_idx = 0
    ax.plot(target_pattern_norm[:, phi_idx], label="Target Pattern", color="blue")
    ax.plot(pred_pattern_norm[:, phi_idx], label="Optimized Pattern", color="orange")
    ax.plot(mf_pattern_norm[:, phi_idx], label="Matched Filter Pattern", color="green")
    ax.set_title(f"Radiation Pattern Comparison\n{theta=:.1f}°, {phi=:.1f}°")
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Power (dB)")
    ax.legend()
    filename = f"optimized_pattern_{theta:.1f}_{phi:.1f}.png"
    fig.savefig(filename, dpi=250, bbox_inches="tight")
    logger.info(f"Saved pattern plot to {filename}")


@app.command()
def dev(theta: float = 30.0, phi: float = 0.0):
    params = load_cst_params()
    steering_angles = np.array([np.radians(theta), np.radians(phi)])
    pattern_shape = params.ref.geps.shape[2:4]

    synth_db = lambda geps, w: py.convert_to_db(py.synthesize_pattern(geps, w))

    w_ref, _ = py.calculate_weights(params.ref.kx, params.ref.ky, steering_angles)
    ref_field = py.synthesize_pattern(params.ref.geps, w_ref, power=False)
    ref_power_db = synth_db(params.ref.geps, w_ref)
    ref_peak_idx = py.unravel_index(np.argmax(ref_power_db), pattern_shape)
    ref_peak = ref_power_db[ref_peak_idx]
    w_ref_power = np.abs(w_ref).sum()

    w_lstsq = py.solve_weights(ref_field, params.dis.geps, alpha=None)
    power_lstsq_db = synth_db(params.dis.geps, w_lstsq)
    lstsq_mse = np.mean(np.square(power_lstsq_db - ref_power_db))
    lstsq_peak_idx = py.unravel_index(np.argmax(power_lstsq_db), pattern_shape)
    lstsq_peak = power_lstsq_db[lstsq_peak_idx]
    w_lstsq_power = np.abs(w_lstsq).sum()

    w_mf = py.matched_filter_weights(np.asarray(params.dis.geps), steering_angles)
    power_mf_db = synth_db(params.dis.geps, w_mf)
    mf_mse = np.mean(np.square(power_mf_db - ref_power_db))
    mf_peak_idx = py.unravel_index(np.argmax(power_mf_db), pattern_shape)
    mf_peak = power_mf_db[mf_peak_idx]
    w_mf_power = np.abs(w_mf).sum()

    w_mf_ref = py.matched_filter_weights(np.asarray(params.ref.geps), steering_angles)
    power_mf_ref_db = synth_db(params.ref.geps, w_mf_ref)
    mf_ref_mse = np.mean(np.square(power_mf_ref_db - ref_power_db))
    mf_peak_ref_idx = py.unravel_index(np.argmax(power_mf_ref_db), pattern_shape)
    mf_peak_ref = power_mf_ref_db[mf_peak_ref_idx]
    w_mf_ref_power = np.abs(w_mf_ref).sum()

    kw = dict(subplot_kw=dict(projection="polar"), layout="compressed")
    fig, ax = plt.subplots(figsize=(8, 8), **kw)
    plot = partial(py.plot_E_plane, phi_idx=0, ax=ax)

    ref_power_title = f"Reference Peak: {ref_peak_idx}° {ref_peak:.2f}dB, Power: {w_ref_power:.2f}, MSE: 0.0"
    lstsq_title = f"LstSq Peak: {lstsq_peak_idx}° {lstsq_peak:.2f}dB, Power: {w_lstsq_power:.2f}, MSE: {lstsq_mse:.3f}"
    mf_ref_title = f"MF Ref Peak: {mf_peak_ref_idx}° {mf_peak_ref:.2f}dB, Power: {w_mf_ref_power:.2f}, MSE: {mf_ref_mse:.3f}"
    mf_title = f"MF Peak: {mf_peak_idx}° {mf_peak:.2f}dB, Power: {w_mf_power:.2f}, MSE: {mf_mse:.3f}"

    plot(ref_power_db, fmt="r-", label=ref_power_title)
    plot(power_lstsq_db, fmt="g-", label=lstsq_title)
    plot(power_mf_db, fmt="b-", label=mf_title)
    plot(power_mf_ref_db, fmt="c-", label=mf_ref_title)

    ax.legend(loc="lower center")
    title = f"Dev Pattern ({theta=:.1f}, {phi=:.1f})"
    fig.suptitle(title, fontweight="bold")
    filename = "gd_dev.png"
    fig.savefig(filename, dpi=250, bbox_inches="tight")
    logger.info(f"Saved random pattern plot to {filename}")

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), layout="compressed")
    fig.suptitle(f"Sine Space ({theta=:.1f}, {phi=:.1f})", fontweight="bold")
    py.plot_sine_space(ref_power_db, ax=axs[0, 0], title=ref_power_title)
    py.plot_sine_space(power_lstsq_db, ax=axs[0, 1], title=lstsq_title)
    py.plot_sine_space(power_mf_ref_db, ax=axs[1, 0], title=mf_ref_title)
    py.plot_sine_space(power_mf_db, ax=axs[1, 1], title=mf_title)

    filename = "gd_dev_sine.png"
    fig.savefig(filename, dpi=250, bbox_inches="tight")
    logger.info(f"Saved sine space plot to {filename}")


if __name__ == "__main__":
    setup_logging()
    app()
