"""
Phased Array Beamforming Optimization

This module optimizes phased array element weights to match radiation patterns
between different environments (e.g., with defects/concealments).

EVALUATION METRICS:
------------------
NMSE (Normalized Mean Square Error):
    Definition: NMSE = mean((pred - target)²) / mean(target²)

    Physical meaning: Ratio of prediction error power to target signal power

    Reported in two forms:
    1. Linear (percentage): NMSE × 100%
       - 2.85% means error power is 2.85% of signal power
       - Lower is better, 0% is perfect

    2. Decibels: NMSE_dB = 10 × log10(NMSE)
       - -15 dB NMSE ≈ 3.2% error
       - -20 dB NMSE ≈ 1.0% error
       - -30 dB NMSE ≈ 0.1% error
       - More negative is better, -∞ dB is perfect

    Interpretation of improvement:
    - Δ = -1 dB means ~20% reduction in error power
    - Δ = -3 dB means ~50% reduction in error power
    - Δ = -10 dB means ~90% reduction in error power

OPTIMIZATION:
------------
- Loss function: MSE in linear power space (loss_scale="linear")
- Linear space provides stable gradients and better convergence
- Do NOT use loss_scale="dB" - it causes divergence due to nonlinear landscape

GRID EVALUATION (VECTORIZED):
-----------------------------
Run grid evaluation using JAX vmap for parallel optimization.

Configuration options:
1. Environment variable: RUN_GRID=quick uv run src/milestone.py
2. Edit RUN_GRID variable in code (line ~726)

Valid values: "quick", "medium", "full", or False (case-insensitive)

Performance (with vectorization via JAX vmap):
- RUN_GRID = "quick"  → ~2-5 seconds (24 points, 1 config, batch_size=24)
  Good for: Testing changes, debugging
  Grid: 3 elevs × 8 azims on Env1_2_rotated with hamming taper
  Speedup: ~18x faster than sequential

- RUN_GRID = "medium" → ~5-10 minutes (360 points, 4 configs, batch_size=32)
  Good for: Evaluating specific environments/tapers
  Grid: 5 elevs × 18 azims on 2 envs × 2 tapers
  Speedup: ~10-15x faster than sequential

- RUN_GRID = "full"   → ~30-60 minutes (7,776 points, 24 configs, batch_size=32)
  Good for: Complete evaluation across all parameters
  Grid: 9 elevs × 36 azims on 6 envs × 2 tapers × 2 learning rates
  Speedup: ~10-15x faster than sequential (was 6-8 hours!)

All modes use:
- loss_scale="linear" exclusively (db removed as it causes divergence)
- evaluate_grid_vectorized() for batch parallel optimization
- JAX JIT compilation for maximum performance

Note: batch_size controls memory usage. Decrease if you run out of memory.
"""

import os
import typing as tp
from functools import partial
from pathlib import Path
from typing import Literal, NamedTuple, get_args

import jax
import jax.numpy as jnp
import numpy as np
from joblib import Memory
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.figure import SubFigure
from matplotlib.projections import PolarAxes

import physics
from tapering import hamming_taper, ideal_steering, uniform_taper

root_dir = Path(__file__).parent.parent
memory_dir = root_dir / ".joblib_cache"
memory = Memory(memory_dir, mmap_mode="r", verbose=0)  # Disk cache for CST files


@memory.cache
def load_cst_file(cst_path: Path) -> np.ndarray:
    """Load CST antenna pattern data from a file."""
    names = (
        "elev_deg",
        "azim_deg",
        "abs_grlz",
        "abs_cross",
        "phase_cross_deg",
        "abs_copol",
        "phase_copol_deg",
        "ax_ratio",
    )
    data = np.genfromtxt(cst_path, skip_header=2, dtype=np.float32, names=names)
    data = data[np.argsort(data, order=["elev_deg", "azim_deg"])]

    phase_cross = np.radians(data["phase_cross_deg"])
    phase_copol = np.radians(data["phase_copol_deg"])
    E_cross = np.sqrt(data["abs_cross"]) * np.exp(1j * phase_cross)
    E_copol = np.sqrt(data["abs_copol"]) * np.exp(1j * phase_copol)
    field = np.stack([E_cross, E_copol], axis=-1)

    field = field.reshape(181, 360, 2)  # (elev, azim, n_pol)
    field = field[:-1]  #  Remove last elev value
    return field


@memory.cache
def load_cst(cst_path: Path) -> np.ndarray:
    print(f"Loading antenna pattern from {cst_path}")
    data = {}
    for path in cst_path.iterdir():
        if "_RG.txt" not in path.name:
            continue  # Skip non-RG files

        i = int(path.stem.split("[")[1].split("]")[0]) - 1
        data[i] = load_cst_file(path)

    data = [v for _, v in sorted(data.items())]
    fields = np.stack(data, axis=0)  # (n_elements, elev, azim, n_pol)
    fields = fields.reshape(4, 4, *fields.shape[1:])  # (n_y, n_x, elev, azim, n_pol)
    fields = fields.transpose(1, 0, 2, 3, 4)  # (n_x, n_y, elev, azim, n_pol)
    return fields


def argmax_nd(x: jax.Array) -> tuple[int, ...]:
    indices = np.unravel_index(np.argmax(x), x.shape)
    indices = tuple(i.item() for i in indices)
    return indices


def to_power(x: jax.Array) -> jax.Array:
    x = jnp.abs(x) ** 2  # Convert to power (magnitude squared)
    x = jnp.sum(x, axis=-1)  # Sum over polarization
    return x


def to_db(x: jax.Array) -> jax.Array:
    return 10 * jnp.log10(x)  # Convert to dB


def to_power_db(x: jax.Array) -> jax.Array:
    return to_db(to_power(x))


def plot_power_db(
    power_db: jax.Array,
    title: str | None = None,
    fig: plt.Figure | SubFigure | None = None,
) -> None:
    indices = argmax_nd(power_db)
    fig = physics.plot_pattern(power_db, clip_min_db=-40.0, fig=fig)
    if title:
        title = f"{title} | Peak {power_db[indices]:.3f} at {indices=}"
        fig.suptitle(title)


def mse_loss(pred_power: jax.Array, target_power: jax.Array) -> jax.Array:
    return jnp.mean((pred_power - target_power) ** 2)


# Type aliases (must be defined before use in function signatures)
LossFn = tp.Callable[[jax.Array, jax.Array], jax.Array]
LossScale = Literal["linear", "db"]
Taper = Literal["hamming", "uniform"]
WeightInit = Literal["env0", "random", "uniform", "zeros"]
Environment = Literal[
    "no_env_rotated",
    "Env1_rotated",
    "Env2_rotated",
    "Env1_1_rotated",
    "Env1_2_rotated",
    "Env2_1_rotated",
    "Env2_2_rotated",
]
GridMode = Literal[False, "quick", "medium", "full"]


@partial(jax.jit, static_argnames=("loss_fn", "loss_scale"))
def train_step(
    w: jnp.ndarray,
    aeps: jax.Array,
    target_power: jax.Array,
    lr: float,
    loss_fn: LossFn = mse_loss,
    loss_scale: LossScale = "db",
) -> tuple[jnp.ndarray, float, float]:
    if loss_scale == "db":
        target_power = to_db(target_power)

    def loss_wrapper(w: jax.Array) -> jax.Array:
        pred_pattern = jnp.einsum("xy,xytpz->tpz", w, aeps)
        pred_power = to_power(pred_pattern)
        if loss_scale == "db":
            pred_power = to_db(pred_power)
        return loss_fn(pred_power, target_power)

    loss, grads = jax.value_and_grad(loss_wrapper)(w)
    grad_norm = jnp.linalg.norm(grads)
    w = w - lr * grads
    return w, loss, grad_norm


def optimize(
    w_init: jax.Array,
    aeps: jax.Array,
    target_power: jax.Array,
    loss_fn: LossFn = mse_loss,
    loss_scale: LossScale = "db",
    lr: float = 1e-5,
    verbose: bool = True,
) -> jax.Array:
    w = w_init

    # Get initial loss
    _, initial_loss, _ = train_step(w, aeps, target_power, 0.0, loss_fn, loss_scale)
    if verbose:
        print(f"Initial loss: {initial_loss:.3f}")

    for step in range(100):
        w, loss, grad_norm = train_step(w, aeps, target_power, lr, loss_fn, loss_scale)

        if verbose and step % 20 == 0:
            print(f"  step {step:3d}, loss: {loss:.3f}")

    if verbose:
        print(f"  step 100, loss: {loss:.3f} (final)")

    pattern_opt = jnp.einsum("xy,xytpz->tpz", w, aeps)
    power_db_opt = to_power_db(pattern_opt)
    return power_db_opt


@partial(jax.jit, static_argnames=("loss_fn", "loss_scale", "n_steps"))
def optimize_batch(
    w_init_batch: jax.Array,  # Shape: (n_batch, nx, ny)
    aeps: jax.Array,  # Shape: (nx, ny, elev, azim, n_pol) - shared across batch
    target_power_batch: jax.Array,  # Shape: (n_batch, elev, azim)
    loss_fn: LossFn = mse_loss,
    loss_scale: LossScale = "linear",
    lr: float = 1e-5,
    n_steps: int = 100,
) -> jax.Array:
    """Vectorized optimization for multiple steering angles in parallel.

    This function uses JAX vmap to optimize multiple weight configurations
    simultaneously, providing significant speedup over sequential optimization.

    Returns:
        power_db_opt_batch: Shape (n_batch, elev, azim) - optimized patterns in dB
    """

    def single_optimize(w_init, target_power):
        """Optimize a single configuration (will be vmapped)."""
        w = w_init

        for _ in range(n_steps):
            w, loss, _ = train_step(w, aeps, target_power, lr, loss_fn, loss_scale)

        pattern_opt = jnp.einsum("xy,xytpz->tpz", w, aeps)
        power_db_opt = to_power_db(pattern_opt)
        return power_db_opt

    # Vectorize over batch dimension
    optimize_vmap = jax.vmap(single_optimize, in_axes=(0, 0))
    power_db_opt_batch = optimize_vmap(w_init_batch, target_power_batch)

    return power_db_opt_batch


class OptParams(NamedTuple):
    taper: Taper
    elev_deg: float
    azim_deg: float
    w: jax.Array
    aeps_env0: jax.Array
    aeps_env1: jax.Array
    power_env0: jax.Array
    power_env1: jax.Array


def init_params(
    env0_name: Environment = "no_env_rotated",
    env1_name: Environment = "Env1_rotated",
    taper: Taper = "uniform",
    weight_init: WeightInit = "env0",
    elev_deg: float = 0.0,
    azim_deg: float = 0.0,
) -> OptParams:
    cst_dir = root_dir / "cst"
    aeps_env0 = load_cst(cst_dir / env0_name)

    # Compute weights based on initialization strategy
    nx, ny = aeps_env0.shape[:2]

    # ALWAYS compute the env0 reference weights (for target pattern)
    if taper == "hamming":
        amplitude = hamming_taper(nx=nx, ny=ny)
    else:
        amplitude = uniform_taper(nx=nx, ny=ny)
    amplitude = amplitude / np.sqrt(np.sum(amplitude**2))  # Normalize power to 1
    phase = ideal_steering(nx=nx, ny=ny, elev_deg=elev_deg, azim_deg=azim_deg)
    w_env0 = amplitude * np.exp(1j * phase)

    # Compute INITIAL weights based on strategy
    if weight_init == "env0":
        # Use env0 optimized weights (original behavior)
        w_init = w_env0
    elif weight_init == "random":
        # Random complex weights with normalized power
        key = jax.random.PRNGKey(42)
        w_real = jax.random.normal(key, shape=(nx, ny))
        w_imag = jax.random.normal(jax.random.split(key)[1], shape=(nx, ny))
        w_init = w_real + 1j * w_imag
        w_init = np.array(w_init)  # Convert to numpy for normalization
        w_init = w_init / np.sqrt(np.sum(np.abs(w_init) ** 2))  # Normalize power to 1
    elif weight_init == "uniform":
        # All ones (uniform amplitude, zero phase)
        w_init = np.ones((nx, ny), dtype=complex)
        w_init = w_init / np.sqrt(np.sum(np.abs(w_init) ** 2))  # Normalize power to 1
    elif weight_init == "zeros":
        # Small random initialization near zero
        key = jax.random.PRNGKey(42)
        w_real = jax.random.normal(key, shape=(nx, ny)) * 0.01
        w_imag = jax.random.normal(jax.random.split(key)[1], shape=(nx, ny)) * 0.01
        w_init = w_real + 1j * w_imag
        w_init = np.array(w_init)
        w_init = w_init / np.sqrt(np.sum(np.abs(w_init) ** 2))  # Normalize power to 1

    # Target pattern: ALWAYS use env0 reference weights in env0 environment
    power_env0 = to_power(np.einsum("xy,xytpz->tpz", w_env0, aeps_env0))

    # Initial pattern in env1: use INITIAL weights in env1 environment
    aeps_env1 = load_cst(cst_dir / env1_name)
    power_env1 = to_power(np.einsum("xy,xytpz->tpz", w_init, aeps_env1))

    return OptParams(
        taper=taper,
        elev_deg=elev_deg,
        azim_deg=azim_deg,
        w=jnp.asarray(w_init),
        aeps_env0=jnp.asarray(aeps_env0),
        aeps_env1=jnp.asarray(aeps_env1),
        power_env0=jnp.asarray(power_env0),
        power_env1=jnp.asarray(power_env1),
    )


class OptResults(NamedTuple):
    power_db_opt: jax.Array
    nmse_env1: float
    nmse_opt: float


# @memory.cache
def run_optimization(
    env0_name: Environment = "no_env_rotated",
    env1_name: Environment = "Env1_rotated",
    taper: Taper = "uniform",
    weight_init: WeightInit = "env0",
    elev_deg: float = 0.0,
    azim_deg: float = 0.0,
    loss_fn: LossFn = mse_loss,
    loss_scale: LossScale = "linear",
    lr: float = 1e-5,
    plot: bool = True,
    verbose: bool = True,
) -> OptResults:
    if verbose:
        print(
            f"Running optimization for elev {elev_deg}°, azim {azim_deg}°, {taper} taper, {weight_init} init"
        )
    params = init_params(
        taper=taper,
        weight_init=weight_init,
        elev_deg=elev_deg,
        azim_deg=azim_deg,
        env0_name=env0_name,
        env1_name=env1_name,
    )
    w, aeps_env1, power_env0 = params.w, params.aeps_env1, params.power_env0
    power_db_opt = optimize(
        w, aeps_env1, power_env0, loss_fn, loss_scale, lr, verbose=verbose
    )

    power_db_env0, power_db_env1 = to_db(params.power_env0), to_db(params.power_env1)

    # Convert optimized pattern to linear power for NMSE calculation
    power_opt = 10 ** (power_db_opt / 10)  # Convert from dB back to linear power
    power_env0 = params.power_env0
    power_env1 = params.power_env1

    # Standard NMSE: MSE normalized by mean power of target
    # NMSE = mean((pred - target)²) / mean(target²)
    mse_env1 = np.mean((power_env1 - power_env0) ** 2)
    mse_opt = np.mean((power_opt - power_env0) ** 2)

    target_power_mean = np.mean(power_env0**2)
    nmse_env1_linear = mse_env1 / target_power_mean
    nmse_opt_linear = mse_opt / target_power_mean

    # Convert to dB for reporting: NMSE_dB = 10 * log10(NMSE)
    # Interpretation:
    #   NMSE_dB = -10 dB means 10% normalized error
    #   NMSE_dB = -20 dB means 1% normalized error
    #   NMSE_dB = -30 dB means 0.1% normalized error
    #   Lower (more negative) is better
    nmse_env1_db = 10 * np.log10(nmse_env1_linear).item()
    nmse_opt_db = 10 * np.log10(nmse_opt_linear).item()
    nmse_improvement_db = nmse_opt_db - nmse_env1_db

    if verbose:
        print("\n=== Evaluation Metrics ===")
        print(
            f"NMSE (naive weights in Env1):     {nmse_env1_db:.3f} dB ({nmse_env1_linear * 100:.2f}%)"
        )
        print(
            f"NMSE (optimized weights in Env1): {nmse_opt_db:.3f} dB ({nmse_opt_linear * 100:.2f}%)"
        )
        print(f"NMSE Improvement (Δ):             {nmse_improvement_db:+.3f} dB")
        if nmse_improvement_db < 0:
            print(f"  → Optimized is {abs(nmse_improvement_db):.3f} dB BETTER")
        else:
            print(f"  → Optimized is {nmse_improvement_db:.3f} dB WORSE")

    if plot:
        fig = plt.figure(figsize=(15, 12), layout="compressed")
        subfigs: np.ndarray = fig.subfigures(3, 1)

        plot_power_db(power_db_env0, fig=subfigs[0], title="No Env")

        title = f"Env 1 | NMSE {nmse_env1_db:.3f} dB"
        plot_power_db(power_db_env1, fig=subfigs[1], title=title)

        title = f"Optimized | NMSE {nmse_opt_db:.3f} dB"
        plot_power_db(power_db_opt, fig=subfigs[2], title=title)

        steer_title = f"Steering Elev {int(elev_deg)}°, Azim {int(azim_deg)}°"
        title = f"{taper} taper | {steer_title}"
        fig.suptitle(title, fontsize=16)

        name = f"patterns_{taper}_elev_{int(elev_deg)}_azim_{int(azim_deg)}.png"
        fig.savefig(name, dpi=200)
        if verbose:
            print(f"Saved figure to {name}")

    return OptResults(
        power_db_opt=power_db_opt,
        nmse_env1=nmse_env1_db,
        nmse_opt=nmse_opt_db,
    )


# Single optimization example (comment out to skip)
# run_optimization(
#     env1_name="Env1_2_rotated",
#     taper="hamming",
#     elev_deg=10,
#     azim_deg=45,
#     loss_scale="linear",
#     lr=1e-5,
# )


def evaluate_grid_vectorized(
    env0_name: Environment = "no_env_rotated",
    env1_name: Environment = "Env1_rotated",
    taper: Taper = "hamming",
    weight_init: WeightInit = "env0",
    loss_fn: LossFn = mse_loss,
    loss_scale: LossScale = "linear",
    lr: float = 1e-5,
    elev_step: int = 5,
    azim_step: int = 10,
    batch_size: int = 32,
) -> None:
    """Vectorized grid evaluation using JAX vmap for parallel optimization.

    Processes multiple steering angles in parallel batches for significant speedup.

    Args:
        weight_init: Weight initialization strategy ("env0", "random", "uniform", "zeros")
        batch_size: Number of optimizations to run in parallel (default 32).
                   Increase for more speedup, decrease if running out of memory.
    """
    import time

    elevs = np.arange(0, 45, elev_step)
    azims = np.arange(0, 360, azim_step)
    n_points = len(elevs) * len(azims)

    print(f"\n{'=' * 60}")
    print("VECTORIZED GRID EVALUATION")
    print(f"{'=' * 60}")
    print(f"Grid: {len(elevs)} elevations × {len(azims)} azimuths = {n_points} points")
    print(f"Batch size: {batch_size} (processing {batch_size} points in parallel)")
    print(f"Environment: {env0_name} vs {env1_name}")
    print(
        f"Taper: {taper}, Weight Init: {weight_init}, Loss: {loss_scale}, LR: {lr:.1e}"
    )

    # Load antenna patterns once (shared across all steering angles)
    cst_dir = root_dir / "cst"
    aeps_env0 = load_cst(cst_dir / env0_name)
    aeps_env1 = load_cst(cst_dir / env1_name)
    nx, ny = aeps_env0.shape[:2]

    # Compute amplitude taper once if using env0 initialization
    if weight_init == "env0":
        if taper == "hamming":
            amplitude = hamming_taper(nx=nx, ny=ny)
        else:
            amplitude = uniform_taper(nx=nx, ny=ny)
        amplitude = amplitude / np.sqrt(np.sum(amplitude**2))
    else:
        amplitude = None  # Will be set per weight_init strategy

    # Create all steering angle pairs
    angle_pairs = [(elev, azim) for elev in elevs for azim in azims]
    n_batches = (len(angle_pairs) + batch_size - 1) // batch_size

    # Storage for results
    nmses_env1_all = []
    nmses_opt_all = []

    start_time = time.time()

    # Process in batches
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(angle_pairs))
        batch_angles = angle_pairs[batch_start:batch_end]

        # Prepare batch data
        w_init_batch = []
        power_env0_batch = []

        for idx, (elev_deg, azim_deg) in enumerate(batch_angles):
            # Generate initial weights based on strategy
            if weight_init == "env0":
                # Use env0 optimized weights with steering
                phase = ideal_steering(
                    nx=nx, ny=ny, elev_deg=elev_deg, azim_deg=azim_deg
                )
                w = amplitude * np.exp(1j * phase)
            elif weight_init == "random":
                # Random complex weights (use different seed per angle for diversity)
                seed = 42 + batch_start + idx
                key = jax.random.PRNGKey(seed)
                w_real = jax.random.normal(key, shape=(nx, ny))
                w_imag = jax.random.normal(jax.random.split(key)[1], shape=(nx, ny))
                w = w_real + 1j * w_imag
                w = np.array(w)
                w = w / np.sqrt(np.sum(np.abs(w) ** 2))
            elif weight_init == "uniform":
                # All ones (uniform amplitude, zero phase)
                w = np.ones((nx, ny), dtype=complex)
                w = w / np.sqrt(np.sum(np.abs(w) ** 2))
            elif weight_init == "zeros":
                # Small random initialization near zero
                seed = 42 + batch_start + idx
                key = jax.random.PRNGKey(seed)
                w_real = jax.random.normal(key, shape=(nx, ny)) * 0.01
                w_imag = (
                    jax.random.normal(jax.random.split(key)[1], shape=(nx, ny)) * 0.01
                )
                w = w_real + 1j * w_imag
                w = np.array(w)
                w = w / np.sqrt(np.sum(np.abs(w) ** 2))

            w_init_batch.append(w)

            # Compute target power using env0 weights (always use env0 for target)
            if weight_init == "env0":
                w_env0 = w  # Already computed above
            else:
                # For other inits, compute env0 pattern separately for target
                phase = ideal_steering(
                    nx=nx, ny=ny, elev_deg=elev_deg, azim_deg=azim_deg
                )
                if taper == "hamming":
                    amp = hamming_taper(nx=nx, ny=ny)
                else:
                    amp = uniform_taper(nx=nx, ny=ny)
                amp = amp / np.sqrt(np.sum(amp**2))
                w_env0 = amp * np.exp(1j * phase)

            power_env0 = to_power(np.einsum("xy,xytpz->tpz", w_env0, aeps_env0))
            power_env0_batch.append(power_env0)

        # Convert to JAX arrays
        w_init_batch = jnp.array(np.stack(w_init_batch))
        power_env0_batch = jnp.array(np.stack(power_env0_batch))
        aeps_env1_jax = jnp.asarray(aeps_env1)

        # Run batch optimization (this is where the magic happens!)
        power_db_opt_batch = optimize_batch(
            w_init_batch,
            aeps_env1_jax,
            power_env0_batch,
            loss_fn,
            loss_scale,
            lr,
            n_steps=100,
        )

        # Compute NMSE for batch
        power_opt_batch = 10 ** (power_db_opt_batch / 10)  # Convert from dB

        for i, (elev_deg, azim_deg) in enumerate(batch_angles):
            # Recompute env0 pattern for this specific angle (for NMSE target)
            phase = ideal_steering(nx=nx, ny=ny, elev_deg=elev_deg, azim_deg=azim_deg)
            if taper == "hamming":
                amp = hamming_taper(nx=nx, ny=ny)
            else:
                amp = uniform_taper(nx=nx, ny=ny)
            amp = amp / np.sqrt(np.sum(amp**2))
            w_env0 = amp * np.exp(1j * phase)
            power_env0 = to_power(np.einsum("xy,xytpz->tpz", w_env0, aeps_env0))
            power_env1 = to_power(np.einsum("xy,xytpz->tpz", w_env0, aeps_env1))

            # Compute NMSE
            mse_env1 = np.mean((power_env1 - power_env0) ** 2)
            mse_opt = np.mean((power_opt_batch[i] - power_env0) ** 2)
            target_power_mean = np.mean(power_env0**2)

            nmse_env1_db = 10 * np.log10(mse_env1 / target_power_mean)
            nmse_opt_db = 10 * np.log10(mse_opt / target_power_mean)

            nmses_env1_all.append(nmse_env1_db)
            nmses_opt_all.append(nmse_opt_db)

        # Progress update
        completed = batch_end
        elapsed = time.time() - start_time
        eta = (elapsed / completed) * (n_points - completed) if completed > 0 else 0
        print(
            f"[Batch {batch_idx + 1}/{n_batches}] Processed {completed}/{n_points} points | "
            f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s"
        )

    # Convert to arrays and reshape
    nmses_env1 = np.array(nmses_env1_all).reshape(len(elevs), len(azims))
    nmses_opt = np.array(nmses_opt_all).reshape(len(elevs), len(azims))
    nmse_diff = nmses_opt - nmses_env1

    # Print summary with enhanced metrics
    print(f"\n{'=' * 60}")
    print("GRID EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Environment: {env0_name} vs {env1_name}")
    print(
        f"Taper: {taper}, Weight Init: {weight_init}, Loss: {loss_scale}, LR: {lr:.1e}"
    )
    print(
        f"Grid: {len(elevs)} elevs × {len(azims)} azims = {len(elevs) * len(azims)} points"
    )

    # Basic statistics
    print("\n[1] NMSE Improvement Distribution:")
    print(f"  Mean:   {nmse_diff.mean():+.3f} dB")
    print(f"  Median: {np.median(nmse_diff):+.3f} dB")
    print(f"  Std:    {nmse_diff.std():.3f} dB")
    print(f"  Min:    {nmse_diff.min():+.3f} dB (best improvement)")
    print(f"  Max:    {nmse_diff.max():+.3f} dB (worst degradation)")

    # Coverage metrics
    improved = nmse_diff < 0
    degraded = nmse_diff > 0
    strong_improvement = nmse_diff < -1.0
    success_rate = 100 * improved.sum() / nmse_diff.size
    strong_rate = 100 * strong_improvement.sum() / nmse_diff.size
    degradation_rate = 100 * degraded.sum() / nmse_diff.size

    print("\n[2] Coverage Metrics:")
    print(
        f"  Success Rate:           {success_rate:.1f}% (angles with improvement, Δ < 0)"
    )
    print(f"  Strong Improvement:     {strong_rate:.1f}% (improvement > 1 dB, Δ < -1)")
    print(
        f"  Degradation Rate:       {degradation_rate:.1f}% (optimization made worse, Δ > 0)"
    )

    # Reliability metrics
    percentiles: np.ndarray = np.percentile(nmse_diff, [5, 25, 75, 95])
    print("\n[3] Reliability Metrics:")
    print(f"  P5  (best 5%):          {percentiles[0]:+.3f} dB")
    print(f"  P25 (best quartile):    {percentiles[1]:+.3f} dB")
    print(f"  P75 (worst quartile):   {percentiles[2]:+.3f} dB")
    print(f"  P95 (worst 5%):         {percentiles[3]:+.3f} dB")

    # Practical impact
    if improved.any():
        mean_improvement_only = nmse_diff[improved].mean()
        print("\n[4] Practical Impact:")
        print(
            f"  Avg Improvement (improved angles only): {mean_improvement_only:+.3f} dB"
        )
        print(f"  Worst-case guarantee (95th percentile): {percentiles[3]:+.3f} dB")

    # Overall assessment
    print("\n[5] Overall Assessment:")
    if success_rate >= 80 and nmse_diff.mean() < -0.5:
        assessment = "EXCELLENT - Optimization reliably improves pattern matching"
    elif success_rate >= 60 and nmse_diff.mean() < -0.2:
        assessment = "GOOD - Optimization helps in most cases"
    elif success_rate >= 50:
        assessment = "MODERATE - Optimization provides modest benefit"
    else:
        assessment = "POOR - Optimization not consistently beneficial"
    print(f"  {assessment}")

    print(f"{'=' * 60}\n")

    # Generate polar plot
    polar_plot = True
    if polar_plot:
        fig, ax = plt.subplots(
            figsize=(8, 8), layout="compressed", subplot_kw={"projection": "polar"}
        )
        axp = tp.cast(PolarAxes, ax)
        azim_grid, elev_grid = np.meshgrid(np.radians(azims), elevs)
        # Use RdBu_r (reversed) so Blue=negative=improvement, Red=positive=degradation
        c = axp.pcolormesh(
            azim_grid, elev_grid, nmse_diff, cmap="RdBu_r", norm=colors.CenteredNorm()
        )
        axp.set_theta_zero_location("N")
        axp.set_theta_direction(-1)
        cbar = fig.colorbar(c, ax=axp, label="ΔNMSE (dB)")
        # Add interpretation to colorbar
        cbar.ax.text(
            0.5,
            -0.1,
            "Blue=Improvement  Red=Degradation",
            transform=cbar.ax.transAxes,
            ha="center",
            fontsize=9,
        )

        envs = f"{env0_name} vs {env1_name}"
        loss_fn_name = getattr(loss_fn, "__name__", "loss").replace("_", " ")
        title = f"ΔNMSE | {taper} taper | {weight_init} init | {envs} | {loss_fn_name} ({loss_scale}) | {lr=:.1e}"
        fig.suptitle(title)

        envs = envs.replace(" ", "_")
        loss_fn_name = getattr(loss_fn, "__name__", "loss")
        name = f"patterns_nmse_{taper}_{weight_init}_init_{envs}_{loss_fn_name}_{loss_scale}_lr_{lr:.1e}.png"
        fig.savefig(name, dpi=200)
        print(f"Saved plot to {name}")


def evaluate_grid(
    env0_name: Environment = "no_env_rotated",
    env1_name: Environment = "Env1_rotated",
    taper: Taper = "hamming",
    loss_fn: LossFn = mse_loss,
    loss_scale: LossScale = "linear",
    lr: float = 1e-5,
    elev_step: int = 5,
    azim_step: int = 10,
) -> None:
    """Evaluate optimization over a grid of steering angles (sequential version).

    Note: Use evaluate_grid_vectorized() for much better performance!

    Args:
        elev_step: Step size for elevation grid (default 5°). Use larger values for faster testing.
        azim_step: Step size for azimuth grid (default 10°). Use larger values for faster testing.
    """
    elevs = np.arange(0, 45, elev_step)
    azims = np.arange(0, 360, azim_step)
    n_points = len(elevs) * len(azims)
    print(
        f"\nEvaluating grid: {len(elevs)} elevations × {len(azims)} azimuths = {n_points} points"
    )
    print(
        "WARNING: Using sequential version. For better performance, use evaluate_grid_vectorized()"
    )
    results = {}

    opt = partial(
        run_optimization,
        env0_name=env0_name,
        env1_name=env1_name,
        taper=taper,
        loss_fn=loss_fn,
        loss_scale=loss_scale,
        lr=lr,
        plot=False,
        verbose=False,  # Suppress per-optimization output
    )

    # Run optimizations with progress tracking
    total = len(elevs) * len(azims)
    completed = 0
    import time

    start_time = time.time()

    for elev in elevs:
        for azim in azims:
            elev_deg, azim_deg = elev.item(), azim.item()
            completed += 1
            elapsed = time.time() - start_time
            eta = (elapsed / completed) * (total - completed) if completed > 0 else 0
            print(
                f"[{completed}/{total}] elev={elev_deg:2.0f}°, azim={azim_deg:3.0f}° | "
                f"Elapsed: {elapsed / 60:.1f}m, ETA: {eta / 60:.1f}m"
            )
            results[(elev, azim)] = opt(elev_deg=elev_deg, azim_deg=azim_deg)

    nmses_env1 = np.array([res.nmse_env1 for res in results.values()]).reshape(
        len(elevs), len(azims)
    )
    nmses_opt = np.array([res.nmse_opt for res in results.values()]).reshape(
        len(elevs), len(azims)
    )

    nmse_diff = nmses_opt - nmses_env1

    # Print summary with enhanced metrics (same as vectorized version)
    print(f"\n{'=' * 60}")
    print("GRID EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Environment: {env0_name} vs {env1_name}")
    print(f"Taper: {taper}, Loss: {loss_scale}, LR: {lr:.1e}")
    print(
        f"Grid: {len(elevs)} elevs × {len(azims)} azims = {len(elevs) * len(azims)} points"
    )

    # Basic statistics
    print("\n[1] NMSE Improvement Distribution:")
    print(f"  Mean:   {nmse_diff.mean():+.3f} dB")
    print(f"  Median: {np.median(nmse_diff):+.3f} dB")
    print(f"  Std:    {nmse_diff.std():.3f} dB")
    print(f"  Min:    {nmse_diff.min():+.3f} dB (best improvement)")
    print(f"  Max:    {nmse_diff.max():+.3f} dB (worst degradation)")

    # Coverage metrics
    improved = nmse_diff < 0
    degraded = nmse_diff > 0
    strong_improvement = nmse_diff < -1.0
    success_rate = 100 * improved.sum() / nmse_diff.size
    strong_rate = 100 * strong_improvement.sum() / nmse_diff.size
    degradation_rate = 100 * degraded.sum() / nmse_diff.size

    print("\n[2] Coverage Metrics:")
    print(
        f"  Success Rate:           {success_rate:.1f}% (angles with improvement, Δ < 0)"
    )
    print(f"  Strong Improvement:     {strong_rate:.1f}% (improvement > 1 dB, Δ < -1)")
    print(
        f"  Degradation Rate:       {degradation_rate:.1f}% (optimization made worse, Δ > 0)"
    )

    # Reliability metrics
    percentiles: np.ndarray = np.percentile(nmse_diff, [5, 25, 75, 95])
    print("\n[3] Reliability Metrics:")
    print(f"  P5  (best 5%):          {percentiles[0]:+.3f} dB")
    print(f"  P25 (best quartile):    {percentiles[1]:+.3f} dB")
    print(f"  P75 (worst quartile):   {percentiles[2]:+.3f} dB")
    print(f"  P95 (worst 5%):         {percentiles[3]:+.3f} dB")

    # Practical impact
    if improved.any():
        mean_improvement_only = nmse_diff[improved].mean()
        print("\n[4] Practical Impact:")
        print(
            f"  Avg Improvement (improved angles only): {mean_improvement_only:+.3f} dB"
        )
        print(f"  Worst-case guarantee (95th percentile): {percentiles[3]:+.3f} dB")

    # Overall assessment
    print("\n[5] Overall Assessment:")
    if success_rate >= 80 and nmse_diff.mean() < -0.5:
        assessment = "EXCELLENT - Optimization reliably improves pattern matching"
    elif success_rate >= 60 and nmse_diff.mean() < -0.2:
        assessment = "GOOD - Optimization helps in most cases"
    elif success_rate >= 50:
        assessment = "MODERATE - Optimization provides modest benefit"
    else:
        assessment = "POOR - Optimization not consistently beneficial"
    print(f"  {assessment}")

    print(f"{'=' * 60}\n")

    basic_plot = False
    if basic_plot:
        fig, ax = plt.subplots(figsize=(15, 5), layout="compressed")
        im2 = ax.imshow(nmse_diff, cmap="viridis", origin="lower")
        ax.set_xticks(np.arange(len(azims)), labels=azims)
        ax.set_yticks(np.arange(len(elevs)), labels=elevs)
        ax.set(title="NMSE Improvement", xlabel="Azim (deg)", ylabel="Elev (deg)")
        fig.colorbar(im2, ax=ax, label="NMSE (dB)")
        name = f"patterns_{taper}_nmse_grid.png"
        fig.savefig(name, dpi=200)
        print(f"Saved plot to {name}")

    polar_plot = True
    if polar_plot:
        fig, ax = plt.subplots(
            figsize=(8, 8), layout="compressed", subplot_kw={"projection": "polar"}
        )
        axp = tp.cast(PolarAxes, ax)  # For type checker
        azim_grid, elev_grid = np.meshgrid(np.radians(azims), elevs)
        # Use RdBu_r (reversed) so Blue=negative=improvement, Red=positive=degradation
        c = axp.pcolormesh(
            azim_grid, elev_grid, nmse_diff, cmap="RdBu_r", norm=colors.CenteredNorm()
        )
        axp.set_theta_zero_location("N")
        axp.set_theta_direction(-1)
        cbar = fig.colorbar(c, ax=axp, label="ΔNMSE (dB)")
        # Add interpretation to colorbar
        cbar.ax.text(
            0.5,
            -0.1,
            "Blue=Improvement  Red=Degradation",
            transform=cbar.ax.transAxes,
            ha="center",
            fontsize=9,
        )

        envs = f"{env0_name} vs {env1_name}"
        loss_fn_name = getattr(loss_fn, "__name__", "loss").replace("_", " ")

        title = f"ΔNMSE | {taper} taper | {envs} | {loss_fn_name} ({loss_scale}) | {lr=:.1e}"
        fig.suptitle(title)

        envs = envs.replace(" ", "_")
        loss_fn_name = getattr(loss_fn, "__name__", "loss")
        name = (
            f"patterns_nmse_{taper}_{envs}_{loss_fn_name}_{loss_scale}_lr_{lr:.1e}.png"
        )
        fig.savefig(name, dpi=200)
        print(f"Saved plot to {name}")


# Grid evaluation configuration
# Set to "quick", "medium", or "full" to run grid evaluation
# Can be overridden by environment variable: RUN_GRID=quick uv run src/milestone.py
_run_grid_env = os.getenv("RUN_GRID", "").lower()
if _run_grid_env in ("quick", "medium", "full"):
    RUN_GRID: GridMode = _run_grid_env  # type: ignore
elif _run_grid_env in ("false", "0", ""):
    RUN_GRID: GridMode = False
else:
    raise ValueError(
        f"Invalid RUN_GRID value: {_run_grid_env!r}. "
        f"Must be one of: False, 'quick', 'medium', 'full'"
    )

# Default value if no environment variable is set
if _run_grid_env == "":
    RUN_GRID = False  # Options: False, "quick", "medium", "full"

if RUN_GRID == "quick":
    # Quick test: ~10-20 seconds with vectorization (was ~5-10 min sequential)
    # 3 elevs × 8 azims = 24 points per config
    # 1 env × 1 taper × 1 lr = 1 config
    # Total: 24 optimizations
    print("\n" + "=" * 60)
    print("QUICK TEST MODE (coarse grid, single environment)")
    print("=" * 60)
    evaluate_grid_vectorized(
        env1_name="Env1_2_rotated",
        taper="hamming",
        loss_scale="linear",
        lr=1e-5,
        elev_step=15,  # 0, 15, 30 → 3 points
        azim_step=45,  # 0, 45, 90, 135, 180, 225, 270, 315 → 8 points
        batch_size=24,  # Process all 24 points in one batch
    )

elif RUN_GRID == "medium":
    # Medium test: ~5-10 minutes with vectorization (was ~1-2 hours sequential)
    # 5 elevs × 18 azims = 90 points per config
    # 2 envs × 2 tapers = 4 configs
    # Total: 360 optimizations
    print("\n" + "=" * 60)
    print("MEDIUM TEST MODE (medium grid, multiple configs)")
    print("=" * 60)
    envs: list[Environment] = ["Env1_2_rotated", "Env2_1_rotated"]
    for env in envs:
        for taper in get_args(Taper):
            evaluate_grid_vectorized(
                env1_name=env,
                taper=taper,
                loss_scale="linear",
                lr=1e-5,
                elev_step=10,  # 0, 10, 20, 30, 40 → 5 points
                azim_step=20,  # 18 points
                batch_size=32,
            )

elif RUN_GRID == "full":
    # Full evaluation: ~30-60 minutes with vectorization (was ~6-8 hours sequential)
    # 9 elevs × 36 azims = 324 points per config
    # 6 envs × 2 tapers × 2 lrs = 24 configs
    # Total: 7,776 optimizations (reduced from 15,552 by removing db loss)
    print("\n" + "=" * 60)
    print("FULL EVALUATION MODE (fine grid, all environments)")
    print("=" * 60)
    envs: list[Environment] = [
        "Env1_rotated",
        "Env2_rotated",
        "Env1_1_rotated",
        "Env1_2_rotated",
        "Env2_1_rotated",
        "Env2_2_rotated",
    ]
    for env in envs:
        for taper in get_args(Taper):
            for lr in [1e-5, 5e-6]:
                evaluate_grid_vectorized(
                    env1_name=env,
                    taper=taper,
                    loss_fn=mse_loss,
                    loss_scale="linear",  # Only use linear (db causes divergence)
                    lr=lr,
                    elev_step=5,  # Full resolution
                    azim_step=10,  # Full resolution
                    batch_size=32,
                )
