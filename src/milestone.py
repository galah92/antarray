import typing as tp
from functools import partial
from pathlib import Path
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from joblib import Memory
from matplotlib import colors
from matplotlib import pyplot as plt
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


def plot_power_db(power_db: jax.Array, title: str | None = None, fig=None) -> None:
    indices = argmax_nd(power_db)
    fig = physics.plot_pattern(power_db, clip_min_db=-40.0, fig=fig)
    if title:
        title = f"{title} | Peak {power_db[indices]:.3f} at {indices=}"
        fig.suptitle(title)


def mse_loss(pred_power: jax.Array, target_power: jax.Array) -> jax.Array:
    return jnp.mean((pred_power - target_power) ** 2)


LossFn = tp.Callable[[jax.Array, jax.Array], jax.Array]
LossScale = tp.Literal["linear", "db"]


@partial(jax.jit, static_argnames=("loss_fn", "loss_scale"))
def train_step(
    w: jnp.ndarray,
    aeps: jax.Array,
    target_power: jax.Array,
    lr: float,
    loss_fn: LossFn = mse_loss,
    loss_scale: LossScale = "db",
) -> tuple[jnp.ndarray, float]:
    if loss_scale == "db":
        target_power = to_db(target_power)

    def loss_wrapper(w: jax.Array) -> jax.Array:
        pred_pattern = jnp.einsum("xy,xytpz->tpz", w, aeps)
        pred_power = to_power(pred_pattern)
        if loss_scale == "db":
            pred_power = to_db(pred_power)
        return loss_fn(pred_power, target_power)

    loss, grads = jax.value_and_grad(loss_wrapper)(w)
    w = w - lr * grads
    # w = w / jnp.linalg.norm(w) * amplitude  # Re-normalize
    return w, loss


def optimize(
    w_init: jax.Array,
    aeps: jax.Array,
    target_power: jax.Array,
    loss_fn: LossFn = mse_loss,
    loss_scale: LossScale = "db",
    lr: float = 1e-5,
) -> jax.Array:
    w = w_init
    for step in range(100):
        w, loss = train_step(w, aeps, target_power, lr, loss_fn, loss_scale)
        if step % 10 == 0:
            print(f"step {step}, loss: {loss:.3f}")

    print(f"Weight norm: {jnp.linalg.norm(w):.3f}")

    pattern_opt = jnp.einsum("xy,xytpz->tpz", w, aeps)
    power_db_opt = to_power_db(pattern_opt)
    return power_db_opt


class OptParams(NamedTuple):
    taper: Literal["hamming", "uniform"]
    elev_deg: float
    azim_deg: float
    w: jax.Array
    aeps_env0: jax.Array
    aeps_env1: jax.Array
    power_env0: jax.Array
    power_env1: jax.Array


def init_params(
    env0_name: str = "no_env_rotated",
    env1_name: str = "Env1_rotated",
    taper: Literal["hamming", "uniform"] = "uniform",
    elev_deg: float = 0.0,
    azim_deg: float = 0.0,
) -> OptParams:
    cst_dir = root_dir / "cst"
    aeps_env0 = load_cst(cst_dir / env0_name)

    # Compute weights
    nx, ny = aeps_env0.shape[:2]
    if taper == "hamming":
        amplitude = hamming_taper(nx=nx, ny=ny)
    else:
        amplitude = uniform_taper(nx=nx, ny=ny)
    amplitude = amplitude / np.sqrt(np.sum(amplitude**2))  # Normalize power to 1

    phase = ideal_steering(nx=nx, ny=ny, elev_deg=elev_deg, azim_deg=azim_deg)
    w = amplitude * np.exp(1j * phase)

    power_env0 = to_power(np.einsum("xy,xytpz->tpz", w, aeps_env0))

    aeps_env1 = load_cst(cst_dir / env1_name)
    power_env1 = to_power(np.einsum("xy,xytpz->tpz", w, aeps_env1))

    return OptParams(
        taper=taper,
        elev_deg=elev_deg,
        azim_deg=azim_deg,
        w=jnp.asarray(w),
        aeps_env0=jnp.asarray(aeps_env0),
        aeps_env1=jnp.asarray(aeps_env1),
        power_env0=jnp.asarray(power_env0),
        power_env1=jnp.asarray(power_env1),
    )


class OptResults(NamedTuple):
    power_db_opt: jax.Array
    mse_env1: float
    mse_opt: float


# @memory.cache
def run_optimization(
    env0_name: str = "no_env_rotated",
    env1_name: str = "Env1_rotated",
    taper: Literal["hamming", "uniform"] = "uniform",
    elev_deg: float = 0.0,
    azim_deg: float = 0.0,
    loss_fn: LossFn = mse_loss,
    loss_scale: LossScale = "db",
    plot: bool = True,
):
    print(f"Running optimization for elev {elev_deg}°, azim {azim_deg}°, {taper} taper")
    params = init_params(
        taper=taper,
        elev_deg=elev_deg,
        azim_deg=azim_deg,
        env0_name=env0_name,
        env1_name=env1_name,
    )
    w, aeps_env1, power_env0 = params.w, params.aeps_env1, params.power_env0
    power_db_opt = optimize(w, aeps_env1, power_env0, loss_fn, loss_scale)

    power_db_env0, power_db_env1 = to_db(params.power_env0), to_db(params.power_env1)
    mse_env1 = np.mean((power_db_env1 - power_db_env0) ** 2).item()
    mse_opt = np.mean((power_db_opt - power_db_env0) ** 2).item()

    if plot:
        fig = plt.figure(figsize=(15, 12), layout="compressed")
        subfigs = fig.subfigures(3, 1)

        plot_power_db(power_db_env0, fig=subfigs[0], title="No Env")

        title = f"Env 1 | MSE {mse_env1:.3f}dB"
        plot_power_db(power_db_env1, fig=subfigs[1], title=title)

        title = f"Optimized | MSE {mse_opt:.3f}dB"
        plot_power_db(power_db_opt, fig=subfigs[2], title=title)

        steer_title = f"Steering Elev {int(elev_deg)}°, Azim {int(azim_deg)}°"
        title = f"{taper} taper | {steer_title}"
        fig.suptitle(title, fontsize=16)

        name = f"patterns_{taper}_elev_{int(elev_deg)}_azim_{int(azim_deg)}.png"
        fig.savefig(name, dpi=200)

    return OptResults(
        power_db_opt=power_db_opt,
        mse_env1=mse_env1,
        mse_opt=mse_opt,
    )


run_optimization(taper="hamming", elev_deg=20, azim_deg=-45)


def evaluate_grid(
    env0_name: str = "no_env_rotated",
    env1_name: str = "Env1_rotated",
    taper: Literal["hamming", "uniform"] = "hamming",
    loss_fn: LossFn = mse_loss,
    loss_scale: LossScale = "db",
) -> None:
    elevs = np.arange(0, 45, 5)
    azims = np.arange(0, 360, 10)
    results = {}

    opt = partial(
        run_optimization,
        env0_name=env0_name,
        env1_name=env1_name,
        taper=taper,
        loss_fn=loss_fn,
        loss_scale=loss_scale,
        plot=False,
    )

    for elev in elevs:
        for azim in azims:
            elev_deg, azim_deg = elev.item(), azim.item()
            results[(elev, azim)] = opt(elev_deg=elev_deg, azim_deg=azim_deg)

    mses_env1 = np.array([res.mse_env1 for res in results.values()]).reshape(
        len(elevs), len(azims)
    )
    mses_opt = np.array([res.mse_opt for res in results.values()]).reshape(
        len(elevs), len(azims)
    )

    mse_diff = mses_opt - mses_env1
    rmse_diff = np.sqrt(np.abs(mse_diff)) * np.sign(mse_diff)
    print(f"Mean RMSE Improvement: {rmse_diff.mean():.3f} dB")
    print(f"Std RMSE Improvement: {rmse_diff.std():.3f} dB")
    print(f"Max RMSE Improvement: {rmse_diff.max():.3f} dB")
    print(f"Min RMSE Improvement: {rmse_diff.min():.3f} dB")

    basic_plot = False
    if basic_plot:
        fig, ax = plt.subplots(figsize=(15, 5), layout="compressed")
        im2 = ax.imshow(rmse_diff, cmap="viridis", origin="lower")
        ax.set_xticks(np.arange(len(azims)), labels=azims)
        ax.set_yticks(np.arange(len(elevs)), labels=elevs)
        ax.set(title="RMAE Improvement", xlabel="Azim (deg)", ylabel="Elev (deg)")
        fig.colorbar(im2, ax=ax, label="RMAE (dB)")
        name = f"patterns_{taper}_rmse_grid.png"
        fig.savefig(name, dpi=200)

    polar_plot = True
    if polar_plot:
        kw = dict(layout="compressed", subplot_kw={"projection": "polar"})
        fig, ax = plt.subplots(figsize=(8, 8), **kw)
        axp = tp.cast(PolarAxes, ax)  # For type checker
        azim_grid, elev_grid = np.meshgrid(np.radians(azims), elevs)
        kw = dict(cmap="RdBu", norm=colors.CenteredNorm())  # Center colormap at 0
        c = axp.pcolormesh(azim_grid, elev_grid, rmse_diff, **kw)
        axp.set_theta_zero_location("N")
        axp.set_theta_direction(-1)
        axp.set_title(
            f"Power Pattern Diff RMAE (dB) ({env0_name} vs {env1_name}), {taper} taper"
        )
        fig.colorbar(c, ax=axp, label="RMAE (dB)")

        envs = f"{env0_name}_vs_{env1_name}"
        loss_fn_name = loss_fn.__name__.replace("_", " ")
        name = f"patterns_{taper}_{envs}_{loss_fn_name}_{loss_scale}.png"
        fig.savefig(name, dpi=200)


# evaluate_grid(env1_name="Env1_rotated", taper="hamming")
# evaluate_grid(env1_name="Env1_rotated", taper="uniform")
# evaluate_grid(env1_name="Env2_rotated", taper="hamming")
# evaluate_grid(env1_name="Env2_rotated", taper="uniform")
