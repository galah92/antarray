from pathlib import Path
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from joblib import Memory
from matplotlib import pyplot as plt

import physics
from tapering import hamming_taper, ideal_steering, uniform_taper  # noqa: F401

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


def to_power_db(x: jax.Array) -> jax.Array:
    x = jnp.abs(x) ** 2  # Convert to power (magnitude squared)
    x = jnp.sum(x, axis=-1)  # Sum over polarization
    x = 10 * jnp.log10(x)  # Convert to dB
    return x


def plot_power_db(power_db: jax.Array, title: str | None = None, fig=None) -> None:
    indices = argmax_nd(power_db)
    fig = physics.plot_pattern(power_db, clip_min_db=-40.0, fig=fig)
    if title:
        title = f"{title} | Peak {power_db[indices]:.3f} at {indices}"
        fig.suptitle(title)


@jax.jit
def train_step(
    w: jnp.ndarray,
    aeps: jax.Array,
    target_power_db: jax.Array,
    lr: float,
) -> tuple[jnp.ndarray, float]:
    def loss_fn(w: jax.Array) -> jax.Array:
        pattern = jnp.einsum("xy,xytpz->tpz", w, aeps)
        power_db = to_power_db(pattern)

        mse = jnp.mean((power_db - target_power_db) ** 2)
        return mse

    loss, grads = jax.value_and_grad(loss_fn)(w)
    w = w - lr * grads
    # w = w / jnp.linalg.norm(w) * amplitude  # Re-normalize
    return w, loss


def optimize(
    w_init: jax.Array,
    aeps: jax.Array,
    target_power_db: jax.Array,
    lr: float = 1e-5,
) -> jax.Array:
    w = w_init
    for step in range(100):
        w, loss = train_step(w, aeps, target_power_db, lr)
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
    power_db_env0: jax.Array
    power_db_env1: jax.Array


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

    power_db_env0 = to_power_db(np.einsum("xy,xytpz->tpz", w, aeps_env0))

    aeps_env1 = load_cst(cst_dir / env1_name)
    power_db_env1 = to_power_db(np.einsum("xy,xytpz->tpz", w, aeps_env1))

    return OptParams(
        taper=taper,
        elev_deg=elev_deg,
        azim_deg=azim_deg,
        w=jnp.asarray(w),
        aeps_env0=jnp.asarray(aeps_env0),
        power_db_env0=jnp.asarray(power_db_env0),
        power_db_env1=jnp.asarray(power_db_env1),
    )


def run(
    taper: Literal["hamming", "uniform"] = "uniform",
    elev_deg: float = 0.0,
    azim_deg: float = 0.0,
):
    params = init_params(taper=taper, elev_deg=elev_deg, azim_deg=azim_deg)
    power_db_opt = optimize(params.w, params.aeps_env0, params.power_db_env1)

    fig = plt.figure(figsize=(15, 12), layout="compressed")
    subfigs = fig.subfigures(3, 1)

    plot_power_db(params.power_db_env0, fig=subfigs[0], title="No Env")

    mse_env0 = np.mean((params.power_db_env0 - params.power_db_env1) ** 2)
    title = f"Env 1 | MSE {mse_env0:.3f}dB"
    plot_power_db(params.power_db_env1, fig=subfigs[1], title=title)

    mse_opt = jnp.mean((power_db_opt - params.power_db_env1) ** 2)
    title = f"Optimized | MSE {mse_opt:.3f}dB"
    plot_power_db(power_db_opt, fig=subfigs[2], title=title)

    name = f"patterns_{taper}_elev_{int(elev_deg)}_azim_{int(azim_deg)}.png"
    fig.savefig(name, dpi=200)


run(azim_deg=1.0, elev_deg=1.0)
