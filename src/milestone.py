from pathlib import Path

import numpy as np
from joblib import Memory

import physics

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


cst_dir = root_dir / "cst" / "no_env_rotated"
aeps = load_cst(cst_dir)

# Compute weights
array_shape = aeps.shape[:2]
amplitude = 1.0 / np.sqrt(np.prod(array_shape))
phase = 0
w = np.full(array_shape, amplitude * np.exp(1j * phase))

x = aeps
x = np.einsum("xy,xytpz->tpz", w, x)  # Weighted sum over elements
x = np.abs(x) ** 2  # Convert to power (magnitude squared)
x = np.sum(x, axis=-1)  # Sum over polarization
x = 10 * np.log10(x)  # Convert to dB
print(f"{x.dtype=}, {x.shape=}, {x.min()=}, {x.max()=}")
fig = physics.plot_pattern(x, clip_min_db=-10.0)
fig.savefig("aep.png", dpi=200)
