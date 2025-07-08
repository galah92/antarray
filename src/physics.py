import logging
import typing
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Literal

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.typing import ArrayLike
from matplotlib.colorizer import ColorizingArtist
from matplotlib.figure import SubFigure
from matplotlib.image import AxesImage
from matplotlib.projections import PolarAxes
from mpl_toolkits.mplot3d import Axes3D

from utils import setup_logging

logger = logging.getLogger(__name__)

root_dir = Path(__file__).parent.parent

DEFAULT_SIM_DIR = root_dir / "openems" / "sim" / "antenna_array"
DEFAULT_SINGLE_ANT_FILENAME = "ff_1x1_60x60_2450_steer_t0_p0.h5"
DEFAULT_SIM_PATH = DEFAULT_SIM_DIR / DEFAULT_SINGLE_ANT_FILENAME


class ArrayConfig:
    """Configuration for antenna array parameters and simulation settings."""

    array_size: tuple[int, int] = (16, 16)
    spacing_mm: tuple[float, float] = (60.0, 60.0)
    freq_hz: float = 2.45e9
    theta_rad: np.ndarray = np.radians(np.arange(180))
    phi_rad: np.ndarray = np.radians(np.arange(360))

    def __init__(
        self,
        array_size: tuple[int, int] = (16, 16),
        spacing_mm: tuple[float, float] = (60.0, 60.0),
        freq_hz: float = 2.45e9,
        theta_rad: np.ndarray | None = None,
        phi_rad: np.ndarray | None = None,
    ):
        self.array_size = array_size
        self.spacing_mm = spacing_mm
        self.freq_hz = freq_hz
        if theta_rad is None:
            theta_rad = np.radians(np.arange(180))
        if phi_rad is None:
            phi_rad = np.radians(np.arange(360))
        self.theta_rad = np.asarray(theta_rad, dtype=np.float32)
        self.phi_rad = np.asarray(phi_rad, dtype=np.float32)


@dataclass(frozen=True)
class OpenEMSData:
    theta_rad: jax.Array
    phi_rad: jax.Array
    r: jax.Array
    Dmax: float
    freq_hz: jax.Array
    E_field: jax.Array
    power_density: jax.Array


@lru_cache
def load_openems_nf2ff(nf2ff_path: Path):
    logger.info(f"Loading antenna pattern from {nf2ff_path}")
    with h5py.File(nf2ff_path, "r") as h5:
        mesh = h5["Mesh"]
        theta_rad, phi_rad, r = mesh["theta"][:], mesh["phi"][:], mesh["r"][:]

        Dmax = h5["nf2ff"].attrs["Dmax"]
        freq_hz = h5["nf2ff"].attrs["Frequency"]

        E_shape = freq_hz.size, phi_rad.size, theta_rad.size
        E_theta = np.empty(E_shape, dtype=complex)
        E_phi = np.empty(E_shape, dtype=complex)

        for freq in range(freq_hz.size):
            E_theta.real[freq] = h5[f"/nf2ff/E_theta/FD/f{freq}_real"][:]
            E_theta.imag[freq] = h5[f"/nf2ff/E_theta/FD/f{freq}_imag"][:]
            E_phi.real[freq] = h5[f"/nf2ff/E_phi/FD/f{freq}_real"][:]
            E_phi.imag[freq] = h5[f"/nf2ff/E_phi/FD/f{freq}_imag"][:]

        # Transpose to (freq, theta, phi)
        E_theta, E_phi = E_theta.transpose(0, 2, 1), E_phi.transpose(0, 2, 1)

        if freq_hz.size == 1:  # Squeeze the frequency dimension
            E_theta, E_phi = np.squeeze(E_theta), np.squeeze(E_phi)
            freq_hz = np.squeeze(freq_hz)

        E_field = np.stack([E_theta, E_phi], axis=-1)  # (freq, theta, phi, 2)
        power_density = np.sum(np.abs(E_field) ** 2, axis=-1)  # (freq, theta, phi)

    return OpenEMSData(theta_rad, phi_rad, r, Dmax, freq_hz, E_field, power_density)


class CstData(typing.NamedTuple):
    config: ArrayConfig
    element_fields: jax.Array


def load_cst_file(cst_path: Path) -> np.ndarray:
    """Load CST antenna pattern data from a file."""
    names = (
        "theta_deg",
        "phi_deg",
        "abs_dir",
        "abs_cross",
        "phase_cross_deg",
        "abs_copol",
        "phase_copol_deg",
        "ax_ratio",
    )
    data = np.genfromtxt(cst_path, skip_header=2, dtype=np.float32, names=names)

    phase_copol = np.radians(data["phase_copol_deg"])
    phase_cross = np.radians(data["phase_cross_deg"])
    E_copol = np.sqrt(data["abs_copol"]) * np.exp(1j * phase_copol)
    E_cross = np.sqrt(data["abs_cross"]) * np.exp(1j * phase_cross)
    field = np.stack([E_copol, E_cross], axis=-1)

    field = field.reshape(360, 181, 2)  #  (phi, theta, n_pol)
    field = field.transpose((1, 0, 2))  #  (theta, phi, n_pol)
    field = field[:-1]  #  Remove last theta value
    return field


@lru_cache
def load_cst(cst_path: Path) -> CstData:
    logger.info(f"Loading antenna pattern from {cst_path}")
    data = {}
    for path in cst_path.iterdir():
        i = int(path.stem.split()[-1][1:-1]) - 1
        data[i] = load_cst_file(path)

    data = [v for _, v in sorted(data.items())]
    fields = np.stack(data, axis=0)
    fields = fields.reshape(4, 4, *fields.shape[1:])

    config = ArrayConfig(array_size=(4, 4), spacing_mm=(75, 75), freq_hz=2.4e9)
    return CstData(config=config, element_fields=fields)


def get_wavenumber(freq_hz: float | None = None) -> float:
    """Calculate the wavenumber for a given frequency in Hz."""
    if freq_hz is None:
        freq_hz = ArrayConfig.freq_hz

    c = 299792458  # Speed of light in m/s
    wavelength = c / freq_hz  # Wavelength in meters
    k = 2 * np.pi / wavelength  # Wavenumber in radians/meter
    return k


@lru_cache
def get_element_positions(
    array_size: tuple[int, int] | None = None,
    spacing_mm: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate element positions using config or explicit parameters."""
    if array_size is None:
        array_size = ArrayConfig.array_size
    if spacing_mm is None:
        spacing_mm = ArrayConfig.spacing_mm

    xn, yn = array_size
    dx_mm, dy_mm = spacing_mm
    # Calculate the x and y positions of the elements in the array in meters, centered around (0, 0)
    x_positions = (np.arange(xn) - (xn - 1) / 2) * dx_mm / 1000
    y_positions = (np.arange(yn) - (yn - 1) / 2) * dy_mm / 1000
    return x_positions, y_positions


@lru_cache
def compute_spatial_phase_coeffs(config: ArrayConfig) -> tuple[np.ndarray, np.ndarray]:
    """Compute spatial phase coefficients (kx, ky) for antenna array elements."""
    k = get_wavenumber(config.freq_hz)
    x_pos, y_pos = get_element_positions(config.array_size, config.spacing_mm)
    kx, ky = k * x_pos, k * y_pos
    return kx, ky


@jax.jit
def calculate_weights(
    kx: ArrayLike,
    ky: ArrayLike,
    steering_angles: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Calculates element weights for given steering angles."""
    steering_angles = jnp.atleast_2d(steering_angles)
    theta, phi = steering_angles[:, 0], steering_angles[:, 1]

    sin_theta = jnp.sin(theta)
    x_phase = jnp.outer(kx, sin_theta * jnp.cos(phi))  # (n_x, n_beams)
    y_phase = jnp.outer(ky, sin_theta * jnp.sin(phi))  # (n_y, n_beams)

    phase_shifts = x_phase[:, None, :] + y_phase[None, :, :]  # (n_x, n_y, n_beams)

    weights = jnp.exp(-1j * phase_shifts)  # (n_x, n_y, n_beams)
    weights = jnp.sum(weights, axis=-1)  # (n_x, n_y), assume no tapering
    weights = weights / np.sqrt(np.prod(weights.shape))  # Normalize weights

    return weights, phase_shifts


def compute_element_fields(
    element_patterns: np.ndarray,
    config: ArrayConfig,
) -> np.ndarray:
    """Create element field basis with geometric phase factors."""
    kx, ky = compute_spatial_phase_coeffs(config)

    sin_theta = np.sin(config.theta_rad)
    sin_phi, cos_phi = np.sin(config.phi_rad), np.cos(config.phi_rad)
    phase_x = np.einsum("t,p,x->tpx", sin_theta, cos_phi, kx)  # (n_theta, n_phi, n_x)
    phase_y = np.einsum("t,p,y->tpy", sin_theta, sin_phi, ky)  # (n_theta, n_phi, n_y)

    geo_phase = phase_x[..., None] + phase_y[..., None, :]  # (n_theta, n_phi, n_x, n_y)
    geo_factor = np.exp(1j * geo_phase)  # (n_theta, n_phi, n_x, n_y)

    element_fields = np.einsum("xytpz,tpxy->xytpz", element_patterns, geo_factor)
    return element_fields


@partial(jax.jit, static_argnames=("power",))
def synthesize_pattern(
    element_fields: ArrayLike,
    weights: ArrayLike,
    power: bool = True,
) -> jax.Array:
    """Synthesizes a pattern from weights using the precomputed basis."""
    total_field = jnp.einsum("xytpz,xy->tpz", element_fields, weights)
    if not power:
        return total_field

    power_pattern = jnp.sum(jnp.abs(total_field) ** 2, axis=-1)
    return power_pattern


@jax.jit
def find_correction_weights(
    target_field: jax.Array,
    element_fields: jax.Array,
    alpha: float | None = 1e-2,
) -> jax.Array:
    """Finds the optimal weights for the distorted array to match a target field using least-squares."""
    n_x, n_y, n_theta, n_phi, n_pol = element_fields.shape
    n_elements = n_x * n_y
    n_points = n_theta * n_phi * n_pol

    A = element_fields.transpose(2, 3, 4, 0, 1).reshape(n_points, n_elements)
    b = target_field.flatten()  # (n_points,)

    if alpha is not None:
        # Solve the Ridge Regression problem: (A^H * A + alpha*I) * w = A^H * b
        A_H = A.T.conj()
        I = jnp.eye(n_elements)
        w = jnp.linalg.solve(A_H @ A + alpha * I, A_H @ b)  #  (n_elements,)
    else:
        # Solve the least-squares problem A * w = b for w
        w = jnp.linalg.lstsq(A, b, rcond=None)[0]  # (n_elements,)

    w = w.reshape(n_x, n_y)  # (n_x, n_y)
    return w


Kind = Literal["cst", "openems", "synthetic"]


@lru_cache
def load_element_patterns(
    config: ArrayConfig,
    kind: Kind = "cst",
    path: Path | None = None,
) -> np.ndarray:
    """Load or simulates element patterns for an array."""
    if kind == "openems":
        if path is None:
            path = DEFAULT_SIM_PATH
        E_field = load_openems_nf2ff(path).E_field  # (n_theta, n_phi, 2)

        reps = config.array_size + (1,) * E_field.ndim
        element_patterns = np.tile(E_field[None, None, ...], reps)
        return element_patterns  # (n_x, n_y, n_theta, n_phi, n_pol)

    if kind == "cst":
        if path is None:
            path = Path(__file__).parents[1] / "cst" / "classic"
        return load_cst(path).element_fields

    if kind == "synthetic":
        theta_size, phi_size = config.theta_rad.size, config.phi_rad.size

        # Base cosine model for field amplitude
        amplitude = np.cos(config.theta_rad)
        amplitude = np.where(config.theta_rad > np.pi / 2, 0, amplitude)
        amplitude = amplitude[:, None] * np.ones((theta_size, phi_size))

        E_field = amplitude[..., None]  # (n_theta, n_phi, 1)

        # Create element patterns for each element in the array
        reps = config.array_size + (1,) * E_field.ndim
        element_patterns = np.tile(E_field[None, None, ...], reps)
        return element_patterns.astype(np.complex64)

    raise ValueError(f"Unknown kind: {kind!r}")


@jax.jit
def normalize_patterns(patterns: ArrayLike) -> jax.Array:
    """Performs peak normalization on a batch of radiation patterns (linear scale)."""
    max_vals = jnp.max(patterns, axis=(1, 2), keepdims=True)
    return patterns / (max_vals + 1e-8)


@partial(jax.jit, static_argnames=("floor_db", "normalize"))
def convert_to_db(
    patterns: ArrayLike,
    floor_db: float | None = None,
    normalize: bool = True,
) -> jax.Array:
    """Converts linear power patterns to dB scale."""
    if normalize:
        patterns = patterns / jnp.max(patterns)  # Normalize

    if floor_db is not None:
        linear_floor = 10.0 ** (floor_db / 10.0)
        patterns = jnp.maximum(patterns, linear_floor)

    return 10.0 * jnp.log10(patterns)  # Convert to dB scale


# =============================================================================
# Plotting Functions
# =============================================================================


def extract_E_plane_cut(pattern: np.ndarray, phi_idx: int = 0) -> np.ndarray:
    """
    Extract the E-plane cut (phi = 0°) from the 3D radiation pattern.
    The input pattern is assumed to be in the range [0, 180°] for theta.
    The output pattern will be in the range [0, 360°] for theta.
    """
    semi_cut1 = pattern[:, phi_idx]  # Extract the E-plane cut at phi_idx
    semi_cut2 = pattern[::-1, (phi_idx + 180) % 360]  # Mirror the other cut
    return np.hstack((semi_cut1, semi_cut2))


def plot_E_plane(
    pattern: np.ndarray,
    fmt: str = "r-",
    *,
    phi_idx: int = 0,
    label: str | None = None,
    title: str | None = None,
    ax: plt.Axes | PolarAxes | None = None,
    filename: str | None = None,
):
    if ax is None:
        fig = plt.figure(constrained_layout=True)
        ax = typing.cast(PolarAxes, fig.add_subplot(projection="polar"))

    pattern_cut = extract_E_plane_cut(pattern, phi_idx=phi_idx)
    theta_rad = np.linspace(0, 2 * np.pi, pattern_cut.size)

    axp = typing.cast(PolarAxes, ax)
    axp.plot(theta_rad, pattern_cut, fmt, linewidth=1, label=label)
    axp.set_thetagrids(np.arange(0, 360, 30))
    # axp.set_rgrids(np.arange(-20, 20, 10))
    # axp.set_rlim(-30, 5)
    axp.set_theta_offset(np.pi / 2)  # make 0 degree at the top
    axp.set_theta_direction(-1)  # clockwise
    axp.set_rlabel_position(45)  # move radial label to the right
    axp.grid(True, linestyle="--")
    # axp.tick_params(labelsize=6)
    if title:
        axp.set_title(title)
    if axp is None:
        fig.set_tight_layout(True)
        if filename:
            fig.savefig(filename, dpi=250)


def extend_pattern_to_360_theta(pattern: np.ndarray) -> np.ndarray:
    """
    Extend radiation pattern from [0,180°] to [0,360°] in theta.
    Useful for plotting 2D E-plane patterns.
    The input pattern is assumed to be in the range [0, 180°] for theta.
    The output pattern will be in the range [0, 360°] for theta.
    """
    _, n_phi = pattern.shape

    phi_indices = (np.arange(n_phi) + n_phi // 2) % n_phi  # Shift phi indices
    extension = pattern[::-1, phi_indices]  # Mirror pattern

    return np.vstack((pattern, extension))


def plot_ff_3d(
    pattern: ArrayLike,
    *,
    theta_rad: ArrayLike | None = None,
    phi_rad: ArrayLike | None = None,
    clip_min_db: float | None = None,
    elev: float | None = None,
    azim: float | None = None,
    title: str = "3D Radiation Pattern",
    ax: plt.Axes | Axes3D | None = None,
):
    if theta_rad is None:
        theta_rad = ArrayConfig.theta_rad
    if phi_rad is None:
        phi_rad = ArrayConfig.phi_rad

    pattern = np.clip(
        pattern, a_min=clip_min_db, a_max=None
    )  # Clip to minimum dB value
    pattern = (pattern - pattern.min()) / np.ptp(pattern)  # Normalize to [0, 1]
    pattern = pattern * 2  # Scale pattern for visualization

    # Calculate cartesian coordinates using the correctly scaled radius
    x = pattern * np.sin(theta_rad)[:, None] * np.cos(phi_rad)[None, :]
    y = pattern * np.sin(theta_rad)[:, None] * np.sin(phi_rad)[None, :]
    z = pattern * np.cos(theta_rad)[:, None]

    if ax is None:
        fig = plt.figure(layout="compressed")
        ax = fig.add_subplot(projection="3d")

    ax3d = typing.cast(Axes3D, ax)
    ax3d.plot_surface(x, y, z, cmap="Spectral_r")
    ax3d.view_init(elev=elev, azim=azim)
    # ax3d.set_box_aspect(None, zoom=1.2)
    ax3d.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 1.8), xticks=[], yticks=[], zticks=[])
    ax3d.set_title(title)


def plot_ff_2d(
    pattern: ArrayLike,
    *,
    theta_rad: ArrayLike | None = None,
    phi_rad: ArrayLike | None = None,
    title: str = "2D Radiation Pattern",
    colorbar: bool = True,
    ax: plt.Axes | None = None,
) -> AxesImage:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))

    if theta_rad is None:
        theta_rad = ArrayConfig.theta_rad
    if phi_rad is None:
        phi_rad = ArrayConfig.phi_rad

    theta_deg, phi_deg = np.degrees(theta_rad), np.degrees(phi_rad)
    extent = (np.min(theta_deg), np.max(theta_deg), np.min(phi_deg), np.max(phi_deg))
    aspect = theta_deg.size / phi_deg.size
    im = ax.imshow(pattern, extent=extent, origin="lower", aspect=aspect)
    ax.set(xticks=theta_deg[::30], yticks=phi_deg[::30], xlabel="θ°", ylabel="φ°")
    ax.set_title(title)

    if colorbar:
        ax.figure.colorbar(im, fraction=0.046, pad=0.04, label="Normalized Dbi")

    return im


def plot_sine_space(
    pattern: ArrayLike,
    *,
    theta_rad: ArrayLike | None = None,
    phi_rad: ArrayLike | None = None,
    title: str = "Sine-Space Radiation Pattern",
    theta_circles: bool = True,
    phi_lines: bool = True,
    colorbar: bool = True,
    ax: plt.Axes | None = None,
) -> ColorizingArtist:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    if theta_rad is None:
        theta_rad = ArrayConfig.theta_rad
    if phi_rad is None:
        phi_rad = ArrayConfig.phi_rad

    u = np.sin(theta_rad)[:, None] * np.cos(phi_rad)
    v = np.sin(theta_rad)[:, None] * np.sin(phi_rad)
    im = ax.contourf(u, v, pattern, levels=128, cmap="viridis")

    axis_args = dict(color="gray", linestyle="--", linewidth=0.9)

    if theta_circles:
        for theta_deg in np.array([30, 60]):
            theta_rad = np.radians(theta_deg)
            radius = np.sin(theta_rad)
            ax.add_patch(plt.Circle((0, 0), radius, fill=False, **axis_args))
            label_offset = np.radians(45)
            x, y = radius * np.cos(label_offset), radius * np.sin(label_offset)
            ax.text(x, y, f"{theta_deg}°", ha="center", va="center", color="gray")

    if phi_lines:
        for phi_deg in np.arange(0, 360, 30):
            phi_rad = np.radians(phi_deg)
            x, y = np.cos(phi_rad), np.sin(phi_rad)
            ax.plot(*np.vstack(([0, 0], [x, y])).T, **axis_args)
            if phi_deg in [0, 90]:
                continue  # Avoid label overlap with the title and colorbar
            ax.text(1.1 * x, 1.1 * y, f"{phi_deg}°", ha="center", va="center")

    ax.add_patch(plt.Circle((0, 0), 1, linewidth=1, fill=False))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim((-1.0, 1.0))
    ax.set_ylim((-1.0, 1.0))
    ax.axis("off")
    ax.set_xlabel("u = sin($\\theta$)cos($\\phi$)")
    ax.set_ylabel("v = sin($\\theta$)sin($\\phi$)")
    ax.set_title(title)

    if colorbar:
        ax.figure.colorbar(im, fraction=0.046, pad=0.04, label="Normalized Dbi")

    return im


def plot_pattern(
    pattern: ArrayLike,
    *,
    theta_rad: ArrayLike | None = None,
    phi_rad: ArrayLike | None = None,
    clip_min_db: float | None = None,
    title: str | None = None,
    colorbar: str | None = None,
    fig: plt.FigureBase | None = None,
) -> plt.FigureBase:
    if theta_rad is None:
        theta_rad = ArrayConfig.theta_rad
    if phi_rad is None:
        phi_rad = ArrayConfig.phi_rad

    if fig is None:
        fig = plt.figure(figsize=(15, 5), layout="compressed")
    axd = fig.subplot_mosaic("ABC", per_subplot_kw={"C": {"projection": "3d"}})
    im = plot_ff_2d(pattern, ax=axd["A"], colorbar=False)
    plot_sine_space(pattern, ax=axd["B"], colorbar=False)
    plot_ff_3d(pattern, clip_min_db=clip_min_db, ax=axd["C"])

    if title is not None:
        fig.suptitle(title)
    if colorbar is not None:
        fig.colorbar(im, ax=axd["B"], label=colorbar)

    return fig


def plot_phase_shifts(
    phase_shifts,
    title: str = "Phase Shifts",
    colorbar: bool = True,
    ax: plt.Axes | None = None,
) -> AxesImage:
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))

    phase_shifts_clipped = (phase_shifts + np.pi) % (2 * np.pi) - np.pi
    im = ax.imshow(
        np.rad2deg(phase_shifts_clipped),
        cmap="twilight_shifted",  # Cyclic colormap suitable for phase values
        origin="lower",
        vmin=-180,
        vmax=180,
    )
    ax.set_xlabel("Element X index")
    ax.set_ylabel("Element Y index")
    ax.set_title(title)

    if colorbar:
        ax.figure.colorbar(im, fraction=0.046, pad=0.04, label="Degrees")

    return im


def steering_repr(steering_angles: np.ndarray):
    arr = steering_angles.T.tolist()
    formatted = (
        "["
        + ", ".join("[" + ", ".join(f"{x:.1f}" for x in row) + "]" for row in arr)
        + "]"
    )
    return f"[θ°, φ°] = {formatted}"


def demo_phase_shifts():
    """Demonstrate phase shift calculations and visualization."""
    config = ArrayConfig()
    kx, ky = compute_spatial_phase_coeffs(config)

    steering_angles = [
        [0, 0],  # Broadside
        [30, 0],  # 30° elevation
        [30, 45],  # 30° elevation, 45° azimuth
        [30, 60],  # 30° elevation, 60° azimuth
        [45, 90],  # 45° elevation, 90° azimuth
        [60, 180],  # 60° elevation, 180° azimuth
    ]
    steering_angles = np.array(steering_angles)
    nrows = 2
    ncols = np.ceil(steering_angles.shape[0] / 2).astype(np.int32)

    # Use new element weight calculator
    phase_shifts_list = []
    for angle in steering_angles:
        _, phase_shifts = calculate_weights(kx, ky, np.radians(angle))
        phase_shifts_list.append(phase_shifts)

    kw = dict(figsize=(15, 10), sharex=True, sharey=True, layout="compressed")
    fig, axes = plt.subplots(nrows, ncols, **kw)

    for i, ax in enumerate(axes.flat[: steering_angles.shape[0]]):
        title = f"θ={steering_angles[i][0]}°, φ={steering_angles[i][1]}°"
        im = plot_phase_shifts(phase_shifts_list[i], title=title, colorbar=False, ax=ax)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Degrees")
    fig.suptitle("Phase Shifts for Different Steering Angles")
    filename = "demo_phase_shifts.png"
    fig.savefig(filename, dpi=250)
    logger.info(f"Saved {filename}")


def demo_openems_patterns():
    """Demonstrate OpenEMS pattern loading with new unified interface."""
    config = ArrayConfig()
    element_patterns = load_element_patterns(config, kind="openems")

    kx, ky = compute_spatial_phase_coeffs(config)
    steering_deg = jnp.array([0.0, 0.0])  # Broadside steering
    weights, _ = calculate_weights(kx, ky, jnp.radians(steering_deg))

    element_fields = compute_element_fields(element_patterns, config)
    power_pattern = synthesize_pattern(element_fields, weights, power=True)
    power_dB = convert_to_db(power_pattern)

    fig = plt.figure(figsize=(15, 5), layout="compressed")
    steering_str = f"θ={steering_deg[0]:.1f}°, φ={steering_deg[1]:.1f}°"
    title = f"OpenEMS Radiation Pattern ({steering_str})"
    plot_pattern(power_dB, clip_min_db=-30, title=title, fig=fig)

    fig_path = "demo_openems.png"
    fig.savefig(fig_path, dpi=250)
    logger.info(f"Saved OpenEMS sample plot to {fig_path}")


def sum_cst():
    cst_path = Path(__file__).parents[1] / "cst"
    cst_data = load_cst(cst_path / "classic")
    fields = cst_data.element_fields
    powers = jnp.sum(jnp.abs(fields) ** 2, axis=-1)
    powers_db = convert_to_db(powers, floor_db=None, normalize=False)
    n_x, n_y = fields.shape[:2]

    # CST sum
    p = (
        Path(__file__).parents[2]
        / "farfield (f=2400) [1[1,0]+2[1,0]+3[1,0]+4[1,0]+5[1,0]+6[1,0]+7[1,0]+8[1,0]+9[1,0]+10[1,0]+11[1,0...].txt"
    )
    g_field = load_cst_file(p)
    g_power = jnp.sum(jnp.abs(g_field) ** 2, axis=-1)
    g_power_db = convert_to_db(g_power, floor_db=None, normalize=False)

    # direct sum of fields
    fields_sum = jnp.sum(fields, axis=(0, 1)) / np.sqrt(n_x * n_y)
    power_sum = jnp.sum(jnp.abs(fields_sum) ** 2, axis=-1)
    power_sum_db = convert_to_db(power_sum, floor_db=None, normalize=False)

    # proper sum of fields
    steering_rad = np.radians([0, 0])
    kx, ky = compute_spatial_phase_coeffs(cst_data.config)
    weights, _ = calculate_weights(kx, ky, steering_rad)
    p_fields = synthesize_pattern(fields, weights, power=False)
    p_power = jnp.sum(jnp.abs(p_fields) ** 2, axis=-1)
    p_power_db = convert_to_db(p_power, floor_db=None, normalize=False)

    kw = dict(subplot_kw=dict(projection="polar"), figsize=(6, 6), layout="compressed")
    fig, ax = plt.subplots(1, 1, **kw)
    plot_E_plane(g_power_db, ax=ax, fmt="r-", label="CST Original")
    plot_E_plane(power_sum_db, ax=ax, fmt="g-", label="Direct Sum")
    plot_E_plane(p_power_db, ax=ax, fmt="b-", label="Proper Sum")
    ax.set_title("CST Element Patterns (E-plane Cut)")
    ax.legend(loc="upper right")
    filename = "cst_compare.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    logger.info(f"Saved CST sum plot to {filename}")

    fig = plt.figure(figsize=(16, 16), layout="compressed")
    kw = dict(sharex=True, sharey=True, subplot_kw=dict(projection="polar"))
    axs = fig.subplots(n_x, n_y, **kw)
    for (i, j), ax in np.ndenumerate(axs):
        title = f"Element ({i + 1}, {j + 1})"
        plot_E_plane(powers_db[i, j], phi_idx=0, title=title, ax=ax)
    fig.suptitle("CST Element Patterns (E-plane Cuts)")
    filename = "cst_sum.png"
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    logger.info(f"Saved CST sum plot to {filename}")


def demo_cst_patterns():
    cst_path = Path(__file__).parents[1] / "cst"
    cst_orig_data = load_cst(cst_path / "classic")
    dist_elem_fields = load_cst(cst_path / "disturbed_5").element_fields

    steering_rad = np.radians([0, 0])
    kx, ky = compute_spatial_phase_coeffs(cst_orig_data.config)
    weights_orig, _ = calculate_weights(kx, ky, steering_rad)

    synthesize_field = partial(synthesize_pattern, power=False)

    def field_to_power(field):
        return jnp.sum(jnp.abs(field) ** 2, axis=-1)

    target_field = synthesize_field(cst_orig_data.element_fields, weights_orig)
    target_power = field_to_power(target_field)
    target_power_db = convert_to_db(target_power, floor_db=None, normalize=False)
    weights_corr = find_correction_weights(target_field, dist_elem_fields)

    dist_field = synthesize_field(dist_elem_fields, weights_orig)
    dist_power = field_to_power(dist_field)
    dist_power_db = convert_to_db(dist_power, floor_db=None, normalize=False)
    corr_field = synthesize_field(dist_elem_fields, weights_corr)
    corr_power = field_to_power(corr_field)
    corr_power_db = convert_to_db(corr_power, floor_db=None, normalize=False)

    phi_rad = cst_orig_data.config.phi_rad

    fig = plt.figure(figsize=(16, 20), layout="constrained")
    subfigs = typing.cast(list[SubFigure], fig.subfigures(4, 1))
    share = dict(sharex=True, sharey=True)
    title_target = "Target (Original Array)"
    title_dist = "Distorted Array (Uncorrected)"
    title_corr = "Distorted Array (Corrected)"

    ax0 = subfigs[0].subplots(1, 3, **share, subplot_kw=dict(projection="polar"))
    phi_idx = np.abs(phi_rad - steering_rad[1]).argmin()
    logger.info(f"Using phi index {phi_idx} for steering angle {steering_rad[1]} rad")
    phi_deg = np.degrees(phi_rad[phi_idx])
    subfigs[0].suptitle(f"E-plane Patterns (Polar Projection) for {phi_deg:.1f}°")
    plot_E_plane(target_power_db, phi_idx=phi_idx, ax=ax0[0], title=title_target)
    plot_E_plane(dist_power_db, phi_idx=phi_idx, ax=ax0[1], title=title_dist)
    plot_E_plane(corr_power_db, phi_idx=phi_idx, ax=ax0[2], title=title_corr)

    ax1 = subfigs[1].subplots(1, 3, **share, subplot_kw=dict(projection="3d"))
    subfigs[1].suptitle("3D Patterns (Cartesian Projection)")
    plot_ff_3d(target_power_db, ax=ax1[0], title=title_target)
    plot_ff_3d(dist_power_db, ax=ax1[1], title=title_dist)
    plot_ff_3d(corr_power_db, ax=ax1[2], title=title_corr)

    ax2 = subfigs[2].subplots(1, 3, **share)
    subfigs[2].suptitle("Sine-Space Patterns")
    im = plot_sine_space(target_power_db, ax=ax2[0], title=title_target, colorbar=False)
    im = plot_sine_space(dist_power_db, ax=ax2[1], title=title_dist, colorbar=False)
    im = plot_sine_space(corr_power_db, ax=ax2[2], title=title_corr, colorbar=False)
    subfigs[2].colorbar(im, ax=ax2, label="Normalized Db")

    ax3 = subfigs[3].subplots(1, 3, **share)
    subfigs[3].suptitle("2D Patterns (Cartesian Projection)")
    im = plot_ff_2d(target_power_db, ax=ax3[0], title=title_target, colorbar=False)
    im = plot_ff_2d(dist_power_db, ax=ax3[1], title=title_dist, colorbar=False)
    im = plot_ff_2d(corr_power_db, ax=ax3[2], title=title_corr, colorbar=False)
    subfigs[3].colorbar(im, ax=ax3, label="Normalized Db")

    steering_deg = np.degrees(steering_rad)
    title = (
        f"Correcting CST Patterns (θ={steering_deg[0]:.1f}°, φ={steering_deg[1]:.1f}°)"
    )
    fig.suptitle(title, fontweight="bold")
    filename = "demo_cst_patterns.png"
    fig.savefig(filename, dpi=250)
    logger.info(f"Saved CST demo plot to {filename}")


if __name__ == "__main__":
    setup_logging()
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        sum_cst()
        demo_phase_shifts()
        demo_openems_patterns()
        demo_cst_patterns()
