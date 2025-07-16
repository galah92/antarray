import logging
import time
import typing
from functools import lru_cache, partial
from pathlib import Path
from typing import Literal, NamedTuple

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.typing import ArrayLike
from joblib import Memory
from matplotlib.colorizer import ColorizingArtist
from matplotlib.image import AxesImage
from matplotlib.projections import PolarAxes
from mpl_toolkits.mplot3d import Axes3D

from utils import setup_logging

logger = logging.getLogger(__name__)

root_dir = Path(__file__).parent.parent
memory_dir = root_dir / ".joblib_cache"
memory = Memory(memory_dir, mmap_mode="r", verbose=0)

DEFAULT_SIM_DIR = root_dir / "openems" / "sim" / "antenna_array"
DEFAULT_SINGLE_ANT_FILENAME = "ff_1x1_60x60_2450_steer_t0_p0.h5"
DEFAULT_SIM_PATH = DEFAULT_SIM_DIR / DEFAULT_SINGLE_ANT_FILENAME

DEFAULT_THETA_RAD = np.radians(np.arange(180))
DEFAULT_PHI_RAD = np.radians(np.arange(360))


class ArrayConfig(NamedTuple):
    """Configuration for antenna array parameters and simulation settings."""

    array_size: tuple[int, int] = (16, 16)
    spacing_mm: tuple[float, float] = (60.0, 60.0)
    freq_hz: float = 2.45e9
    theta_rad: np.ndarray = DEFAULT_THETA_RAD
    phi_rad: np.ndarray = DEFAULT_PHI_RAD

    def __hash__(self):
        # We cannot hash numpy arrays
        return hash((self.array_size, self.spacing_mm, self.freq_hz))


class OpenEMSData(NamedTuple):
    theta_rad: jax.Array
    phi_rad: jax.Array
    r: jax.Array
    Dmax: float
    freq_hz: jax.Array
    E_field: jax.Array
    power_density: jax.Array


@lru_cache
def load_openems_nf2ff(nf2ff_path: Path) -> OpenEMSData:
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


Kind = Literal["cst", "openems", "synthetic"]


class ElementPatternData(NamedTuple):
    """Hold element patterns and configuration.

    Always contains GEPs (Geometric Element Patterns) with spatial phase factors applied.
    This ensures consistent behavior regardless of the data source.
    """

    geps: np.ndarray  # GEPs: (n_x, n_y, n_theta, n_phi, n_pol)
    config: ArrayConfig
    source: Kind


def load_element_patterns_from_cst(cst_path: Path | None = None) -> ElementPatternData:
    """Load CST antenna patterns from a directory."""
    if cst_path is None:
        cst_path = Path(__file__).parents[1] / "cst" / "classic"
    geps = load_cst(cst_path)
    # As Given by Snir
    cst_config = ArrayConfig(array_size=(4, 4), spacing_mm=(75, 75), freq_hz=2.4e9)
    return ElementPatternData(geps=geps, config=cst_config, source="cst")


def load_element_patterns_from_openems(
    array_size: tuple[int, int] = (16, 16),
    spacing_mm: tuple[float, float] = (60.0, 60.0),
    path: Path | None = None,
) -> ElementPatternData:
    """Load OpenEMS antenna patterns from simulation data."""
    if path is None:
        path = DEFAULT_SIM_PATH
    openems_data = load_openems_nf2ff(path)

    E_field = openems_data.E_field  # (freq, n_theta, n_phi, 2)
    reps = array_size + (1,) * E_field.ndim
    aeps = np.tile(E_field[None, None, ...], reps)

    openems_config = ArrayConfig(
        array_size=aeps.shape[:2],
        spacing_mm=spacing_mm,
        freq_hz=openems_data.freq_hz.item(),
        theta_rad=openems_data.theta_rad,
        phi_rad=openems_data.phi_rad,
    )

    # Convert AEPs to GEPs by computing geometric phase factors
    geps = compute_geps(aeps, openems_config)
    return ElementPatternData(geps=geps, config=openems_config, source="openems")


def load_element_patterns(
    kind: Kind = "cst",
    path: Path | None = None,
    config: ArrayConfig | None = None,
) -> ElementPatternData:
    """Factory function to create ElementPatternData from a specified kind."""
    if kind == "cst":
        return load_element_patterns_from_cst(path)
    elif kind == "openems":
        return load_element_patterns_from_openems(path=path)
    elif kind == "synthetic":
        if config is None:
            config = ArrayConfig()

        theta_size, phi_size = config.theta_rad.size, config.phi_rad.size

        # Base cosine model for field amplitude
        amplitude = np.cos(config.theta_rad)
        amplitude = np.where(config.theta_rad > np.pi / 2, 0, amplitude)
        amplitude = amplitude[:, None] * np.ones((theta_size, phi_size))

        E_field = amplitude[..., None]  # (n_theta, n_phi, 1)

        # Create element patterns for each element in the array
        reps = config.array_size + (1,) * E_field.ndim
        aeps = np.tile(E_field[None, None, ...], reps)
        aeps = aeps.astype(np.complex64)

        # Convert AEPs to GEPs
        geps = compute_geps(aeps, config)
        return ElementPatternData(geps=geps, config=config, source=kind)
    else:
        raise ValueError(f"Unknown kind: {kind!r}")


@memory.cache
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


@memory.cache
def load_cst(cst_path: Path) -> np.ndarray:
    logger.info(f"Loading antenna pattern from {cst_path}")
    data = {}
    for path in cst_path.iterdir():
        i = int(path.stem.split()[-1][1:-1]) - 1
        data[i] = load_cst_file(path)

    data = [v for _, v in sorted(data.items())]
    fields = np.stack(data, axis=0)
    fields = fields.reshape(4, 4, *fields.shape[1:])
    return fields


def get_wavenumber(freq_hz: float) -> float:
    """Calculate the wavenumber for a given frequency."""
    c = 299792458  # Speed of light in m/s
    wavelength = c / freq_hz  # Wavelength in meters
    k = 2 * np.pi / wavelength  # Wavenumber in radians/meter
    return k


@lru_cache
def get_element_positions(
    array_size: tuple[int, int],
    spacing_mm: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the x and y positions of the elements in the array, centered around (0, 0)."""
    (xn, yn), (dx_mm, dy_mm) = array_size, spacing_mm
    x_positions = (np.arange(xn) - (xn - 1) / 2) * dx_mm / 1000
    y_positions = (np.arange(yn) - (yn - 1) / 2) * dy_mm / 1000
    return x_positions, y_positions


@lru_cache
def get_element_positions_2d(
    array_size: tuple[int, int],
    spacing_mm: tuple[float, float],
) -> np.ndarray:
    """Get 2D element positions for non-separable calculations.

    Returns:
        positions: Array of shape (n_x, n_y, 2) with [x, y] coordinates for each element
    """
    x_positions, y_positions = get_element_positions(array_size, spacing_mm)

    # Create 2D grid of positions
    x_grid, y_grid = np.meshgrid(x_positions, y_positions, indexing="ij")
    positions = np.stack([x_grid, y_grid], axis=-1)  # (n_x, n_y, 2)

    return positions


@lru_cache
def compute_spatial_phase_coeffs(config: ArrayConfig) -> tuple[np.ndarray, np.ndarray]:
    """Compute spatial phase coefficients (kx, ky) for antenna array elements (separable case)."""
    k = get_wavenumber(config.freq_hz)
    x_pos, y_pos = get_element_positions(config.array_size, config.spacing_mm)
    kx, ky = k * x_pos, k * y_pos
    return kx, ky


@lru_cache
def compute_spatial_coeffs(config: ArrayConfig) -> np.ndarray:
    """Compute spatial coefficients for non-separable calculations.

    Returns:
        spatial_coeffs: Array of shape (n_x, n_y, 2) with [k*x, k*y] for each element
    """
    k = get_wavenumber(config.freq_hz)
    positions = get_element_positions_2d(config.array_size, config.spacing_mm)
    spatial_coeffs = k * positions  # (n_x, n_y, 2)
    return spatial_coeffs


@jax.jit
def calculate_weights(
    kx: ArrayLike,
    ky: ArrayLike,
    steering_angles: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Calculates element weights for given steering angles (separable case)."""
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


@jax.jit
def calculate_weights_2d(
    spatial_coeffs: ArrayLike,
    steering_angles: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Calculates element weights for given steering angles (non-separable case).

    Args:
        spatial_coeffs: Array of shape (n_x, n_y, 2) with [k*x, k*y] for each element
        steering_angles: Array of shape (n_beams, 2) with [theta, phi] for each beam

    Returns:
        weights: Array of shape (n_x, n_y) for single beam or (n_x, n_y, n_beams) for multiple
        phase_shifts: Array of shape (n_x, n_y) for single beam or (n_x, n_y, n_beams) for multiple
    """
    steering_angles = jnp.atleast_2d(steering_angles)
    theta, phi = steering_angles.T

    steerings = jnp.sin(theta) * jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=0)
    phase_shifts = spatial_coeffs @ steerings  # (n_x, n_y, n_beams)

    weights = jnp.exp(-1j * phase_shifts)  # (n_x, n_y, n_beams)
    weights = jnp.sum(weights, axis=-1)  # (n_x, n_y), assume no tapering
    weights = weights / np.sqrt(np.prod(weights.shape))  # Normalize weights

    # Return weights as (n_x, n_y) and phase_shifts as (n_x, n_y, n_beams)
    return weights, phase_shifts


def compute_geps(
    aeps: np.ndarray,
    config: ArrayConfig,
    separable: bool = True,
) -> np.ndarray:
    """Create geometric element patterns (GEPs) with spatial phase factors."""
    sin_theta = np.sin(config.theta_rad)
    sin_phi, cos_phi = np.sin(config.phi_rad), np.cos(config.phi_rad)
    if separable:
        kx, ky = compute_spatial_phase_coeffs(config)
        phase_x = np.einsum("t,p,x->tpx", sin_theta, cos_phi, kx)
        phase_y = np.einsum("t,p,y->tpy", sin_theta, sin_phi, ky)

        # (n_theta, n_phi, n_x, n_y)
        geo_phase = phase_x[..., None] + phase_y[..., None, :]
    else:
        spatial_coeffs = compute_spatial_coeffs(config)  # (n_x, n_y, 2)
        u = sin_theta[:, None] * cos_phi[None, :]  # (n_theta, n_phi)
        v = sin_theta[:, None] * sin_phi[None, :]  # (n_theta, n_phi)
        uv = np.stack([u, v], axis=-1)  # (n_theta, n_phi, 2)

        # Compute geometric phase for each element and direction
        geo_phase = np.einsum("xyk,tpk->tpxy", spatial_coeffs, uv)

    geo_factor = np.exp(1j * geo_phase)  # (n_theta, n_phi, n_x, n_y)
    geps = np.einsum("xytpz,tpxy->xytpz", aeps, geo_factor)
    return geps


@partial(jax.jit, static_argnames=("power",))
def synthesize_pattern(
    geps: ArrayLike,
    weights: ArrayLike,
    power: bool = True,
) -> jax.Array:
    """Synthesizes a pattern from weights using the precomputed GEPs (geometric element patterns)."""
    total_field = jnp.einsum("xytpz,xy->tpz", geps, weights)
    if not power:
        return total_field

    power_pattern = jnp.sum(jnp.abs(total_field) ** 2, axis=-1)
    return power_pattern


@jax.jit
def solve_weights(
    target_field: jax.Array,
    geps: jax.Array,
    alpha: float | None = 1e-2,
) -> jax.Array:
    """Finds the optimal weights for the distorted array to match a target field using least-squares."""
    n_x, n_y, n_theta, n_phi, n_pol = geps.shape
    n_elements = n_x * n_y
    n_points = n_theta * n_phi * n_pol

    A = geps.transpose(2, 3, 4, 0, 1).reshape(n_points, n_elements)
    b = target_field.flatten()  # (n_points,)

    if alpha is not None:
        # Solve the Ridge Regression problem: (A^H * A + alpha*I) * w = A^H * b
        # Cost_normalized = (1 / n_points) * ||A*w - b||² + alpha * ||w||²
        A_H = A.T.conj()
        # Scale down the matrix products by n_points
        A_H_A = (A_H @ A) / n_points
        A_H_b = (A_H @ b) / n_points
        I = jnp.eye(n_elements)
        w = jnp.linalg.solve(A_H_A + alpha * I, A_H_b)  #  (n_elements,)
    else:
        # Solve the least-squares problem A * w = b for w
        w = jnp.linalg.lstsq(A, b, rcond=None)[0]  # (n_elements,)

    w = w.reshape(n_x, n_y)  # (n_x, n_y)
    return w


@jax.jit
def normalize_patterns(patterns: ArrayLike) -> jax.Array:
    """Performs peak normalization on a batch of radiation patterns (linear scale)."""
    max_vals = jnp.max(patterns, axis=(1, 2), keepdims=True)
    return patterns / (max_vals + 1e-8)


@partial(jax.jit, static_argnames=("floor_db", "normalize"))
def convert_to_db(
    patterns: ArrayLike,
    floor_db: float | None = None,
    normalize: bool = False,
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
    theta_rad: np.ndarray = DEFAULT_THETA_RAD,
    phi_rad: np.ndarray = DEFAULT_PHI_RAD,
    clip_min_db: float | None = None,
    elev: float | None = None,
    azim: float | None = None,
    title: str = "3D Radiation Pattern",
    ax: plt.Axes | Axes3D | None = None,
):
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
    theta_rad: np.ndarray = DEFAULT_THETA_RAD,
    phi_rad: np.ndarray = DEFAULT_PHI_RAD,
    title: str = "2D Radiation Pattern",
    colorbar: bool = True,
    ax: plt.Axes | None = None,
) -> AxesImage:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))

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
    theta_rad: np.ndarray = DEFAULT_THETA_RAD,
    phi_rad: np.ndarray = DEFAULT_PHI_RAD,
    title: str = "Sine-Space Radiation Pattern",
    theta_circles: bool = True,
    phi_lines: bool = True,
    colorbar: bool = True,
    ax: plt.Axes | None = None,
) -> ColorizingArtist:
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    u = np.sin(theta_rad)[:, None] * np.cos(phi_rad)
    v = np.sin(theta_rad)[:, None] * np.sin(phi_rad)
    im = ax.contourf(u, v, pattern, levels=128, cmap="viridis")

    axis_args = dict(color="gray", linestyle="--", linewidth=0.9)

    if theta_circles:
        for theta_deg in [30, 60]:
            theta_rad = np.radians(theta_deg)
            radius = np.sin(theta_rad)
            ax.add_patch(plt.Circle((0, 0), radius, fill=False, **axis_args))
            label_offset = np.radians(45)
            x, y = radius * np.cos(label_offset), radius * np.sin(label_offset)
            ax.text(x, y, f"{theta_deg}°", ha="center", va="center", color="gray")

    if phi_lines:
        for phi_deg in range(0, 360, 30):
            phi_rad = np.radians(phi_deg)
            offset = np.pi / 2
            x, y = np.cos(-phi_rad + offset), np.sin(-phi_rad + offset)
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
    theta_rad: np.ndarray = DEFAULT_THETA_RAD,
    phi_rad: np.ndarray = DEFAULT_PHI_RAD,
    clip_min_db: float | None = None,
    title: str | None = None,
    colorbar: str | None = None,
    fig: plt.FigureBase | None = None,
) -> plt.FigureBase:
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
    element_data = load_element_patterns_from_openems()
    config = element_data.config

    kx, ky = compute_spatial_phase_coeffs(config)
    steering_deg = jnp.array([0.0, 0.0])  # Broadside steering
    weights, _ = calculate_weights(kx, ky, jnp.radians(steering_deg))

    power_pattern = synthesize_pattern(element_data.geps, weights, power=True)
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
    element_fields = load_cst(cst_path / "classic")
    powers = jnp.sum(jnp.abs(element_fields) ** 2, axis=-1)
    powers_db = convert_to_db(powers, floor_db=None, normalize=False)
    n_x, n_y = element_fields.shape[:2]

    # CST sum
    p = (
        Path(__file__).parents[2]
        / "farfield (f=2400) [1[1,0]+2[1,0]+3[1,0]+4[1,0]+5[1,0]+6[1,0]+7[1,0]+8[1,0]+9[1,0]+10[1,0]+11[1,0...].txt"
    )
    g_field = load_cst_file(p)
    g_power = jnp.sum(jnp.abs(g_field) ** 2, axis=-1)
    g_power_db = convert_to_db(g_power, floor_db=None, normalize=False)

    # direct sum of fields
    fields_sum = jnp.sum(element_fields, axis=(0, 1)) / np.sqrt(n_x * n_y)
    power_sum = jnp.sum(jnp.abs(fields_sum) ** 2, axis=-1)
    power_sum_db = convert_to_db(power_sum, floor_db=None, normalize=False)

    steering_rad = np.radians([0, 0])
    cst_config = ArrayConfig(array_size=(4, 4), spacing_mm=(75, 75), freq_hz=2.4e9)
    kx, ky = compute_spatial_phase_coeffs(cst_config)
    weights, _ = calculate_weights(kx, ky, steering_rad)

    p_power = synthesize_pattern(element_fields, weights)
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
    orig_data = load_element_patterns_from_cst()
    cst_path = Path(__file__).parents[1] / "cst"
    dist_data = load_element_patterns_from_cst(cst_path / "disturbed_5")

    steering_deg = np.array([0.0, 0.0])
    steering_rad = np.radians(steering_deg)
    kx, ky = compute_spatial_phase_coeffs(orig_data.config)
    w_orig, _ = calculate_weights(kx, ky, steering_rad)

    to_db = partial(convert_to_db, normalize=False)

    target_field = synthesize_pattern(orig_data.geps, w_orig, power=False)
    target_power = jnp.sum(jnp.abs(target_field) ** 2, axis=-1)
    target_power_db = to_db(target_power)

    w_corr = solve_weights(target_field, dist_data.geps, alpha=None)

    dist_power_db = to_db(synthesize_pattern(dist_data.geps, w_orig))
    corr_power_db = to_db(synthesize_pattern(dist_data.geps, w_corr))

    # Always use the phi slice where the peak occurs for best visualization
    assert abs(steering_rad[1]) < 1e-6, "Only theta steering supported"
    phi_idx = np.unravel_index(np.argmax(target_power_db), target_power_db.shape)[1]
    phi_deg = np.degrees(orig_data.config.phi_rad[phi_idx])
    logger.info(f"{phi_idx=} (φ={phi_deg:.1f}°)")

    # Calculate MSE metrics
    dist_mse = np.mean(np.square(dist_power_db - target_power_db))
    corr_mse = np.mean(np.square(corr_power_db - target_power_db))
    logger.info(f"{dist_mse=:.2f}, {corr_mse=:.2f}")

    theta_rad, phi_rad = orig_data.config.theta_rad, orig_data.config.phi_rad
    target_metrics = compute_pattern_metrics(target_power_db, theta_rad, phi_rad)
    dist_metrics = compute_pattern_metrics(dist_power_db, theta_rad, phi_rad)
    corr_metrics = compute_pattern_metrics(corr_power_db, theta_rad, phi_rad)

    kw = dict(subplot_kw=dict(projection="polar"), layout="compressed")
    fig, ax = plt.subplots(figsize=(8, 8), **kw)

    plot = partial(plot_E_plane, phi_idx=phi_idx, ax=ax)
    plot(
        target_power_db,
        fmt="r-",
        label=f"Target Pattern (Peak={target_metrics['peak_power_db']:.1f}dB)",
    )
    plot(
        dist_power_db,
        fmt="g-",
        label=f"Distorted Pattern (Peak={dist_metrics['peak_power_db']:.1f}dB)",
    )
    plot(
        corr_power_db,
        fmt="b-",
        label=f"Corrected Pattern (Peak={corr_metrics['peak_power_db']:.1f}dB)",
    )
    ax.legend(loc="lower center")

    total_power_orig = np.sum(np.square(np.abs(w_orig)))
    total_power_corr = np.sum(np.square(np.abs(w_corr)))
    logger.info(f"{total_power_orig=:.2f}, {total_power_corr=:.2f}")

    title = f"CST Patterns (θ={steering_deg[0]:.1f}°, φ={steering_deg[1]:.1f}°)"
    fig.suptitle(title, fontweight="bold")
    filename = f"demo_cst_patterns_{steering_deg[0]:.0f}_{steering_deg[1]:.0f}.png"
    fig.savefig(filename, dpi=250, bbox_inches="tight")
    logger.info(f"Saved CST demo plot to {filename}")


def compute_pattern_metrics(
    pattern_db: np.ndarray,
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
) -> dict:
    """Compute comprehensive pattern analysis metrics.

    Args:
        pattern_db: Pattern in dB scale (n_theta, n_phi)
        theta_rad: Theta angles in radians
        phi_rad: Phi angles in radians
        phi_idx: Phi index for E-plane cut analysis

    Returns:
        Dictionary with pattern metrics for both cut and full pattern
    """
    theta_deg, phi_deg = np.degrees(theta_rad), np.degrees(phi_rad)

    global_peak_idx = np.unravel_index(np.argmax(pattern_db), pattern_db.shape)
    peak_power_db = pattern_db[global_peak_idx]
    peak_theta_deg = theta_deg[global_peak_idx[0]]
    peak_phi_deg = phi_deg[global_peak_idx[1]]

    # Global sidelobe level - find maximum sidelobe outside main beam region
    # Define main beam region as ±10° around peak (adjustable)
    main_beam_theta_width = 10.0  # degrees
    main_beam_phi_width = 10.0  # degrees

    theta_mask = np.abs(theta_deg - peak_theta_deg) <= main_beam_theta_width
    phi_mask = np.abs(phi_deg - peak_phi_deg) <= main_beam_phi_width

    # Handle phi wraparound at 0°/360°
    if peak_phi_deg < main_beam_phi_width:
        phi_mask |= phi_deg >= (360 - (main_beam_phi_width - peak_phi_deg))
    elif peak_phi_deg > (360 - main_beam_phi_width):
        phi_mask |= phi_deg <= (main_beam_phi_width - (360 - peak_phi_deg))

    # Create 2D main beam mask
    main_beam_mask = theta_mask[:, None] & phi_mask[None, :]
    sidelobe_region = pattern_db[~main_beam_mask]
    ssl_db = peak_power_db - np.max(sidelobe_region)

    # 3D directivity estimate using solid angle integration
    # Convert to linear scale for integration
    pattern = 10 ** (pattern_db / 10)

    # Compute solid angle weights: sin(θ) * dθ * dφ
    dtheta = np.diff(theta_rad)[0] if len(theta_rad) > 1 else np.pi / 180
    dphi = np.diff(phi_rad)[0] if len(phi_rad) > 1 else 2 * np.pi / 360

    # sin(θ) weighting for spherical coordinates
    solid_angle_weights = np.sin(theta_rad)[:, None] * dtheta * dphi

    # Ensure weights have same shape as pattern
    solid_angle_weights = np.broadcast_to(solid_angle_weights, pattern.shape)

    # Total radiated power (integrate over sphere)
    total_power = np.sum(pattern * solid_angle_weights)

    # Directivity = 4π * peak_intensity / total_power
    peak_linear = 10 ** (peak_power_db / 10)
    directivity = 4 * np.pi * peak_linear / total_power
    directivity_db = 10 * np.log10(directivity)

    # Pattern efficiency (what fraction of power is in main beam)
    main_beam_power = np.sum(
        pattern[main_beam_mask] * solid_angle_weights[main_beam_mask]
    )
    pattern_efficiency = main_beam_power / total_power

    return {
        "peak_power_db": peak_power_db,
        "peak_theta_deg": peak_theta_deg,
        "peak_phi_deg": peak_phi_deg,
        "ssl_db": ssl_db,
        "directivity_db": directivity_db,
        "pattern_efficiency": pattern_efficiency,
    }


def test_random_sampling():
    """Simplified random weight sampling test - loads CST data and runs until Ctrl-C."""
    # Load data
    orig_data = load_element_patterns_from_cst()
    cst_path = Path(__file__).parents[1] / "cst"
    dist_data = load_element_patterns_from_cst(cst_path / "disturbed_5")

    # Create target pattern
    steering_rad = np.radians([0.0, 0.0])
    kx, ky = compute_spatial_phase_coeffs(orig_data.config)
    w_orig, _ = calculate_weights(kx, ky, steering_rad)
    target_field = synthesize_pattern(orig_data.geps, w_orig, power=False)
    target_power_db = convert_to_db(
        np.sum(np.abs(target_field) ** 2, axis=-1), normalize=False
    )
    target_peak = np.max(target_power_db)
    peak_idx = np.unravel_index(np.argmax(target_power_db), target_power_db.shape)

    to_db = partial(convert_to_db, normalize=False)
    w_lstsq = solve_weights(target_field, dist_data.geps, alpha=None)
    power_lstsq_db = to_db(synthesize_pattern(dist_data.geps, w_lstsq))
    lstsq_mse = np.mean(np.square(power_lstsq_db - target_power_db))
    lstsq_peak = np.max(power_lstsq_db[peak_idx])
    logger.info(
        f"LstSq MSE: {lstsq_mse:.3f} | LstSq Peak: {lstsq_peak:.1f}dB | Target Peak: {target_peak:.1f}dB"
    )

    array_size = dist_data.geps.shape[:2]
    best_mse, iteration = float("inf"), 0
    start_time = time.time()

    try:
        while True:
            iteration += 1
            # Generate random weights
            # amplitude = np.ones(array_size)
            amplitude = np.random.uniform(size=array_size)
            amplitude = amplitude / np.sum(amplitude) * np.prod(array_size)
            phase = np.random.uniform(high=2 * np.pi, size=array_size)
            w = amplitude * np.exp(1j * phase)
            w = w / np.sqrt(np.prod(array_size))  # Normalize to sqrt(N)

            rand_pattern_db = to_db(synthesize_pattern(dist_data.geps, w))
            rand_mse = np.mean(np.square(rand_pattern_db - target_power_db))
            rand_peak = np.max(rand_pattern_db[peak_idx])

            if rand_mse < best_mse:
                best_mse = rand_mse
                logger.info(
                    f"Iter {iteration:>5d} | {rand_mse=:.3f} | {rand_peak=:.1f}dB"
                )

                kw = dict(subplot_kw=dict(projection="polar"), layout="compressed")
                fig, ax = plt.subplots(figsize=(8, 8), **kw)
                plot = partial(plot_E_plane, phi_idx=0, ax=ax)

                label = f"Target (Peak: {target_peak:.1f}dB)"
                plot(target_power_db, fmt="r-", label=label)

                label = f"LstSq (Peak: {lstsq_peak:.1f}dB, MSE: {lstsq_mse:.3f})"
                plot(power_lstsq_db, fmt="g-", label=label)

                label = f"Rand (Peak: {rand_peak:.1f}dB, MSE: {rand_mse:.3f})"
                plot(rand_pattern_db, fmt="b-", label=label)

                ax.legend(loc="lower center")
                title = f"Rand Pattern ({iteration=}"
                fig.suptitle(title, fontweight="bold")
                filename = f"rand_pattern_{iteration}.png"
                fig.savefig(filename, dpi=250, bbox_inches="tight")
                logger.info(f"Saved random pattern plot to {filename}")

    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - start_time
        rate = iteration / elapsed
        logger.info(
            f"Iter {iteration:>5d} | Best MSE: {best_mse:.3f} | LstSq MSE: {lstsq_mse:.3f} | Rate: {rate:.0f} iter/s | Elapsed: {elapsed:.1f}s"
        )


if __name__ == "__main__":
    setup_logging()
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        # sum_cst()
        # demo_phase_shifts()
        # demo_openems_patterns()
        # demo_cst_patterns()
        test_random_sampling()
