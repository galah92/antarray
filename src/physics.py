import logging
import typing
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.typing import ArrayLike
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


# @dataclass(frozen=True)
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


@lru_cache
def load_cst(cst_path: Path) -> CstData:
    logger.info(f"Loading antenna pattern from {cst_path}")
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
    data = {}
    for path in cst_path.iterdir():
        i = int(path.stem.split()[-1][1:-1]) - 1
        data[i] = np.genfromtxt(path, skip_header=2, dtype=np.float32, names=names)

    data = [v for _, v in sorted(data.items())]
    data = np.stack(data, axis=0)  # (16, 181 * 360, ...)

    E_cross = data["abs_cross"] * np.exp(1j * np.radians(data["phase_cross_deg"]))
    E_copol = data["abs_copol"] * np.exp(1j * np.radians(data["phase_copol_deg"]))
    fields = np.stack([E_copol, E_cross], axis=-1)

    fields = fields.reshape(-1, 360, 181, 2)  #  (n_element, phi, theta, n_pol)
    fields = fields.transpose((0, 2, 1, 3))  #  (n_element, theta, phi, n_pol)
    fields = fields[:, :-1]  #  Remove last theta value
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

    return weights, phase_shifts


def make_element_weight_calculator(config: ArrayConfig) -> Callable:
    """Factory to create a function for calculating element weights."""
    kx, ky = compute_spatial_phase_coeffs(config)
    kx, ky = jnp.asarray(kx), jnp.asarray(ky)
    calculate = partial(calculate_weights, kx, ky)
    return calculate


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

    element_fields = np.einsum("xytpz,tpxy->tpzxy", element_patterns, geo_factor)
    return element_fields


@jax.jit
def synthesize_pattern(element_fields: ArrayLike, weights: ArrayLike) -> jax.Array:
    """Synthesizes a pattern from weights using the precomputed basis."""
    total_field = jnp.einsum("xytpz,xy->tpz", element_fields, weights)
    power_pattern = jnp.sum(jnp.abs(total_field) ** 2, axis=-1)
    return power_pattern


def make_pattern_synthesizer(
    element_patterns: np.ndarray,
    config: ArrayConfig,
) -> Callable:
    """Factory to create a pattern synthesis function."""
    element_fields = compute_element_fields(element_patterns, config)
    element_fields = jnp.asarray(element_fields)
    synthesize = partial(synthesize_pattern, element_fields)
    return synthesize


@lru_cache
def load_element_patterns(
    config: ArrayConfig,
    openems_path: Path | None = None,
) -> np.ndarray:
    """Simulates element patterns for ideal array, with optional OpenEMS data."""
    if openems_path is not None:
        E_field = load_openems_nf2ff(openems_path).E_field  # (n_theta, n_phi, 2)

        E_field = np.tile(E_field[..., None, None], config.array_size)
        return E_field  # (n_x, n_y, n_theta, n_phi, n_pol)

    # Use existing synthetic generation (unchanged)
    theta_size, phi_size = config.theta_rad.size, config.phi_rad.size

    # Base cosine model for field amplitude
    amplitude = np.cos(config.theta_rad)
    amplitude = np.where(config.theta_rad > np.pi / 2, 0, amplitude)
    amplitude = amplitude[:, None] * np.ones((theta_size, phi_size))

    E_field = amplitude[:, :, None]  # (n_theta, n_phi, 1)
    E_field = np.tile(E_field[..., None, None], config.array_size)
    return E_field.astype(np.complex64)  # (n_x, n_y, n_theta, n_phi, n_pol)


def make_physics_setup(
    config: ArrayConfig,
    openems_path: Path | None = None,
) -> tuple[Callable, Callable]:
    """Creates the physics simulation setup with optional OpenEMS data support."""

    element_patterns = load_element_patterns(config, openems_path=openems_path)
    synthesize_pattern = make_pattern_synthesizer(element_patterns, config)

    compute_element_weights = make_element_weight_calculator(config)

    return synthesize_pattern, compute_element_weights


@jax.jit
def normalize_patterns(patterns: ArrayLike) -> jax.Array:
    """Performs peak normalization on a batch of radiation patterns (linear scale)."""
    max_vals = jnp.max(patterns, axis=(1, 2), keepdims=True)
    return patterns / (max_vals + 1e-8)


@partial(jax.jit, static_argnames=("floor_db",))
def convert_to_db(patterns: ArrayLike, floor_db: float | None = None) -> jax.Array:
    """Converts linear power patterns to normalized dB scale."""
    normalized = patterns / jnp.max(patterns)  # Normalize

    if floor_db is not None:
        linear_floor = 10.0 ** (floor_db / 10.0)
        normalized = jnp.maximum(normalized, linear_floor)

    return 10.0 * jnp.log10(normalized)


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
    label: str | None = None,
    title: str | None = None,
    ax: PolarAxes | None = None,
    filename: str | None = None,
):
    if ax is None:
        fig = plt.figure(constrained_layout=True)
        ax = typing.cast(PolarAxes, fig.add_subplot(projection="polar"))

    pattern_cut = extract_E_plane_cut(pattern, phi_idx=0)
    theta_rad = np.linspace(0, 2 * np.pi, pattern_cut.size)

    ax.plot(theta_rad, pattern_cut, fmt, linewidth=1, label=label)
    ax.set_thetagrids(np.arange(0, 360, 30))
    ax.set_rgrids(np.arange(-20, 20, 10))
    ax.set_rlim(-25, 15)
    ax.set_theta_offset(np.pi / 2)  # make 0 degree at the top
    ax.set_theta_direction(-1)  # clockwise
    ax.set_rlabel_position(90)  # move radial label to the right
    ax.grid(True, linestyle="--")
    ax.tick_params(labelsize=6)
    if title:
        ax.set_title(title)
    if ax is None:
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
    theta_rad: ArrayLike,
    phi_rad: ArrayLike,
    pattern: ArrayLike,
    *,
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
    theta_rad: ArrayLike,
    phi_rad: ArrayLike,
    pattern: ArrayLike,
    *,
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
    theta_rad: ArrayLike,
    phi_rad: ArrayLike,
    pattern: ArrayLike,
    *,
    title: str = "Sine-Space Radiation Pattern",
    theta_circles: bool = True,
    phi_lines: bool = True,
    colorbar: bool = True,
    ax: plt.Axes | None = None,
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

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

    return ax


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
    im = plot_ff_2d(theta_rad, phi_rad, pattern, ax=axd["A"], colorbar=False)
    plot_sine_space(theta_rad, phi_rad, pattern, ax=axd["B"], colorbar=False)
    plot_ff_3d(theta_rad, phi_rad, pattern, clip_min_db=clip_min_db, ax=axd["C"])

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


# =============================================================================
# Demo Functions
# =============================================================================


def demo_phase_shifts():
    """Demonstrate phase shift calculations and visualization."""
    config = ArrayConfig()
    compute_element_weights = make_element_weight_calculator(config)

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
        _, phase_shifts = compute_element_weights(np.radians(angle))
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

    synthesize_ideal, compute_element_weights = make_physics_setup(
        config, openems_path=DEFAULT_SIM_PATH
    )

    steering_deg = jnp.array([30, 45])
    steering_angle = jnp.radians(steering_deg)
    weights, _ = compute_element_weights(steering_angle)
    power_pattern = synthesize_ideal(weights)
    power_dB = convert_to_db(power_pattern)

    fig = plt.figure(figsize=(15, 5), layout="compressed")
    steering_str = f"θ={np.degrees(steering_angle[0]):.1f}°, φ={np.degrees(steering_angle[1]):.1f}°"
    title = f"OpenEMS Radiation Pattern ({steering_str})"
    plot_pattern(power_dB, clip_min_db=-30, title=title, fig=fig)

    fig_path = "test_openems.png"
    fig.savefig(fig_path, dpi=250)
    logger.info(f"Saved OpenEMS sample plot to {fig_path}")


def demo_physics_patterns():
    """Demonstrate the physics simulation functions."""
    steering_angle = jnp.array([jnp.pi / 6, jnp.pi / 4])  # 30°, 45°

    config = ArrayConfig()
    synthesize_ideal, compute_element_weights = make_physics_setup(config)
    weights, _ = compute_element_weights(steering_angle)
    ideal_pattern = synthesize_ideal(weights)

    floor_db = -60.0  # dB floor for clipping
    linear_floor = 10.0 ** (floor_db / 10.0)
    ideal_pattern = 10.0 * jnp.log10(jnp.maximum(ideal_pattern, linear_floor))

    fig = plt.figure(figsize=(15, 10), layout="compressed")
    subfigs = typing.cast(list[SubFigure], fig.subfigures(2, 1))

    plot_pattern(ideal_pattern, title="Ideal Pattern", clip_min_db=-10, fig=subfigs[0])

    steering_str = f"θ={np.degrees(steering_angle[0]):.1f}°, φ={np.degrees(steering_angle[1]):.1f}°"
    fig.suptitle(f"Ideal Patterns ({steering_str})")
    filename = "demo_physics.png"
    fig.savefig(filename, dpi=250)
    logger.info(f"Saved {filename}")


if __name__ == "__main__":
    setup_logging()
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        demo_phase_shifts()
        demo_openems_patterns()
        demo_physics_patterns()
