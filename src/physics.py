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


def get_wavenumber(
    freq_hz: float | None = None, config: ArrayConfig | None = None
) -> float:
    """Calculate the wavenumber for a given frequency in Hz."""
    if freq_hz is None:
        freq_hz = config.freq_hz if config is not None else ArrayConfig.freq_hz

    c = 299792458  # Speed of light in m/s
    wavelength = c / freq_hz  # Wavelength in meters
    k = 2 * np.pi / wavelength  # Wavenumber in radians/meter
    return k


@lru_cache(maxsize=1)
def get_element_positions(
    array_size: tuple[int, int] | None = None,
    spacing_mm: tuple[float, float] | None = None,
    config: ArrayConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate element positions using config or explicit parameters."""
    if config is not None:
        array_size = config.array_size
        spacing_mm = config.spacing_mm
    elif array_size is None or spacing_mm is None:
        # Use defaults from ArrayConfig
        array_size = ArrayConfig.array_size
        spacing_mm = ArrayConfig.spacing_mm

    xn, yn = array_size
    dx_mm, dy_mm = spacing_mm
    # Calculate the x and y positions of the elements in the array in meters, centered around (0, 0)
    x_positions = (np.arange(xn) - (xn - 1) / 2) * dx_mm / 1000
    y_positions = (np.arange(yn) - (yn - 1) / 2) * dy_mm / 1000
    return x_positions, y_positions


root_dir = Path(__file__).parent.parent

DEFAULT_SIM_DIR = root_dir / "openems" / "sim" / "antenna_array"
DEFAULT_SINGLE_ANT_FILENAME = "ff_1x1_60x60_2450_steer_t0_p0.h5"
DEFAULT_SIM_PATH = DEFAULT_SIM_DIR / DEFAULT_SINGLE_ANT_FILENAME


@jax.jit
def calculate_weights(
    kx: ArrayLike,
    ky: ArrayLike,
    steering_angle_rad: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """Calculates analytical weights for given steering angles."""
    steering_angle_rad = jnp.atleast_2d(steering_angle_rad)
    theta_steer, phi_steer = steering_angle_rad[:, 0], steering_angle_rad[:, 1]

    sin_theta = jnp.sin(theta_steer)
    ux = sin_theta * jnp.cos(phi_steer)  # (n_beams,)
    uy = sin_theta * jnp.sin(phi_steer)  # (n_beams,)

    x_phase, y_phase = jnp.outer(kx, ux), jnp.outer(ky, uy)  # (n_x, n_beams)
    phase_shifts = x_phase[:, None, :] + y_phase[None, :, :]  # (n_x, n_y, n_beams)

    weights = jnp.exp(-1j * phase_shifts)  # (n_x, n_y, n_beams)
    weights = jnp.sum(weights, axis=-1)  # (n_x, n_y), assume no tapering

    return weights, phase_shifts


def create_analytical_weight_calculator(config: ArrayConfig | None = None) -> Callable:
    """Factory to create a function for calculating analytical weights."""
    config = config or ArrayConfig()

    k = get_wavenumber(config=config)
    x_pos, y_pos = get_element_positions(config=config)
    kx, ky = k * x_pos, k * y_pos
    kx, ky = jnp.asarray(kx), jnp.asarray(ky)

    calculate = partial(calculate_weights, kx, ky)
    return calculate


@jax.jit
def synthesize_pattern(element_field_basis: ArrayLike, weights: ArrayLike) -> jax.Array:
    """Synthesizes a pattern from weights using the precomputed basis."""
    total_field = jnp.einsum("xy,tpzxy->tpz", weights, element_field_basis)
    power_pattern = jnp.sum(jnp.abs(total_field) ** 2, axis=-1)
    return power_pattern


def create_pattern_synthesizer(
    element_patterns: jax.Array,
    config: ArrayConfig,
) -> Callable:
    """Factory to create a pattern synthesis function."""

    k = get_wavenumber(config=config)
    x_pos, y_pos = get_element_positions(config=config)
    k_pos_x, k_pos_y = k * x_pos, k * y_pos

    sin_theta = jnp.sin(config.theta_rad)
    ux = sin_theta[:, None] * jnp.cos(config.phi_rad)[None, :]
    uy = sin_theta[:, None] * jnp.sin(config.phi_rad)[None, :]

    phase_x, phase_y = ux[..., None] * k_pos_x, uy[..., None] * k_pos_y
    geo_phase = phase_x[..., None] + phase_y[:, :, None, :]
    geo_factor = jnp.exp(1j * geo_phase)

    element_field_basis = jnp.einsum("xytpz,tpxy->tpzxy", element_patterns, geo_factor)

    synthesize = partial(synthesize_pattern, element_field_basis)
    return synthesize


def create_element_patterns(
    config: ArrayConfig,
    openems_path: Path | None = None,
) -> jax.Array:
    """Simulates element patterns for ideal array, with optional OpenEMS data."""
    if openems_path is not None:
        E_field = load_openems_nf2ff(openems_path).E_field  # (n_theta, n_phi, 2)

        reps = config.array_size + (1,) * E_field.ndim
        E_field = jnp.tile(E_field, reps)  # (n_x, n_y, n_theta, n_phi, n_polarization)
        return E_field

    # Use existing synthetic generation (unchanged)
    theta_size, phi_size = config.theta_rad.size, config.phi_rad.size

    # Base cosine model for field amplitude
    amplitude = jnp.cos(config.theta_rad)
    amplitude = amplitude.at[config.theta_rad > np.pi / 2].set(0)
    amplitude = amplitude[:, None] * jnp.ones((theta_size, phi_size))

    E_field = amplitude[:, :, None]  # (n_theta, n_phi, 1)
    reps = config.array_size + (1,) * E_field.ndim
    E_field = jnp.tile(E_field, reps)  # (n_x, n_y, n_theta, n_phi, n_polarization)
    return E_field.astype(jnp.complex64)


def create_physics_setup(
    config: ArrayConfig,
    openems_path: Path | None = None,
) -> tuple[Callable, Callable]:
    """Creates the physics simulation setup with optional OpenEMS data support."""

    element_patterns = create_element_patterns(config, openems_path=openems_path)
    synthesize_pattern = create_pattern_synthesizer(element_patterns, config)

    compute_analytical = create_analytical_weight_calculator(config)

    return synthesize_pattern, compute_analytical


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
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))

    theta_deg, phi_deg = np.rad2deg(theta_rad), np.rad2deg(phi_rad)
    extent = (np.min(theta_deg), np.max(theta_deg), np.min(phi_deg), np.max(phi_deg))
    aspect = theta_deg.size / phi_deg.size
    im = ax.imshow(pattern, extent=extent, origin="lower", aspect=aspect)
    ax.set_xlabel("θ°")
    ax.set_ylabel("φ°")
    ax.set_title(title)

    if colorbar:
        ax.figure.colorbar(im, fraction=0.046, pad=0.04, label="Normalized Dbi")


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
    fig: plt.Figure | SubFigure | None = None,
):
    if theta_rad is None:
        theta_rad = ArrayConfig.theta_rad
    if phi_rad is None:
        phi_rad = ArrayConfig.phi_rad
    if fig is None:
        fig = plt.figure(figsize=(15, 5), layout="compressed")
    if title is not None:
        fig.suptitle(title)
    axd = fig.subplot_mosaic("ABC", per_subplot_kw={"C": {"projection": "3d"}})
    plot_ff_2d(theta_rad, phi_rad, pattern, ax=axd["A"])
    plot_sine_space(theta_rad, phi_rad, pattern, ax=axd["B"])
    plot_ff_3d(theta_rad, phi_rad, pattern, clip_min_db=clip_min_db, ax=axd["C"])


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
    compute_analytical = create_analytical_weight_calculator(config)

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

    # Use new analytical calculator
    phase_shifts_list = []
    for angle in steering_angles:
        _, phase_shifts = compute_analytical(np.radians(angle))
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

    synthesize_ideal, compute_analytical = create_physics_setup(
        config, openems_path=DEFAULT_SIM_PATH
    )

    steering_deg = jnp.array([30, 45])
    steering_angle = jnp.radians(steering_deg)
    weights, _ = compute_analytical(steering_angle)
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
    synthesize_ideal, compute_analytical = create_physics_setup(config)
    weights, _ = compute_analytical(steering_angle)
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
