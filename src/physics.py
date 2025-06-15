import logging
import typing
from collections.abc import Callable
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
from matplotlib.image import AxesImage
from matplotlib.projections import PolarAxes
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger(__name__)

plt.rcParams["figure.constrained_layout.use"] = True


class ArrayConfig:
    """Configuration for antenna array parameters and simulation settings."""

    array_size: tuple[int, int] = (16, 16)
    spacing_mm: tuple[float, float] = (60.0, 60.0)
    freq_hz: float = 2.45e9
    pattern_shape: tuple[int, int] = (180, 360)  # (theta, phi)


@dataclass(frozen=True)
class OpenEMSData:
    theta_rad: jax.Array
    phi_rad: jax.Array
    r: jax.Array
    Dmax: float
    freq_hz: jax.Array
    E_field: jax.Array
    E_norm: jax.Array


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
        E_norm = np.sqrt(np.abs(E_theta) ** 2 + np.abs(E_phi) ** 2)

    return OpenEMSData(theta_rad, phi_rad, r, Dmax, freq_hz, E_field, E_norm)


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

DEFAULT_DATASET_DIR = root_dir / "dataset"
DEFAULT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DATASET_NAME = "farfield_dataset.h5"


def check_grating_lobes(freq=None, spacing_mm=None, config=None, verbose=False):
    """Check for potential grating lobes in an antenna array based on element spacing."""
    if config is not None:
        freq = config.freq_hz
        spacing_mm = config.spacing_mm
    elif freq is None or spacing_mm is None:
        freq = ArrayConfig.freq_hz if freq is None else freq
        spacing_mm = ArrayConfig.spacing_mm if spacing_mm is None else spacing_mm

    c = 299792458
    wavelength = c / freq
    wavelength_mm = wavelength * 1000

    dx, dy = spacing_mm
    dx_lambda = dx / wavelength_mm
    dy_lambda = dy / wavelength_mm

    if dx_lambda <= 0.5:
        dx_critical = 90
    else:
        dx_critical = np.rad2deg(np.arcsin(1 / dx_lambda - 1))

    if dy_lambda <= 0.5:
        dy_critical = 90
    else:
        dy_critical = np.rad2deg(np.arcsin(1 / dy_lambda - 1))

    dx_critical_angle = dx_critical if dx_lambda > 0.5 else None
    dy_critical_angle = dy_critical if dy_lambda > 0.5 else None
    has_grating_lobes = dx_lambda > 0.5 or dy_lambda > 0.5

    if verbose:
        logger.info("Array spacing check:")
        logger.info(f"Wavelength: {wavelength_mm:.2f} mm")
        logger.info(f"Element spacing: {dx_lambda:.1f}λ x {dy_lambda:.1f}λ")

    if has_grating_lobes:
        logger.info("WARNING: Grating lobes will be visible when steering beyond:")
        if dx_critical_angle is not None:
            logger.info(f"  - {dx_critical_angle:.1f}° in the X direction")
        if dy_critical_angle is not None:
            logger.info(f"  - {dy_critical_angle:.1f}° in the Y direction")


def calc_array_params(
    array_size: tuple[int, int] = ArrayConfig.array_size,
    spacing_mm: tuple[float, float] = ArrayConfig.spacing_mm,
    *,
    theta_rad: np.ndarray | None = None,
    phi_rad: np.ndarray | None = None,
    sim_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, jax.Array, jax.Array]:
    """Calculate array parameters using explicit parameters or defaults."""
    theta_rad = theta_rad if theta_rad is not None else np.radians(np.arange(180))
    phi_rad = phi_rad if phi_rad is not None else np.radians(np.arange(360))
    sim_path = sim_path or DEFAULT_SIM_PATH

    nf2ff = load_openems_nf2ff(sim_path)
    E_field, Dmax, freq_hz = nf2ff.E_field, nf2ff.Dmax, nf2ff.freq_hz
    E_field = E_field[: theta_rad.size, ...]
    Dmax_array = Dmax * np.prod(array_size)

    check_grating_lobes(freq_hz, spacing_mm)

    k = get_wavenumber(freq_hz)
    x_pos, y_pos = get_element_positions(array_size, spacing_mm)
    kx, ky = k * x_pos, k * y_pos

    geo_exp = calc_geo_exp(theta_rad, phi_rad, kx, ky)

    precomputed = jnp.einsum("tpc,tpxy->tpcxy", E_field, geo_exp)
    precomputed = precomputed / np.prod(array_size)

    taper = calc_taper(array_size)
    return kx, ky, taper, precomputed, Dmax_array


@lru_cache(maxsize=1)
def calc_taper(
    array_size: tuple[int, int] = (16, 16),
    taper_type: Literal["uniform", "hamming", "taylor"] = "uniform",
) -> np.ndarray:
    if taper_type == "hamming":
        window_func = np.hamming
    elif taper_type == "taylor":
        # Simple approximation of Taylor window using Kaiser
        window_func = partial(np.kaiser, beta=3)
    else:
        window_func = np.ones  # Uniform taper

    window_x, window_y = window_func(array_size[0]), window_func(array_size[1])
    taper = np.outer(window_x, window_y)  # Multiply the 1D windows
    return taper.astype(np.complex64)


@jax.jit
def calc_phase_shifts(
    kx: ArrayLike,
    ky: ArrayLike,
    steering_rad: ArrayLike,
) -> jax.Array:
    """Calculate phase shifts for each element in the array based on steering angles."""
    theta_steering, phi_steering = steering_rad.T

    sin_theta = jnp.sin(theta_steering)
    ux = sin_theta * jnp.cos(phi_steering)
    uy = sin_theta * jnp.sin(phi_steering)

    x_phase = kx[:, None] * ux[None, :]
    y_phase = ky[:, None] * uy[None, :]

    phase_shifts = x_phase[:, None, :] + y_phase[None, :, :]
    phase_shifts = phase_shifts % (2 * jnp.pi)

    return phase_shifts


@jax.jit
def calc_geo_exp(
    theta_rad: ArrayLike,
    phi_rad: ArrayLike,
    kx: ArrayLike,  # Wavenumber-scaled x-positions of array elements
    ky: ArrayLike,  # Wavenumber-scaled y-positions of array elements
) -> jax.Array:
    """
    Calculate the geometric phase terms for the array factor calculation.
    """
    # x and y components of the geometric phase: (len(theta_rad), len(phi_rad))
    sin_theta = jnp.sin(theta_rad)[:, None]
    ux = sin_theta * jnp.cos(phi_rad)
    uy = sin_theta * jnp.sin(phi_rad)

    # Geometric phase terms for each element and angle: (len(theta), len(phi), xn, yn)
    x_geo_phase = kx[None, None, :, None] * ux[:, :, None, None]
    y_geo_phase = ky[None, None, None, :] * uy[:, :, None, None]

    geo_phase = x_geo_phase + y_geo_phase  # Geometric phase terms
    geo_exp = jnp.exp(1j * geo_phase)  # Complex exponential of the geometric phases

    geo_exp = jnp.asarray(geo_exp)  # Convert to JAX array for JIT compilation

    return geo_exp


def normalize_rad_pattern(pattern: ArrayLike, Dmax: float) -> jax.Array:
    """
    Normalize the radiation pattern to dBi.
    """
    pattern = pattern / jnp.max(jnp.abs(pattern))
    pattern = 20 * jnp.log10(jnp.abs(pattern)) + 10.0 * jnp.log10(Dmax)
    return pattern


@jax.jit
def rad_pattern_from_geo_and_excitations(
    precomputed: ArrayLike,
    Dmax_array: float,
    w: ArrayLike,
) -> jax.Array:
    E_total = jnp.einsum("xy,tpcxy->tpc", w, precomputed)

    E_total = jnp.abs(E_total) ** 2
    E_total = jnp.sum(E_total, axis=-1)
    E_total = jnp.sqrt(E_total)

    E_norm = normalize_rad_pattern(E_total, Dmax_array)
    return E_norm


@jax.jit
def rad_pattern_from_geo(
    kx: ArrayLike,
    ky: ArrayLike,
    taper: ArrayLike,
    precomputed: ArrayLike,
    Dmax_array: ArrayLike,
    steering_rad: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute radiation pattern for given steering angles.
    """
    phase_shifts = calc_phase_shifts(kx, ky, steering_rad)

    excitations = jnp.einsum("xy,xys->xy", taper, jnp.exp(-1j * phase_shifts))

    E_norm = rad_pattern_from_geo_and_excitations(precomputed, Dmax_array, excitations)
    return E_norm, excitations


def precompute_array_contributions(
    element_patterns: jax.Array,  # Shape: (16, 16, n_theta, n_phi) - your simulated patterns
    kx: jax.Array,  # From your existing calc_array_params
    ky: jax.Array,  # From your existing calc_array_params
    theta_rad: jax.Array,  # From your dataset
    phi_rad: jax.Array,  # From your dataset
) -> jax.Array:
    """
    Precompute element_patterns * exp(1j * geometric_phase) for all elements.
    Reuses your existing kx, ky arrays.

    Returns:
        Shape (16, 16, n_theta, n_phi) - precomputed element contributions
    """
    # Direction cosines (reusing your existing approach)
    sin_theta = jnp.sin(theta_rad)
    cos_phi = jnp.cos(phi_rad)
    sin_phi = jnp.sin(phi_rad)

    u = sin_theta[:, None] * cos_phi[None, :]  # (n_theta, n_phi)
    v = sin_theta[:, None] * sin_phi[None, :]  # (n_theta, n_phi)

    # Geometric phase using your existing kx, ky
    x_phase = kx[:, None, None] * u[None, :, :]  # (16, n_theta, n_phi)
    y_phase = ky[:, None, None] * v[None, :, :]  # (16, n_theta, n_phi)

    # (16, 16, n_theta, n_phi)
    geometric_phase = x_phase[:, None, :, :] + y_phase[None, :, :, :]

    # Combine with element patterns
    return element_patterns * jnp.exp(1j * geometric_phase)


@jax.jit
def array_synthesis(
    precomputed_contributions: jax.Array,  # From precompute_array_contributions
    excitations: jax.Array,  # Shape: (16, 16) - complex excitations
) -> jax.Array:
    """
    Synthesize array pattern from precomputed contributions and excitations.

    Returns:
        Shape: (n_theta, n_phi) - power pattern
    """
    total_field = jnp.sum(
        excitations[:, :, None, None] * precomputed_contributions, axis=(0, 1)
    )

    return jnp.abs(total_field) ** 2


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
    E_norm,
    Dmax,
    fmt: str = "r-",
    *,
    normalize: bool = False,
    label: str | None = None,
    title: str | None = None,
    ax: PolarAxes | None = None,
    filename: str | None = None,
):
    if ax is None:
        fig = plt.figure(constrained_layout=True)
        ax = typing.cast(PolarAxes, fig.add_subplot(projection="polar"))

    E_norm_cut = extract_E_plane_cut(E_norm, phi_idx=0)
    theta_rad = np.linspace(0, 2 * np.pi, E_norm_cut.size)

    if normalize:
        E_norm_cut = normalize_rad_pattern(E_norm_cut, Dmax)

    ax.plot(theta_rad, E_norm_cut, fmt, linewidth=1, label=label)
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
            fig.savefig(filename, dpi=600)


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
    hide_backlobe: bool = True,
    elev: float | None = None,
    azim: float | None = None,
    title: str = "3D Radiation Pattern",
    ax: Axes3D | None = None,
):
    pattern = pattern.clip(min=0)  # Clip negative values to 0

    # Calculate cartesian coordinates
    x = pattern * np.sin(theta_rad)[:, None] * np.cos(phi_rad)[None, :]
    y = pattern * np.sin(theta_rad)[:, None] * np.sin(phi_rad)[None, :]
    z = pattern * np.cos(theta_rad)[:, None]

    if hide_backlobe:
        z = np.array(z)  # Ensure z is a NumPy array and thus writable
        z[z < 0] = np.nan  # Set backlobe values to NaN

    if ax is None:
        fig = plt.figure(constrained_layout=True)
        ax = typing.cast(Axes3D, fig.add_subplot(projection="3d"))

    ax.plot_surface(x, y, z, cmap="Spectral_r")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(None, zoom=1.2)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_zlim(0, 30)
    ax.set_title(title)


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
    im = ax.contourf(u, v, pattern, levels=128, cmap="magma_r")

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


def test_plot_ff_3d():
    steering_deg = np.array([[15, 15], [30, 120], [45, 210]])

    array_params = calc_array_params(array_size=(16, 16), spacing_mm=(60, 60))
    E_norm, _ = rad_pattern_from_geo(*array_params, np.radians(steering_deg))
    E_norm = np.asarray(E_norm)

    theta_rad, phi_rad = np.radians(np.arange(180)), np.radians(np.arange(360))
    fig, axs = plt.subplots(1, 3, figsize=[18, 6], layout="constrained")

    plot_ff_2d(theta_rad, phi_rad, E_norm, ax=axs[0])
    plot_sine_space(theta_rad, phi_rad, E_norm, ax=axs[1])

    axs[2].remove()
    axs[2] = fig.add_subplot(1, 3, 3, projection="3d")
    plot_ff_3d(theta_rad, phi_rad, E_norm, ax=axs[2])

    steering_str = steering_repr(steering_deg.T)
    phase_shift_title = f"Radiation Pattern with Phase Shifts {steering_str}"
    fig.suptitle(phase_shift_title)

    fig_path = "test.png"
    fig.savefig(fig_path, dpi=600)
    logger.info(f"Saved sample plot to {fig_path}")


def create_analytical_weight_calculator(config: ArrayConfig | None = None) -> Callable:
    """Factory to create a function for calculating analytical weights."""
    config = config or ArrayConfig()

    k = get_wavenumber(config=config)
    x_pos, y_pos = get_element_positions(config=config)
    k_pos_x, k_pos_y = k * x_pos, k * y_pos

    @jax.jit
    def calculate(steering_angle_rad: ArrayLike) -> tuple[jax.Array, jax.Array]:
        """Calculates ideal, analytical weights for a given steering angle."""
        theta_steer, phi_steer = steering_angle_rad[0], steering_angle_rad[1]

        ux = jnp.sin(theta_steer) * jnp.cos(phi_steer)
        uy = jnp.sin(theta_steer) * jnp.sin(phi_steer)

        x_phase, y_phase = k_pos_x * ux, k_pos_y * uy
        phase_shifts = jnp.add.outer(x_phase, y_phase)

        return jnp.exp(-1j * phase_shifts), phase_shifts

    return calculate


def create_pattern_synthesizer(
    element_patterns: jax.Array,
    config: ArrayConfig,
) -> Callable:
    """Factory to create a pattern synthesis function."""
    config = config or ArrayConfig()

    @jax.jit
    def precompute_basis(raw_patterns):
        k = get_wavenumber(config=config)
        x_pos, y_pos = get_element_positions(config=config)
        k_pos_x, k_pos_y = k * x_pos, k * y_pos

        theta_size, phi_size = config.pattern_shape
        theta_rad = jnp.radians(jnp.arange(theta_size))
        phi_rad = jnp.radians(jnp.arange(phi_size))

        sin_theta = jnp.sin(theta_rad)
        ux = sin_theta[:, None] * jnp.cos(phi_rad)[None, :]
        uy = sin_theta[:, None] * jnp.sin(phi_rad)[None, :]

        phase_x, phase_y = ux[..., None] * k_pos_x, uy[..., None] * k_pos_y
        geo_phase = phase_x[..., None] + phase_y[:, :, None, :]
        geo_factor = jnp.exp(1j * geo_phase)

        return jnp.einsum("xytpz,tpxy->tpzxy", raw_patterns, geo_factor)

    element_field_basis = precompute_basis(element_patterns)

    @jax.jit
    def synthesize(weights: ArrayLike) -> jax.Array:
        """Synthesizes a pattern from weights using the precomputed basis."""
        total_field = jnp.einsum("xy,tpzxy->tpz", weights, element_field_basis)
        power_pattern = jnp.sum(jnp.abs(total_field) ** 2, axis=-1)
        return power_pattern

    return synthesize


def create_element_patterns(
    config: ArrayConfig, key: jax.Array, is_embedded: bool
) -> jax.Array:
    """Simulates element patterns for either ideal or embedded array."""
    theta_size, phi_size = config.pattern_shape
    theta = jnp.radians(jnp.arange(theta_size))

    # Base cosine model for field amplitude
    base_field_amp = jnp.cos(theta)
    base_field_amp = base_field_amp.at[theta > np.pi / 2].set(0)
    base_field_amp = base_field_amp[:, None] * jnp.ones((theta_size, phi_size))

    if not is_embedded:
        ideal_field = base_field_amp[None, None, :, :, None]
        ideal_field = jnp.tile(ideal_field, (*config.array_size, 1, 1, 1))
        return ideal_field.astype(jnp.complex64)

    # Embedded case: simulate distortion
    num_pols = 2
    final_shape = (*config.array_size, *config.pattern_shape, num_pols)
    low_res_shape = (*config.array_size, 10, 20, num_pols)

    key, amp_key, phase_key = jax.random.split(key, 3)

    amp_dist_low_res = jax.random.uniform(
        amp_key, low_res_shape, minval=0.5, maxval=1.5
    )
    amp_distortion = jax.image.resize(amp_dist_low_res, final_shape, method="bicubic")

    phase_dist_low_res = jax.random.uniform(phase_key, low_res_shape, maxval=2 * np.pi)
    phase_distortion = jax.image.resize(
        phase_dist_low_res, final_shape, method="bicubic"
    )

    distorted_amplitude = base_field_amp[None, None, ..., None] * amp_distortion
    distorted_field = distorted_amplitude * jnp.exp(1j * phase_distortion)

    return distorted_field.astype(jnp.complex64)


def create_physics_setup(key: jax.Array, config: ArrayConfig | None = None):
    """Creates the physics simulation setup (patterns and synthesizers)."""
    config = config or ArrayConfig()

    key, ideal_key, embedded_key = jax.random.split(key, 3)

    ideal_patterns = create_element_patterns(config, ideal_key, is_embedded=False)
    embedded_patterns = create_element_patterns(config, embedded_key, is_embedded=True)

    synthesize_ideal = create_pattern_synthesizer(ideal_patterns, config)
    synthesize_embedded = create_pattern_synthesizer(embedded_patterns, config)
    compute_analytical = create_analytical_weight_calculator(config)

    return synthesize_ideal, synthesize_embedded, compute_analytical


def demo_phase_shifts():
    """Demonstrate phase shift calculations and visualization."""
    config = ArrayConfig()
    k = get_wavenumber(config=config)
    x_pos, y_pos = get_element_positions(config=config)
    kx, ky = k * x_pos, k * y_pos

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

    phase_shifts = calc_phase_shifts(kx, ky, np.radians(steering_angles))
    phase_shifts = phase_shifts.transpose(2, 0, 1)  # (n_steering, n_theta, n_phi)

    kw = dict(figsize=(15, 10), sharex=True, sharey=True, layout="compressed")
    fig, axes = plt.subplots(nrows, ncols, **kw)

    for i, ax in enumerate(axes.flat[: steering_angles.shape[0]]):
        title = f"θ={steering_angles[i][0]}°, φ={steering_angles[i][1]}°"
        im = plot_phase_shifts(phase_shifts[i], title=title, colorbar=False, ax=ax)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Degrees")
    fig.suptitle("Phase Shifts for Different Steering Angles")
    filename = "demo_phase_shifts.png"
    fig.savefig(filename, dpi=250)
    logger.info(f"Saved {filename}")


def demo_tapers():
    """Demonstrate different taper functions."""
    array_size = ArrayConfig().array_size
    taper_types = ["uniform", "hamming", "taylor"]

    kw = dict(figsize=(15, 5), sharex=True, sharey=True, layout="compressed")
    fig, axes = plt.subplots(1, len(taper_types), **kw)

    for i, taper_type in enumerate(taper_types):
        taper = calc_taper(array_size, taper_type)
        im = axes[i].imshow(np.abs(taper), cmap="viridis", origin="lower")
        axes[i].set_title(f"{taper_type.capitalize()} Taper")
        axes[i].set_xlabel("Element X")
        axes[i].set_ylabel("Element Y")

    fig.colorbar(im, ax=axes, label="Taper Amplitude")
    fig.suptitle("Different Array Taper Functions")
    filename = "demo_tapers.png"
    fig.savefig(filename, dpi=250)
    logger.info(f"Saved {filename}")


def demo_simple_patterns():
    """Create simple synthetic patterns for demonstration."""
    # Create simple test patterns
    theta_rad = np.linspace(0, np.pi, 180)
    phi_rad = np.linspace(0, 2 * np.pi, 360)

    # Pattern 1: Simple cosine pattern
    theta_grid, phi_grid = np.meshgrid(theta_rad, phi_rad, indexing="ij")
    pattern1 = 20 * np.cos(theta_grid) ** 2
    pattern1[theta_grid > np.pi / 2] = 0  # Set values beyond 90° to 0
    # pattern1 = 20 * np.log10(np.maximum(pattern1, 1e-4))
    # pattern1 = np.maximum(pattern1, -40)  # Floor at -40 dB
    # Convert to dB scale
    pattern1 = 20 * np.log10(np.maximum(pattern1, 1e-4)) + 20  # Add 20 dB offset

    # Pattern 2: Directional pattern
    pattern2 = 25 * np.cos(theta_grid) ** 4 * (1 + 0.5 * np.cos(2 * phi_grid))
    pattern2[theta_grid > np.pi / 2] = 0  # Set values beyond 90° to 0
    # pattern2 = np.maximum(pattern2, -40)

    patterns = [pattern1, pattern2]
    titles = ["Cosine² Pattern", "Directional Pattern"]

    for i, (pattern, title) in enumerate(zip(patterns, titles)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="compressed")
        plot_ff_2d(theta_rad, phi_rad, pattern, ax=axes[0])
        plot_sine_space(theta_rad, phi_rad, pattern, ax=axes[1])

        axes[2].remove()
        axes[2] = fig.add_subplot(1, 3, 3, projection="3d")
        plot_ff_3d(theta_rad, phi_rad, pattern, ax=axes[2])

        fig.suptitle(title)
        filename = f"demo_pattern_{i + 1}.png"
        fig.savefig(filename, dpi=250)
        logger.info(f"Saved {filename}")


def demo_physics_functions():
    """Demonstrate the physics simulation functions."""
    steering_angle = jnp.array([jnp.pi / 6, jnp.pi / 4])  # 30°, 45°

    key = jax.random.key(42)
    synthesize_ideal, synthesize_embedded, compute_analytical = create_physics_setup(
        key
    )
    weights, phase_shifts = compute_analytical(steering_angle)
    ideal_pattern = synthesize_ideal(weights)
    embedded_pattern = synthesize_embedded(weights)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), layout="compressed")

    theta_rad = jnp.linspace(0, jnp.pi, ideal_pattern.shape[0])
    phi_rad = jnp.linspace(0, 2 * jnp.pi, ideal_pattern.shape[1])

    # Ideal pattern plots
    title = "Ideal Pattern (2D)"
    plot_ff_2d(theta_rad, phi_rad, ideal_pattern, title=title, ax=axes[0, 0])
    title = "Ideal Pattern (Sine Space)"
    plot_sine_space(theta_rad, phi_rad, ideal_pattern, title=title, ax=axes[0, 1])
    axes[0, 2].remove()
    axes[0, 2] = fig.add_subplot(2, 3, 3, projection="3d")
    title = "Ideal Pattern (3D)"
    plot_ff_3d(theta_rad, phi_rad, ideal_pattern, title=title, ax=axes[0, 2])

    # Embedded pattern plots
    title = "Embedded Pattern (2D)"
    plot_ff_2d(theta_rad, phi_rad, embedded_pattern, title=title, ax=axes[1, 0])
    title = "Embedded Pattern (Sine Space)"
    plot_sine_space(theta_rad, phi_rad, embedded_pattern, title=title, ax=axes[1, 1])
    axes[1, 2].remove()
    axes[1, 2] = fig.add_subplot(2, 3, 6, projection="3d")
    title = "Embedded Pattern (3D)"
    plot_ff_3d(theta_rad, phi_rad, embedded_pattern, title=title, ax=axes[1, 2])

    fig.suptitle(
        f"Physics Demo: Ideal vs Embedded Patterns (θ={np.degrees(steering_angle[0]):.1f}°, φ={np.degrees(steering_angle[1]):.1f}°)"
    )
    filename = "demo_physics.png"
    fig.savefig(filename, dpi=250)
    logger.info(f"Saved {filename}")

    # Plot phase shifts
    fig, ax = plt.subplots(figsize=(8, 6), layout="compressed")
    plot_phase_shifts(
        phase_shifts,
        title=f"Analytical Phase Shifts (θ={np.degrees(steering_angle[0]):.1f}°, φ={np.degrees(steering_angle[1]):.1f}°)",
        ax=ax,
    )
    fig.savefig("demo_phase_shifts_analytical.png", dpi=250)
    logger.info("Saved demo_phase_shifts_analytical.png")


if __name__ == "__main__":
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        demo_phase_shifts()
        demo_tapers()
        # demo_simple_patterns()
        demo_physics_functions()
