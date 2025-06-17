import logging
import typing
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

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

    @jax.jit
    def precompute_basis(raw_patterns):
        k = get_wavenumber(config=config)
        x_pos, y_pos = get_element_positions(config=config)
        k_pos_x, k_pos_y = k * x_pos, k * y_pos

        sin_theta = jnp.sin(config.theta_rad)
        ux = sin_theta[:, None] * jnp.cos(config.phi_rad)[None, :]
        uy = sin_theta[:, None] * jnp.sin(config.phi_rad)[None, :]

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


def add_embedding_effects(base_patterns: jax.Array, key: jax.Array) -> jax.Array:
    """Add realistic element-to-element variations to OpenEMS base patterns."""
    # Apply small random amplitude/phase distortions to simulate embedding effects
    key, amp_key, phase_key = jax.random.split(key, 3)

    # Small variations (5-10%) to simulate coupling between elements
    amp_variations = 1.0 + 0.05 * jax.random.normal(
        amp_key, base_patterns.shape[:2] + (1, 1, 1)
    )
    phase_variations = 0.1 * jax.random.normal(
        phase_key, base_patterns.shape[:2] + (1, 1, 1)
    )

    return base_patterns * amp_variations * jnp.exp(1j * phase_variations)


def create_element_patterns(
    config: ArrayConfig,
    key: jax.Array,
    is_embedded: bool,
    openems_path: Path | None = None,
) -> jax.Array:
    """Simulates element patterns for either ideal or embedded array, with optional OpenEMS data."""
    if openems_path is not None:
        # Load OpenEMS data and convert to unified format
        openems_data = load_openems_nf2ff(openems_path)
        single_element = openems_data.E_field  # Shape: (n_theta, n_phi, 2)

        # Tile across array positions to create 5D format: (array_x, array_y, n_theta, n_phi, n_polarizations)
        base_patterns = jnp.tile(
            single_element[None, None, ...], (*config.array_size, 1, 1, 1)
        )

        if is_embedded:
            # Add element-specific distortions to simulate coupling
            return add_embedding_effects(base_patterns, key)
        else:
            return base_patterns
    else:
        # Use existing synthetic generation (unchanged)
        theta_size, phi_size = config.theta_rad.size, config.phi_rad.size

        # Base cosine model for field amplitude
        base_field_amp = jnp.cos(config.theta_rad)
        base_field_amp = base_field_amp.at[config.theta_rad > np.pi / 2].set(0)
        base_field_amp = base_field_amp[:, None] * jnp.ones((theta_size, phi_size))

        if not is_embedded:
            ideal_field = base_field_amp[None, None, :, :, None]
            ideal_field = jnp.tile(ideal_field, (*config.array_size, 1, 1, 1))
            return ideal_field.astype(jnp.complex64)

        # Embedded case: simulate distortion
        num_pols = 2
        final_shape = (*config.array_size, theta_size, phi_size, num_pols)
        low_res_shape = (*config.array_size, 10, 20, num_pols)

        key, amp_key, phase_key = jax.random.split(key, 3)

        amp_dist_low_res = jax.random.uniform(
            amp_key, low_res_shape, minval=0.5, maxval=1.5
        )
        amp_distortion = jax.image.resize(
            amp_dist_low_res, final_shape, method="bicubic"
        )

        phase_dist_low_res = jax.random.uniform(
            phase_key, low_res_shape, maxval=2 * np.pi
        )
        phase_distortion = jax.image.resize(
            phase_dist_low_res, final_shape, method="bicubic"
        )

        distorted_amplitude = base_field_amp[None, None, ..., None] * amp_distortion
        distorted_field = distorted_amplitude * jnp.exp(1j * phase_distortion)

        return distorted_field.astype(jnp.complex64)


def create_physics_setup(
    key: jax.Array,
    config: ArrayConfig | None = None,
    openems_path: Path | None = None,
):
    """Creates the physics simulation setup with optional OpenEMS data support."""
    config = config or ArrayConfig()

    key, ideal_key, embedded_key = jax.random.split(key, 3)

    # Always use synthetic for ideal patterns (clean reference)
    ideal_patterns = create_element_patterns(
        config, ideal_key, is_embedded=False, openems_path=openems_path
    )

    # Use OpenEMS or synthetic for embedded patterns based on parameter
    embedded_patterns = create_element_patterns(
        config, embedded_key, is_embedded=True, openems_path=openems_path
    )

    synthesize_ideal = create_pattern_synthesizer(ideal_patterns, config)
    synthesize_embedded = create_pattern_synthesizer(embedded_patterns, config)
    compute_analytical = create_analytical_weight_calculator(config)

    return synthesize_ideal, synthesize_embedded, compute_analytical


@jax.jit
def normalize_patterns(patterns: ArrayLike) -> jax.Array:
    """Performs peak normalization on a batch of radiation patterns (linear scale)."""
    max_vals = jnp.max(patterns, axis=(1, 2), keepdims=True)
    return patterns / (max_vals + 1e-8)


@jax.jit
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
    clip_min_db: float | None = None,
    elev: float | None = None,
    azim: float | None = None,
    title: str = "3D Radiation Pattern",
    ax: Axes3D | None = None,
):
    pattern = np.clip(pattern, min=clip_min_db)  # Clip to minimum dB value
    pattern = (pattern - pattern.min()) / np.ptp(pattern)  # Normalize to [0, 1]
    pattern = pattern * 2  # Scale pattern for visualization

    # Calculate cartesian coordinates using the correctly scaled radius
    x = pattern * np.sin(theta_rad)[:, None] * np.cos(phi_rad)[None, :]
    y = pattern * np.sin(theta_rad)[:, None] * np.sin(phi_rad)[None, :]
    z = pattern * np.cos(theta_rad)[:, None]

    if ax is None:
        fig = plt.figure(layout="compressed")
        ax = typing.cast(Axes3D, fig.add_subplot(projection="3d"))

    ax.plot_surface(x, y, z, cmap="Spectral_r")
    ax.view_init(elev=elev, azim=azim)
    # ax.set_box_aspect(None, zoom=1.2)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 1.8), xticks=[], yticks=[], zticks=[])
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
    key = jax.random.key(42)
    config = ArrayConfig()

    synthesize_ideal, _, compute_analytical = create_physics_setup(
        key, config, openems_path=DEFAULT_SIM_PATH
    )

    # Test steering angle
    steering_deg = jnp.array([30, 45])
    steering_angle = jnp.radians(steering_deg)
    weights, _ = compute_analytical(steering_angle)
    power_pattern = synthesize_ideal(weights)
    power_dB = convert_to_db(power_pattern)

    theta_rad, phi_rad = config.theta_rad, config.phi_rad

    fig = plt.figure(figsize=(15, 5), layout="compressed")
    axd = fig.subplot_mosaic("ABC", per_subplot_kw={"C": {"projection": "3d"}})

    plot_ff_2d(theta_rad, phi_rad, power_dB, ax=axd["A"])
    plot_sine_space(theta_rad, phi_rad, power_dB, ax=axd["B"])
    plot_ff_3d(theta_rad, phi_rad, power_dB, clip_min_db=-30, ax=axd["C"])

    steering_str = f"θ={np.degrees(steering_angle[0]):.1f}°, φ={np.degrees(steering_angle[1]):.1f}°"
    phase_shift_title = f"OpenEMS Radiation Pattern ({steering_str})"
    fig.suptitle(phase_shift_title)

    fig_path = "test_openems.png"
    fig.savefig(fig_path, dpi=250)
    logger.info(f"Saved OpenEMS sample plot to {fig_path}")


def demo_simple_patterns():
    """Create simple synthetic patterns for demonstration."""
    # Create simple test patterns
    theta_rad = np.linspace(0, np.pi, 180)
    phi_rad = np.linspace(0, 2 * np.pi, 360)

    # Pattern 1: Simple cosine pattern
    theta_grid, phi_grid = np.meshgrid(theta_rad, phi_rad, indexing="ij")
    pattern1 = 20 * np.cos(theta_grid) ** 2
    pattern1[theta_grid > np.pi / 2] = 0  # Set values beyond 90° to 0

    # Pattern 2: Directional pattern
    pattern2 = 25 * np.cos(theta_grid) ** 4 * (1 + 0.5 * np.cos(2 * phi_grid))
    pattern2[theta_grid > np.pi / 2] = 0  # Set values beyond 90° to 0

    patterns = [pattern1, pattern2]
    titles = ["Cosine² Pattern", "Directional Pattern"]

    for i, (pattern, title) in enumerate(zip(patterns, titles)):
        fig = plt.figure(figsize=(15, 5), layout="compressed")
        axd = fig.subplot_mosaic("ABC", per_subplot_kw={"C": {"projection": "3d"}})

        power_dB = convert_to_db(pattern)
        plot_ff_2d(theta_rad, phi_rad, power_dB, ax=axd["A"])
        plot_sine_space(theta_rad, phi_rad, power_dB, ax=axd["B"])
        plot_ff_3d(theta_rad, phi_rad, power_dB, clip_min_db=-30, ax=axd["C"])

        fig.suptitle(title)
        filename = f"demo_pattern_{i + 1}.png"
        fig.savefig(filename, dpi=250)
        logger.info(f"Saved {filename}")


def demo_physics_patterns():
    """Demonstrate the physics simulation functions."""
    steering_angle = jnp.array([jnp.pi / 6, jnp.pi / 4])  # 30°, 45°

    key = jax.random.key(42)
    synthesize_ideal, synthesize_embedded, compute_analytical = create_physics_setup(
        key
    )
    weights, phase_shifts = compute_analytical(steering_angle)
    ideal_pattern = synthesize_ideal(weights)
    embedded_pattern = synthesize_embedded(weights)

    floor_db = -60.0  # dB floor for clipping
    linear_floor = 10.0 ** (floor_db / 10.0)
    ideal_pattern = 10.0 * jnp.log10(jnp.maximum(ideal_pattern, linear_floor))
    embedded_pattern = 10.0 * jnp.log10(jnp.maximum(embedded_pattern, linear_floor))

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
        demo_simple_patterns()
        demo_openems_patterns()
        demo_physics_patterns()
