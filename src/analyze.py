import logging
from functools import lru_cache, partial
from pathlib import Path
from typing import Literal

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from jax.typing import ArrayLike

logger = logging.getLogger(__name__)


@lru_cache
def read_nf2ff(nf2ff_path: Path):
    logger.info(f"Loading antenna pattern from {nf2ff_path}")
    with h5py.File(nf2ff_path, "r") as h5:
        mesh = h5["Mesh"]
        theta_rad, phi_rad, r = mesh["theta"][:], mesh["phi"][:], mesh["r"][:]

        Dmax = h5["nf2ff"].attrs["Dmax"]
        freqs = h5["nf2ff"].attrs["Frequency"]

        E_shape = freqs.size, phi_rad.size, theta_rad.size
        E_theta = np.empty(E_shape, dtype=complex)
        E_phi = np.empty(E_shape, dtype=complex)

        for freq in range(freqs.size):
            E_theta.real[freq] = h5[f"/nf2ff/E_theta/FD/f{freq}_real"][:]
            E_theta.imag[freq] = h5[f"/nf2ff/E_theta/FD/f{freq}_imag"][:]
            E_phi.real[freq] = h5[f"/nf2ff/E_phi/FD/f{freq}_real"][:]
            E_phi.imag[freq] = h5[f"/nf2ff/E_phi/FD/f{freq}_imag"][:]

        # Transpose to (freq, theta, phi)
        E_theta, E_phi = E_theta.transpose(0, 2, 1), E_phi.transpose(0, 2, 1)

        if freqs.size == 1:  # Squeeze the frequency dimension
            E_theta, E_phi = np.squeeze(E_theta), np.squeeze(E_phi)
            freqs = np.squeeze(freqs)

        E_field = np.stack([E_theta, E_phi], axis=0)  # (2, freq, theta, phi)
        E_norm = np.sqrt(np.abs(E_theta) ** 2 + np.abs(E_phi) ** 2)

    return {
        "theta_rad": theta_rad,
        "phi_rad": phi_rad,
        "r": r,
        "Dmax": Dmax,
        "freq": freqs,
        "E_field": E_field,
        "E_norm": E_norm,
    }


def get_wavenumber(freq_hz: float) -> float:
    """
    Calculate the wavenumber for a given frequency in Hz.
    The wavenumber is defined as k = 2π/λ, where λ is the wavelength.
    The wavelength is calculated as λ = c/f, where c is the speed of light.
    """
    c = 299792458  # Speed of light in m/s
    wavelength = c / freq_hz  # Wavelength in meters
    k = 2 * np.pi / wavelength  # Wavenumber in radians/meter
    return k


@lru_cache(maxsize=1)
def get_element_positions(
    array_size: tuple[int, int],
    spacing_mm: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    xn, yn = array_size
    dx_mm, dy_mm = spacing_mm
    # Calculate the x and y positions of the elements in the array in meters, centered around (0, 0)
    x_positions = (np.arange(xn) - (xn - 1) / 2) * dx_mm / 1000
    y_positions = (np.arange(yn) - (yn - 1) / 2) * dy_mm / 1000
    return x_positions, y_positions


DEFAULT_SIM_DIR = Path.cwd() / "src" / "sim" / "antenna_array"
DEFAULT_SINGLE_ANT_FILENAME = "ff_1x1_60x60_2450_steer_t0_p0.h5"
DEFAULT_SIM_PATH = DEFAULT_SIM_DIR / DEFAULT_SINGLE_ANT_FILENAME


def check_grating_lobes(freq, dx, dy, verbose=False):
    """Check for potential grating lobes in an antenna array based on element spacing."""

    # Calculate wavelength
    c = 299792458  # Speed of light in m/s
    wavelength = c / freq  # Wavelength in meters
    wavelength_mm = wavelength * 1000  # Wavelength in mm

    # Check spacing in terms of wavelength
    dx_lambda = dx / wavelength_mm
    dy_lambda = dy / wavelength_mm

    # Calculate critical angles where grating lobes start to appear
    # For visible grating lobes: d/λ > 1/(1+|sin(θ)|)
    if dx_lambda <= 0.5:
        dx_critical = 90  # No grating lobes for spacing <= λ/2
    else:
        dx_critical = np.rad2deg(np.arcsin(1 / dx_lambda - 1))

    if dy_lambda <= 0.5:
        dy_critical = 90  # No grating lobes for spacing <= λ/2
    else:
        dy_critical = np.rad2deg(np.arcsin(1 / dy_lambda - 1))

    dx_critical_angle = dx_critical if dx_lambda > 0.5 else None
    dy_critical_angle = dy_critical if dy_lambda > 0.5 else None
    has_grating_lobes = dx_lambda > 0.5 or dy_lambda > 0.5

    if verbose:
        logger.info("Array spacing check:")
        logger.info(f"Wavelength: {'wavelength_mm':.2f} mm")
        logger.info(f"Element spacing: {'dx_lambda':.1f}λ x {'dy_lambda':.1f}λ")

    if has_grating_lobes:
        logger.info("WARNING: Grating lobes will be visible when steering beyond:")
        if dx_critical_angle is not None:
            logger.info(f"  - {'dx_critical_angle':.1f}° in the X direction")
        if dy_critical_angle is not None:
            logger.info(f"  - {'dy_critical_angle':.1f}° in the Y direction")


def calc_array_params(
    array_size: tuple[int, int] = (16, 16),
    spacing_mm: tuple[float, float] = (60, 60),
    *,
    theta_rad: np.ndarray = np.radians(np.arange(180)),
    phi_rad: np.ndarray = np.radians(np.arange(360)),
    sim_path: Path = DEFAULT_SIM_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    nf2ff = read_nf2ff(sim_path)
    E_field, Dmax, freq_hz = nf2ff["E_field"], nf2ff["Dmax"], nf2ff["freq"]
    Dmax_array = Dmax * np.prod(array_size)  # Scale Dmax for the array size

    check_grating_lobes(freq_hz, *spacing_mm)

    k = get_wavenumber(freq_hz)
    x_pos, y_pos = get_element_positions(array_size, spacing_mm)
    kx, ky = k * x_pos, k * y_pos  # Wavenumber-scaled positions

    taper = calc_taper(array_size)
    geo_exp = calc_geo_exp(theta_rad, phi_rad, kx, ky)

    if theta_rad.size < E_field.shape[2]:
        E_field = E_field[:, : theta_rad.size, :]  # Trim E_field to match theta_rad

    # Assuming E_field has shape (2, theta, phi) for a single element,
    # we broadcast it to all elements.
    E_field = E_field[:, None, None, :, :]
    precomputed = E_field * geo_exp
    # Rearrange to (theta, phi, 2, xn, yn)
    # TODO: calculate it like that from the start
    precomputed = precomputed.transpose(3, 4, 0, 1, 2)

    # Normalize by the total number of elements.
    xn, yn = geo_exp.shape[:2]
    precomputed = precomputed / (xn * yn)

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
    kx: ArrayLike,  # Wavenumber-scaled x-positions of array elements
    ky: ArrayLike,  # Wavenumber-scaled y-positions of array elements
    steering_rad: ArrayLike,  # Steering angles in radians, (n_angles, 2) for (theta, phi)
) -> Array:
    """
    Calculate phase shifts for each element in the array based on steering angles.
    Computes the phase shifts for each element based on the steering angles
    and the positions of the elements in the array.
    """
    theta_steering, phi_steering = steering_rad.T  # (n_angles, 2)

    # Calculate steering vector components
    sin_theta = jnp.sin(theta_steering)
    ux = sin_theta * jnp.cos(phi_steering)
    uy = sin_theta * jnp.sin(phi_steering)

    x_phase = kx[:, None] * ux[None, :]  # (xn,), (n_angles,) -> (xn, n_angles)
    y_phase = ky[:, None] * uy[None, :]  # (yn,), (n_angles,) -> (yn, n_angles)

    # Combine phases: (xn, n_angles), (yn, n_angles) -> (xn, yn, n_angles)
    phase_shifts = x_phase[:, None, :] + y_phase[None, :, :]
    phase_shifts = phase_shifts % (2 * jnp.pi)  # Normalize to [0, 2π)

    return phase_shifts


@jax.jit
def calc_geo_exp(
    theta_rad: ArrayLike,
    phi_rad: ArrayLike,
    kx: ArrayLike,  # Wavenumber-scaled x-positions of array elements
    ky: ArrayLike,  # Wavenumber-scaled y-positions of array elements
) -> Array:
    """
    Calculate the geometric phase terms for the array factor calculation.
    """
    # x and y components of the geometric phase: (len(theta_rad), len(phi_rad))
    sin_theta = jnp.sin(theta_rad)[:, None]
    ux = sin_theta * jnp.cos(phi_rad)
    uy = sin_theta * jnp.sin(phi_rad)

    # Geometric phase terms for each element and angle: (xn, yn, len(theta), len(phi))
    x_geo_phase = kx[:, None, None, None] * ux[None, None, :, :]
    y_geo_phase = ky[None, :, None, None] * uy[None, None, :, :]

    geo_phase = x_geo_phase + y_geo_phase  # Geometric phase terms
    geo_exp = jnp.exp(1j * geo_phase)  # Complex exponential of the geometric phases

    geo_exp = jnp.asarray(geo_exp)  # Convert to JAX array for JIT compilation

    return geo_exp


def normalize_rad_pattern(pattern: ArrayLike, Dmax: float) -> Array:
    """
    Normalize the radiation pattern to dBi.
    """
    pattern = pattern / jnp.max(jnp.abs(pattern))
    pattern = 20 * jnp.log10(jnp.abs(pattern)) + 10.0 * jnp.log10(Dmax)
    return pattern


@jax.jit
def rad_pattern_from_geo_and_excitations(
    precomputed: ArrayLike,  # Precomputed array exponential terms, shape (2, xn, yn, theta, phi)
    Dmax_array: float,
    w: ArrayLike,  # Element excitations, shape (xn, yn, n_angles)
) -> tuple[Array, Array]:
    E_total = jnp.einsum("xy,tpcxy->tpc", w, precomputed)

    E_total = jnp.abs(E_total) ** 2

    # E_total.shape == (theta, phi, 2) or (theta, phi, 1).
    # If it's the former, E_total_theta & E_total_phi are the theta and phi components of the electric field.
    # In both cases, we want the norm and can sum.
    E_total = jnp.sum(E_total, axis=-1)

    # TODO: this is probably not needed because the power density is proportional to the magnitude of the E-field *squared*.
    E_total = jnp.sqrt(E_total)

    E_norm = normalize_rad_pattern(E_total, Dmax_array)
    return E_norm, w


@jax.jit
def rad_pattern_from_geo_and_phase_shifts(
    taper: ArrayLike,
    precomputed: ArrayLike,  # Precomputed array exponential terms, shape (2, xn, yn, theta, phi)
    Dmax_array: float,
    phase_shifts: ArrayLike = np.array([[0]]),
) -> tuple[Array, Array]:
    excitations = taper * jnp.exp(-1j * phase_shifts)

    E_norm, excitations = rad_pattern_from_geo_and_excitations(
        precomputed, Dmax_array, excitations
    )
    return E_norm


@jax.jit
def rad_pattern_from_geo(
    kx: ArrayLike,
    ky: ArrayLike,
    taper: ArrayLike,
    precomputed: ArrayLike,
    Dmax_array: ArrayLike,
    steering_rad: ArrayLike,
) -> tuple[Array, Array]:
    """
    Compute radiation pattern for given steering angles.
    """
    phase_shifts = calc_phase_shifts(kx, ky, steering_rad)

    # This assumes the taper is constant across all angles.
    # If taper is a 2D array, it should match the shape of phase_shifts.
    excitations = jnp.einsum("xy,xys->xy", taper, jnp.exp(-1j * phase_shifts))

    E_norm, excitations = rad_pattern_from_geo_and_excitations(
        precomputed, Dmax_array, excitations
    )
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
    ax: plt.Axes | None = None,
    filename: str | None = None,
):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

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
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    pattern: np.ndarray,
    *,
    hide_backlobe: bool = True,
    elev: float | None = None,
    azim: float | None = None,
    title: str = "3D Radiation Pattern",
    ax: plt.Axes | None = None,
) -> plt.Axes:
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
        ax = fig.add_subplot(projection="3d")

    ax.plot_surface(x, y, z, cmap="Spectral_r")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect(None, zoom=1.2)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_zlim(0, 30)
    ax.set_title(title)


def plot_ff_2d(
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    pattern: np.ndarray,
    *,
    title: str = "2D Radiation Pattern",
    colorbar: bool = True,
    ax: plt.Axes | None = None,
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))

    theta_deg, phi_deg = np.rad2deg(theta_rad), np.rad2deg(phi_rad)
    extent = [np.min(theta_deg), np.max(theta_deg), np.min(phi_deg), np.max(phi_deg)]
    aspect = theta_deg.size / phi_deg.size
    im = ax.imshow(pattern, extent=extent, origin="lower", aspect=aspect)
    ax.set_xlabel("θ°")
    ax.set_ylabel("φ°")
    ax.set_title(title)

    if colorbar:
        ax.get_figure().colorbar(im, fraction=0.046, pad=0.04, label="Normalized Dbi")


def plot_sine_space(
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    pattern: np.ndarray,
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
            loc = radius * np.array([np.cos(label_offset), np.sin(label_offset)])
            ax.text(*loc, f"{theta_deg}°", ha="center", va="center", color="gray")

    if phi_lines:
        for phi_deg in np.arange(0, 360, 30):
            phi_rad = np.radians(phi_deg)
            phi_sine = np.array([np.cos(phi_rad), np.sin(phi_rad)])
            ax.plot(*np.vstack(([0, 0], phi_sine)).T, **axis_args)
            if phi_deg in [0, 90]:
                continue  # Avoid label overlap with the title and colorbar
            ax.text(*(1.1 * phi_sine), f"{phi_deg}°", ha="center", va="center")

    ax.add_patch(plt.Circle((0, 0), 1, linewidth=1, fill=False))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.axis("off")
    ax.set_xlabel("u = sin($\\theta$)cos($\\phi$)")
    ax.set_ylabel("v = sin($\\theta$)sin($\\phi$)")
    ax.set_title(title)

    if colorbar:
        ax.get_figure().colorbar(im, fraction=0.046, pad=0.04, label="Normalized Dbi")

    return ax


def plot_phase_shifts(
    phase_shifts,
    title: str = "Phase Shifts",
    colorbar: bool = True,
    ax: plt.Axes | None = None,
):
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
        ax.get_figure().colorbar(im, fraction=0.046, pad=0.04, label="Degrees")


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
    fig, axs = plt.subplots(1, 3, figsize=[18, 6])

    plot_ff_2d(theta_rad, phi_rad, E_norm, ax=axs[0])
    plot_sine_space(theta_rad, phi_rad, E_norm, ax=axs[1])

    axs[2].remove()
    axs[2] = fig.add_subplot(1, 3, 3, projection="3d")
    plot_ff_3d(theta_rad, phi_rad, E_norm, ax=axs[2])

    steering_str = steering_repr(steering_deg.T)
    phase_shift_title = f"Radiation Pattern with Phase Shifts {steering_str}"
    fig.suptitle(phase_shift_title)
    fig.set_tight_layout(True)

    fig_path = "test.png"
    fig.savefig(fig_path, dpi=600, bbox_inches="tight")
    logger.info(f"Saved sample plot to {fig_path}")


# test_plot_ff_3d()
