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


@lru_cache
def read_nf2ff(nf2ff_path: Path):
    print(f"Loading antenna pattern from {nf2ff_path}")
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

        E_norm = np.sqrt(np.abs(E_phi) ** 2 + np.abs(E_theta) ** 2)

    return {
        "theta_rad": theta_rad,
        "phi_rad": phi_rad,
        "r": r,
        "Dmax": Dmax,
        "freq": freqs,
        "E_theta": E_theta,
        "E_phi": E_phi,
        "E_norm": E_norm,
    }


def get_wavenumber(freq_hz: float) -> float:
    c = 299792458  # Speed of light in m/s
    wavelength = c / freq_hz  # Wavelength in meters
    k = 2 * np.pi / wavelength  # Wavenumber in radians/meter
    return k


@lru_cache(maxsize=1)
def get_element_positions(
    xn: int,
    yn: int,
    dx_mm: float = 60,
    dy_mm: float = 60,
) -> tuple[np.ndarray, np.ndarray]:
    # Calculate the x and y positions of the elements in the array in meters, centered around (0, 0)
    x_positions = (np.arange(xn) - (xn - 1) / 2) * dx_mm / 1000
    y_positions = (np.arange(yn) - (yn - 1) / 2) * dy_mm / 1000
    return x_positions, y_positions


def calc_array_params(
    xn: int = 16,
    yn: int = 16,
    dx_mm: float = 60,
    dy_mm: float = 60,
    freq_hz: float = 2.45e9,
) -> np.ndarray:
    k = get_wavenumber(freq_hz)
    x_pos, y_pos = get_element_positions(xn, yn, dx_mm, dy_mm)
    return k, x_pos, y_pos


@lru_cache(maxsize=1)
def calc_taper(
    xn: int = 16,
    yn: int = 16,
    taper_type: Literal["uniform", "hamming", "taylor"] = "uniform",
) -> np.ndarray:
    if taper_type == "hamming":
        window_x, window_y = np.hamming(xn), np.hamming(yn)
    elif taper_type == "taylor":
        # Simple approximation of Taylor window using Kaiser
        window_x, window_y = np.kaiser(xn, 3), np.kaiser(yn, 3)
    else:
        window_x, window_y = np.ones(xn), np.ones(yn)

    return np.outer(window_x, window_y)  # Create 2D taper by multiplying the 1D windows


@jax.jit
def calc_phase_shifts(
    k: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    steering_angles_deg: ArrayLike,
) -> Array:
    """
    Calculate phase shifts for each element in the array based on steering angles.
    Computes the phase shifts for each element based on the steering angles
    and the positions of the elements in the array.
    """
    # Convert to radians
    steering_angles_rad = jnp.radians(steering_angles_deg)
    theta_steering, phi_steering = steering_angles_rad.T  # (n_angles, 2)

    # Calculate steering vector components
    sin_theta = jnp.sin(theta_steering)
    ux = sin_theta * jnp.cos(phi_steering)
    uy = sin_theta * jnp.sin(phi_steering)

    x_phase = jnp.einsum("i,k->ik", x_pos, ux)  # (xn,), (n_angles,) -> (xn, n_angles)
    y_phase = jnp.einsum("j,k->jk", y_pos, uy)  # (yn,), (n_angles,) -> (yn, n_angles)

    # Combine phases: (xn, n_angles), (yn, n_angles) -> (xn, yn, n_angles)
    phase_shifts = k * (x_phase[:, None, :] + y_phase[None, :, :])
    phase_shifts = phase_shifts % (2 * jnp.pi)  # Normalize to [0, 2π)

    return phase_shifts


@jax.jit
def calc_excitations_from_steering(
    k: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
    taper: ArrayLike,
    steering_angles_deg: ArrayLike,
) -> Array:
    """
    Calculate element excitations for a given array configuration and steering angles.
    Computes the phase shifts for each element based on the steering angles
    and applies the tapering to compute the excitations.
    """
    phase_shifts = calc_phase_shifts(k, x_pos, y_pos, steering_angles_deg)

    # Calculate excitations: (xn, yn), (xn, yn, n_angles) -> (xn, yn)
    excitations = jnp.einsum("ij,ijk->ij", taper, jnp.exp(1j * phase_shifts))

    return excitations


@jax.jit
def calc_excitations_from_phase_shifts(
    taper: ArrayLike,
    phase_shifts: ArrayLike,
) -> Array:
    """
    Computes the excitations for each element based on the provided phase shifts
    and applies the tapering.
    """
    # Calculate excitations: (xn, yn), (xn, yn, n_angles) -> (xn, yn)
    excitations = jnp.einsum("ij,ijk->ij", taper, jnp.exp(1j * phase_shifts))

    return excitations


def calc_geo_exp(
    theta_rad: ArrayLike,
    phi_rad: ArrayLike,
    k: ArrayLike,
    x_pos: ArrayLike,
    y_pos: ArrayLike,
) -> Array:
    """
    Calculate the geometric phase terms for the array factor calculation.
    """
    # x and y components of the geometric phase: (len(theta_rad), len(phi_rad))
    sin_theta = jnp.sin(theta_rad)[:, None]
    ux = k * sin_theta * jnp.cos(phi_rad)
    uy = k * sin_theta * jnp.sin(phi_rad)

    # Geometric phase terms for each element and angle: (xn, yn, len(theta), len(phi))
    x_geo_phase = x_pos[:, None, None, None] * ux[None, None, :, :]
    y_geo_phase = y_pos[None, :, None, None] * uy[None, None, :, :]

    geo_phase = x_geo_phase + y_geo_phase  # Geometric phase terms for all elements
    geo_exp = jnp.exp(1j * geo_phase)  # Complex exponential of the geometric phase

    geo_exp = jnp.asarray(geo_exp)  # Convert to JAX array for JIT compilation

    return geo_exp


@jax.jit
def calc_array_factor(
    geo_exp: ArrayLike,
    excitations: ArrayLike | None = None,
) -> Array:
    """
    Calculates the array factor given the element excitations.
    Excitations should be an (xn, yn) complex NumPy array.
    If excitations is None, all elements are assumed to have 1 + 0j excitation.
    """
    xn, yn = geo_exp.shape[:2]
    if excitations is None:
        excitations = jnp.ones((xn, yn), dtype=jnp.complex64)

    # The array factor sum term for each element is Excitation * exp(j * GeometricPhase)
    # However, if the Excitation is defined as A_n * exp(j * alpha_n), and the array
    # factor includes a *subtraction* of this phase, like A_n * exp(j * (G_n - alpha_n)),
    # then this is equivalent to A_n * exp(j G_n) * exp(-j alpha_n) = Excitation.conjugate() * exp(j G_n).
    weighted_exp_terms = excitations[:, :, None, None].conjugate() * geo_exp

    # Sum contributions from all elements along the element dimensions (xn, yn).
    # Shape: (len(theta), len(phi))
    AF = jnp.sum(weighted_exp_terms, axis=(0, 1))

    # Normalize by the total number of elements.
    AF = AF / (xn * yn)

    return AF


def run_array_factor(
    E_theta: ArrayLike,
    E_phi: ArrayLike,
    array_factor: ArrayLike,
) -> Array:
    """
    Calculate the electric field norm from the array factor and single element fields.
    """
    E_theta_array, E_phi_array = array_factor * E_theta, array_factor * E_phi
    E_norm_array = jnp.sqrt(jnp.abs(E_theta_array) ** 2 + jnp.abs(E_phi_array) ** 2)
    return E_norm_array


def normalize_rad_pattern(pattern: ArrayLike, Dmax: float) -> Array:
    """
    Normalize the radiation pattern to dBi.
    """
    pattern = pattern / jnp.max(jnp.abs(pattern))
    pattern = 20 * jnp.log10(jnp.abs(pattern)) + 10.0 * jnp.log10(Dmax)
    return pattern


@jax.jit
def rad_pattern_from_geo(
    k: jnp.ndarray,
    x_pos: jnp.ndarray,
    y_pos: jnp.ndarray,
    taper: jnp.ndarray,
    geo_exp: jnp.ndarray,
    E_theta: jnp.ndarray,
    E_phi: jnp.ndarray,
    Dmax_array: jnp.ndarray,
    steering_angles: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute radiation pattern for given steering angles.
    """
    excitations = calc_excitations_from_steering(
        k, x_pos, y_pos, taper, steering_angles
    )
    AF = calc_array_factor(geo_exp, excitations)
    E_norm = run_array_factor(E_theta, E_phi, AF)
    E_norm = normalize_rad_pattern(E_norm, Dmax_array)
    return E_norm, excitations


def rad_pattern_from_single_elem(
    E_theta: np.ndarray,
    E_phi: np.ndarray,
    Dmax: float,
    freq_hz: float = 2.45e9,
    xn: int = 16,
    yn: int = 16,
    dx_mm: float = 60,
    dy_mm: float = 60,
    steering_deg: np.ndarray = np.array([[0, 0]]),
    taper_type: Literal["uniform", "hamming", "taylor"] = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    theta_rad, phi_rad = np.radians(np.arange(180)), np.radians(np.arange(360))

    k, x_pos, y_pos = calc_array_params(xn, yn, dx_mm, dy_mm, freq_hz)
    taper = calc_taper(xn, yn, taper_type)
    geo_exp = calc_geo_exp(theta_rad, phi_rad, k, x_pos, y_pos)
    Dmax_array = Dmax * (xn * yn)

    E_norm, excitations = rad_pattern_from_geo(
        k,
        x_pos,
        y_pos,
        taper,
        geo_exp,
        E_theta,
        E_phi,
        Dmax_array,
        steering_deg,
    )
    return E_norm, excitations


def rad_pattern_from_single_elem_and_phase_shifts(
    E_theta: np.ndarray,
    E_phi: np.ndarray,
    Dmax: float,
    freq_hz: float = 2.45e9,
    xn: int = 16,
    yn: int = 16,
    dx_mm: float = 60,
    dy_mm: float = 60,
    phase_shifts: np.ndarray = np.array([[0]]),
) -> tuple[np.ndarray, np.ndarray]:
    theta_rad, phi_rad = np.radians(np.arange(180)), np.radians(np.arange(360))

    k, x_pos, y_pos = calc_array_params(xn, yn, dx_mm, dy_mm, freq_hz)
    taper = calc_taper(xn, yn)
    geo_exp = calc_geo_exp(theta_rad, phi_rad, k, x_pos, y_pos)
    Dmax_array = Dmax * (xn * yn)

    excitations = calc_excitations_from_phase_shifts(taper, phase_shifts)
    AF = calc_array_factor(geo_exp, excitations)
    E_norm = run_array_factor(E_theta, E_phi, AF)
    E_norm = normalize_rad_pattern(E_norm, Dmax_array)
    return E_norm, excitations


def rad_pattern_from_geo_and_phase_shifts(
    taper: jnp.ndarray,
    geo_exp: jnp.ndarray,
    E_theta: jnp.ndarray,
    E_phi: jnp.ndarray,
    Dmax_array: float,
    phase_shifts: np.ndarray = np.array([[0]]),
) -> tuple[np.ndarray, np.ndarray]:
    excitations = calc_excitations_from_phase_shifts(taper, phase_shifts)
    AF = calc_array_factor(geo_exp, excitations)
    E_norm = run_array_factor(E_theta, E_phi, AF)
    E_norm = normalize_rad_pattern(E_norm, Dmax_array)
    return E_norm, excitations


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


def plot_sim_and_af(
    sim_dir,
    freq_hz,
    xns,
    yn,
    dxs,
    *,
    freq_idx=0,  # index of the frequency to plot
    steering_theta_deg=0,
    steering_phi_deg=0,
    figname: bool | str = True,
):
    """
    Plot comparison between OpenEMS simulation and array factor calculation,
    optionally with beam steering.

    Parameters:
    -----------
    sim_dir : Path
        Directory containing OpenEMS simulation results
    freq : float
        Operating frequency in Hz
    xns : list or numpy.ndarray
        Number of elements in the x-direction to plot
    yn : int
        Number of elements in the y-direction
    dxs : list or numpy.ndarray
        Element spacings to plot in millimeters
    figname : str, optional
        Filename to save the figure, if None the figure is not saved
    steering_theta : float, optional
        Steering elevation angle in degrees (default: 0)
    steering_phi : float, optional
        Steering azimuth angle in degrees (default: 0)
    """
    if not sim_dir:
        sim_dir = Path.cwd() / "src" / "sim" / "antenna_array"

    xns = np.array(xns)  # number of antennas in x direction
    dxs = np.array(dxs)  # distance between antennas in mm

    # Load the single antenna pattern (for array factor calculation)
    single_antenna_filename = f"ff_1x1_60x60_{freq_hz / 1e6:n}_steer_t0_p0.h5"
    nf2ff = read_nf2ff(sim_dir / single_antenna_filename)
    E_theta, E_phi = nf2ff["E_theta"][freq_idx], nf2ff["E_phi"][freq_idx]
    Dmax_single = nf2ff["Dmax"]

    elem_params = E_theta, E_phi, Dmax_single, freq_hz
    rad_pattern_from_array_params = partial(rad_pattern_from_single_elem, *elem_params)

    # Create a figure with polar subplots
    fig, axs = plt.subplots(
        nrows=len(xns),
        ncols=len(dxs),
        figsize=4 * np.array([len(dxs), len(xns)]),
        subplot_kw={"projection": "polar"},
    )
    axs = np.array([axs]).flatten() if min(len(xns), len(dxs)) == 1 else axs.flatten()

    # Loop through combinations and create combined plots
    for i, xn in enumerate(xns):
        for j, dx_mm in enumerate(dxs):
            ax = axs[i * len(dxs) + j]
            dy_mm = dx_mm

            # Plot OpenEMS full simulation
            filename = f"ff_{xn}x{yn}_{dx_mm}x{dy_mm}_{freq_hz / 1e6:n}_steer_t{steering_theta_deg}_p{steering_phi_deg}.h5"
            try:
                nf2ff = read_nf2ff(sim_dir / filename)
            except (FileNotFoundError, KeyError):  # Could not load simulation data
                pass

            E_norm = nf2ff["E_norm"][freq_idx]
            Dmax = nf2ff["Dmax"]
            label = "OpenEMS Simulation"
            plot_E_plane(E_norm=E_norm, Dmax=Dmax, normalize=True, label=label, ax=ax)

            steering_deg = np.array([[steering_theta_deg, steering_phi_deg]])
            E_norm, _ = rad_pattern_from_array_params(
                xn, yn, dx_mm, dy_mm, steering_deg
            )
            label = "Array Factor"
            plot_E_plane(E_norm=E_norm, Dmax=Dmax, fmt="g--", label=label, ax=ax)

            c = 299_792_458
            lambda0 = c / freq_hz
            lambda0_mm = lambda0 * 1e3
            freq_ghz = freq_hz / 1e9

            title = f"{xn}x{yn} array, {dx_mm}x{dy_mm}mm, {freq_ghz:n}GHz, {dx_mm / lambda0_mm:.2f}λ, steering: θ={steering_theta_deg}°, φ={steering_phi_deg}°"
            ax.set_title(title, fontsize=8)

    fig.legend(["OpenEMS Simulation", "Array Factor"], fontsize=8)
    fig.suptitle("OpenEMS Simulation vs Array Factor Comparison", y=0.99)

    fig.set_tight_layout(True)
    if figname:
        if figname is True:
            figname = (
                f"ff_{freq_ghz:.0f}GHz_steer_t{steering_theta_deg}_p{steering_phi_deg}"
            )
        fig.savefig(figname, dpi=600)


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
    im = ax.imshow(pattern, extent=extent, origin="lower")
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
    im = ax.contourf(u, v, pattern, levels=128)

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
    return f"[θ°, φ°] = {steering_angles.T.tolist()}"


def test_plot_ff_3d():
    steering_deg = np.array([[15, 15], [30, 120], [45, 210]])

    sim_dir = Path.cwd() / "src" / "sim" / "antenna_array"
    freq_hz = 2.45e9
    freq_idx = 0

    single_antenna_filename = f"ff_1x1_60x60_{freq_hz / 1e6:n}_steer_t0_p0.h5"
    nf2ff = read_nf2ff(sim_dir / single_antenna_filename)
    theta_rad, phi_rad = nf2ff["theta_rad"], nf2ff["phi_rad"]
    E_theta, E_phi = nf2ff["E_theta"][freq_idx], nf2ff["E_phi"][freq_idx]
    Dmax = nf2ff["Dmax"]

    E_norm, _ = rad_pattern_from_single_elem(
        E_theta,
        E_phi,
        Dmax,
        freq_hz,
        xn=16,
        yn=16,
        dx_mm=60,
        dy_mm=60,
        steering_deg=steering_deg,
    )
    E_norm = np.asarray(E_norm)

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
    print(f"Saved sample plot to {fig_path}")


# test_plot_ff_3d()
