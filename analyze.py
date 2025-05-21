from functools import lru_cache
from pathlib import Path
from typing import Literal

import h5py
import matplotlib.pyplot as plt
import numpy as np


@lru_cache
def read_nf2ff(nf2ff_path: Path):
    print(f"Loading antenna pattern from {nf2ff_path}")
    with h5py.File(nf2ff_path, "r") as h5:
        mesh = h5["Mesh"]
        phi, theta, r = mesh["phi"][:], mesh["theta"][:], mesh["r"][:]

        Dmax = h5["nf2ff"].attrs["Dmax"]
        freqs = h5["nf2ff"].attrs["Frequency"]

        E_shape = freqs.size, phi.size, theta.size
        E_theta = np.empty(E_shape, dtype=complex)
        E_phi = np.empty(E_shape, dtype=complex)

        for freq in range(freqs.size):
            E_theta.real[freq] = h5[f"/nf2ff/E_theta/FD/f{freq}_real"][:]
            E_theta.imag[freq] = h5[f"/nf2ff/E_theta/FD/f{freq}_imag"][:]
            E_phi.real[freq] = h5[f"/nf2ff/E_phi/FD/f{freq}_real"][:]
            E_phi.imag[freq] = h5[f"/nf2ff/E_phi/FD/f{freq}_imag"][:]

        # Transpose to (freq, theta, phi)
        E_theta = E_theta.transpose(0, 2, 1)
        E_phi = E_phi.transpose(0, 2, 1)

        E_norm = np.sqrt(np.abs(E_phi) ** 2 + np.abs(E_theta) ** 2)

    return {
        "theta": theta,
        "phi": phi,
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


def get_element_positions(
    xn: int,
    yn: int,
    dx_mm: float = 60,
    dy_mm: float = 60,
) -> tuple[np.ndarray, np.ndarray]:
    dx_m, dy_m = dx_mm / 1000, dy_mm / 1000  # Convert element spacing from mm to meters
    x_positions = (np.arange(xn) - (xn - 1) / 2) * dx_m
    y_positions = (np.arange(yn) - (yn - 1) / 2) * dy_m
    return x_positions, y_positions


class ArrayFactorCalculator:
    """
    Calculate the array factor for a given array configuration.
    Pre-calculate terms dependent on geometry and observation angles
    to efficiently compute the array factor for different excitations.
    """

    def __init__(
        self,
        theta: np.ndarray,  # Array of observation theta angles (radians)
        phi: np.ndarray,  # Array of observation phi angles (radians)
        xn: int = 16,  # Number of elements in the x-direction
        yn: int = 16,  # Number of elements in the x-direction
        dx_mm: float = 60,
        dy_mm: float = 60,
        freq_hz: float = 2.45e9,
    ):
        k = get_wavenumber(freq_hz)
        x_pos, y_pos = get_element_positions(xn, yn, dx_mm, dy_mm)

        # Terms for x and y components of the geometric phase.
        # Shape: (len(theta), len(phi))
        sin_theta = np.sin(theta)[:, None]
        ux = k * sin_theta * np.cos(phi)
        uy = k * sin_theta * np.sin(phi)

        # Precompute the geometric phase terms for each element and angle.
        # Shape: (xn, yn, len(theta), len(phi)).
        x_geo_phase = x_pos[:, None, None, None] * ux[None, None, :, :]
        y_geo_phase = y_pos[None, :, None, None] * uy[None, None, :, :]

        # Combine the geometric phase terms for all elements.
        geo_phase = x_geo_phase + y_geo_phase
        # Complex exponential of the geometric phase terms.
        self.geo_exp = np.exp(1j * geo_phase)

    def __call__(self, excitations: np.ndarray | None = None) -> np.ndarray:
        """
        Calculates the array factor given the element excitations.
        Excitations should be an (xn, yn) complex NumPy array.
        If excitations is None, all elements are assumed to have 1 + 0j excitation.
        """
        xn, yn = self.geo_exp.shape[:2]
        if excitations is None:
            excitations = np.ones((xn, yn), dtype=complex)

        # The array factor sum term for each element is Excitation * exp(j * GeometricPhase)
        # However, if the Excitation is defined as A_n * exp(j * alpha_n), and the array
        # factor includes a *subtraction* of this phase, like A_n * exp(j * (G_n - alpha_n)),
        # then this is equivalent to A_n * exp(j G_n) * exp(-j alpha_n) = Excitation.conjugate() * exp(j G_n).
        weighted_exp_terms = excitations[:, :, None, None].conjugate() * self.geo_exp

        # Sum contributions from all elements along the element dimensions (xn, yn).
        # Shape: (len(theta), len(phi))
        AF = np.sum(weighted_exp_terms, axis=(0, 1))

        # Normalize by the total number of elements.
        AF = AF / (xn * yn)

        return AF


def calc_phase_shifts(
    theta_steering: float = 0.0,
    phi_steering: float = 0.0,
    xn: int = 16,
    yn: int = 16,
    dx_mm: float = 60,
    dy_mm: float = 60,
    freq: float = 2.45e9,
) -> np.ndarray:
    k = get_wavenumber(freq)
    x_pos, y_pos = get_element_positions(xn, yn, dx_mm, dy_mm)

    # Convert degrees to radians
    theta_steering, phi_steering = np.radians(theta_steering), np.radians(phi_steering)

    # Steering components of a unit vector pointing in the steered direction
    sin_theta_steering = np.sin(theta_steering)
    ux = sin_theta_steering * np.cos(phi_steering)
    uy = sin_theta_steering * np.sin(phi_steering)

    # Path difference = (r_element_vector) dot (unit_steering_vector)
    # Simplified as: x_element * ux + y_element * uy
    path_difference = (x_pos * ux)[:, None] + (y_pos * uy)

    phase_shifts = -k * path_difference

    phase_shifts = -phase_shifts  # FIXME: Should this be negated or not?
    phase_shifts = phase_shifts % (2 * np.pi)  # Normalize to [0, 2π)

    return phase_shifts


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


def calc_excitation(
    steering_theta: float = 0.0,
    steering_phi: float = 0.0,
    xn: int = 16,
    yn: int = 16,
    dx_mm: float = 60,
    dy_mm: float = 60,
    freq: float = 2.45e9,
    taper_type: Literal["uniform", "hamming", "taylor"] = "uniform",
) -> np.ndarray:
    """
    Calculate the element excitations for a given array configuration.
    The excitations are calculated based on the steering angles and taper type.
    """
    phase_shifts = calc_phase_shifts(
        steering_theta, steering_phi, xn, yn, dx_mm, dy_mm, freq
    )
    taper = calc_taper(xn, yn, taper_type)
    excitations = taper * np.exp(1j * phase_shifts)
    return excitations


def plot_ff_polar(
    E_norm,
    Dmax,
    theta,
    *,
    normalize: bool = True,
    label: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    filename: str | None = None,
):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    if normalize:
        E_norm = normalize_pattern(E_norm, Dmax)

    ax.plot(theta, E_norm, "r-", linewidth=1, label=label)
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


def plot_sim_and_af(
    sim_dir,
    freq,
    xns,
    yn,
    dxs,
    *,
    freq_idx=0,  # index of the frequency to plot
    steering_theta=0,
    steering_phi=0,
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
    single_antenna_filename = f"farfield_1x1_60x60_{freq / 1e6:n}_steer_t0_p0.h5"
    nf2ff = read_nf2ff(sim_dir / single_antenna_filename)
    theta, phi = nf2ff["theta"], nf2ff["phi"]
    E_theta_single, E_phi_single = nf2ff["E_theta"][freq_idx], nf2ff["E_phi"][freq_idx]
    Dmax_single = nf2ff["Dmax"]

    phi_idx = np.argmin(np.abs(phi - steering_phi))  # Index of steering phi

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
        for j, dx in enumerate(dxs):
            ax = axs[i * len(dxs) + j]
            dy = dx

            # Plot OpenEMS full simulation
            filename = f"farfield_{xn}x{yn}_{dx}x{dx}_{freq / 1e6:n}_steer_t{steering_theta}_p{steering_phi}.h5"
            try:
                nf2ff = read_nf2ff(sim_dir / filename)
            except (FileNotFoundError, KeyError):  # Could not load simulation data
                pass

            plot_ff_polar(
                E_norm=nf2ff["E_norm"][freq_idx, :, phi_idx],
                Dmax=nf2ff["Dmax"],
                theta=theta,
                label="OpenEMS Simulation",
                ax=ax,
            )

            excitations = calc_excitation(
                steering_theta, steering_phi, xn, yn, dx, dy, freq, taper_type="uniform"
            )
            AF = ArrayFactorCalculator(theta, phi, xn, yn, dx, dy, freq)(excitations)

            # Calculate radiation pattern for the array factor
            E_theta_array = AF * E_theta_single
            E_phi_array = AF * E_phi_single
            E_norm_array = np.sqrt(
                np.abs(E_theta_array) ** 2 + np.abs(E_phi_array) ** 2
            )
            E_norm_array = E_norm_array[:, phi_idx]  # Select theta slice

            Dmax_array = Dmax_single * (xn * yn)
            E_norm_array_db = normalize_pattern(E_norm_array, Dmax_array)
            ax.plot(theta, E_norm_array_db, "g--", linewidth=1, label="Array Factor")

            c = 299_792_458
            lambda0 = c / freq
            lambda0_mm = lambda0 * 1e3
            freq_ghz = freq / 1e9

            title = f"{xn}x{yn} array, {dx}x{dy}mm, {freq_ghz:n}GHz, {dx / lambda0_mm:.2f}λ, steering: θ={steering_theta}°, φ={steering_phi}°"
            ax.set_title(title, fontsize=8)

    fig.legend(["OpenEMS Simulation", "Array Factor"], fontsize=8)
    fig.suptitle("OpenEMS Simulation vs Array Factor Comparison", y=0.99)

    fig.set_tight_layout(True)
    if figname:
        if figname is True:
            figname = f"ff_{freq_ghz:.0f}GHz_steer_t{steering_theta}_p{steering_phi}"
        fig.savefig(figname, dpi=600)


def normalize_pattern(pattern: np.ndarray, Dmax: float) -> np.ndarray:
    """
    Normalize the radiation pattern to dBi.
    """
    pattern = pattern / np.max(np.abs(pattern))
    pattern = 20 * np.log10(np.abs(pattern)) + 10.0 * np.log10(Dmax)
    return pattern


def plot_ff_3d(
    theta: np.ndarray,
    phi: np.ndarray,
    pattern: np.ndarray,
    *,
    title: str = "3D Radiation Pattern",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    pattern = pattern.clip(min=0)  # Clip negative values to 0
    theta, phi = np.meshgrid(theta, phi)

    # Calculate cartesian coordinates
    x = pattern * np.sin(theta) * np.cos(phi)
    y = pattern * np.sin(theta) * np.sin(phi)
    z = pattern * np.cos(theta)

    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")

    ax.plot_surface(x, y, z, cmap="Spectral_r")
    ax.view_init(elev=20.0, azim=-100)
    ax.set_aspect("equalxy")
    ax.set_box_aspect(None, zoom=1.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_title(title)


def test_plot_ff_3d():
    steering_theta = 0.0
    steering_phi = 0.0

    freq = 2.45e9
    sim_dir = Path.cwd() / "src" / "sim" / "antenna_array"
    freq_idx = 0
    xn, yn = 16, 16
    dx, dy = 60, 60

    single_antenna_filename = f"farfield_1x1_60x60_{freq / 1e6:n}_steer_t0_p0.h5"
    nf2ff = read_nf2ff(sim_dir / single_antenna_filename)
    theta, phi = nf2ff["theta"], nf2ff["phi"]
    E_theta_single, E_phi_single = nf2ff["E_theta"][freq_idx], nf2ff["E_phi"][freq_idx]
    Dmax_single = nf2ff["Dmax"]

    # Convert theta from elevation to polar angle (inclination from Z-axis)
    # Polar angle: 0 (Z-axis) to pi (-Z-axis)
    # theta = np.pi/2 - theta

    excitations = calc_excitation(
        steering_theta, steering_phi, xn, yn, dx, dy, freq, taper_type="uniform"
    )
    AF = ArrayFactorCalculator(theta, phi, xn, yn, dx, dy, freq)(excitations)

    # Calculate radiation pattern for the array factor
    E_theta_array = AF * E_theta_single
    E_phi_array = AF * E_phi_single
    E_norm_array = np.sqrt(np.abs(E_theta_array) ** 2 + np.abs(E_phi_array) ** 2)

    Dmax_array = Dmax_single * (xn * yn)
    E_norm_array_db = normalize_pattern(E_norm_array, Dmax_array)

    plot_ff_3d(
        theta=theta,
        phi=phi,
        pattern=E_norm_array_db.T,
        title=f"3D Radiation Pattern, {xn}x{yn} array, {dx}x{dy}mm, {freq / 1e9:.0f}GHz",
    )
    plt.show()


def plot_ff_2d(
    theta: np.ndarray,
    phi: np.ndarray,
    pattern: np.ndarray,
    *,
    title: str = "2D Radiation Pattern",
    colorbar: bool = True,
    ax: plt.Axes | None = None,
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))

    theta_deg, phi_deg = np.rad2deg(theta), np.rad2deg(phi)
    extent = [np.min(theta_deg), np.max(theta_deg), np.min(phi_deg), np.max(phi_deg)]
    im = ax.imshow(pattern, extent=extent, origin="lower")
    ax.set_xlabel("Theta (degrees)")
    ax.set_ylabel("Phi (degrees)")
    ax.set_title(title)

    if colorbar:
        ax.get_figure().colorbar(im, fraction=0.046, pad=0.04, label="Normalized Dbi")


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
