from functools import lru_cache
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


@lru_cache
def read_nf2ff(nf2ff_path: Path):
    print(f"Loading single antenna pattern from {nf2ff_path}")
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


def array_factor(
    theta: np.ndarray,  # Array of observation theta angles (radians)
    phi: np.ndarray,  # Array of observation phi angles (radians)
    xn: int,
    yn: int,
    dx_mm: float = 60,
    dy_mm: float = 60,
    freq_hz: float = 2.45e9,  # Changed to freq_hz for clarity in units
    excitations: np.ndarray | None = None,  # (xn, yn) complex array of excitations
) -> np.ndarray:
    k = get_wavenumber(freq_hz)

    theta, phi = np.asarray(theta), np.asarray(phi)

    dx_m = dx_mm / 1000  # Convert from mm to meters
    dy_m = dy_mm / 1000  # Convert from mm to meters

    # Initialize default excitations if none provided
    if excitations is None:
        excitations = np.ones((xn, yn), dtype=complex)

    phase_shifts = np.angle(excitations)  # Phase shifts from excitations

    # Initialize output array based on input shapes
    if theta.ndim == 1 and phi.ndim == 1:
        # Create a meshgrid for broadcasting
        THETA, PHI = np.meshgrid(theta, phi, indexing="ij")
        AF = np.zeros((theta.size, phi.size), dtype=complex)
    else:
        # Assume pre-meshed inputs
        THETA, PHI = theta, phi
        AF = np.zeros_like(THETA, dtype=complex)

    # Calculate array factor
    sin_theta = np.sin(THETA)
    cos_phi = np.cos(PHI)
    sin_phi = np.sin(PHI)

    # Phase differences
    psi_x = k * dx_m * sin_theta * cos_phi
    psi_y = k * dy_m * sin_theta * sin_phi

    # Element positions
    x_pos = np.arange(xn) - (xn - 1) / 2
    y_pos = np.arange(yn) - (yn - 1) / 2

    # Calculate array factor by summing contributions from each element
    for ix in range(xn):
        for iy in range(yn):
            phase = x_pos[ix] * psi_x + y_pos[iy] * psi_y - phase_shifts[ix, iy]
            AF += np.exp(1j * phase)

    AF = AF / (xn * yn)  # Normalize by total number of elements

    return AF


def array_factor2(
    theta: np.ndarray,  # Array of observation theta angles (radians)
    phi: np.ndarray,  # Array of observation phi angles (radians)
    xn: int,
    yn: int,
    dx_mm: float = 60,
    dy_mm: float = 60,
    freq_hz: float = 2.45e9,  # Changed to freq_hz for clarity in units
    excitations: np.ndarray | None = None,  # (xn, yn) complex array of excitations
) -> np.ndarray:
    k = get_wavenumber(freq_hz)

    dx_m, dy_m = dx_mm / 1000, dy_mm / 1000  # Convert element spacing from mm to meters

    # 1. Element Positions (Centered)
    x_positions_m = (np.arange(xn) - (xn - 1) / 2.0) * dx_m  # Shape (xn,)
    y_positions_m = (np.arange(yn) - (yn - 1) / 2.0) * dy_m  # Shape (yn,)

    # Create 2D meshgrid for element coordinates (xn, yn)
    x_coords, y_coords = np.meshgrid(x_positions_m, y_positions_m, indexing="ij")

    # 2. Observation Angles (Meshgrid for all combinations)
    # Convert input degrees to radians
    theta_rad, phi_rad = np.radians(theta), np.radians(phi)

    # Create 2D meshgrids for observation angles (len(theta), len(phi))
    theta_grid, phi_grid = np.meshgrid(theta_rad, phi_rad, indexing="ij")

    # 3. Calculate Direction Cosines (ux, uy) for all observation points
    sin_theta_obs = np.sin(theta_grid)
    u_x_obs = sin_theta_obs * np.cos(phi_grid)  # Shape (len(theta), len(phi))
    u_y_obs = sin_theta_obs * np.sin(phi_grid)  # Shape (len(theta), len(phi))

    # 4. Handle Excitations
    if excitations is None:
        element_excitations = np.ones((xn, yn), dtype=np.complex128)
    else:
        element_excitations = np.asarray(excitations, dtype=np.complex128)
        if element_excitations.shape != (xn, yn):
            raise ValueError(
                f"Excitations must be of shape ({xn}, {yn}), but got {element_excitations.shape}"
            )

    # 5. Core Array Factor Calculation: Sum over elements for each observation point
    # Expand observation angle terms to match element dims at the end for broadcasting:
    u_x_obs_expanded = u_x_obs[:, :, None, None]  # (len(theta), len(phi), 1, 1)
    u_y_obs_expanded = u_y_obs[:, :, None, None]  # (len(theta), len(phi), 1, 1)

    # x_coords and y_coords are (xn, yn)
    # They will broadcast to (1, 1, xn, yn) for the multiplication
    arg_exp = k * (x_coords * u_x_obs_expanded + y_coords * u_y_obs_expanded)

    # Multiply by element excitations and sum over elements
    # element_excitations is (xn, yn), it broadcasts to (1, 1, xn, yn)
    af_pattern = np.sum(element_excitations * np.exp(1j * arg_exp), axis=(-2, -1))

    return af_pattern


def calculate_phase_shifts(
    xn: int,
    yn: int,
    dx_mm: float = 60,
    dy_mm: float = 60,
    freq: float = 2.45e9,
    theta_steering: float = 0.0,
    phi_steering: float = 0.0,
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

            phase_shifts = calculate_phase_shifts(
                xn, yn, dx, dy, freq, steering_theta, steering_phi
            )
            excitations = np.exp(1j * phase_shifts)
            AF = array_factor(theta, phi, xn, yn, dx, dy, freq, excitations)

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
