from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import h5py


def read_nf2ff(nf2ff_path: Path):
    with h5py.File(nf2ff_path, "r") as h5:
        mesh = h5["Mesh"]
        phi, theta, r = mesh["phi"][:], mesh["theta"][:], mesh["r"][:]

        Dmax = h5["nf2ff"].attrs["Dmax"]
        freq = h5["nf2ff"].attrs["Frequency"]

        E_shape = freq.size, phi.size, theta.size
        E_phi = np.empty(E_shape, dtype=complex)
        E_theta = np.empty(E_shape, dtype=complex)

        E_phi_ = h5["nf2ff"]["E_phi"]["FD"]
        E_theta_ = h5["nf2ff"]["E_theta"]["FD"]
        for i in range(freq.size):
            real, imag = E_phi_[f"f{i}_real"][:], E_phi_[f"f{i}_imag"][:]
            E_phi[i] = real + 1j * imag
            real, imag = E_theta_[f"f{i}_real"][:], E_theta_[f"f{i}_imag"][:]
            E_theta[i] = real + 1j * imag

        E_norm = np.sqrt(np.abs(E_phi) ** 2 + np.abs(E_theta) ** 2)

    return {
        "theta": theta,
        "phi": phi,
        "r": r,
        "Dmax": Dmax,
        "freq": freq,
        "E_phi": E_phi,
        "E_theta": E_theta,
        "E_norm": E_norm,
    }


def plot_ff_2d(nf2ff, ax: plt.Axes | None = None):
    theta = np.rad2deg(nf2ff["theta"])
    E_norm = nf2ff["E_norm"][0]
    Dmax = nf2ff["Dmax"]
    E_norm = 20.0 * np.log10(E_norm / np.max(E_norm)) + 10.0 * np.log10(Dmax[:, None])

    if ax is None:
        ax = plt.figure().add_subplot()
    ax.plot(theta, np.squeeze(E_norm[0]), "k-", linewidth=2, label="xz-plane")
    ax.plot(theta, np.squeeze(E_norm[1]), "r--", linewidth=2, label="yz-plane")
    ax.set_xlabel("Theta (deg)")
    ax.set_ylabel("Directivity (dBi)")
    ax.set_title("Directivity Plot")
    ax.legend()
    ax.grid()


def plot_ff_polar(
    E_norm,
    Dmax,
    theta,
    *,
    title: str | None = None,
    ax: plt.Axes | None = None,
    filename: str | None = None,
):
    E_norm = E_norm / np.max(np.abs(E_norm))
    E_norm = 20 * np.log10(np.abs(E_norm)) + 10.0 * np.log10(Dmax)

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    ax.plot(theta, E_norm, "r-", linewidth=1)
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


def array_factor(theta, phi, freq, xn, yn, dx, dy, phase_shifts=None):
    """
    Calculate the array factor of a rectangular antenna array with phase shifts for beamforming.

    Parameters:
    -----------
    theta : float or numpy.ndarray
        Elevation angle(s) in radians
    phi : float or numpy.ndarray
        Azimuth angle(s) in radians
    freq : float
        Operating freq in Hz
    xn : int
        Number of elements in the x-direction
    yn : int
        Number of elements in the y-direction
    dx : float
        Element spacing in the x-direction in millimeters
    dy : float
        Element spacing in the y-direction in millimeters
    phase_shifts : numpy.ndarray, optional
        Phase shifts for each element in radians, with shape (xn, yn).
        If None, no phase shifting is applied.

    Returns:
    --------
    numpy.ndarray
        Array factor magnitude
    """
    # Convert input to numpy arrays if they aren't already
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    # Calculate wavelength and convert spacing to meters
    c = 299792458  # Speed of light in m/s
    wavelength = c / freq  # Wavelength in meters
    dx_m = dx / 1000  # Convert from mm to meters
    dy_m = dy / 1000  # Convert from mm to meters

    # Wave number
    k = 2 * np.pi / wavelength

    # Initialize default phase shifts if none provided
    if phase_shifts is None:
        phase_shifts = np.zeros((xn, yn))

    # Ensure phase_shifts has correct dimensions
    if phase_shifts.shape != (xn, yn):
        raise ValueError(f"Phase shifts must have shape ({xn}, {yn})")

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
    x_positions = np.arange(xn) - (xn - 1) / 2
    y_positions = np.arange(yn) - (yn - 1) / 2

    # Calculate array factor by summing contributions from each element
    for ix in range(xn):
        for iy in range(yn):
            # Phase for this element (including phase shift for beamforming)
            phase = (
                x_positions[ix] * psi_x + y_positions[iy] * psi_y - phase_shifts[ix, iy]
            )
            # Add contribution to array factor
            AF += np.exp(1j * phase)

    # Normalize by total number of elements
    AF = AF / (xn * yn)

    return np.abs(AF)


def calculate_phase_shifts(xn, yn, dx, dy, freq, steering_theta, steering_phi):
    """
    Calculate phase shifts for antenna elements to steer the beam in a specific direction.

    Parameters:
    -----------
    xn : int
        Number of elements in the x-direction
    yn : int
        Number of elements in the y-direction
    dx : float
        Element spacing in the x-direction in millimeters
    dy : float
        Element spacing in the y-direction in millimeters
    freq : float
        Operating frequency in Hz
    steering_theta : float
        Steering elevation angle in degrees
    steering_phi : float
        Steering azimuth angle in degrees

    Returns:
    --------
    numpy.ndarray
        Phase shifts for each element in radians, with shape (xn, yn)
    """
    # Convert angles to radians
    steering_theta_rad = np.deg2rad(steering_theta)
    steering_phi_rad = np.deg2rad(steering_phi)

    # Calculate wavelength and convert spacing to meters
    c = 299792458  # Speed of light in m/s
    wavelength = c / freq  # Wavelength in meters
    dx_m = dx / 1000  # Convert from mm to meters
    dy_m = dy / 1000  # Convert from mm to meters

    # Wave number
    k = 2 * np.pi / wavelength

    # Element positions
    x_positions = np.arange(xn).reshape(-1, 1) - (xn - 1) / 2
    y_positions = np.arange(yn).reshape(1, -1) - (yn - 1) / 2

    # Calculate phase shifts
    sin_theta = np.sin(steering_theta_rad)
    phase_x = k * dx_m * sin_theta * np.cos(steering_phi_rad) * x_positions
    phase_y = k * dy_m * sin_theta * np.sin(steering_phi_rad) * y_positions

    phase_shifts = phase_x + phase_y

    return phase_shifts


def plot_sim_and_af(
    sim_dir,
    freq,
    xns,
    yn,
    dxs,
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
    single_antenna_nf2ff = read_nf2ff(sim_dir / single_antenna_filename)
    single_E_norm = single_antenna_nf2ff["E_norm"][0][0]
    single_Dmax = single_antenna_nf2ff["Dmax"]
    theta, phi = single_antenna_nf2ff["theta"], single_antenna_nf2ff["phi"]

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

            # Plot 1: OpenEMS full simulation (in red)
            filename = f"farfield_{xn}x{yn}_{dx}x{dx}_{freq / 1e6:n}_steer_t{steering_theta}_p{steering_phi}.h5"
            try:
                openems_nf2ff = read_nf2ff(sim_dir / filename)
                openems_E_norm = openems_nf2ff["E_norm"][0][0]
                openems_Dmax = openems_nf2ff["Dmax"]

                # Normalize and calculate dB for OpenEMS
                openems_norm = openems_E_norm / np.max(np.abs(openems_E_norm))
                openems_db = 20 * np.log10(np.abs(openems_norm)) + 10.0 * np.log10(
                    openems_Dmax
                )
                ax.plot(
                    theta, openems_db, "r-", linewidth=1, label="OpenEMS Simulation"
                )
            except (FileNotFoundError, KeyError):  # Could not load simulation data
                pass

            # Plot 2: Array Factor calculation with beamforming
            # Calculate phase shifts for beamforming
            phase_shifts = calculate_phase_shifts(
                xn, yn, dx, dy, freq, steering_theta, steering_phi
            )

            # Calculate array factor with phase shifts
            AF = array_factor(theta, phi[0], freq, xn, yn, dx, dy, phase_shifts)
            array_factor_E_norm = single_E_norm * AF.T

            # Normalize and calculate dB for Array Factor
            af_norm = array_factor_E_norm / np.max(np.abs(array_factor_E_norm))
            array_Dmax = single_Dmax * (xn * yn)
            af_db = 20 * np.log10(np.abs(af_norm)) + 10.0 * np.log10(array_Dmax)

            ax.plot(theta, af_db, "g--", linewidth=1, label="Array Factor")

            # Plot settings
            ax.set_thetagrids(np.arange(30, 360, 30))
            ax.set_rgrids(np.arange(-20, 20, 10))
            ax.set_rlim(-25, 15)
            ax.set_theta_offset(np.pi / 2)  # make 0 degree at the top
            ax.set_theta_direction(-1)  # clockwise
            ax.set_rlabel_position(90)  # move radial label to the right
            ax.grid(True, linestyle="--")
            ax.tick_params(labelsize=6)

            c = 299_792_458
            lambda0 = c / freq
            lambda0_mm = lambda0 * 1e3
            freq_ghz = freq / 1e9

            title = f"{xn}x{yn} array, {dx}x{dy}mm, {freq_ghz:n}GHz, {dx / lambda0_mm:.2f}λ, steering: θ={steering_theta}°, φ={steering_phi}°"
            ax.set_title(title, fontsize=8)

    # Create legend with appropriate labels
    legend_labels = ["OpenEMS Simulation"]
    legend_labels.append("Array Factor")
    fig.legend(legend_labels, fontsize=8)

    # Add beamforming info to suptitle if applicable
    fig.suptitle("OpenEMS Simulation vs Array Factor Comparison", y=0.99)

    fig.set_tight_layout(True)
    if figname:
        if figname is True:
            figname = f"ff_{freq_ghz:.0f}GHz_steer_t{steering_theta}_p{steering_phi}"
        fig.savefig(figname, dpi=600)


def plot_ff_3d(
    theta: np.ndarray,
    phi: np.ndarray,
    E_norm: np.ndarray,
    freq: float,
    *,
    freq_index: int = 0,
    logscale: float | None = None,
    normalize: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    # Extract and normalize E-field data if requested
    if normalize or logscale is not None:
        E_far = E_norm[freq_index] / np.max(E_norm[freq_index])
    else:
        E_far = E_norm[freq_index]

    # Apply logarithmic scaling if requested
    if logscale is not None:
        E_far = 20 * np.log10(np.abs(E_far)) / -logscale + 1
        E_far = E_far * (E_far > 0)  # Clamp negative values

    # Create coordinate meshgrid
    theta, phi = np.meshgrid(theta, phi, indexing="xy")

    E_far[E_far < 0] = 0  # Clip negative values to 0

    # Calculate cartesian coordinates
    x = E_far * np.sin(theta) * np.cos(phi)
    y = E_far * np.sin(theta) * np.sin(phi)
    z = E_far * np.cos(theta)

    # Create 3D plot
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")

    ax.plot_surface(x, y, z, cmap="Spectral_r")

    # Configure plot settings
    ax.view_init(elev=20.0, azim=-100)
    ax.set_aspect("equalxy")
    ax.set_box_aspect(None, zoom=1.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)

    ax.set_title("3D Far Field Pattern")

    return ax
