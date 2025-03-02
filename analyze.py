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


def array_factor(theta, phi, freq, xn, yn, dx, dy):
    """
    Calculate the array factor of a rectangular antenna array.

    Parameters:
    -----------
    theta : float or numpy.ndarray
        Elevation angle(s) in radians
    phi : float or numpy.ndarray
        Azimuth angle(s) in radians
    xn : int
        Number of elements in the x-direction
    yn : int
        Number of elements in the y-direction
    dx : float
        Element spacing in the x-direction in millimeters
    dy : float
        Element spacing in the y-direction in millimeters
    freq : float
        Operating freq in Hz

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
            # Phase for this element
            phase = x_positions[ix] * psi_x + y_positions[iy] * psi_y
            # Add contribution to array factor
            AF += np.exp(1j * phase)

    # Normalize by total number of elements
    AF = AF / (xn * yn)

    return np.abs(AF)


def plot_sim_and_af(sim_dir, freq, xns, yn, dxs, figname: str | None = None):
    if not sim_dir:
        sim_dir = Path.cwd() / "src" / "sim" / "antenna_array"

    freq = 2.45e9  # frequency in Hz
    xns = np.array([1, 2, 4])  # number of antennas in x direction
    yn = 1  # number of antennas in y direction
    dxs = np.array([60, 90])  # distance between antennas in mm

    # Load the single antenna pattern (for array factor calculation)
    single_antenna_filename = f"farfield_1x1_60x60_{freq / 1e6:n}.h5"
    single_antenna_nf2ff = read_nf2ff(sim_dir / single_antenna_filename)
    single_E_norm = single_antenna_nf2ff["E_norm"][0][0]
    single_Dmax = single_antenna_nf2ff["Dmax"]
    theta, phi = single_antenna_nf2ff["theta"], single_antenna_nf2ff["phi"]

    # Create a figure with polar subplots
    fig, axs = plt.subplots(
        nrows=xns.size,
        ncols=dxs.size,
        figsize=4 * np.array([dxs.size, xns.size]),
        subplot_kw={"projection": "polar"},
    )
    axs = axs.flatten()

    # Loop through combinations and create combined plots
    for i, xn in enumerate(xns):
        for j, dx in enumerate(dxs):
            ax = axs[i * dxs.size + j]
            dy = dx

            # Plot 1: OpenEMS full simulation (in red)
            filename = f"farfield_{xn}x{yn}_{dx}x{dx}_{freq / 1e6:n}.h5"
            openems_nf2ff = read_nf2ff(sim_dir / filename)
            openems_E_norm = openems_nf2ff["E_norm"][0][0]
            openems_Dmax = openems_nf2ff["Dmax"]

            # Normalize and calculate dB for OpenEMS
            openems_norm = openems_E_norm / np.max(np.abs(openems_E_norm))
            openems_db = 20 * np.log10(np.abs(openems_norm)) + 10.0 * np.log10(
                openems_Dmax
            )
            ax.plot(theta, openems_db, "r-", linewidth=1, label="OpenEMS Simulation")

            # Plot 2: Array Factor calculation (in blue)
            AF = array_factor(theta, phi[0], freq, xn, yn, dx, dy)
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
            title = (
                f"{xn}x{yn} array, {dx}x{dy}mm, {freq_ghz:n}GHz, {dx / lambda0_mm:.2f}λ"
            )
            ax.set_title(title, fontsize=8)

    fig.legend(["OpenEMS Simulation", "Array Factor"], fontsize=8)
    fig.suptitle("OpenEMS Simulation vs Array Factor Comparison", y=0.99)
    fig.set_tight_layout(True)
    if figname:
        fig.savefig(figname, dpi=600)


def plot_ff_3d(
    nf2ff: dict[str, object],
    *,
    freq_index: int = 0,
    logscale: float | None = None,
    normalize: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot normalized 3D far field pattern.

    Args:
        nf2ff: Dictionary containing the output of calc_nf2ff function with keys:
            - E_norm: List of numpy arrays containing E-field norm data
            - freq: Array of frequencies
            - theta: Array of theta coordinates
            - phi: Array of phi coordinates
            - Dmax: Array of maximum directivities (optional)
        freq_index: Index of frequency to plot (default: 0)
        logscale: If set, shows far field with logarithmic scale and sets
                 the dB value for point of origin. Values below will be clamped.
        normalize: Whether to normalize linear plot (default: False).
                 Note: Log-plot is always normalized.

    Returns:
        matplotlib.pyplot.Axes object

    Example:
        >>> plot_ff_3d(nf2ff, freq_index=0, logscale=-20)
    """
    # Extract and normalize E-field data if requested
    if normalize or logscale is not None:
        E_far = nf2ff["E_norm"][freq_index] / np.max(nf2ff["E_norm"][freq_index])
    else:
        E_far = nf2ff["E_norm"][freq_index]

    # Apply logarithmic scaling if requested
    freq = nf2ff["freq"][freq_index]
    if logscale is not None:
        E_far = 20 * np.log10(E_far) / -logscale + 1
        E_far = E_far * (E_far > 0)  # Clamp negative values
        title = f"Electrical far field [dB] @ f = {freq:.2e} Hz"
    elif not normalize:
        title = f"Electrical far field [V/m] @ f = {freq:.2e} Hz"
    else:
        title = f"Normalized electrical far field @ f = {freq:.2e} Hz"

    # Create coordinate meshgrid
    theta, phi = np.meshgrid(nf2ff["theta"], nf2ff["phi"], indexing="xy")

    # Calculate cartesian coordinates
    x = E_far * np.sin(theta) * np.cos(phi)
    y = E_far * np.sin(theta) * np.sin(phi)
    z = E_far * np.cos(theta)

    # Create 3D plot
    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")
    c = plt.cm.viridis(E_far / np.max(E_far))
    surf = ax.plot_surface(x, y, z, facecolors=c, shade=False)

    # Configure plot settings
    ax.set_box_aspect([1, 1, 1])

    # Add colorbar
    if logscale is not None:
        ticks = np.linspace(0, np.max(E_far), 9)
        _ticklabels = np.linspace(
            logscale, 10 * np.log10(nf2ff.get("Dmax", [1])[freq_index]), 9
        )
        plt.colorbar(
            surf,
            ax=ax,
            ticks=ticks,
            format=lambda x, _: f"{float(x):.1f}",
        )
    else:
        plt.colorbar(surf, ax=ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_title(title)

    return ax


def example_usage():
    """Example showing how to use the plot_ff_3d function."""
    # Create sample data (normally this would come from calc_nf2ff)
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2 * np.pi, 2)

    # Create sample E_norm data (dipole-like pattern)
    theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing="xy")
    E_norm = np.sin(theta_mesh)

    # Create sample nf2ff dictionary
    nf2ff = {
        "E_norm": [E_norm],  # List containing one array
        "freq": [1e9],  # 1 GHz
        "theta": theta,
        "phi": phi,
        "Dmax": [1.5],
    }

    # Plot with different options
    plot_ff_3d(nf2ff)
    # plt.figure()
    # plot_ff_3d(nf2ff, logscale=-20)
    # plt.figure()
    # plot_ff_3d(nf2ff, normalize=True)

    plt.show()


if __name__ == "__main__":
    example_usage()
