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


def plot_ff_2d(nf2ff):
    theta = np.rad2deg(nf2ff["theta"])
    E_norm = nf2ff["E_norm"][0]
    Dmax = nf2ff["Dmax"]
    E_norm = 20.0 * np.log10(E_norm / np.max(E_norm)) + 10.0 * np.log10(Dmax[:, None])

    plt.plot(theta, np.squeeze(E_norm[0]), "k-", linewidth=2, label="xz-plane")
    plt.plot(theta, np.squeeze(E_norm[1]), "r--", linewidth=2, label="yz-plane")
    plt.xlabel("Theta (deg)")
    plt.ylabel("Directivity (dBi)")
    plt.title("Directivity Plot")
    plt.legend()
    plt.grid()


def plot_ff_3d(
    nf2ff: dict[str, object],
    *,
    freq_index: int = 0,
    logscale: float | None = None,
    normalize: bool = False,
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    c = plt.cm.viridis(E_far / np.max(E_far))
    surf = ax.plot_surface(x, y, z, facecolors=c, shade=False)

    # Configure plot settings
    ax.set_box_aspect([1, 1, 1])
    ax.axis("off")

    # Add colorbar
    if logscale is not None:
        ticks = np.linspace(0, np.max(E_far), 9)
        ticklabels = np.linspace(
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
