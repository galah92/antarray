"""
Example script to visualize beamforming patterns in antenna arrays
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import analyze


def plot_beamforming_comparison(xn=4, yn=1, dx=90, freq=2.45e9, angles=None):
    """
    Plot array factor for different beam steering angles.

    Parameters:
    -----------
    xn : int
        Number of elements in x-direction
    yn : int
        Number of elements in y-direction
    dx : float
        Element spacing in millimeters
    freq : float
        Operating frequency in Hz
    angles : list
        List of steering angles in degrees to compare
    """
    if angles is None:
        angles = [0, 15, 30, 45, 60]

    # Path to simulation results
    sim_dir = Path.cwd() / "src" / "sim" / "antenna_array"

    # Load the single antenna pattern (for array factor calculation)
    single_antenna_filename = f"farfield_1x1_60x60_{freq / 1e6:n}.h5"
    try:
        single_antenna_nf2ff = analyze.read_nf2ff(sim_dir / single_antenna_filename)
        single_E_norm = single_antenna_nf2ff["E_norm"][0][0]
        single_Dmax = single_antenna_nf2ff["Dmax"]
        theta, phi = single_antenna_nf2ff["theta"], single_antenna_nf2ff["phi"]
    except (FileNotFoundError, KeyError) as e:
        print(f"Could not load single antenna pattern: {e}")
        # Create dummy data for demonstration
        theta = np.linspace(-np.pi, np.pi, 361)
        phi = np.array([0, np.pi / 2])
        single_E_norm = np.ones_like(theta)
        single_Dmax = np.array([1.5])

    # Create figure for polar plots
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": "polar"})

    # Plot array factor for each steering angle
    colors = plt.cm.viridis(np.linspace(0, 1, len(angles)))

    for i, angle in enumerate(angles):
        # Calculate phase shifts for this steering angle
        phase_shifts = analyze.calculate_phase_shifts(xn, yn, dx, dx, freq, angle, 0)

        # Calculate array factor with phase shifts
        AF = analyze.array_factor(
            theta, phi[0], freq, xn, yn, dx, dx, phase_shifts.reshape(xn, yn)
        )
        array_factor_E_norm = single_E_norm * AF.T

        # Normalize and calculate dB
        af_norm = array_factor_E_norm / np.max(np.abs(array_factor_E_norm))
        array_Dmax = single_Dmax * (xn * yn)
        af_db = 20 * np.log10(np.abs(af_norm)) + 10.0 * np.log10(array_Dmax)

        # Plot this steering angle
        ax.plot(theta, af_db, color=colors[i], linewidth=2, label=f"{angle}°")

    # Plot settings
    # ax.set_thetagrids(np.arange(-180, 180, 30))
    ax.set_thetagrids(np.arange(0, 360, 30))
    ax.set_rgrids(np.arange(-20, 20, 10))
    ax.set_rlim(-25, 15)
    ax.set_theta_offset(np.pi / 2)  # make 0 degree at the top
    ax.set_theta_direction(-1)  # clockwise
    ax.set_rlabel_position(90)  # move radial label to the right
    ax.grid(True, linestyle="--")

    # Title and legend
    c = 299_792_458
    lambda0 = c / freq
    lambda0_mm = lambda0 * 1e3
    freq_ghz = freq / 1e9
    title = (
        f"{xn}x{yn} array, {dx}mm spacing ({dx / lambda0_mm:.2f}λ), {freq_ghz:.2f}GHz"
    )
    ax.set_title(title, pad=20)

    # Create legend outside the plot
    plt.legend(title="Steering Angles", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(
        f"beamforming_comparison_{xn}x{yn}_{dx}mm.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    # Plot beamforming comparison for a 4x1 array with 90mm spacing
    plot_beamforming_comparison(xn=4, yn=1, dx=90, angles=[0, 15, 30, 45, 60])

    # Plot beamforming comparison for different spacings
    for dx in [60, 90]:
        plot_beamforming_comparison(xn=4, yn=1, dx=dx)
