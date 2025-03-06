from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py

from analyze import read_nf2ff


def generate_element_phase_shifts(xn, yn, method="random", **kwargs):
    """
    Generate phase shifts for individual antenna elements using different methods.

    Parameters:
    -----------
    xn : int
        Number of elements in x direction
    yn : int
        Number of elements in y direction
    method : str
        Method to generate phase shifts:
        - "random": random phases between min_phase and max_phase
        - "gradient": linear gradient across array
        - "zones": divide array into zones with different phases
        - "beamforming": proper phase shifts for beam steering

    kwargs:
        For "random": min_phase, max_phase (in degrees)
        For "gradient": start_phase, end_phase (in degrees)
        For "zones": n_zones, phase_values (list of phases in degrees)
        For "beamforming": theta_steering, phi_steering (in degrees), freq (in Hz),
                          dx, dy (element spacing in mm)

    Returns:
    --------
    numpy.ndarray
        Phase shifts for each element in radians, shape (xn, yn)
    """
    if method == "random":
        min_phase = np.deg2rad(kwargs.get("min_phase", -180))
        max_phase = np.deg2rad(kwargs.get("max_phase", 180))
        return np.random.uniform(min_phase, max_phase, size=(xn, yn))

    elif method == "gradient":
        start_phase = np.deg2rad(kwargs.get("start_phase", -180))
        end_phase = np.deg2rad(kwargs.get("end_phase", 180))
        x_gradient = np.linspace(start_phase, end_phase, xn)
        y_gradient = np.linspace(start_phase, end_phase, yn)
        return x_gradient[:, np.newaxis] + y_gradient[np.newaxis, :] / 2

    elif method == "zones":
        n_zones = kwargs.get("n_zones", 4)
        phase_values = kwargs.get(
            "phase_values", np.deg2rad(np.linspace(-180, 180, n_zones))
        )

        # Create zones
        zones = np.zeros((xn, yn), dtype=int)
        zone_size_x = xn // int(np.sqrt(n_zones))
        zone_size_y = yn // int(np.sqrt(n_zones))

        for i in range(int(np.sqrt(n_zones))):
            for j in range(int(np.sqrt(n_zones))):
                zone_idx = i * int(np.sqrt(n_zones)) + j
                if zone_idx < n_zones:
                    x_start = i * zone_size_x
                    x_end = (
                        (i + 1) * zone_size_x if i < int(np.sqrt(n_zones)) - 1 else xn
                    )
                    y_start = j * zone_size_y
                    y_end = (
                        (j + 1) * zone_size_y if j < int(np.sqrt(n_zones)) - 1 else yn
                    )
                    zones[x_start:x_end, y_start:y_end] = zone_idx

        # Map zones to phase values
        phase_shifts = np.zeros((xn, yn))
        for i in range(n_zones):
            phase_shifts[zones == i] = phase_values[i]

        return phase_shifts

    elif method == "beamforming":
        # Get parameters for beamforming
        theta_steering = kwargs.get("theta_steering", 0.0)  # degrees
        phi_steering = kwargs.get("phi_steering", 0.0)  # degrees
        freq = kwargs.get("freq", 2.45e9)  # Hz
        dx = kwargs.get("dx", 60)  # mm
        dy = kwargs.get("dy", 60)  # mm

        # Calculate wavelength and convert spacing to meters
        c = 299792458  # Speed of light in m/s
        wavelength = c / freq  # Wavelength in meters
        dx_m = dx / 1000  # Convert from mm to meters
        dy_m = dy / 1000  # Convert from mm to meters

        # Convert angles to radians
        theta_rad = np.deg2rad(theta_steering)
        phi_rad = np.deg2rad(phi_steering)

        # Wave number
        k = 2 * np.pi / wavelength

        # Element positions (centered around origin)
        x_positions = (np.arange(xn) - (xn - 1) / 2) * dx_m
        y_positions = (np.arange(yn) - (yn - 1) / 2) * dy_m

        # Calculate phase shifts for beamforming
        sin_theta = np.sin(theta_rad)
        cos_phi = np.cos(phi_rad)
        sin_phi = np.sin(phi_rad)

        # Create 2D arrays of positions
        x_grid, y_grid = np.meshgrid(x_positions, y_positions, indexing="ij")

        # Phase shifts to steer the beam
        # IMPORTANT: Using positive sign for beamforming - we want to advance the phase
        # to compensate for path differences so the waves arrive in phase in the steering direction
        phase_shifts = k * sin_theta * (x_grid * cos_phi + y_grid * sin_phi)

        return phase_shifts

    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_array_factor_with_element_phases(
    theta, phi, freq, xn, yn, dx, dy, phase_shifts
):
    """
    Calculate array factor with individual element phase shifts.

    Parameters:
    -----------
    theta : numpy.ndarray
        Elevation angle(s) in radians
    phi : numpy.ndarray
        Azimuth angle(s) in radians
    freq : float
        Operating frequency in Hz
    xn : int
        Number of elements in x direction
    yn : int
        Number of elements in y direction
    dx : float
        Element spacing in x direction (mm)
    dy : float
        Element spacing in y direction (mm)
    phase_shifts : numpy.ndarray
        Phase shift for each element in radians, shape (xn, yn)

    Returns:
    --------
    numpy.ndarray
        Array factor magnitude with shape (len(phi), len(theta))
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

    # Create meshgrid for calculations
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

    # Calculate sin(theta) and sin/cos(phi) once
    sin_theta = np.sin(THETA)
    cos_phi = np.cos(PHI)
    sin_phi = np.sin(PHI)

    # Element positions (centered around origin)
    x_positions = np.arange(xn) - (xn - 1) / 2
    y_positions = np.arange(yn) - (yn - 1) / 2

    # Reshape arrays for broadcasting
    # Make sin_theta, cos_phi, sin_phi shape: (len(phi), len(theta), 1, 1)
    sin_theta = sin_theta.T[:, :, np.newaxis, np.newaxis]
    cos_phi = cos_phi.T[:, :, np.newaxis, np.newaxis]
    sin_phi = sin_phi.T[:, :, np.newaxis, np.newaxis]

    # Make x_positions shape: (1, 1, xn, 1) and y_positions shape: (1, 1, 1, yn)
    x_positions = x_positions.reshape(1, 1, xn, 1)
    y_positions = y_positions.reshape(1, 1, 1, yn)

    # Make phase_shifts shape: (1, 1, xn, yn)
    phase_shifts = phase_shifts.reshape(1, 1, xn, yn)

    # Calculate phase terms for all elements at once
    psi_x = k * dx_m * sin_theta * cos_phi
    psi_y = k * dy_m * sin_theta * sin_phi

    # Compute the total phase for all elements and all angles at once
    total_phase = x_positions * psi_x + y_positions * psi_y - phase_shifts

    # Sum the complex exponentials across all elements
    AF = np.sum(np.exp(1j * total_phase), axis=(2, 3))

    # Normalize by total number of elements
    AF = AF / (xn * yn)

    return np.abs(AF)


def array_factor_partial_phase(theta, phi, freq, xn, yn, dx, dy):
    """
    Parameters:
    -----------
    theta : numpy.ndarray
        Elevation angle(s) in radians
    phi : numpy.ndarray
        Azimuth angle(s) in radians
    freq : float
        Operating frequency in Hz
    xn : int
        Number of elements in x direction
    yn : int
        Number of elements in y direction
    dx : float
        Element spacing in x direction (mm)
    dy : float
        Element spacing in y direction (mm)

    Returns:
    --------
    numpy.ndarray
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

    # Create meshgrid for calculations
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

    # Calculate sin(theta) and sin/cos(phi) once
    sin_theta = np.sin(THETA)
    cos_phi = np.cos(PHI)
    sin_phi = np.sin(PHI)

    # Element positions (centered around origin)
    x_positions = np.arange(xn) - (xn - 1) / 2
    y_positions = np.arange(yn) - (yn - 1) / 2

    # Reshape arrays for broadcasting
    # Make sin_theta, cos_phi, sin_phi shape: (len(phi), len(theta), 1, 1)
    sin_theta = sin_theta.T[:, :, np.newaxis, np.newaxis]
    cos_phi = cos_phi.T[:, :, np.newaxis, np.newaxis]
    sin_phi = sin_phi.T[:, :, np.newaxis, np.newaxis]

    # Make x_positions shape: (1, 1, xn, 1) and y_positions shape: (1, 1, 1, yn)
    x_positions = x_positions.reshape(1, 1, xn, 1)
    y_positions = y_positions.reshape(1, 1, 1, yn)

    # Calculate phase terms for all elements at once
    psi_x = k * dx_m * sin_theta * cos_phi
    psi_y = k * dy_m * sin_theta * sin_phi

    # Compute the partial phase for all elements and all angles at once
    partial_phase = x_positions * psi_x + y_positions * psi_y
    return partial_phase


def array_factor_partial_and_shift(xn, yn, partial_phase, phase_shifts):
    # Make phase_shifts shape: (1, 1, xn, yn)
    phase_shifts = phase_shifts.reshape(1, 1, xn, yn)

    # Compute the total phase for all elements and all angles at once
    total_phase = partial_phase - phase_shifts

    # Sum the complex exponentials across all elements
    AF = np.sum(np.exp(1j * total_phase), axis=(2, 3))

    # Normalize by total number of elements
    AF = AF / (xn * yn)

    return np.abs(AF)


# Add this function to help understand grating lobes
def check_grating_lobes(freq, dx, dy):
    """
    Check for potential grating lobes in an antenna array based on element spacing.

    Parameters:
    -----------
    freq : float
        Operating frequency in Hz
    dx : float
        Element spacing in x direction (mm)
    dy : float
        Element spacing in y direction (mm)

    Returns:
    --------
    dict
        Information about potential grating lobes
    """
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

    return {
        "wavelength_mm": wavelength_mm,
        "dx_lambda": dx_lambda,
        "dy_lambda": dy_lambda,
        "dx_critical_angle": dx_critical if dx_lambda > 0.5 else None,
        "dy_critical_angle": dy_critical if dy_lambda > 0.5 else None,
        "has_grating_lobes": dx_lambda > 0.5 or dy_lambda > 0.5,
    }


def generate_dataset(
    sim_dir_path: Path,
    outfile: Path,
    single_antenna_filename: str,
    n_samples: int = 1000,
    phase_method: str = "random",
):
    """
    Generate a dataset of farfield radiation patterns with individual element phase shifts.

    Parameters:
    -----------
    sim_dir_path : Path
        Directory containing OpenEMS simulation results with single antenna pattern
    output_dir_path : Path
        Directory to save the generated dataset
    single_antenna_filename : str
        Filename of the single antenna pattern H5 file
    n_samples : int
        Number of samples to generate
    phase_method : str
        Method to generate phase shifts: "random", "gradient", "zones", "beamforming"
    """
    # Fixed parameters for the 16x16 array
    xn = yn = 16  # 16x16 array
    dx = dy = 60  # 60x60 mm spacing
    freq = 2.45e9  # 2.45 GHz

    # Check for grating lobes with the given parameters
    grating_info = check_grating_lobes(freq, dx, dy)
    print("Array spacing check:")
    print(f"Wavelength: {grating_info['wavelength_mm']:.2f} mm")
    print(
        f"Element spacing: {grating_info['dx_lambda']:.3f}λ x {grating_info['dy_lambda']:.3f}λ"
    )
    if grating_info["has_grating_lobes"]:
        print("WARNING: Grating lobes will be visible when steering beyond:")
        if grating_info["dx_critical_angle"] is not None:
            print(f"  - {grating_info['dx_critical_angle']:.1f}° in the X direction")
        if grating_info["dy_critical_angle"] is not None:
            print(f"  - {grating_info['dy_critical_angle']:.1f}° in the Y direction")
    else:
        print("No grating lobes expected (element spacing <= λ/2)")

    # Load the single antenna pattern
    single_antenna_path = sim_dir_path / single_antenna_filename
    print(f"Loading single antenna pattern from {single_antenna_path}")
    single_antenna_nf2ff = read_nf2ff(single_antenna_path)

    # Extract necessary data
    single_E_norm = single_antenna_nf2ff["E_norm"][0]
    single_Dmax = single_antenna_nf2ff["Dmax"][0]  # Assuming single frequency
    theta = single_antenna_nf2ff["theta"]
    phi = single_antenna_nf2ff["phi"]

    # Initialize dataset arrays
    n_theta = len(theta)
    n_phi = len(phi)

    partial_phase = array_factor_partial_phase(theta, phi, freq, xn, yn, dx, dy)

    # Generate dataset
    print(f"Generating dataset with {n_samples} samples...")
    steering_info = []  # Store steering angles for beamforming

    # Create output directory if it doesn't exist
    outfile.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(outfile, "x") as h5f:
        h5f.create_dataset("theta", data=theta)
        h5f.create_dataset("phi", data=phi)

        # Store metadata
        h5f.attrs["array_size"] = f"{xn}x{yn}"
        h5f.attrs["spacing"] = f"{dx}x{dy}mm"
        h5f.attrs["frequency"] = freq
        h5f.attrs["description"] = (
            "Farfield radiation patterns dataset with individual element phase shifts"
        )
        h5f.attrs["phase_method"] = phase_method

        # Store 2D radiation patterns
        patterns = h5f.create_dataset("patterns", shape=(n_samples, n_phi, n_theta))
        # Store individual element phase shifts
        labels = h5f.create_dataset("labels", shape=(n_samples, xn, yn))

        for i in tqdm(range(n_samples)):
            # Generate phase shifts for each element
            if phase_method == "random":
                phase_shifts = generate_element_phase_shifts(
                    xn, yn, "random", min_phase=-180, max_phase=180
                )
            elif phase_method == "gradient":
                # Generate a random gradient direction and strength for variety
                start_phase = np.random.uniform(-180, 0)
                end_phase = np.random.uniform(0, 180)
                phase_shifts = generate_element_phase_shifts(
                    xn, yn, "gradient", start_phase=start_phase, end_phase=end_phase
                )
            elif phase_method == "zones":
                n_zones = np.random.choice([4, 9, 16])  # 2x2, 3x3 or 4x4 zones
                phase_values = np.random.uniform(-np.pi, np.pi, size=n_zones)
                phase_shifts = generate_element_phase_shifts(
                    xn, yn, "zones", n_zones=n_zones, phase_values=phase_values
                )
            elif phase_method == "beamforming":
                # Generate random steering angles with wider range to make effect more visible
                theta_steering = np.random.uniform(-70, 70)  # -70° to 70° elevation
                phi_steering = np.random.choice(
                    [0, 90, 180, 270]
                )  # Use cardinal directions for clearer visualization

                phase_shifts = generate_element_phase_shifts(
                    xn,
                    yn,
                    "beamforming",
                    theta_steering=theta_steering,
                    phi_steering=phi_steering,
                    freq=freq,
                    dx=dx,
                    dy=dy,
                )
                steering_info.append((theta_steering, phi_steering))
            else:
                # Mix of methods for more variety
                method = np.random.choice(["random", "beamforming", "zones"])
                if method == "beamforming":
                    theta_steering = np.random.uniform(-60, 60)
                    phi_steering = np.random.uniform(0, 360)
                    phase_shifts = generate_element_phase_shifts(
                        xn,
                        yn,
                        method,
                        theta_steering=theta_steering,
                        phi_steering=phi_steering,
                        freq=freq,
                        dx=dx,
                        dy=dy,
                    )
                else:
                    phase_shifts = generate_element_phase_shifts(xn, yn, method)

            # Calculate array factor for all phi and theta values at once
            AF = array_factor_partial_and_shift(xn, yn, partial_phase, phase_shifts)

            # Multiply by single element pattern to get total pattern
            # The shape of AF is (n_phi, n_theta) after the calculation
            total_pattern = single_E_norm[0] * AF

            # Normalize
            total_pattern = total_pattern / np.max(np.abs(total_pattern))

            # Convert to dB (normalized directivity)
            array_gain = single_Dmax * (xn * yn)  # Theoretical array gain
            total_pattern_db = 20 * np.log10(np.abs(total_pattern)) + 10.0 * np.log10(
                array_gain
            )

            # Store pattern and label
            patterns[i] = total_pattern_db
            labels[i] = phase_shifts  # Store the phase shifts as labels

        # Store steering info if using beamforming
        if phase_method == "beamforming" and steering_info:
            h5f.create_dataset("steering_info", data=np.array(steering_info))


def load_dataset(dataset_file: Path):
    """
    Load the generated dataset.

    Parameters:
    -----------
    dataset_file : Path
        Path to the dataset H5 file

    Returns:
    --------
    dict
        Dictionary containing patterns, labels, and metadata
    """
    with h5py.File(dataset_file, "r") as h5f:
        dataset = {
            "patterns": h5f["patterns"][:],
            "labels": h5f["labels"][:],
            "theta": h5f["theta"][:],
            "phi": h5f["phi"][:],
        }

        # Load metadata
        for key in h5f.attrs:
            dataset[key] = h5f.attrs[key]

    return dataset


def plot_samples(dataset, n_samples=5, output_dir=None):
    """
    Visualize a few samples from the dataset.

    Parameters:
    -----------
    dataset : dict
        Dataset loaded with load_dataset
    n_samples : int
        Number of samples to visualize
    """
    # Choose n_samples random indices
    indices = np.random.choice(
        len(dataset["patterns"]),
        size=min(n_samples, len(dataset["patterns"])),
        replace=False,
    )

    theta = dataset["theta"]

    for idx in indices:
        # Create figure with three subplots: pattern, phase shifts, and polar pattern
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        pattern = dataset["patterns"][idx]
        phase_shifts = dataset["labels"][idx]

        # Plot radiation pattern at phi=0
        phi_idx = np.argmin(np.abs(dataset["phi"]))  # phi=0 cut
        axs[0].plot(np.rad2deg(theta), pattern[phi_idx])
        axs[0].set_title("Radiation Pattern (dB) at phi=0°")
        axs[0].set_xlabel("Theta (degrees)")
        axs[0].set_ylabel("Directivity (dBi)")
        axs[0].grid(True)
        # axs[0].set_xlim([-180, 180])
        axs[0].set_ylim([-30, np.max(pattern) + 1])

        # Plot phase shifts
        im = axs[1].imshow(
            np.rad2deg(phase_shifts),
            cmap="viridis",
            origin="lower",
            interpolation="nearest",
            vmin=-180,
            vmax=180,
        )

        # Add steering info if available
        if "steering_info" in dataset and idx < len(dataset["steering_info"]):
            theta_s, phi_s = dataset["steering_info"][idx]
            axs[1].set_title(f"Phase Shifts (θ={theta_s:.1f}°, φ={phi_s:.1f}°)")
        else:
            axs[1].set_title("Element Phase Shifts (degrees)")

        axs[1].set_xlabel("Element Y index")
        axs[1].set_ylabel("Element X index")
        plt.colorbar(im, ax=axs[1])

        # Plot polar pattern
        axs[2] = plt.subplot(1, 3, 3, projection="polar")
        phi_idx = np.argmin(np.abs(dataset["phi"]))  # phi=0 cut
        norm_pattern = pattern[phi_idx] - np.max(pattern)  # Normalize to 0 dB max

        # Map theta from [0,pi] to [0,2pi] for polar plot
        plot_theta = theta  # Radial coordinate
        plot_r = norm_pattern  # Pattern value

        axs[2].plot(plot_theta, plot_r)
        axs[2].set_theta_zero_location("N")  # 0 degrees at the top
        axs[2].set_theta_direction(-1)  # clockwise
        axs[2].set_rlim(-40, 5)  # dB limits
        axs[2].set_title("Polar Pattern (phi=0°)")
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()
        if output_dir:
            plt.savefig(output_dir / f"sample_{idx}.png")


if __name__ == "__main__":
    # Default simulation directory
    sim_dir = Path.cwd() / "src" / "sim" / "antenna_array"
    outfile = Path.cwd() / "dataset" / "farfield_dataset.h5"

    # Filename format typically matches what's used in analyze.py
    single_antenna_filename = "farfield_1x1_60x60_2450_steer_t0_p0.h5"

    # Generate dataset
    generate_dataset(
        sim_dir_path=sim_dir,
        outfile=outfile,
        single_antenna_filename=single_antenna_filename,
        n_samples=10_000,
        phase_method="beamforming",
    )

    # Load and visualize the generated dataset
    dataset = load_dataset(outfile)
    plot_samples(dataset, n_samples=1, output_dir=outfile.parent)
