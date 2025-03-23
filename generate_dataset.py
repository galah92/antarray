from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib import animation
from tqdm import tqdm

import analyze
from analyze import read_nf2ff

DEFAULT_SIM_DIR = Path.cwd() / "src" / "sim" / "antenna_array"
DEFAULT_DATASET_DIR = Path.cwd() / "dataset"
DEFAULT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUTFILE = "farfield_dataset.h5"
DEFAULT_SINGLE_ANT_FILENAME = "farfield_1x1_60x60_2450_steer_t0_p0.h5"


app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


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

        # Wave number
        k = 2 * np.pi / wavelength

        # Element positions (centered around origin)
        x_positions = (np.arange(xn) - (xn - 1) / 2) * dx_m
        y_positions = (np.arange(yn) - (yn - 1) / 2) * dy_m

        # Create 2D arrays of positions
        x_grid, y_grid = np.meshgrid(x_positions, y_positions, indexing="ij")

        # Set amplitude
        amplitude = 1.0

        # Remove NaN values from steering angles
        theta_steering = theta_steering[~np.isnan(theta_steering)]
        phi_steering = phi_steering[~np.isnan(phi_steering)]

        # Convert angles to radians
        thetas = np.deg2rad(theta_steering)
        phis = np.deg2rad(phi_steering)

        combined_excitation = np.zeros((xn, yn), dtype=complex)

        for theta, phi in zip(thetas, phis):
            # Calculate phase shifts for this beam
            phase_shift = (
                k * np.sin(theta) * (x_grid * np.cos(phi) + y_grid * np.sin(phi))
            )

            # Add this beam's excitation to the combined excitation
            combined_excitation += amplitude * np.exp(1j * phase_shift)

        # Extract phase from combined excitation
        combined_phase = np.angle(combined_excitation)

        return combined_phase

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


@app.command()
def generate(
    n_samples: int = 1_000,
    phase_method: str = "beamforming",
    theta_steering: float | None = None,  # Degrees
    phi_steering: float | None = None,  # Degrees
    sim_dir_path: Path = DEFAULT_SIM_DIR,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    outfile: Path = DEFAULT_OUTFILE,
    overwrite: bool = False,
    single_antenna_filename: str = DEFAULT_SINGLE_ANT_FILENAME,
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
    # print("Array spacing check:")
    # print(f"Wavelength: {grating_info['wavelength_mm']:.2f} mm")
    # print(
    #     f"Element spacing: {grating_info['dx_lambda']:.1f}λ x {grating_info['dy_lambda']:.1f}λ"
    # )
    if grating_info["has_grating_lobes"]:
        print("WARNING: Grating lobes will be visible when steering beyond:")
        if grating_info["dx_critical_angle"] is not None:
            print(f"  - {grating_info['dx_critical_angle']:.1f}° in the X direction")
        if grating_info["dy_critical_angle"] is not None:
            print(f"  - {grating_info['dy_critical_angle']:.1f}° in the Y direction")

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

    # Create output directory if it doesn't exist
    outfile = dataset_dir / outfile
    outfile.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if overwrite else "x"
    with h5py.File(outfile, mode) as h5f:
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

        if phase_method == "beamforming":
            # Store steering info
            steering_info = h5f.create_dataset("steering_info", shape=(n_samples, 2))

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
                if theta_steering is None:
                    # Generate random steering angles with wider range to make effect more visible
                    theta_steering = np.random.uniform(-70, 70)  # -70° to 70° elevation
                if phi_steering is None:
                    # Use cardinal directions for clearer visualization
                    phi_steering = np.random.choice([0, 90, 180, 270])

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

            if phase_method == "beamforming":
                steering_info[i] = [theta_steering, phi_steering]

            # Calculate array factor for all phi and theta values at once
            AF = array_factor_partial_and_shift(xn, yn, partial_phase, phase_shifts)

            # Multiply by single element pattern to get total pattern
            # The shape of AF is (n_phi, n_theta) after the calculation
            total_pattern = single_E_norm[0] * AF

            # Normalize
            total_pattern = total_pattern / np.max(np.abs(total_pattern))

            # Convert to dB (normalized directivity)
            array_gain = single_Dmax * (xn * yn)  # Theoretical array gain
            array_gain_db = 10.0 * np.log10(array_gain)
            total_pattern_db = 20 * np.log10(np.abs(total_pattern)) + array_gain_db

            # Store pattern and label
            patterns[i] = total_pattern_db
            labels[i] = phase_shifts  # Store the phase shifts as labels


@app.command()
def plot_dataset_phase_shifts(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    dataset_name: Path = "ff_beamforming.h5",
    outfile: Path = "beamforming_phase_shifts.gif",
):
    with h5py.File(dataset_dir / dataset_name, "r") as h5f:
        labels = h5f["labels"]
        steering_info = h5f["steering_info"]

        fig, ax = plt.subplots()
        fig.set_tight_layout(True)

        ax.set_xlabel("Element X index")
        ax.set_ylabel("Element Y index")
        title = ax.set_title("Phase Shifts")

        # Create a colorbar
        sm = plt.cm.ScalarMappable(cmap="twilight_shifted")
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Degrees")
        cbar.set_ticks(np.linspace(0, 1, 7))
        cbar.set_ticklabels(np.linspace(-180, 180, 7, dtype=np.int32))

        im = ax.imshow(
            np.zeros_like(labels[0]),
            cmap="twilight_shifted",  # Cyclic colormap for phase values
            origin="lower",
            vmin=-180,
            vmax=180,
        )

        def animate(i):
            print(f"Plotting {i}")
            j = i * 111

            # Update the data for the plot
            phase_shifts_clipped = (labels[j] + np.pi) % (2 * np.pi) - np.pi
            im.set_data(np.rad2deg(phase_shifts_clipped))

            # Update the title with steering info
            theta_s, phi_s = steering_info[j]
            text = f"Phase Shifts (θ={theta_s:03.1f}°, φ={phi_s:03.1f}°) {i:04d}"
            title.set_text(text)

            return im, title

        frames = len(labels)
        frames = 111 * 2
        ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True)

        # To save the animation using Pillow as a gif
        writer = animation.PillowWriter(fps=20, bitrate=1800)
        ani.save(dataset_dir / outfile, writer=writer)


@app.command()
def explore_dataset_phase_shifts(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    dataset_name: Path = "ff_beamforming.h5",
):
    with h5py.File(dataset_dir / dataset_name, "r") as h5f:
        phi_steering = np.arange(-55, 55 + 1)
        n_phi = phi_steering.size
        n_blocks = 111
        labels = h5f["labels"]
        blocks = labels[: n_blocks * n_phi, ...].reshape(n_blocks, n_phi, 16, 16)

        def foo(blocks):
            blocks_diff = np.diff(blocks, axis=0)

            blocks_diffdiff = np.sum(np.diff(blocks_diff, axis=0), axis=(1, 2, 3))
            print(blocks_diffdiff)
            print(blocks_diffdiff.max())
            print(blocks_diff[0, :2, :2, :2])
            print("------------------------")
            print(blocks_diff[1, :2, :2, :2])
            print("------------------------")
            print(blocks_diff[2, :2, :2, :2])

        phi_blocks = blocks
        foo(phi_blocks)
        theta_blocks = blocks.transpose(1, 0, 2, 3)
        foo(theta_blocks)


@app.command()
def generate_beamforming(
    n_samples: int = 1_000,
    theta_start: float = -65,  # Degrees
    theta_end: float = 65,  # Degrees
    phi_start: float = -65,  # Degrees
    phi_end: float = 65,  # Degrees
    max_n_beams: int = 1,
    sim_dir_path: Path = DEFAULT_SIM_DIR,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    outfile: Path = DEFAULT_OUTFILE,
    overwrite: bool = False,
    single_antenna_filename: str = DEFAULT_SINGLE_ANT_FILENAME,
):
    # Fixed parameters for the 16x16 array
    xn = yn = 16  # 16x16 array
    dx = dy = 60  # 60x60 mm spacing
    freq = 2.45e9  # 2.45 GHz

    # Check for grating lobes with the given parameters
    grating_info = check_grating_lobes(freq, dx, dy)
    # print("Array spacing check:")
    # print(f"Wavelength: {grating_info['wavelength_mm']:.2f} mm")
    # print(
    #     f"Element spacing: {grating_info['dx_lambda']:.1f}λ x {grating_info['dy_lambda']:.1f}λ"
    # )
    if grating_info["has_grating_lobes"]:
        print("WARNING: Grating lobes will be visible when steering beyond:")
        if grating_info["dx_critical_angle"] is not None:
            print(f"  - {grating_info['dx_critical_angle']:.1f}° in the X direction")
        if grating_info["dy_critical_angle"] is not None:
            print(f"  - {grating_info['dy_critical_angle']:.1f}° in the Y direction")

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

    # Create output directory if it doesn't exist
    outfile = dataset_dir / outfile
    outfile.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if overwrite else "x"
    with h5py.File(outfile, mode) as h5f:
        h5f.create_dataset("theta", data=theta)
        h5f.create_dataset("phi", data=phi)

        # Store metadata
        h5f.attrs["array_size"] = f"{xn}x{yn}"
        h5f.attrs["spacing"] = f"{dx}x{dy}mm"
        h5f.attrs["frequency"] = freq
        h5f.attrs["phase_method"] = "beamforming"

        # Store 2D radiation patterns
        patterns = h5f.create_dataset("patterns", shape=(n_samples, n_phi, n_theta))
        # Store individual element phase shifts
        labels = h5f.create_dataset("labels", shape=(n_samples, xn, yn))
        # Store steering info
        steering_info = h5f.create_dataset(
            "steering_info", shape=(n_samples, 2, max_n_beams)
        )

        for i in tqdm(range(n_samples)):
            # Generate random steering angles within the specified range
            theta_steering = np.random.uniform(theta_start, theta_end, size=max_n_beams)
            phi_steering = np.random.uniform(phi_start, phi_end, size=max_n_beams)

            # Choose a random number of beams to simulate and fill the rest with NaN
            n_beams = np.random.randint(max_n_beams) + 1
            theta_steering[n_beams:] = np.nan
            phi_steering[n_beams:] = np.nan

            # Generate phase shifts for each element
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
            steering_info[i] = [theta_steering, phi_steering]

            # Calculate array factor for all phi and theta values at once
            AF = array_factor_partial_and_shift(xn, yn, partial_phase, phase_shifts)

            # Multiply by single element pattern to get total pattern
            # The shape of AF is (n_phi, n_theta) after the calculation
            total_pattern = single_E_norm[0] * AF

            # Normalize
            total_pattern = total_pattern / np.max(np.abs(total_pattern))

            # Convert to dB (normalized directivity)
            array_gain = single_Dmax * (xn * yn)  # Theoretical array gain
            array_gain_db = 10.0 * np.log10(array_gain)
            total_pattern_db = 20 * np.log10(np.abs(total_pattern)) + array_gain_db

            # Store pattern and label
            patterns[i] = total_pattern_db
            labels[i] = phase_shifts  # Store the phase shifts as labels


@app.command()
def plot_beamforming_angles(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    dataset_name: Path = "ff_beamforming.h5",
    outfile: Path | None = None,
):
    with h5py.File(dataset_dir / dataset_name, "r") as h5f:
        theta, phi = h5f["steering_info"][:].T

    fig, ax = plt.subplots()

    ax.scatter(theta, phi, s=0.3)
    ax.set_xlabel("Theta (degrees)")
    ax.set_ylabel("Phi (degrees)")
    ax.set_title(f"Beamforming Angles ({dataset_name})")
    ax.set_aspect("equal", adjustable="box")

    filename = outfile or dataset_name.with_suffix("")
    fig.savefig(dataset_dir / filename, dpi=600)


def ff_from_phase_shifts(
    phase_shifts: np.ndarray,
    sim_dir_path: Path = DEFAULT_SIM_DIR,
    single_antenna_filename: str = DEFAULT_SINGLE_ANT_FILENAME,
):
    # Fixed parameters for the 16x16 array
    xn = yn = 16  # 16x16 array
    dx = dy = 60  # 60x60 mm spacing
    freq = 2.45e9  # 2.45 GHz

    # Check for grating lobes with the given parameters
    grating_info = check_grating_lobes(freq, dx, dy)
    # print("Array spacing check:")
    # print(f"Wavelength: {grating_info['wavelength_mm']:.2f} mm")
    # print(
    #     f"Element spacing: {grating_info['dx_lambda']:.1f}λ x {grating_info['dy_lambda']:.1f}λ"
    # )
    if grating_info["has_grating_lobes"]:
        print("WARNING: Grating lobes will be visible when steering beyond:")
        if grating_info["dx_critical_angle"] is not None:
            print(f"  - {grating_info['dx_critical_angle']:.1f}° in the X direction")
        if grating_info["dy_critical_angle"] is not None:
            print(f"  - {grating_info['dy_critical_angle']:.1f}° in the Y direction")

    # Load the single antenna pattern
    single_antenna_path = sim_dir_path / single_antenna_filename
    print(f"Loading single antenna pattern from {single_antenna_path}")
    single_antenna_nf2ff = read_nf2ff(single_antenna_path)

    # Extract necessary data
    single_E_norm = single_antenna_nf2ff["E_norm"][0]
    single_Dmax = single_antenna_nf2ff["Dmax"][0]  # Assuming single frequency
    theta = single_antenna_nf2ff["theta"]
    phi = single_antenna_nf2ff["phi"]

    partial_phase = array_factor_partial_phase(theta, phi, freq, xn, yn, dx, dy)

    # Calculate array factor for all phi and theta values at once
    AF = array_factor_partial_and_shift(xn, yn, partial_phase, phase_shifts)

    # Multiply by single element pattern to get total pattern
    # The shape of AF is (n_phi, n_theta) after the calculation
    total_pattern = single_E_norm[0] * AF

    # Normalize
    total_pattern = total_pattern / np.max(np.abs(total_pattern))

    # Convert to dB (normalized directivity)
    array_gain = single_Dmax * (xn * yn)  # Theoretical array gain
    array_gain_db = 10.0 * np.log10(array_gain)
    total_pattern_db = 20 * np.log10(np.abs(total_pattern)) + array_gain_db

    return total_pattern_db


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
            "steering_info": h5f.get("steering_info", np.array([]))[:],
        }

        # Load metadata
        for key in h5f.attrs:
            dataset[key] = h5f.attrs[key]

    return dataset


DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR / "plots"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_samples(
    n_samples: int = 1,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    dataset_name: Path = DEFAULT_OUTFILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
):
    """
    Visualize a few samples from the dataset.

    Parameters:
    -----------
    dataset : dict
        Dataset loaded with load_dataset
    n_samples : int
        Number of samples to visualize
    """
    dataset_path = dataset_dir / dataset_name
    # Choose n_samples random indices
    dataset = load_dataset(dataset_path)
    indices = np.random.choice(
        len(dataset["patterns"]),
        size=min(n_samples, len(dataset["patterns"])),
        replace=False,
    )

    for idx in indices:
        plot_sample(idx, dataset_path, output_dir)


@app.command()
def plot_sample(
    idx: int,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    dataset_name: Path = DEFAULT_OUTFILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
):
    """
    Visualize a single sample from the dataset.

    Parameters:
    -----------
    dataset : dict
        Dataset loaded with load_dataset
    idx : int
        Index of the sample to visualize
    """
    dataset_path = dataset_dir / dataset_name
    dataset = load_dataset(dataset_path)
    theta, phi = dataset["theta"], dataset["phi"]
    freq = dataset["frequency"]

    # Create figure with three subplots: pattern, phase shifts, and polar pattern
    fig, axs = plt.subplots(1, 3, figsize=[18, 6])

    pattern = dataset["patterns"][idx]
    phase_shifts = dataset["labels"][idx]

    # pattern[pattern < 0] = 0  # Clip negative values to 0

    analyze.plot_phase_shifts(phase_shifts, title="Phase Shifts", ax=axs[0])
    analyze.plot_ff_2d(pattern, theta, phi, ax=axs[1])

    axs[2].remove()
    axs[2] = fig.add_subplot(1, 3, 3, projection="3d")
    pattern_3d = pattern[None, ...]  # Add freq dimension (first one) for 3D plot
    analyze.plot_ff_3d(theta, phi, pattern_3d, freq=freq, ax=axs[2])

    steering_info = dataset["steering_info"][idx]
    thetas_s, phis_s = steering_info
    thetas_s, phis_s = thetas_s[~np.isnan(thetas_s)], phis_s[~np.isnan(phis_s)]
    thetas_s = np.array2string(thetas_s, precision=2, separator=", ")
    phis_s = np.array2string(phis_s, precision=2, separator=", ")
    phase_shift_title = f"Phase Shifts (θ={thetas_s}°, φ={phis_s}°)"
    fig.suptitle(phase_shift_title)

    fig.set_tight_layout(True)
    if output_dir:
        fig.savefig(output_dir / f"sample_{idx}.png", dpi=600, bbox_inches="tight")


@app.command()
def count_beams(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    dataset_name: Path = "rand_bf_2d.h5",
):
    """
    Count the number of samples with single beam and multiple beams in the dataset.

    Parameters:
    -----------
    dataset_dir : Path
        Directory containing the dataset
    dataset_name : Path
        Name of the dataset file
    """
    dataset_path = dataset_dir / dataset_name
    print(f"Analyzing dataset: {dataset_path}")

    try:
        with h5py.File(dataset_path, "r") as h5f:
            if "steering_info" not in h5f:
                print("This dataset does not contain steering information.")
                return

            steering_info = h5f["steering_info"][:]
            total_samples = steering_info.shape[0]

            # Check if steering_info has the expected shape
            if len(steering_info.shape) < 3:
                print(
                    "Steering info doesn't have the expected format for beam counting."
                )
                return

            # Count samples with different number of beams
            # NaN values in theta angles indicate unused beams
            beam_counts = {}

            for i in range(total_samples):
                # Get theta values for this sample
                thetas = steering_info[i, 0, :]
                # Count non-NaN values to determine number of beams
                num_beams = np.sum(~np.isnan(thetas))

                # Update the counter
                if num_beams in beam_counts:
                    beam_counts[num_beams] += 1
                else:
                    beam_counts[num_beams] = 1

            print(f"\nTotal samples in dataset: {total_samples}")
            print("\nBeam distribution:")
            for num_beams, count in sorted(beam_counts.items()):
                percentage = (count / total_samples) * 100
                print(
                    f"  {num_beams} beam{'s' if num_beams != 1 else ''}: {count} samples ({percentage:.1f}%)"
                )

    except FileNotFoundError:
        print(f"Dataset file not found: {dataset_path}")
    except Exception as e:
        print(f"Error analyzing dataset: {e}")


if __name__ == "__main__":
    app()
