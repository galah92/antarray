import subprocess as sp
from functools import partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
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
DEFAULT_DATASET_NAME = "farfield_dataset.h5"
DEFAULT_SINGLE_ANT_FILENAME = "farfield_1x1_60x60_2450_steer_t0_p0.h5"


app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.command()
def simulate(sim_path: str = "antenna_array.py"):
    image_name = "openems-image"

    cmd = f"""
	docker run -it --rm \
		-e DISPLAY=host.docker.internal:0 \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v ./src:/app/ \
		-v /tmp:/tmp/ \
		{image_name} \
		python3 /app/{sim_path}
	"""
    sp.run(cmd, shell=True)


def generate_element_amplitudes(xn, yn, method="uniform", **kwargs):
    """
    Generate amplitude values for individual antenna elements using different methods.

    Parameters:
    -----------
    xn : int
        Number of elements in x direction
    yn : int
        Number of elements in y direction
    method : str
        Method to generate amplitudes:
        - "uniform": uniform amplitude for all elements (default 1.0)
        - "random": random amplitudes between min_amp and max_amp
        - "taper": apply tapering function (Hamming, Hanning, etc.)
        - "zones": divide array into zones with different amplitudes
        - "failure": randomly set some elements to zero to simulate failures

    kwargs:
        For "uniform": amplitude (default 1.0)
        For "random": min_amp, max_amp (default 0.5, 1.0)
        For "taper": taper_type (hamming, hanning, taylor, blackman, etc.)
        For "zones": n_zones, amplitude_values (list of amplitudes)
        For "failure": failure_rate (percentage of elements to fail)

    Returns:
    --------
    numpy.ndarray
        Amplitude values for each element, shape (xn, yn)
    """
    if method == "uniform":
        amplitude = kwargs.get("amplitude", 1.0)
        return np.ones((xn, yn)) * amplitude

    elif method == "random":
        min_amp = kwargs.get("min_amp", 0.5)
        max_amp = kwargs.get("max_amp", 1.0)
        return np.random.uniform(min_amp, max_amp, size=(xn, yn))

    elif method == "taper":
        taper_type = kwargs.get("taper_type", "hamming")

        # Create 1D window functions
        if taper_type == "hamming":
            window_x = np.hamming(xn)
            window_y = np.hamming(yn)
        elif taper_type == "hanning":
            window_x = np.hanning(xn)
            window_y = np.hanning(yn)
        elif taper_type == "blackman":
            window_x = np.blackman(xn)
            window_y = np.blackman(yn)
        elif taper_type == "taylor":
            # Simple approximation of Taylor window using Kaiser
            window_x = np.kaiser(xn, 3)
            window_y = np.kaiser(yn, 3)
        else:
            # Default to Hamming
            window_x = np.hamming(xn)
            window_y = np.hamming(yn)

        # Create 2D taper by multiplying the 1D windows
        return np.outer(window_x, window_y)

    elif method == "zones":
        n_zones = kwargs.get("n_zones", 4)
        amplitude_values = kwargs.get(
            "amplitude_values", np.linspace(0.5, 1.0, n_zones)
        )

        # Create zones (similar to phase shift zones)
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

        # Map zones to amplitude values
        amplitudes = np.zeros((xn, yn))
        for i in range(n_zones):
            amplitudes[zones == i] = amplitude_values[i]

        return amplitudes

    elif method == "failure":
        failure_rate = kwargs.get("failure_rate", 0.05)  # 5% failure by default
        amplitudes = np.ones((xn, yn))

        # Calculate number of elements to fail
        num_elements = xn * yn
        num_failures = int(num_elements * failure_rate)

        # Choose random elements to fail
        failure_indices = np.random.choice(
            num_elements, size=num_failures, replace=False
        )

        # Set failed elements to zero amplitude
        flat_amplitudes = amplitudes.flatten()
        flat_amplitudes[failure_indices] = 0.0

        return flat_amplitudes.reshape((xn, yn))

    else:
        raise ValueError(f"Unknown amplitude method: {method}")


def generate_element_phase_shifts(
    xn,
    yn,
    theta_steering=0.0,
    phi_steering=0.0,
    freq=2.45e9,
    dx=60,
    dy=60,
):
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
        phase_shift = k * np.sin(theta) * (x_grid * np.cos(phi) + y_grid * np.sin(phi))

        # Add this beam's excitation to the combined excitation
        combined_excitation += amplitude * np.exp(1j * phase_shift)

    # Extract phase from combined excitation
    combined_phase = np.angle(combined_excitation)

    return combined_phase


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
    sin_theta = sin_theta.T[:, :, None, None]
    cos_phi = cos_phi.T[:, :, None, None]
    sin_phi = sin_phi.T[:, :, None, None]

    # Make x_positions shape: (1, 1, xn, 1) and y_positions shape: (1, 1, 1, yn)
    x_positions = x_positions.reshape(1, 1, xn, 1)
    y_positions = y_positions.reshape(1, 1, 1, yn)

    # Calculate phase terms for all elements at once
    psi_x = k * dx_m * sin_theta * cos_phi
    psi_y = k * dy_m * sin_theta * sin_phi

    # Compute the partial phase for all elements and all angles at once
    partial_phase = x_positions * psi_x + y_positions * psi_y
    return partial_phase


@partial(jax.jit, static_argnums=(0, 1))
def array_factor_partial_and_shift(
    xn, yn, partial_phase, phase_shifts, amplitudes=None
):
    """Calculate array factor with individual element phase shifts and amplitudes."""
    # Make phase_shifts shape: (1, 1, xn, yn)
    phase_shifts = phase_shifts.reshape(1, 1, xn, yn)

    # Use uniform amplitudes if none provided
    if amplitudes is None:
        amplitudes = jnp.ones((xn, yn))

    # Make amplitudes shape: (1, 1, xn, yn)
    amplitudes = amplitudes.reshape(1, 1, xn, yn)

    # Compute the total phase for all elements and all angles at once
    total_phase = partial_phase - phase_shifts

    # Sum the complex exponentials across all elements, including amplitude weights
    AF = jnp.sum(amplitudes * jnp.exp(1j * total_phase), axis=(2, 3))

    # Normalize by total sum of amplitudes
    AF = AF / jnp.sum(amplitudes)

    return jnp.abs(AF)


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
        print("Array spacing check:")
        print(f"Wavelength: {'wavelength_mm':.2f} mm")
        print(f"Element spacing: {'dx_lambda':.1f}λ x {'dy_lambda':.1f}λ")

    if has_grating_lobes:
        print("WARNING: Grating lobes will be visible when steering beyond:")
        if dx_critical_angle is not None:
            print(f"  - {'dx_critical_angle':.1f}° in the X direction")
        if dy_critical_angle is not None:
            print(f"  - {'dy_critical_angle':.1f}° in the Y direction")


@app.command()
def plot_dataset_phase_shifts(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    dataset_name: Path = "ff_beamforming.h5",
    gif_name: Path = "beamforming_phase_shifts.gif",
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
        ani.save(dataset_dir / gif_name, writer=writer)


@app.command()
def generate_beamforming(
    n_samples: int = 1_000,
    theta_start: float = -65,  # Degrees
    theta_end: float = 65,  # Degrees
    phi_start: float = -65,  # Degrees
    phi_end: float = 65,  # Degrees
    max_n_beams: int = 1,
    amplitude_method: str = "uniform",  # New parameter
    sim_dir_path: Path = DEFAULT_SIM_DIR,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    dataset_name: Path = DEFAULT_DATASET_NAME,
    overwrite: bool = False,
    single_antenna_filename: str = DEFAULT_SINGLE_ANT_FILENAME,
):
    """
    Generate a dataset of farfield radiation patterns with individual element phase shifts and amplitudes.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    theta_start, theta_end : float
        Range of steering angles in theta (elevation) in degrees
    phi_start, phi_end : float
        Range of steering angles in phi (azimuth) in degrees
    max_n_beams : int
        Maximum number of beams to simulate
    amplitude_method : str
        Method to generate amplitude values: "uniform", "random", "taper", "zones", "failure"
    """
    # Fixed parameters for the 16x16 array
    xn = yn = 16  # 16x16 array
    dx = dy = 60  # 60x60 mm spacing
    freq = 2.45e9  # 2.45 GHz

    check_grating_lobes(freq, dx, dy)

    nf2ff_path = sim_dir_path / single_antenna_filename
    nf2ff = read_nf2ff(nf2ff_path)

    single_E_norm = nf2ff["E_norm"][0]
    single_Dmax = nf2ff["Dmax"][0]  # Assuming single frequency
    theta, phi = nf2ff["theta"], nf2ff["phi"]
    n_theta, n_phi = len(theta), len(phi)

    partial_phase = array_factor_partial_phase(theta, phi, freq, xn, yn, dx, dy)

    print(f"Generating dataset with {n_samples} samples...")

    mode = "w" if overwrite else "x"
    with h5py.File(dataset_dir / dataset_name, mode) as h5f:
        h5f.create_dataset("theta", data=theta)
        h5f.create_dataset("phi", data=phi)

        patterns = h5f.create_dataset("patterns", shape=(n_samples, n_phi, n_theta))
        labels = h5f.create_dataset("labels", shape=(n_samples, xn, yn))
        amplitudes = h5f.create_dataset("amplitudes", shape=(n_samples, xn, yn))
        steering_info = h5f.create_dataset(
            "steering_info", shape=(n_samples, 2, max_n_beams)
        )

        # Choose a random number of beams to simulate for each sample
        n_beams_sq = np.square(np.arange(max_n_beams) + 1)
        n_beams_prob = n_beams_sq / np.sum(n_beams_sq)  # consider softmax instead
        print(f"{n_beams_prob=}")
        n_beams = np.random.choice(max_n_beams, size=n_samples, p=n_beams_prob) + 1

        # Generate random steering angles within the specified range
        steering_size = (n_samples, max_n_beams)
        theta_steerings = np.random.uniform(theta_start, theta_end, size=steering_size)
        phi_steerings = np.random.uniform(phi_start, phi_end, size=steering_size)

        # Generate amplitude parameters based on method
        if amplitude_method == "random":
            # For each sample, choose a random range for amplitudes
            min_amps = np.random.uniform(0.3, 0.7, size=n_samples)
            max_amps = np.random.uniform(0.8, 1.0, size=n_samples)
        elif amplitude_method == "taper":
            # Choose random taper types for variety
            taper_types = np.random.choice(
                ["hamming", "hanning", "blackman", "taylor"], size=n_samples
            )
        elif amplitude_method == "zones":
            # Random number of zones for each sample
            n_zones_list = np.random.choice([4, 9, 16], size=n_samples)
        elif amplitude_method == "failure":
            # Random failure rates
            failure_rates = np.random.uniform(0.01, 0.1, size=n_samples)

        for i in tqdm(range(n_samples)):
            theta_steering, phi_steering = theta_steerings[i], phi_steerings[i]
            # Set steering angles to NaN for unused beams
            theta_steering[n_beams[i] :] = np.nan
            phi_steering[n_beams[i] :] = np.nan

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

            # Generate amplitudes based on the selected method
            if amplitude_method == "uniform":
                element_amplitudes = generate_element_amplitudes(xn, yn, "uniform")
            elif amplitude_method == "random":
                element_amplitudes = generate_element_amplitudes(
                    xn, yn, "random", min_amp=min_amps[i], max_amp=max_amps[i]
                )
            elif amplitude_method == "taper":
                element_amplitudes = generate_element_amplitudes(
                    xn, yn, "taper", taper_type=taper_types[i]
                )
            elif amplitude_method == "zones":
                element_amplitudes = generate_element_amplitudes(
                    xn, yn, "zones", n_zones=n_zones_list[i]
                )
            elif amplitude_method == "failure":
                element_amplitudes = generate_element_amplitudes(
                    xn, yn, "failure", failure_rate=failure_rates[i]
                )
            else:
                # Default to uniform amplitudes
                element_amplitudes = generate_element_amplitudes(xn, yn, "uniform")

            # Store amplitude values
            amplitudes[i] = element_amplitudes

            # Calculate array factor for all phi and theta values at once
            partial_phase = jnp.asarray(partial_phase)
            phase_shifts = jnp.asarray(phase_shifts)
            element_amplitudes = jnp.asarray(element_amplitudes)
            AF = array_factor_partial_and_shift(
                xn, yn, partial_phase, phase_shifts, element_amplitudes
            )

            # Multiply by single element pattern to get total pattern
            # The shape of AF is (n_phi, n_theta) after the calculation
            total_pattern = single_E_norm[0] * AF

            # Normalize
            total_pattern = total_pattern / np.max(np.abs(total_pattern))

            # Convert to dB (normalized directivity)
            array_gain = single_Dmax * np.sum(
                element_amplitudes
            )  # Adjusted for amplitudes
            array_gain_db = 10.0 * np.log10(array_gain)
            total_pattern_db = 20 * np.log10(np.abs(total_pattern)) + array_gain_db

            # Store pattern and label
            patterns[i] = total_pattern_db
            labels[i] = phase_shifts  # Store the phase shifts as labels


def ff_from_phase_shifts(
    phase_shifts: np.ndarray,
    sim_dir_path: Path = DEFAULT_SIM_DIR,
    single_antenna_filename: str = DEFAULT_SINGLE_ANT_FILENAME,
):
    # Fixed parameters for the 16x16 array
    xn = yn = 16  # 16x16 array
    dx = dy = 60  # 60x60 mm spacing
    freq = 2.45e9  # 2.45 GHz

    check_grating_lobes(freq, dx, dy)

    nf2ff_path = sim_dir_path / single_antenna_filename
    nf2ff = read_nf2ff(nf2ff_path)

    single_E_norm = nf2ff["E_norm"][0]
    single_Dmax = nf2ff["Dmax"][0]  # Assuming single frequency
    theta = nf2ff["theta"]
    phi = nf2ff["phi"]

    partial_phase = array_factor_partial_phase(theta, phi, freq, xn, yn, dx, dy)
    AF = array_factor_partial_and_shift(xn, yn, partial_phase, phase_shifts)

    # Multiply by single element pattern to get total pattern
    total_pattern = single_E_norm[0] * AF

    # Normalize
    total_pattern = total_pattern / np.max(np.abs(total_pattern))

    # Convert to dB (normalized directivity)
    array_gain = single_Dmax * (xn * yn)  # Theoretical array gain
    array_gain_db = 10.0 * np.log10(array_gain)
    total_pattern_db = 20 * np.log10(np.abs(total_pattern)) + array_gain_db

    return total_pattern_db


@app.command()
def plot_sample(
    idx: int,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    dataset_name: Path = DEFAULT_DATASET_NAME,
):
    """
    Visualize a single sample from the dataset.
    """
    dataset_path = dataset_dir / dataset_name
    with h5py.File(dataset_path, "r") as h5f:
        pattern = h5f["patterns"][idx]
        phase_shifts = h5f["labels"][idx]
        steering_info = h5f["steering_info"][idx]
        theta, phi = h5f["theta"][:], h5f["phi"][:]

        # Get amplitudes if they exist
        amplitudes = h5f["amplitudes"][idx] if "amplitudes" in h5f else None

    fig, axs = plt.subplots(
        1,
        4 if amplitudes is not None else 3,
        figsize=[24 if amplitudes is not None else 18, 6],
    )

    pattern = pattern.clip(min=0)

    analyze.plot_phase_shifts(phase_shifts, ax=axs[0])

    # Plot amplitudes if available
    if amplitudes is not None:
        cax = axs[1].imshow(amplitudes, origin="lower", cmap="viridis", vmin=0, vmax=1)
        axs[1].set_title("Element Amplitudes")
        axs[1].set_xlabel("Element X index")
        axs[1].set_ylabel("Element Y index")
        plt.colorbar(cax, ax=axs[1])
        analyze.plot_ff_2d(theta, phi, pattern, ax=axs[2])

        axs[3].remove()
        axs[3] = fig.add_subplot(1, 4, 4, projection="3d")
        analyze.plot_ff_3d(theta, phi, pattern, ax=axs[3])
    else:
        analyze.plot_ff_2d(theta, phi, pattern, ax=axs[1])

        axs[2].remove()
        axs[2] = fig.add_subplot(1, 3, 3, projection="3d")
        analyze.plot_ff_3d(theta, phi, pattern, ax=axs[2])

    steering_str = steering_repr(steering_info)
    phase_shift_title = f"Phase Shifts ({steering_str})"
    fig.suptitle(phase_shift_title)

    fig.set_tight_layout(True)

    sample_path = dataset_dir / f"sample_{idx}.png"
    fig.savefig(sample_path, dpi=600, bbox_inches="tight")
    print(f"Saved sample plot to {sample_path}")


def steering_repr(steering_angles: np.ndarray):
    thetas_s, phis_s = steering_angles
    thetas_s, phis_s = thetas_s[~np.isnan(thetas_s)], phis_s[~np.isnan(phis_s)]
    thetas_s = np.array2string(thetas_s, precision=2, separator=", ")
    phis_s = np.array2string(phis_s, precision=2, separator=", ")
    return f"θ={thetas_s}°, φ={phis_s}°"


@app.command()
def extract_n_beam_samples(
    n_beams: int,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    input_dataset: str = "rand_bf_2d_4k.h5",
    output_dataset: str = "rand_bf_2d_only_2k.h5",
):
    """
    Extracts samples with exactly n beams from a dataset and saves them to a new dataset file.

    Parameters:
    -----------
    dataset_dir : Path
        Directory containing the input dataset and where the output will be saved
    input_dataset : str
        Filename of the input dataset
    output_dataset : str
        Filename for the output dataset containing only n_beams-beam samples
    """
    input_path = dataset_dir / input_dataset
    output_path = dataset_dir / output_dataset

    print(f"Extracting {n_beams}-beam samples from {input_path} to {output_path}")

    # Open the input dataset
    with h5py.File(input_path, "r") as src_h5f:
        # Check if steering_info exists
        if "steering_info" not in src_h5f:
            print("Error: Input dataset doesn't contain steering information.")
            return

        # Get the steering information to identify n_beams-beam samples
        steering_info = src_h5f["steering_info"][:]

        # Find samples with exactly n_beams beams (n_beams non-NaN values in theta angles)
        samples_with_two_beams = []
        for i in range(steering_info.shape[0]):
            thetas = steering_info[i, 0, :]  # First row contains theta values
            num_beams = np.sum(~np.isnan(thetas))
            if num_beams == n_beams:
                samples_with_two_beams.append(i)

        # If no samples with n_beams beams found
        if not samples_with_two_beams:
            print(f"No samples with exactly {n_beams} beams found in the dataset.")
            return

        print(
            f"Found {len(samples_with_two_beams)} samples with exactly {n_beams} beams."
        )

        # Create output dataset
        with h5py.File(output_path, "w") as dst_h5f:
            # Copy theta and phi arrays directly
            dst_h5f.create_dataset("theta", data=src_h5f["theta"][:])
            dst_h5f.create_dataset("phi", data=src_h5f["phi"][:])

            # Get shapes for the new datasets
            n_samples = len(samples_with_two_beams)
            patterns_shape = (n_samples,) + src_h5f["patterns"].shape[1:]
            labels_shape = (n_samples,) + src_h5f["labels"].shape[1:]
            steering_shape = (n_samples,) + src_h5f["steering_info"].shape[1:]

            # Create datasets in the output file
            dst_patterns = dst_h5f.create_dataset("patterns", shape=patterns_shape)
            dst_labels = dst_h5f.create_dataset("labels", shape=labels_shape)
            dst_steering = dst_h5f.create_dataset("steering_info", shape=steering_shape)

            # Copy selected samples
            for new_idx, old_idx in enumerate(samples_with_two_beams):
                dst_patterns[new_idx] = src_h5f["patterns"][old_idx]
                dst_labels[new_idx] = src_h5f["labels"][old_idx]
                dst_steering[new_idx] = src_h5f["steering_info"][old_idx]

            # Copy attributes (metadata)
            for key in src_h5f.attrs:
                dst_h5f.attrs[key] = src_h5f.attrs[key]

            # Add additional metadata
            dst_h5f.attrs["description"] = (
                f"Dataset containing only samples with exactly {n_beams} beams"
            )
            dst_h5f.attrs["source_dataset"] = input_dataset

    print(
        f"Successfully created {n_beams}-beam dataset with {n_samples} samples at {output_path}"
    )


@app.command()
def unify_datasets(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    dataset1_name: str = "rand_bf_2d_only_20k.h5",
    dataset2_name: str = "rand_bf_2d_only_40k.h5",
    output_dataset: str = "rand_bf_2d_only_60k.h5",
    chunk_size: int = 1000,
):
    """
    Unify two dataset files into one, processing in chunks to handle large files.

    Parameters:
    -----------
    dataset_dir : Path
        Directory containing the datasets
    dataset1_name : str
        Filename of the first dataset
    dataset2_name : str
        Filename of the second dataset
    output_dataset : str
        Filename for the unified dataset
    chunk_size : int
        Number of samples to process at a time
    """
    dataset1_path = dataset_dir / dataset1_name
    dataset2_path = dataset_dir / dataset2_name
    output_path = dataset_dir / output_dataset

    print(f"Unifying datasets: {dataset1_path} and {dataset2_path} into {output_path}")

    with h5py.File(dataset1_path, "r") as ds1, h5py.File(dataset2_path, "r") as ds2:
        # Ensure theta and phi are identical in both datasets
        if not np.array_equal(ds1["theta"][:], ds2["theta"][:]) or not np.array_equal(
            ds1["phi"][:], ds2["phi"][:]
        ):
            print("Error: Theta and Phi arrays do not match between the datasets.")
            return

        # Get dataset sizes
        n_samples1 = ds1["patterns"].shape[0]
        n_samples2 = ds2["patterns"].shape[0]
        n_theta = ds1["patterns"].shape[2]
        n_phi = ds1["patterns"].shape[1]
        xn, yn = ds1["labels"].shape[1:]

        # Create the unified dataset
        with h5py.File(output_path, "w") as unified_ds:
            unified_ds.create_dataset("theta", data=ds1["theta"][:])
            unified_ds.create_dataset("phi", data=ds1["phi"][:])
            unified_ds.create_dataset(
                "patterns", shape=(n_samples1 + n_samples2, n_phi, n_theta)
            )
            unified_ds.create_dataset("labels", shape=(n_samples1 + n_samples2, xn, yn))
            unified_ds.create_dataset(
                "steering_info",
                shape=(n_samples1 + n_samples2,) + ds1["steering_info"].shape[1:],
            )

            # Copy attributes from the first dataset
            for key in ds1.attrs:
                unified_ds.attrs[key] = ds1.attrs[key]

            # Add metadata about the unification
            unified_ds.attrs["description"] = (
                f"Unified dataset combining {dataset1_name} and {dataset2_name}"
            )
            unified_ds.attrs["source_datasets"] = f"{dataset1_name}, {dataset2_name}"

            # Process and copy data in chunks
            def copy_in_chunks(src_ds, dst_ds, start_idx, chunk_size):
                for i in range(0, src_ds.shape[0], chunk_size):
                    end_idx = min(i + chunk_size, src_ds.shape[0])
                    dst_ds[start_idx : start_idx + (end_idx - i)] = src_ds[i:end_idx]
                    start_idx += end_idx - i
                return start_idx

            start_idx = 0
            start_idx = copy_in_chunks(
                ds1["patterns"], unified_ds["patterns"], start_idx, chunk_size
            )
            start_idx = copy_in_chunks(
                ds2["patterns"], unified_ds["patterns"], start_idx, chunk_size
            )

            start_idx = 0
            start_idx = copy_in_chunks(
                ds1["labels"], unified_ds["labels"], start_idx, chunk_size
            )
            start_idx = copy_in_chunks(
                ds2["labels"], unified_ds["labels"], start_idx, chunk_size
            )

            start_idx = 0
            start_idx = copy_in_chunks(
                ds1["steering_info"], unified_ds["steering_info"], start_idx, chunk_size
            )
            start_idx = copy_in_chunks(
                ds2["steering_info"], unified_ds["steering_info"], start_idx, chunk_size
            )

    print(f"Successfully unified datasets into {output_path}")


if __name__ == "__main__":
    app()
