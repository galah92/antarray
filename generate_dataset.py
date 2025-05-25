import subprocess as sp
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
DEFAULT_DATASET_NAME = "farfield_dataset.h5"
DEFAULT_SINGLE_ANT_FILENAME = "ff_1x1_60x60_2450_steer_t0_p0.h5"


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
def generate_beamforming(
    n_samples: int = 1_000,
    theta_end: float = 65,  # Degrees
    max_n_beams: int = 1,
    sim_dir_path: Path = DEFAULT_SIM_DIR,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    dataset_name: Path = DEFAULT_DATASET_NAME,
    overwrite: bool = False,
    single_antenna_filename: str = DEFAULT_SINGLE_ANT_FILENAME,
):
    # Fixed parameters for the 16x16 array
    xn = yn = 16  # 16x16 array
    dx = dy = 60  # 60x60 mm spacing
    freq_hz = 2.45e9  # 2.45 GHz

    check_grating_lobes(freq_hz, dx, dy)

    nf2ff = read_nf2ff(sim_dir_path / single_antenna_filename)

    freq_idx = 0  # index of the frequency to plot
    theta_rad, phi_rad = nf2ff["theta_rad"], nf2ff["phi_rad"]
    E_theta_single, E_phi_single = nf2ff["E_theta"][freq_idx], nf2ff["E_phi"][freq_idx]
    Dmax_single = nf2ff["Dmax"][freq_idx]
    Dmax_array = Dmax_single * (xn * yn)

    print(f"Generating dataset with {n_samples} samples")

    mode = "w" if overwrite else "x"
    with h5py.File(dataset_dir / dataset_name, mode) as h5f:
        h5f.create_dataset("theta", data=theta_rad)
        h5f.create_dataset("phi", data=phi_rad)

        patterns_ds = h5f.create_dataset(
            "patterns", shape=(n_samples, theta_rad.size, phi_rad.size)
        )
        excitations_ds = h5f.create_dataset(
            "excitations", shape=(n_samples, xn, yn), dtype=np.complex64
        )
        steering_ds = h5f.create_dataset("steering", shape=(n_samples, 2, max_n_beams))

        # Choose a random number of beams to simulate for each sample
        n_beams_sq = np.square(np.arange(max_n_beams) + 1)
        n_beams_prob = n_beams_sq / np.sum(n_beams_sq)  # consider softmax instead
        n_beams = np.random.choice(max_n_beams, size=n_samples, p=n_beams_prob) + 1

        # Generate random steering angles within the specified range
        steering_size = (n_samples, max_n_beams)
        theta_steerings = np.random.uniform(0, theta_end, size=steering_size)
        phi_steerings = np.random.uniform(0, 360, size=steering_size)

        ex_calc = analyze.ExcitationCalculator(xn, yn, dx, dy, freq_hz)
        af_calc = analyze.ArrayFactorCalculator(
            theta_rad, phi_rad, xn, yn, dx, dy, freq_hz
        )

        for i in tqdm(range(n_samples)):
            theta_steering, phi_steering = theta_steerings[i], phi_steerings[i]
            # Set steering angles to NaN for unused beams
            theta_steering[n_beams[i] :] = np.nan
            phi_steering[n_beams[i] :] = np.nan

            # Calculate the excitations for the current steering angles using superposition
            excitations = np.zeros((xn, yn), dtype=excitations_ds.dtype)
            for i_beam in range(n_beams[i]):
                excitations += ex_calc(theta_steering[i_beam], phi_steering[i_beam])

            AF = af_calc(excitations=excitations)
            E_norm = analyze.run_array_factor(E_theta_single, E_phi_single, AF)
            E_norm = analyze.normalize_pattern(E_norm, Dmax_array)

            patterns_ds[i] = E_norm
            excitations_ds[i] = excitations
            steering_ds[i] = [theta_steering, phi_steering]


@app.command()
def plot_dataset_phase_shifts(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    dataset_name: Path = "ff_beamforming.h5",
    gif_name: Path = "beamforming_phase_shifts.gif",
):
    with h5py.File(dataset_dir / dataset_name, "r") as h5f:
        excitations = h5f["excitations"]
        steering = h5f["steering"]

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
        cbar.set_tickexcitations(np.linspace(-180, 180, 7, dtype=np.int32))

        im = np.zeros_like(np.angle(excitations[0]))
        cmap = "twilight_shifted"  # Cyclic colormap for phase values
        im = ax.imshow(im, cmap=cmap, origin="lower", vmin=-180, vmax=180)

        def animate(i):
            print(f"Plotting {i}")
            j = i * 111

            # Update the data for the plot
            phase_shifts = np.angle(excitations[j])
            phase_shifts_clipped = (phase_shifts + np.pi) % (2 * np.pi) - np.pi
            im.set_data(np.rad2deg(phase_shifts_clipped))

            # Update the title with steering info
            theta_s, phi_s = steering[j]
            text = f"Phase Shifts (θ={theta_s:03.1f}°, φ={phi_s:03.1f}°) {i:04d}"
            title.set_text(text)

            return im, title

        frames = len(excitations)
        frames = 111 * 2
        ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True)

        # To save the animation using Pillow as a gif
        writer = animation.PillowWriter(fps=20, bitrate=1800)
        ani.save(dataset_dir / gif_name, writer=writer)


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
        phase_shifts = np.angle(h5f["excitations"][idx])
        steering = h5f["steering"][idx]
        theta, phi = h5f["theta"][:], h5f["phi"][:]

    fig, axs = plt.subplots(1, 3, figsize=[18, 6])

    pattern = pattern.clip(min=0)

    analyze.plot_phase_shifts(phase_shifts, ax=axs[0])

    analyze.plot_ff_2d(theta, phi, pattern, ax=axs[1])

    axs[2].remove()
    axs[2] = fig.add_subplot(1, 3, 3, projection="3d")
    analyze.plot_ff_3d(theta, phi, pattern, ax=axs[2])

    steering_str = steering_repr(steering)
    phase_shift_title = f"Phase Shifts ({steering_str})"
    fig.suptitle(phase_shift_title)
    fig.set_tight_layout(True)

    sample_path = dataset_dir / f"sample_{idx}.png"
    fig.savefig(sample_path, dpi=600, bbox_inches="tight")
    print(f"Saved sample plot to {sample_path}")


def ff_from_phase_shifts(
    phase_shifts: np.ndarray,
    sim_dir_path: Path = DEFAULT_SIM_DIR,
    single_antenna_filename: str = DEFAULT_SINGLE_ANT_FILENAME,
):
    # Fixed parameters for the 16x16 array
    xn = yn = 16  # 16x16 array
    dx = dy = 60  # 60x60 mm spacing
    freq_hz = 2.45e9  # 2.45 GHz

    check_grating_lobes(freq_hz, dx, dy)

    nf2ff = read_nf2ff(sim_dir_path / single_antenna_filename)

    freq_idx = 0  # index of the frequency to plot
    theta_rad, phi_rad = nf2ff["theta_rad"], nf2ff["phi_rad"]
    E_theta_single, E_phi_single = nf2ff["E_theta"][freq_idx], nf2ff["E_phi"][freq_idx]
    Dmax_single = nf2ff["Dmax"][freq_idx]
    Dmax_array = Dmax_single * (xn * yn)

    ex_calc = analyze.ExcitationCalculator(xn, yn, dx, dy, freq_hz)
    af_calc = analyze.ArrayFactorCalculator(theta_rad, phi_rad, xn, yn, dx, dy, freq_hz)
    AF = af_calc(excitations=ex_calc(phase_shifts))
    E_norm = analyze.run_array_factor(E_theta_single, E_phi_single, AF)
    E_norm = analyze.normalize_pattern(E_norm, Dmax_array)

    return E_norm


def steering_repr(steering_angles: np.ndarray):
    thetas_s, phis_s = steering_angles
    thetas_s, phis_s = thetas_s[~np.isnan(thetas_s)], phis_s[~np.isnan(phis_s)]
    thetas_s = np.array2string(thetas_s, precision=2, separator=", ")
    phis_s = np.array2string(phis_s, precision=2, separator=", ")
    return f"θ={thetas_s}°, φ={phis_s}°"


if __name__ == "__main__":
    app()
