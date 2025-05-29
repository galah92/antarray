import logging
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

logger = logging.getLogger(__name__)

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


def get_beams_prob(max_n_beams: int = 1):
    """
    Calculate the probability distribution for the number of beams.
    The probability is proportional to the square of the number of beams.
    """
    n_beams_sq = np.square(np.arange(max_n_beams) + 1)
    n_beams_prob = n_beams_sq / np.sum(n_beams_sq)  # Normalize to sum to 1
    return n_beams_prob


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
    array_size = (16, 16)
    theta_rad = np.radians(np.arange(180))
    phi_rad = np.radians(np.arange(360))

    logger.info(f"Generating dataset with {n_samples} samples")

    array_params = analyze.calc_array_params2(
        array_size=array_size,
        spacing_mm=(60, 60),
        theta_rad=theta_rad,
        phi_rad=phi_rad,
        sim_path=sim_dir_path / single_antenna_filename,
    )

    # Create a function to calculate E_norm and excitations for given steering angles
    static_params = jax.tree_util.tree_map(jnp.asarray, array_params)  # Convert to JAX
    rad_pattern_from_steering = partial(analyze.rad_pattern_from_geo, *static_params)

    # Choose a random number of beams to simulate for each sample
    n_beams_prob = get_beams_prob(max_n_beams)
    n_beams = np.random.choice(max_n_beams, size=n_samples, p=n_beams_prob) + 1

    # Generate random steering angles within the specified range
    steering_size = (n_samples, max_n_beams)
    theta_steerings = np.random.uniform(0, theta_end, size=steering_size)
    phi_steerings = np.random.uniform(0, 360, size=steering_size)
    steerings = np.dstack([theta_steerings, phi_steerings])

    mode = "w" if overwrite else "x"
    with h5py.File(dataset_dir / dataset_name, mode) as h5f:
        h5f.create_dataset("theta", data=theta_rad)
        h5f.create_dataset("phi", data=phi_rad)

        patterns_shape = (n_samples, theta_rad.size, phi_rad.size)
        patterns_ds = h5f.create_dataset("patterns", shape=patterns_shape)
        ex_shape = (n_samples, *array_size)
        ex_ds = h5f.create_dataset("excitations", shape=ex_shape, dtype=np.complex64)

        for i in tqdm(range(n_samples)):
            steer = steerings[i]
            steer[n_beams[i] :] = np.nan  # Set steering angles to NaN for unused beams

            valid_steer = np.radians(steer[~np.isnan(steer).any(axis=1)])
            E_norm, excitations = rad_pattern_from_steering(jnp.asarray(valid_steer))

            patterns_ds[i] = E_norm
            ex_ds[i] = excitations

        h5f.create_dataset("steering", data=steerings)


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
        cbar.set_ticklabels(np.linspace(-180, 180, 7, dtype=np.int32))

        im = np.zeros_like(np.angle(excitations[0]))
        cmap = "twilight_shifted"  # Cyclic colormap for phase values
        im = ax.imshow(im, cmap=cmap, origin="lower", vmin=-180, vmax=180)

        def animate(i):
            logger.debug(f"Plotting {i}")
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

    steering_str = analyze.steering_repr(steering)
    phase_shift_title = f"Phase Shifts ({steering_str})"
    fig.suptitle(phase_shift_title)
    fig.set_tight_layout(True)

    sample_path = dataset_dir / f"sample_{idx}.png"
    fig.savefig(sample_path, dpi=600, bbox_inches="tight")
    logger.info(f"Saved sample plot to {sample_path}")


def ff_from_phase_shifts(
    phase_shifts: np.ndarray,
    sim_dir_path: Path = DEFAULT_SIM_DIR,
    single_antenna_filename: str = DEFAULT_SINGLE_ANT_FILENAME,
):
    _, _, taper, geo_exp, E_field, Dmax_array = analyze.calc_array_params2(
        array_size=(16, 16),
        spacing_mm=(60, 60),
        sim_path=sim_dir_path / single_antenna_filename,
    )

    E_norm, _ = analyze.rad_pattern_from_geo_and_phase_shifts(
        taper,
        geo_exp,
        E_field,
        Dmax_array,
        phase_shifts,
    )

    return E_norm


if __name__ == "__main__":
    app()
