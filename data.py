import logging
import subprocess as sp
from functools import partial
from pathlib import Path
from typing import NamedTuple

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


DEFAULT_ARRAY_SIZE = (16, 16)
DEFAULT_SPACING_MM = (60, 60)
DEFAULT_THETA_END = 65.0
DEFAULT_MAX_N_BEAMS = 4


class DataBatch(NamedTuple):
    radiation_patterns: jax.Array  # (n_theta, n_phi, 3) - pattern & trig encoding
    phase_shifts: jax.Array  # (array_x, array_y)
    steering_angles: jax.Array  # (n_beams, 2) - theta, phi in radians


class Dataset:
    def __init__(
        self,
        batch_size: int = 512,
        limit: int = None,
        theta_end: float = DEFAULT_THETA_END,
        array_size: tuple[int, int] = DEFAULT_ARRAY_SIZE,
        spacing_mm: tuple[float, float] = DEFAULT_SPACING_MM,
        max_n_beams: int = DEFAULT_MAX_N_BEAMS,
        sim_dir_path: Path = DEFAULT_SIM_DIR,
        key: jax.Array = None,
        prefetch: bool = True,
        clip: bool = True,
        normalize: bool = True,
        front_hemisphere: bool = True,
        radiation_pattern_max=30,  # Maximum radiation pattern value in dB observed
        trig_encoding: bool = True,
    ):
        self.batch_size = batch_size
        self.limit = limit
        self.count = 0

        self.theta_end = jnp.radians(theta_end)
        self.sim_dir_path = sim_dir_path
        self.prefetch = prefetch
        self.clip = clip
        self.normalize = normalize
        self.radiation_pattern_max = radiation_pattern_max
        self.trig_encoding = trig_encoding

        if key is None:
            key = jax.random.key(0)
        self.key = key

        # Load and prepare array parameters
        array_params = analyze.calc_array_params2(
            array_size=array_size,
            spacing_mm=spacing_mm,
            theta_rad=jnp.radians(jnp.arange(90 if front_hemisphere else 180)),
            phi_rad=jnp.radians(jnp.arange(360)),
            sim_path=sim_dir_path / DEFAULT_SINGLE_ANT_FILENAME,
        )

        # Convert to JAX arrays and make them static
        self.array_params = [jnp.asarray(param) for param in array_params]

        # Beam probability distribution
        self.n_beams_prob = get_beams_prob(max_n_beams)

        self.rad_pattern_from_steering = partial(
            analyze.rad_pattern_from_geo,
            *self.array_params,
        )

        # Precompute trigonometric encoding channels
        phi_rad = jnp.arange(360) * jnp.pi / 180
        self.sin_phi = jnp.sin(phi_rad)[None, :] * jnp.ones((90, 1))
        self.cos_phi = jnp.cos(phi_rad)[None, :] * jnp.ones((90, 1))

        self.vmapped_generate_sample = jax.vmap(self.generate_sample)

        self._prefetched_batch = None
        if self.prefetch:
            self._prefetched_batch = self.generate_batch()

    def generate_sample(self, key: jax.Array) -> DataBatch:
        key1, key2 = jax.random.split(key)

        # Generate random steering angles for multiple beams
        n_beams = 2
        theta_steering = jax.random.uniform(key1, (n_beams,)) * self.theta_end
        phi_steering = jax.random.uniform(key2, (n_beams,)) * (2 * jnp.pi)
        steering_angles = jnp.stack((theta_steering, phi_steering), axis=-1)

        radiation_pattern, excitations = self.rad_pattern_from_steering(steering_angles)
        phase_shifts = jnp.angle(excitations)

        if self.clip:
            radiation_pattern = jnp.clip(radiation_pattern, a_min=0.0)

        if self.normalize:
            radiation_pattern = radiation_pattern / self.radiation_pattern_max

        if self.trig_encoding:
            arrays = [radiation_pattern, self.sin_phi, self.cos_phi]
            radiation_pattern = jnp.stack(arrays, axis=-1)

        return DataBatch(radiation_pattern, phase_shifts, steering_angles)

    def generate_batch(self) -> DataBatch:
        self.key, batch_key = jax.random.split(self.key)
        sample_keys = jax.random.split(batch_key, self.batch_size)
        samples = self.vmapped_generate_sample(sample_keys)
        return samples

    def __next__(self) -> DataBatch:
        if self.limit is not None and self.count >= self.limit:
            raise StopIteration
        self.count += 1

        if self.prefetch:
            current_batch = self._prefetched_batch
            self._prefetched_batch = self.generate_batch()
            return current_batch
        else:
            return self.generate_batch()

    def __iter__(self):
        self.count = 0
        return self


@app.command()
def generate_beamforming(
    n_samples: int = 1_000,
    theta_end: float = 65,  # Degrees
    max_n_beams: int = 1,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    dataset_name: Path = DEFAULT_DATASET_NAME,
    overwrite: bool = False,
    seed: int = 42,
):
    array_size = (16, 16)
    theta_rad = np.radians(np.arange(90))
    phi_rad = np.radians(np.arange(360))
    batch_size = min(512, n_samples)

    dataset = Dataset(
        batch_size=batch_size,
        limit=np.ceil(n_samples / batch_size).astype(np.int64),
        array_size=array_size,
        spacing_mm=(60, 60),
        theta_end=theta_end,
        max_n_beams=max_n_beams,
        key=jax.random.key(seed),
        clip=False,
        normalize=False,
        trig_encoding=False,
    )

    logger.info(f"Generating dataset with {n_samples} samples")

    mode = "w" if overwrite else "x"
    with h5py.File(dataset_dir / dataset_name, mode) as h5f:
        h5f.create_dataset("theta", data=theta_rad)
        h5f.create_dataset("phi", data=phi_rad)

        patterns_shape = (n_samples, theta_rad.size, phi_rad.size)
        patterns_ds = h5f.create_dataset("patterns", shape=patterns_shape)
        ex_shape = (n_samples, *array_size)
        ex_ds = h5f.create_dataset("excitations", shape=ex_shape, dtype=np.complex64)

        for i, batch in tqdm(enumerate(dataset), total=dataset.limit):
            n = min(batch_size, n_samples - i)  # Handle the last batch
            patterns_ds[i : i + n] = batch.radiation_patterns[:n]
            ex_ds[i : i + n] = batch.phase_shifts[:n]

        # h5f.create_dataset("steering", data=steerings)


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
