import logging
import subprocess as sp
from functools import partial
from pathlib import Path
from typing import NamedTuple

import cyclopts
import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from tqdm import tqdm

import physics

logger = logging.getLogger(__name__)

app = cyclopts.App()

root_dir = Path(__file__).parent.parent

DEFAULT_DATASET_DIR = root_dir / "dataset"
DEFAULT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DATASET_NAME = "farfield_dataset.h5"


@app.command()
def simulate(sim_path: str = "antenna_array.py"):
    image_name = "openems-image"

    cmd = f"""
	docker run -it --rm \
		-e DISPLAY=host.docker.internal:0 \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v ./openems:/app/ \
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


class DataBatch(NamedTuple):
    radiation_patterns: jax.Array  # (n_theta, n_phi, 3) - pattern & trig encoding
    phase_shifts: jax.Array  # (array_x, array_y)
    steering_angles: jax.Array  # (n_beams, 2) - theta, phi in radians


def create_sample_generator(
    theta_end: float,
    rad_pattern_from_steering,
    transform_fn,
):
    """Create a JIT-compiled sample generation function."""

    @jax.jit
    def generate_sample(key: jax.Array) -> DataBatch:
        key1, key2 = jax.random.split(key)

        # Generate random steering angles for multiple beams
        n_beams = 2
        theta_steering = jax.random.uniform(key1, (n_beams,)) * theta_end
        phi_steering = jax.random.uniform(key2, (n_beams,)) * (2 * jnp.pi)
        steering_angles = jnp.stack((theta_steering, phi_steering), axis=-1)

        radiation_pattern, excitations = rad_pattern_from_steering(steering_angles)
        phase_shifts = jnp.angle(excitations)
        radiation_pattern = transform_fn(radiation_pattern)

        return DataBatch(radiation_pattern, phase_shifts, steering_angles)

    return generate_sample


def create_radiation_pattern_transform(
    clip: bool = True,
    normalize: bool = True,
    radiation_pattern_max: float = 30.0,
    trig_encoding: bool = True,
):
    """Create a JIT-compiled radiation pattern transformation function."""

    @jax.jit
    def transform(
        radiation_pattern: ArrayLike,
        sin_phi: ArrayLike,
        cos_phi: ArrayLike,
    ) -> jax.Array:
        if clip:
            radiation_pattern = jnp.clip(radiation_pattern, a_min=0.0)

        if normalize:
            radiation_pattern = radiation_pattern / radiation_pattern_max

        if trig_encoding:
            channels = [radiation_pattern, sin_phi, cos_phi]
            radiation_pattern = jnp.stack(channels, axis=-1)

        return radiation_pattern

    # Precompute trigonometric encoding channels
    n_theta = 180
    phi_rad = jnp.arange(360) * jnp.pi / 180
    sin_phi = jnp.tile(jnp.sin(phi_rad), reps=(n_theta, 1))
    cos_phi = jnp.tile(jnp.cos(phi_rad), reps=(n_theta, 1))

    # Return a specialized transform function with sin_phi/cos_phi baked in
    return jax.jit(partial(transform, sin_phi=sin_phi, cos_phi=cos_phi))


class Dataset:
    def __init__(
        self,
        batch_size: int = 512,
        limit: int | None = None,
        prefetch: bool = True,
        array_size: tuple[int, int] = (16, 16),
        spacing_mm: tuple[float, float] = (60, 60),
        theta_end: float = 65.0,
        max_n_beams: int = 4,
        clip: bool = True,
        normalize: bool = True,
        radiation_pattern_max: float = 30.0,  # Maximum radiation pattern value in dB observed
        trig_encoding: bool = True,
        key: jax.Array | None = None,
        kind: physics.Kind = "cst",
    ):
        self.batch_size = batch_size
        self.limit = limit
        self.count = 0
        self.prefetch = prefetch

        self.array_size = array_size
        self.spacing_mm = spacing_mm
        self.theta_end = jnp.radians(theta_end)
        self.kind = kind

        self.clip = clip
        self.normalize = normalize
        self.radiation_pattern_max = radiation_pattern_max
        self.trig_encoding = trig_encoding

        if key is None:
            key = jax.random.key(0)
        self.key = key

        config = physics.ArrayConfig(array_size=array_size, spacing_mm=spacing_mm)
        element_data = physics.load_aeps(config, kind=self.kind)
        aeps = element_data.aeps
        config = element_data.config  # Use the config from the loaded data
        kx, ky = physics.compute_spatial_phase_coeffs(config)
        element_fields = physics.compute_element_fields(aeps, config)

        def unified_pattern_from_steering(steering_angles):
            """Generate patterns using unified physics interface."""
            weights, _ = physics.calculate_weights(kx, ky, steering_angles)
            power_pattern = physics.synthesize_pattern(
                element_fields, weights, power=True
            )
            return power_pattern, weights

        self.rad_pattern_from_steering = unified_pattern_from_steering

        self.transform_fn = create_radiation_pattern_transform(
            clip=clip,
            normalize=normalize,
            radiation_pattern_max=radiation_pattern_max,
            trig_encoding=trig_encoding,
        )

        self.generate_sample = create_sample_generator(
            self.theta_end,
            self.rad_pattern_from_steering,
            self.transform_fn,
        )

        self.generate_batch = jax.jit(
            lambda key: jax.vmap(self.generate_sample)(
                jax.random.split(key, self.batch_size)
            )
        )

        self.prefetched_batch = None
        if self.prefetch:
            self.key, batch_key = jax.random.split(self.key)
            self.prefetched_batch = self.generate_batch(batch_key)

    def __next__(self) -> DataBatch:
        if self.limit is not None and self.count >= self.limit:
            raise StopIteration
        self.count += 1

        self.key, batch_key = jax.random.split(self.key)
        if self.prefetch:
            current_batch = self.prefetched_batch
            self.prefetched_batch = self.generate_batch(batch_key)
            return current_batch
        else:
            return self.generate_batch(batch_key)

    def __iter__(self):
        self.count = 0
        return self


@app.command()
def generate_beamforming(
    n_samples: int = 1_000,
    theta_end: float = 65,  # Degrees
    max_n_beams: int = 1,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    dataset_name: str = DEFAULT_DATASET_NAME,
    overwrite: bool = False,
    seed: int = 42,
    kind: physics.Kind = "cst",
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
        kind=kind,
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
            k = i * batch_size
            n = min(batch_size, n_samples - k)  # Handle the last batch
            patterns_ds[k : k + n] = batch.radiation_patterns[:n]
            ex_ds[k : k + n] = batch.phase_shifts[:n]


if __name__ == "__main__":
    app()
