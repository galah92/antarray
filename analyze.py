from functools import lru_cache, partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


@lru_cache
def read_nf2ff(nf2ff_path: Path):
    print(f"Loading single antenna pattern from {nf2ff_path}")
    with h5py.File(nf2ff_path, "r") as h5:
        mesh = h5["Mesh"]
        phi, theta, r = mesh["phi"][:], mesh["theta"][:], mesh["r"][:]

        Dmax = h5["nf2ff"].attrs["Dmax"]
        freqs = h5["nf2ff"].attrs["Frequency"]

        E_shape = freqs.size, phi.size, theta.size
        E_theta = np.empty(E_shape, dtype=complex)
        E_phi = np.empty(E_shape, dtype=complex)

        for freq in range(freqs.size):
            E_theta.real[freq] = h5[f"/nf2ff/E_theta/FD/f{freq}_real"][:]
            E_theta.imag[freq] = h5[f"/nf2ff/E_theta/FD/f{freq}_imag"][:]
            E_phi.real[freq] = h5[f"/nf2ff/E_phi/FD/f{freq}_real"][:]
            E_phi.imag[freq] = h5[f"/nf2ff/E_phi/FD/f{freq}_imag"][:]

        # Transpose to (freq, theta, phi)
        E_theta = E_theta.transpose(0, 2, 1)
        E_phi = E_phi.transpose(0, 2, 1)

        E_norm = np.sqrt(np.abs(E_phi) ** 2 + np.abs(E_theta) ** 2)

    return {
        "theta": theta,
        "phi": phi,
        "r": r,
        "Dmax": Dmax,
        "freq": freqs,
        "E_theta": E_theta,
        "E_phi": E_phi,
        "E_norm": E_norm,
    }


def get_wavenumber(freq_hz: float) -> float:
    c = 299792458  # Speed of light in m/s
    wavelength = c / freq_hz  # Wavelength in meters
    k = 2 * np.pi / wavelength  # Wavenumber in radians/meter
    return k


def get_element_positions(
    xn: int,
    yn: int,
    dx_mm: float = 60,
    dy_mm: float = 60,
) -> tuple[np.ndarray, np.ndarray]:
    dx_m, dy_m = dx_mm / 1000, dy_mm / 1000  # Convert element spacing from mm to meters
    x_positions = (np.arange(xn) - (xn - 1) / 2) * dx_m
    y_positions = (np.arange(yn) - (yn - 1) / 2) * dy_m
    return x_positions, y_positions


def array_factor(
    theta: np.ndarray,  # Array of observation theta angles (radians)
    phi: np.ndarray,  # Array of observation phi angles (radians)
    xn: int,
    yn: int,
    dx_mm: float = 60,
    dy_mm: float = 60,
    freq_hz: float = 2.45e9,
    excitations: np.ndarray | None = None,
) -> np.ndarray:
    k = get_wavenumber(freq_hz)

    theta, phi = np.asarray(theta), np.asarray(phi)

    dx_m = dx_mm / 1000  # Convert from mm to meters
    dy_m = dy_mm / 1000  # Convert from mm to meters

    # Element positions (centered around origin)
    x_pos = np.arange(xn) - (xn - 1) / 2
    y_pos = np.arange(yn) - (yn - 1) / 2

    # Create a meshgrid for broadcasting
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

    # Phase differences
    sin_theta = np.sin(THETA)
    psi_x = k * dx_m * sin_theta * np.cos(PHI)
    psi_y = k * dy_m * sin_theta * np.sin(PHI)

    # Initialize default excitations if none provided
    if excitations is None:
        excitations = np.ones((xn, yn), dtype=complex)

    phase_shifts = np.angle(excitations)

    # Calculate array factor by summing contributions from each element
    AF = np.zeros((theta.size, phi.size), dtype=complex)
    for ix in range(xn):
        for iy in range(yn):
            phase = x_pos[ix] * psi_x + y_pos[iy] * psi_y - phase_shifts[ix, iy]
            AF += np.exp(1j * phase)

    AF = AF / (xn * yn)  # Normalize by total number of elements

    return AF


class ArrayFactorCalculator:
    """
    A class to calculate the array factor for a given array configuration.
    It pre-calculates terms dependent on geometry and observation angles
    to efficiently compute the array factor for different excitations.
    """

    def __init__(
        self,
        theta: np.ndarray,  # Array of observation theta angles (radians)
        phi: np.ndarray,  # Array of observation phi angles (radians)
        xn: int,
        yn: int,
        dx_mm: float = 60,
        dy_mm: float = 60,
        freq_hz: float = 2.45e9,
    ):
        self.theta = np.asarray(theta)
        self.phi = np.asarray(phi)
        self.xn = xn
        self.yn = yn
        self.dx_mm = dx_mm
        self.dy_mm = dy_mm
        self.freq_hz = freq_hz

        self._precompute_terms()

    def _precompute_terms(self):
        """
        Pre-computes terms that depend only on the array geometry
        and observation angles. These include k, element positions,
        and the phase components (psi_x, psi_y) for each element.
        """
        self.k = get_wavenumber(self.freq_hz)

        dx_m = self.dx_mm / 1000  # Convert from mm to meters
        dy_m = self.dy_mm / 1000  # Convert from mm to meters

        # Element positions (centered around origin)
        self.x_pos = np.arange(self.xn) - (self.xn - 1) / 2
        self.y_pos = np.arange(self.yn) - (self.yn - 1) / 2

        # Create a meshgrid for broadcasting
        THETA, PHI = np.meshgrid(self.theta, self.phi, indexing="ij")

        # Phase differences per unit step in x and y (psi_x_unit, psi_y_unit)
        sin_theta = np.sin(THETA)
        self.psi_x_unit = self.k * dx_m * sin_theta * np.cos(PHI)
        self.psi_y_unit = self.k * dy_m * sin_theta * np.sin(PHI)

        # Precompute the 'geometry-dependent' phase term for each element and angle
        # This will be (x_pos[ix] * psi_x_unit + y_pos[iy] * psi_y_unit)
        self.geometry_phase_terms = np.zeros(
            (self.xn, self.yn, self.theta.size, self.phi.size), dtype=complex
        )

        for ix in range(self.xn):
            for iy in range(self.yn):
                self.geometry_phase_terms[ix, iy, :, :] = (
                    self.x_pos[ix] * self.psi_x_unit + self.y_pos[iy] * self.psi_y_unit
                )

    def calculate_af(self, excitations: np.ndarray | None = None) -> np.ndarray:
        """
        Calculates the array factor given the element excitations.
        Assumes excitations are (xn, yn) complex array.
        If excitations is None, all elements are assumed to have 1 + 0j excitation.
        """
        # Initialize default excitations if none provided
        if excitations is None:
            excitations = np.ones((self.xn, self.yn), dtype=complex)
        elif excitations.shape != (self.xn, self.yn):
            raise ValueError(
                f"Excitations must be a ({self.xn}, {self.yn}) array, "
                f"but got {excitations.shape}"
            )

        # Ensure excitations are complex
        excitations = excitations.astype(complex)

        # Extract phase shifts from excitations
        phase_shifts = np.angle(excitations)

        # Initialize array factor sum
        AF = np.zeros((self.theta.size, self.phi.size), dtype=complex)

        # Iterate through elements, applying the excitation phase
        for ix in range(self.xn):
            for iy in range(self.yn):
                # The total phase for this element is the precomputed geometry phase
                # minus the excitation phase shift for this element.
                total_phase = (
                    self.geometry_phase_terms[ix, iy, :, :] - phase_shifts[ix, iy]
                )
                AF += excitations[ix, iy] * np.exp(
                    1j * total_phase
                )  # Multiply by magnitude as well

        AF = AF / (self.xn * self.yn)  # Normalize by total number of elements

        return AF


def array_factor3(
    theta: np.ndarray,
    phi: np.ndarray,
    xn: int,
    yn: int,
    dx_mm: float = 60,
    dy_mm: float = 60,
    freq_hz: float = 2.45e9,
    excitations: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculate the array factor for a given array configuration.
    This function uses JAX for efficient computation.
    """
    # Create an instance of the ArrayFactorCalculator
    af_calculator = ArrayFactorCalculator(theta, phi, xn, yn, dx_mm, dy_mm, freq_hz)

    # Calculate the array factor using the precomputed terms
    AF = af_calculator.calculate_af(excitations)

    return AF


def calculate_phase_shifts(
    xn: int,
    yn: int,
    dx_mm: float = 60,
    dy_mm: float = 60,
    freq: float = 2.45e9,
    theta_steering: float = 0.0,
    phi_steering: float = 0.0,
) -> np.ndarray:
    k = get_wavenumber(freq)
    x_pos, y_pos = get_element_positions(xn, yn, dx_mm, dy_mm)

    # Convert degrees to radians
    theta_steering, phi_steering = np.radians(theta_steering), np.radians(phi_steering)

    # Steering components of a unit vector pointing in the steered direction
    sin_theta_steering = np.sin(theta_steering)
    ux = sin_theta_steering * np.cos(phi_steering)
    uy = sin_theta_steering * np.sin(phi_steering)

    # Path difference = (r_element_vector) dot (unit_steering_vector)
    # Simplified as: x_element * ux + y_element * uy
    path_difference = (x_pos * ux)[:, None] + (y_pos * uy)

    phase_shifts = -k * path_difference

    phase_shifts = -phase_shifts  # FIXME: Should this be negated or not?
    phase_shifts = phase_shifts % (2 * np.pi)  # Normalize to [0, 2π)

    return phase_shifts


def plot_ff_polar(
    E_norm,
    Dmax,
    theta,
    *,
    normalize: bool = True,
    label: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    filename: str | None = None,
):
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    if normalize:
        E_norm = normalize_pattern(E_norm, Dmax)

    ax.plot(theta, E_norm, "r-", linewidth=1, label=label)
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


def plot_sim_and_af(
    sim_dir,
    freq,
    xns,
    yn,
    dxs,
    *,
    freq_idx=0,  # index of the frequency to plot
    steering_theta=0,
    steering_phi=0,
    figname: bool | str = True,
):
    """
    Plot comparison between OpenEMS simulation and array factor calculation,
    optionally with beam steering.

    Parameters:
    -----------
    sim_dir : Path
        Directory containing OpenEMS simulation results
    freq : float
        Operating frequency in Hz
    xns : list or numpy.ndarray
        Number of elements in the x-direction to plot
    yn : int
        Number of elements in the y-direction
    dxs : list or numpy.ndarray
        Element spacings to plot in millimeters
    figname : str, optional
        Filename to save the figure, if None the figure is not saved
    steering_theta : float, optional
        Steering elevation angle in degrees (default: 0)
    steering_phi : float, optional
        Steering azimuth angle in degrees (default: 0)
    """
    if not sim_dir:
        sim_dir = Path.cwd() / "src" / "sim" / "antenna_array"

    xns = np.array(xns)  # number of antennas in x direction
    dxs = np.array(dxs)  # distance between antennas in mm

    # Load the single antenna pattern (for array factor calculation)
    single_antenna_filename = f"farfield_1x1_60x60_{freq / 1e6:n}_steer_t0_p0.h5"
    nf2ff = read_nf2ff(sim_dir / single_antenna_filename)
    theta, phi = nf2ff["theta"], nf2ff["phi"]
    E_theta_single, E_phi_single = nf2ff["E_theta"][freq_idx], nf2ff["E_phi"][freq_idx]
    Dmax_single = nf2ff["Dmax"]

    phi_idx = np.argmin(np.abs(phi - steering_phi))  # Index of steering phi

    # Create a figure with polar subplots
    fig, axs = plt.subplots(
        nrows=len(xns),
        ncols=len(dxs),
        figsize=4 * np.array([len(dxs), len(xns)]),
        subplot_kw={"projection": "polar"},
    )
    axs = np.array([axs]).flatten() if min(len(xns), len(dxs)) == 1 else axs.flatten()

    # Loop through combinations and create combined plots
    for i, xn in enumerate(xns):
        for j, dx in enumerate(dxs):
            ax = axs[i * len(dxs) + j]
            dy = dx

            # Plot OpenEMS full simulation
            filename = f"farfield_{xn}x{yn}_{dx}x{dx}_{freq / 1e6:n}_steer_t{steering_theta}_p{steering_phi}.h5"
            try:
                nf2ff = read_nf2ff(sim_dir / filename)
            except (FileNotFoundError, KeyError):  # Could not load simulation data
                pass

            plot_ff_polar(
                E_norm=nf2ff["E_norm"][freq_idx, :, phi_idx],
                Dmax=nf2ff["Dmax"],
                theta=theta,
                label="OpenEMS Simulation",
                ax=ax,
            )

            phase_shifts = calculate_phase_shifts(
                xn, yn, dx, dy, freq, steering_theta, steering_phi
            )
            excitations = np.exp(1j * phase_shifts)
            AF = array_factor3(theta, phi, xn, yn, dx, dy, freq, excitations)

            # Calculate radiation pattern for the array factor
            E_theta_array = AF * E_theta_single
            E_phi_array = AF * E_phi_single
            E_norm_array = np.sqrt(
                np.abs(E_theta_array) ** 2 + np.abs(E_phi_array) ** 2
            )
            E_norm_array = E_norm_array[:, phi_idx]  # Select theta slice

            Dmax_array = Dmax_single * (xn * yn)
            E_norm_array_db = normalize_pattern(E_norm_array, Dmax_array)
            ax.plot(theta, E_norm_array_db, "g--", linewidth=1, label="Array Factor")

            c = 299_792_458
            lambda0 = c / freq
            lambda0_mm = lambda0 * 1e3
            freq_ghz = freq / 1e9

            title = f"{xn}x{yn} array, {dx}x{dy}mm, {freq_ghz:n}GHz, {dx / lambda0_mm:.2f}λ, steering: θ={steering_theta}°, φ={steering_phi}°"
            ax.set_title(title, fontsize=8)

    fig.legend(["OpenEMS Simulation", "Array Factor"], fontsize=8)
    fig.suptitle("OpenEMS Simulation vs Array Factor Comparison", y=0.99)

    fig.set_tight_layout(True)
    if figname:
        if figname is True:
            figname = f"ff_{freq_ghz:.0f}GHz_steer_t{steering_theta}_p{steering_phi}"
        fig.savefig(figname, dpi=600)


def normalize_pattern(pattern: np.ndarray, Dmax: float) -> np.ndarray:
    """
    Normalize the radiation pattern to dBi.
    """
    pattern = pattern / np.max(np.abs(pattern))
    pattern = 20 * np.log10(np.abs(pattern)) + 10.0 * np.log10(Dmax)
    return pattern


def plot_ff_3d(
    theta: np.ndarray,
    phi: np.ndarray,
    pattern: np.ndarray,
    *,
    title: str = "3D Radiation Pattern",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    pattern = pattern.clip(min=0)  # Clip negative values to 0
    theta, phi = np.meshgrid(theta, phi)

    # Calculate cartesian coordinates
    x = pattern * np.sin(theta) * np.cos(phi)
    y = pattern * np.sin(theta) * np.sin(phi)
    z = pattern * np.cos(theta)

    if ax is None:
        ax = plt.figure().add_subplot(projection="3d")

    ax.plot_surface(x, y, z, cmap="Spectral_r")
    ax.view_init(elev=20.0, azim=-100)
    ax.set_aspect("equalxy")
    ax.set_box_aspect(None, zoom=1.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_title(title)


def plot_ff_2d(
    theta: np.ndarray,
    phi: np.ndarray,
    pattern: np.ndarray,
    *,
    title: str = "2D Radiation Pattern",
    colorbar: bool = True,
    ax: plt.Axes | None = None,
):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))

    theta_deg, phi_deg = np.rad2deg(theta), np.rad2deg(phi)
    extent = [np.min(theta_deg), np.max(theta_deg), np.min(phi_deg), np.max(phi_deg)]
    im = ax.imshow(pattern, extent=extent, origin="lower")
    ax.set_xlabel("Theta (degrees)")
    ax.set_ylabel("Phi (degrees)")
    ax.set_title(title)

    if colorbar:
        ax.get_figure().colorbar(im, fraction=0.046, pad=0.04, label="Normalized Dbi")


def plot_phase_shifts(
    phase_shifts,
    title: str = "Phase Shifts",
    colorbar: bool = True,
    ax: plt.Axes | None = None,
):
    if not ax:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))

    phase_shifts_clipped = (phase_shifts + np.pi) % (2 * np.pi) - np.pi
    im = ax.imshow(
        np.rad2deg(phase_shifts_clipped),
        cmap="twilight_shifted",  # Cyclic colormap suitable for phase values
        origin="lower",
        vmin=-180,
        vmax=180,
    )
    ax.set_xlabel("Element X index")
    ax.set_ylabel("Element Y index")
    ax.set_title(title)

    if colorbar:
        ax.get_figure().colorbar(im, fraction=0.046, pad=0.04, label="Degrees")
