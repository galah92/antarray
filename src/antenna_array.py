from pathlib import Path
import numpy as np

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import EPS0, C0

### General parameter setup
filename = Path(__file__).stem
sim_path = Path(__file__).parent / "sim" / filename
sim_path.mkdir(parents=True, exist_ok=True)

### Antenna array parameters
patch_width = 32.5  # patch width (resonant length) in x-direction
patch_length = 32.5  # patch length in y-direction

# setup FDTD parameter & excitation function
f0 = 2.45e9  # center frequency
fc = 0.5e9  # 20 dB corner frequency


def calculate_phase_shifts(n_x, n_y, d_x, d_y, freq, steering_theta, steering_phi):
    """
    Calculate phase shifts for antenna elements to steer the beam in a specific direction.

    Parameters:
    -----------
    n_x : int
        Number of elements in the x-direction
    n_y : int
        Number of elements in the y-direction
    d_x : float
        Element spacing in the x-direction in mm
    d_y : float
        Element spacing in the y-direction in mm
    freq : float
        Operating frequency in Hz
    steering_theta : float
        Steering elevation angle in degrees
    steering_phi : float
        Steering azimuth angle in degrees

    Returns:
    --------
    numpy.ndarray
        Phase shifts for each element in radians, with shape (n_x*n_y)
    """
    # Convert angles to radians
    steering_theta_rad = np.deg2rad(steering_theta)
    steering_phi_rad = np.deg2rad(steering_phi)

    # Calculate wavelength and convert spacing to meters
    c = 299792458  # Speed of light in m/s
    wavelength = c / freq  # Wavelength in meters
    dx_m = d_x / 1000  # Convert from mm to meters
    dy_m = d_y / 1000  # Convert from mm to meters

    # Wave number
    k = 2 * np.pi / wavelength

    # Initialize phase shifts array (in linear order, not matrix)
    phase_shifts = np.zeros(n_x * n_y)

    # Element positions
    x_positions = (np.arange(n_x) - (n_x - 1) / 2) * d_x / 1000
    y_positions = (np.arange(n_y) - (n_y - 1) / 2) * d_y / 1000

    # Calculate phase shifts
    sin_theta = np.sin(steering_theta_rad)

    # Populate the phase shift array
    idx = 0
    for x_pos in x_positions:
        for y_pos in y_positions:
            # Calculate phase shift for this element
            phase_x = k * x_pos * sin_theta * np.cos(steering_phi_rad)
            phase_y = k * y_pos * sin_theta * np.sin(steering_phi_rad)
            phase_shifts[idx] = phase_x + phase_y
            idx += 1

    return phase_shifts


def simulate(
    n_x: int,
    n_y: int,
    d_x: float | None = None,
    d_y: float | None = None,
    steering_theta: float = 0,
    steering_phi: float = 0,
):
    # define array size and dimensions
    if d_x is None:
        d_x = patch_width * 3
    if d_y is None:
        d_y = patch_length * 3

    print(f"Simulating {n_x}x{n_y} array with {d_x=:.2f} and {d_y=:.2f}")
    if steering_theta != 0 or steering_phi != 0:
        print(f"Beam steering: theta={steering_theta}°, phi={steering_phi}°")

    # substrate setup
    substrate_epsR = 3.38
    substrate_kappa = 0  # 1e-3 * 2 * np.pi * 2.45e9 * EPS0 * substrate_epsR
    substrate_width = 60 + (n_x - 1) * d_x
    substrate_length = 60 + (n_y - 1) * d_y
    substrate_thickness = 1.524
    substrate_cells = 4

    # setup feeding
    feed_pos = -5.5  # feeding position in x-direction
    feed_width = 2  # width of the feeding line
    feed_R = 50  # feed resistance

    # size of the simulation box around the array
    SimBox = np.array([50 + substrate_width, 50 + substrate_length, 25])

    ### FDTD setup
    ## * Limit the simulation to 30k timesteps
    ## * Define a reduced end criteria of -50dB
    FDTD = openEMS(NrTS=30000, EndCriteria=1e-5)
    FDTD.SetGaussExcite(f0, fc)

    # Set boundary conditions
    FDTD.SetBoundaryCond(["MUR", "MUR", "MUR", "MUR", "MUR", "MUR"])

    ### Setup CSXCAD geometry & mesh
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)

    # setup the mesh
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3)  # all lengths in mm
    mesh_res = C0 / (f0 + fc) / 1e-3 / 20  # cell size: lambda/20

    ### Generate mesh
    # initialize the base mesh with the "air-box" dimensions
    mesh.AddLine(
        "x", [-SimBox[0] / 2, SimBox[0] / 2, -substrate_width / 2, substrate_width / 2]
    )
    mesh.AddLine(
        "y",
        [-SimBox[1] / 2, SimBox[1] / 2, -substrate_length / 2, substrate_length / 2],
    )
    mesh.AddLine(
        "z",
        [-SimBox[2] / 2]
        + list(np.linspace(0, substrate_thickness, substrate_cells + 1))
        + [SimBox[2]],
    )

    ant_midX = (np.arange(n_x) - (n_x - 1) / 2) * d_x
    ant_midY = (np.arange(n_y) - (n_y - 1) / 2) * d_y

    # Add mesh lines for patches and feeds
    for midX in ant_midX:
        for midY in ant_midY:
            # feeding mesh
            mesh.AddLine("x", [midX + feed_pos])
            mesh.AddLine("y", [midY - feed_width / 2, midY + feed_width / 2])

            # patch mesh with 2/3 - 1/3 rule
            mesh.AddLine(
                "x",
                [
                    midX - patch_width / 2 - mesh_res / 2 * 0.66,
                    midX - patch_width / 2 + mesh_res / 2 * 0.33,
                    midX + patch_width / 2 + mesh_res / 2 * 0.66,
                    midX + patch_width / 2 - mesh_res / 2 * 0.33,
                ],
            )
            mesh.AddLine(
                "y",
                [
                    midY - patch_length / 2 - mesh_res / 2 * 0.66,
                    midY - patch_length / 2 + mesh_res / 2 * 0.33,
                    midY + patch_length / 2 + mesh_res / 2 * 0.66,
                    midY + patch_length / 2 - mesh_res / 2 * 0.33,
                ],
            )

    # Create smooth mesh
    mesh.SmoothMeshLines("all", mesh_res, 1.4)

    ### Create substrate
    substrate = CSX.AddMaterial(
        "substrate", epsilon=substrate_epsR, kappa=substrate_kappa
    )
    start = [-substrate_width / 2, -substrate_length / 2, 0]
    stop = [substrate_width / 2, substrate_length / 2, substrate_thickness]
    substrate.AddBox(priority=0, start=start, stop=stop)

    ### Create ground (same size as substrate)
    gnd = CSX.AddMetal("gnd")  # create a perfect electric conductor (PEC)
    start[2] = 0
    stop[2] = 0
    gnd.AddBox(start=start, stop=stop, priority=10)

    ### Create patches and feeds
    patch = CSX.AddMetal("patch")

    # Calculate phase shifts for beam steering
    phase_shifts = np.zeros(n_x * n_y)
    if steering_theta != 0 or steering_phi != 0:
        phase_shifts = calculate_phase_shifts(
            n_x, n_y, d_x, d_y, f0, steering_theta, steering_phi
        )

    # Convert phase shifts to time delays
    # For a phase shift φ, the time delay t = φ/(2πf)
    time_delays = phase_shifts / (2 * np.pi * f0)

    ports = []
    port_idx = 0
    for midX in ant_midX:
        for midY in ant_midY:
            # Create patch
            start = [
                midX - patch_width / 2,
                midY - patch_length / 2,
                substrate_thickness,
            ]
            stop = [
                midX + patch_width / 2,
                midY + patch_length / 2,
                substrate_thickness,
            ]
            patch.AddBox(priority=10, start=start, stop=stop)

            # Add feed port with phase shift
            port_start = [midX + feed_pos - feed_width / 2, midY - feed_width / 2, 0]
            port_stop = [
                midX + feed_pos + feed_width / 2,
                midY + feed_width / 2,
                substrate_thickness,
            ]

            port = FDTD.AddLumpedPort(
                port_idx + 1,
                feed_R,
                port_start,
                port_stop,
                "z",
                excite=True,
                priority=5,
                delay=time_delays[port_idx],
            )
            ports.append(port)
            port_idx += 1

    ### Add the nf2ff recording box
    nf2ff = FDTD.CreateNF2FFBox()

    ### Run the simulation
    post_proc_only = False
    if not post_proc_only:
        FDTD.Run(sim_path)

    return sim_path, nf2ff, ports


def postprocess(sim_path, nf2ff, f0, ports, outfile=None):
    """
    Process OpenEMS simulation results

    Parameters:
    -----------
    sim_path : Path
        Path to simulation results
    nf2ff : object
        Far field calculation object
    f0 : float
        Center frequency
    ports : list
        List of port objects
    outfile : str
        Output filename
    """
    # Calculate total input power
    P_in = 0
    for port in ports:
        port.CalcPort(sim_path, f0)
        P_in += 0.5 * port.uf_tot * np.conj(port.if_tot)

    # Plot S11 for first port
    port1 = ports[0]
    s11 = port1.uf_ref / port1.uf_inc
    _s11_dB = 20.0 * np.log10(np.abs(s11))

    # Calculate far field
    theta = np.arange(-90.0, 90.0, 1.0)
    phi = np.arange(-90.0, 90.0, 1.0)

    print("Calculating 3D far field...")
    _nf2ff_3d = nf2ff.CalcNF2FF(
        sim_path,
        f0,
        theta,
        phi,
        center=[0, 0, 1e-3],
        outfile=outfile,
    )


if __name__ == "__main__":
    ants = [(16, 16)]
    d_ant = [60]
    steering_thetas = [0, 15, 30, 45]
    steering_phis = [0]

    # Run standard simulations without beam steering
    for n_x, n_y in ants:
        for d in d_ant:
            for steering_theta in steering_thetas:
                for steering_phi in steering_phis:
                    outfile = f"farfield_{n_x}x{n_y}_{d}x{d}_{f0 / 1e6:n}_steer_t{steering_theta}_p{steering_phi}.h5"
                    if (sim_path / outfile).exists():
                        print(f"Skipping {outfile}")
                        continue
                    sim_path, nf2ff, ports = simulate(
                        n_x=n_x,
                        n_y=n_y,
                        d_x=d,
                        d_y=d,
                        steering_theta=steering_theta,
                        steering_phi=steering_phi,
                    )
                    postprocess(sim_path, nf2ff, f0, ports, outfile)
