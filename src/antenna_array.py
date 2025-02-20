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
patch_width = 32  # patch width (resonant length) in x-direction
patch_length = 40  # patch length in y-direction

# setup FDTD parameter & excitation function
f0 = 0  # center frequency
fc = 3e9  # 20 dB corner frequency


def simulate(xn: int, yn: int):
    # define array size and dimensions
    x_spacing = patch_width * 3
    y_spacing = patch_length * 3

    # substrate setup
    substrate_epsR = 3.38
    substrate_kappa = 1e-3 * 2 * np.pi * 2.45e9 * EPS0 * substrate_epsR
    substrate_width = 60 + (xn - 1) * x_spacing
    substrate_length = 60 + (yn - 1) * y_spacing
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

    ant_midX = (xn / 2 - np.arange(xn) - 0.5) * x_spacing
    ant_midY = (yn / 2 - np.arange(yn) - 0.5) * y_spacing

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
    port_number = 1

    ports = []
    i = 0
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

            # Add feed port
            port_start = [midX + feed_pos - feed_width / 2, midY - feed_width / 2, 0]
            port_stop = [
                midX + feed_pos + feed_width / 2,
                midY + feed_width / 2,
                substrate_thickness,
            ]
            port = FDTD.AddLumpedPort(
                port_number,
                feed_R,
                port_start,
                port_stop,
                "z",
                excite=True,
                priority=5,
                # delay=np.exp(-1j * 2 * np.pi * f0 * 0.5),
            )
            ports.append(port)
            port_number += 1
            i += 1

    ### Add the nf2ff recording box
    nf2ff = FDTD.CreateNF2FFBox()

    ### Run the simulation
    post_proc_only = False
    if not post_proc_only:
        FDTD.Run(sim_path)

    return sim_path, nf2ff, ports


def postprcess(sim_path, nf2ff, xn, yn, f0, fc, ports, outfile=str | None):
    ### Post-processing and plotting
    f = np.linspace(max(1e9, f0 - fc), f0 + fc, 501)

    # Calculate total input power
    P_in = 0
    for port_nr in range(1, xn * yn + 1):
        port = ports[port_nr - 1]
        port.CalcPort(sim_path, f)
        P_in += 0.5 * port.uf_tot * np.conj(port.if_tot)

    # Plot S11 for first port
    port1 = ports[0]
    s11 = port1.uf_ref / port1.uf_inc
    s11_dB = 20.0 * np.log10(np.abs(s11))

    # Find resonance frequency
    # idx = np.where((s11_dB < -10) & (s11_dB == np.min(s11_dB)))[0]
    # if not len(idx) == 1:
    #     raise Exception("No resonance frequency found for far-field calulation")
    # f_res = f[idx[0]]
    # Claude's help
    idx = np.argmin(s11_dB)  # Find minimum S11 location
    f_res = f[idx]
    print(f"Best match frequency: {f_res / 1e9:.2f} GHz with S11: {s11_dB[idx]:.2f} dB")

    # Proceed even if match isn't perfect
    if s11_dB[idx] > -5:
        print(
            "Warning: Poor impedance match, S11 minimum is only {:.2f} dB".format(
                s11_dB[idx]
            )
        )

    phi = np.arange(-180.0, 180.0, 2.0)
    theta = np.arange(-180.0, 180.0, 2.0)
    print("Calculating 3D far field...")
    _nf2ff_3d = nf2ff.CalcNF2FF(
        sim_path,
        f_res,
        theta,
        phi,
        center=[0, 0, 1e-3],
        outfile=outfile,
    )


xn, yn = 4, 4
sim_path, nf2ff, ports = simulate(xn=xn, yn=yn)
postprcess(sim_path, nf2ff, xn, yn, f0, fc, ports, outfile=f"farfield_{xn}_{yn}.h5")

xn, yn = 1, 1
sim_path, nf2ff, ports = simulate(xn=xn, yn=yn)
postprcess(sim_path, nf2ff, xn, yn, f0, fc, ports, outfile=f"farfield_{xn}_{yn}.h5")
