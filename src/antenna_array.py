from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import EPS0, C0

### General parameter setup
filename = Path(__file__).stem
sim_path = Path(Path.cwd(), filename)

### Antenna array parameters
# patch width (resonant length) in x-direction
patch_width = 32.86
# patch length in y-direction
patch_length = 41.37

# define array size and dimensions
array = {
    "xn": 4,
    "yn": 4,
    "x_spacing": patch_width * 3,
    "y_spacing": patch_length * 3,
}

# substrate setup
substrate_epsR = 3.38
substrate_kappa = 1e-3 * 2 * np.pi * 2.45e9 * EPS0 * substrate_epsR
substrate_width = 60 + (array["xn"] - 1) * array["x_spacing"]
substrate_length = 60 + (array["yn"] - 1) * array["y_spacing"]
substrate_thickness = 1.524
substrate_cells = 4

# setup feeding
feed_pos = -5.5
feed_width = 2
feed_R = 50  # feed resistance

# size of the simulation box around the array
SimBox = np.array([50 + substrate_width, 50 + substrate_length, 25])

# setup FDTD parameter & excitation function
f0 = 0  # center frequency
fc = 3e9  # 20 dB corner frequency

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
    "y", [-SimBox[1] / 2, SimBox[1] / 2, -substrate_length / 2, substrate_length / 2]
)
mesh.AddLine(
    "z",
    [-SimBox[2] / 2]
    + list(np.linspace(0, substrate_thickness, substrate_cells + 1))
    + [SimBox[2]],
)

# Add mesh lines for patches and feeds
for xn in range(array["xn"]):
    for yn in range(array["yn"]):
        midX = (array["xn"] / 2 - xn - 0.5) * array["x_spacing"]
        midY = (array["yn"] / 2 - yn - 0.5) * array["y_spacing"]

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
substrate = CSX.AddMaterial("substrate", epsilon=substrate_epsR, kappa=substrate_kappa)
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
for xn in range(array["xn"]):
    for yn in range(array["yn"]):
        midX = (array["xn"] / 2 - xn - 0.5) * array["x_spacing"]
        midY = (array["yn"] / 2 - yn - 0.5) * array["y_spacing"]

        # Create patch
        start = [midX - patch_width / 2, midY - patch_length / 2, substrate_thickness]
        stop = [midX + patch_width / 2, midY + patch_length / 2, substrate_thickness]
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
            excite=1.0,
            priority=5,
            delay=np.exp(-1j * 2 * np.pi * f0 * 0.5),
        )
        ports.append(port)
        port_number += 1

### Add the nf2ff recording box
nf2ff = FDTD.CreateNF2FFBox()

### Run the simulation
post_proc_only = False
if not post_proc_only:
    FDTD.Run(sim_path, cleanup=True)

### Post-processing and plotting
f = np.linspace(max(1e9, f0 - fc), f0 + fc, 501)

# Calculate total input power
P_in = 0
for port_nr in range(1, array["xn"] * array["yn"] + 1):
    port = ports[port_nr - 1]
    port.CalcPort(sim_path, f)
    P_in += 0.5 * port.uf_tot * np.conj(port.if_tot)

# Plot S11 for first port
port1 = ports[0]
s11 = port1.uf_ref / port1.uf_inc
s11_dB = 20.0 * np.log10(np.abs(s11))

plt.figure()
plt.plot(f / 1e9, s11_dB, "k-", linewidth=2, label="$S_{11}$")
plt.grid()
plt.legend()
plt.ylabel("S-Parameter (dB)")
plt.xlabel("Frequency (GHz)")
plt.savefig("S11.png")

# Find resonance frequency
idx = np.where((s11_dB < -10) & (s11_dB == np.min(s11_dB)))[0]
if not len(idx) == 1:
    raise Exception("No resonance frequency found for far-field calulation")

f_res = f[idx[0]]

# Calculate far field at phi=0 and phi=90 degrees
theta = np.arange(-180, 180, 2)
phi = [0, 90]
print("Calculating far field at phi=[0 90] deg...")
nf2ff_res = nf2ff.CalcNF2FF(sim_path, f_res, theta, phi, center=[0, 0, 1e-3])

Dlog = 10 * np.log10(nf2ff_res.Dmax)

plt.figure()
E_norm, Dmax = nf2ff_res.E_norm, nf2ff_res.Dmax
E_norm = 20.0 * np.log10(E_norm[0] / np.max(E_norm[0])) + 10.0 * np.log10(Dmax[0])
plt.plot(theta, np.squeeze(E_norm[:, 0]), "k-", linewidth=2, label="xz-plane")
plt.plot(theta, np.squeeze(E_norm[:, 1]), "r--", linewidth=2, label="yz-plane")
plt.grid()
plt.ylabel("Directivity (dBi)")
plt.xlabel("Theta (deg)")
plt.title(f"Frequency: {f_res / 1e9:.2f} GHz")
plt.legend()
plt.savefig("Directivity.png")

draw_3d_pattern = True
if draw_3d_pattern:
    phi = np.arange(-180.0, 180.0, 2.0)
    theta = np.arange(-180.0, 180.0, 2.0)
    print("Calculating 3D far field...")
    nf2ff_3d = nf2ff.CalcNF2FF(sim_path, f_res, theta, phi, center=[0, 0, 1e-3])
    # plotFF3D(nf2ff_3d)

print("Done.")


def array_factor_sum(xn, yn, theta, phi, f_res, nf2ff, center):
    AF = np.zeros((len(theta), len(phi)), dtype=complex)
    for xn in range(array["xn"]):
        for yn in range(array["yn"]):
            midX = (array["xn"] / 2 - xn - 0.5) * array["x_spacing"]
            midY = (array["yn"] / 2 - yn - 0.5) * array["y_spacing"]

            # Calculate the far field of the single patch
            nf2ff_res = nf2ff.CalcNF2FF(
                sim_path, f_res, theta, phi, center=[midX, midY, 1e-3]
            )

            # Calculate the array factor
            AF += nf2ff_res.E_norm

    return AF


AF = array_factor_sum(
    array["xn"], array["yn"], theta, phi, f_res, nf2ff, center=[0, 0, 1e-3]
)
AF_dB = 20 * np.log10(np.abs(AF) / np.max(np.abs(AF)))


def plot_array_factor(AF_dB, theta, phi):
    plt.figure()
    plt.pcolormesh(theta, phi, AF_dB.T, shading="gouraud")
    plt.colorbar()
    plt.xlabel("Theta (deg)")
    plt.ylabel("Phi (deg)")
    plt.title("Array Factor (dB)")
    plt.savefig("ArrayFactor.png")
