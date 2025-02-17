from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import h5py


def read_nf2ff(nf2ff_path: Path):
    nf2ff = {}
    with h5py.File(nf2ff_path, "r") as h5:
        nf2ff["theta"] = h5["Mesh"]["theta"][:]
        nf2ff["phi"] = h5["Mesh"]["phi"][:]
        nf2ff["r"] = h5["Mesh"]["r"][:]

        E_phi = h5["nf2ff"]["E_phi"]["FD"]
        E_phi = E_phi["f0_real"][:] + 1j * E_phi["f0_imag"][:]
        nf2ff["E_phi"] = E_phi

        E_theta = h5["nf2ff"]["E_theta"]["FD"]
        E_theta = E_theta["f0_real"][:] + 1j * E_theta["f0_imag"][:]
        nf2ff["E_theta"] = E_theta

        E_norm = np.sqrt(np.abs(E_phi) ** 2 + np.abs(E_theta) ** 2)
        nf2ff["E_norm"] = E_norm

    return nf2ff


def plot_directivity(nf2ff):
    theta = np.rad2deg(nf2ff["theta"])
    E_norm = nf2ff["E_norm"]
    print(np.max(E_norm, axis=1))
    E_norm_scaled = 20.0 * np.log10(E_norm / np.max(E_norm))

    plt.plot(theta, np.squeeze(E_norm_scaled[0]), "k-", linewidth=2, label="xz-plane")
    plt.plot(theta, np.squeeze(E_norm_scaled[1]), "r--", linewidth=2, label="yz-plane")
    plt.xlabel("Theta (deg)")
    plt.ylabel("Directivity (dBi)")
    plt.title("Directivity Plot")
    plt.legend()
    plt.grid()
