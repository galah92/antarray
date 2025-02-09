import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import h5py
from pathlib import Path

def plotFF3D_from_h5(h5_filename, *args):
    """
    Plots normalized 3D far field pattern from an HDF5 file.

    Args:
        h5_filename: Path to the HDF5 file.
        *args: Variable input arguments as key-value pairs.

    Returns:
        h: The surface plot object.
    """

    with h5py.File(h5_filename, 'r') as f:  # Open the HDF5 file in read mode
        # Read data from the HDF5 file.  Adapt these to your actual HDF5 structure.
        theta = f['Mesh']['theta'][:]  # Assuming 'theta' is a dataset in the HDF5 file
        phi = f['Mesh']['phi'][:]      # Assuming 'phi' is a dataset
        E_norm = []
        for i in range(len(f['E_norm'])): # Assuming E_norm is a group of datasets, one for each frequency
            E_norm.append(f['E_norm'][f'E{i+1}'][:]) #E1, E2, E3...
        freq = f['freq'][:] # Assuming 'freq' is a dataset
        Dmax = f['Dmax'][:] # Assuming 'Dmax' is a dataset

        nf2ff = {
            'theta': theta,
            'phi': phi,
            'E_norm': E_norm,
            'freq': freq,
            'Dmax': Dmax
        }


    # defaults
    logscale = None
    freq_index = 0  # Python indexing starts at 0
    normalize = False

    # Process variable arguments
    n = 0
    while n < len(args):
        if args[n] == 'logscale':
            logscale = args[n+1]
        elif args[n] == 'freq_index':
            freq_index = args[n+1] -1 # Adjust for python 0-based indexing
        elif args[n] == 'normalize':
            normalize = args[n+1]
        else:
            print(f"Warning: Unknown argument key: '{args[n]}'")
        n += 2

    if (normalize or logscale is not None):
        E_far = nf2ff['E_norm'][freq_index] / np.max(nf2ff['E_norm'][freq_index])
    else:
        E_far = nf2ff['E_norm'][freq_index]

    if logscale is not None:
        E_far = 20*np.log10(E_far)/-logscale + 1
        E_far = E_far * (E_far > 0)
        titletext = f'electrical far field [dB] @ f = {nf2ff["freq"][freq_index]} Hz'
    elif normalize:
        titletext = f'normalized electrical far field @ f = {nf2ff["freq"][freq_index]} Hz'
    else:
        titletext = f'electrical far field [V/m] @ f = {nf2ff["freq"][freq_index]} Hz'

    theta, phi = np.meshgrid(nf2ff['theta'], nf2ff['phi'])
    x = E_far * np.sin(theta) * np.cos(phi)
    y = E_far * np.sin(theta) * np.sin(phi)
    z = E_far * np.cos(theta)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    h = ax.plot_surface(x, y, z, facecolors=cm.jet(E_far), linewidth=0, antialiased=False)
    ax.set_axis_off()

    if logscale is not None:
        cbar = fig.colorbar(h, shrink=0.75, pad=0.05, ticks=np.linspace(0, np.max(E_far), 9))
        cbar.ax.set_yticklabels([f'{val:.1f}' for val in np.linspace(logscale, 10*np.log10(nf2ff['Dmax'][freq_index]), 9)])
    else:
        cbar = fig.colorbar(h, shrink=0.75, pad=0.05)

    ax.set_title(titletext)

    plt.show()
    return h


# Example usage:
h5_file = "src/Simp_Patch/nf2ff.h5"  # Replace with your HDF5 file name
h = plotFF3D_from_h5(h5_file, 'freq_index', 2, 'logscale', -20)
#plotFF3D_from_h5(h5_file, 'freq_index', 1, 'normalize', True)