# -*- coding: utf-8 -*-
"""
 Patch Antenna Array Tutorial

 This example demonstrates how to:
  - calculate the reflection coefficient of a patch antenna array
 
 Tested with
  - python 3.10
  - openEMS v0.0.34+

 Translated from MATLAB version by Thorsten Liebig
"""

import os
import tempfile
from pylab import *

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import *

### General parameter setup
Sim_Path = os.path.join(tempfile.gettempdir(), 'patch_array')

post_proc_only = False
draw_3d_pattern = False  # this may take a while
use_pml = False         # use pml boundaries instead of mur

### Antenna array parameters
# patch width (resonant length) in x-direction
patch_width = 32.86
# patch length in y-direction
patch_length = 41.37

# define array size and dimensions
array = {
    'xn': 4,
    'yn': 4,
    'x_spacing': patch_width * 3,
    'y_spacing': patch_length * 3
}

# substrate setup
substrate_epsR = 3.38
substrate_kappa = 1e-3 * 2*pi*2.45e9 * EPS0*substrate_epsR
substrate_width = 60 + (array['xn']-1) * array['x_spacing']
substrate_length = 60 + (array['yn']-1) * array['y_spacing']
substrate_thickness = 1.524
substrate_cells = 4

# setup feeding
feed_pos = -5.5
feed_width = 2
feed_R = 50  # feed resistance

# size of the simulation box around the array
SimBox = np.array([50+substrate_width, 50+substrate_length, 25])

### FDTD setup
## * Limit the simulation to 30k timesteps
## * Define a reduced end criteria of -50dB
FDTD = openEMS(NrTS=30000, EndCriteria=1e-5)
f0 = 0  # center frequency
fc = 3e9  # 20 dB corner frequency
FDTD.SetGaussExcite(f0, fc)

# Set boundary conditions
if use_pml:
    bc = ['PML_8'] * 6
else:
    bc = ['MUR'] * 6
FDTD.SetBoundaryCond(bc)

### Setup CSXCAD geometry & mesh
CSX = ContinuousStructure()
FDTD.SetCSX(CSX)

# setup the mesh
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1e-3)  # all lengths in mm
mesh_res = C0/(f0+fc)/1e-3/20  # cell size: lambda/20

### Generate mesh
# initialize the base mesh with the "air-box" dimensions
mesh.AddLine('x', [-SimBox[0]/2, SimBox[0]/2, -substrate_width/2, substrate_width/2])
mesh.AddLine('y', [-SimBox[1]/2, SimBox[1]/2, -substrate_length/2, substrate_length/2])
mesh.AddLine('z', [-SimBox[2]/2] + list(linspace(0, substrate_thickness, substrate_cells+1)) + [SimBox[2]])

# Add mesh lines for patches and feeds
for xn in range(array['xn']):
    for yn in range(array['yn']):
        midX = (array['xn']/2 - xn - 0.5) * array['x_spacing']
        midY = (array['yn']/2 - yn - 0.5) * array['y_spacing']
        
        # feeding mesh
        mesh.AddLine('x', [midX + feed_pos])
        mesh.AddLine('y', [midY - feed_width/2, midY + feed_width/2])
        
        # patch mesh with 2/3 - 1/3 rule
        mesh.AddLine('x', [
            midX - patch_width/2 - mesh_res/2*0.66,
            midX - patch_width/2 + mesh_res/2*0.33,
            midX + patch_width/2 + mesh_res/2*0.66,
            midX + patch_width/2 - mesh_res/2*0.33
        ])
        mesh.AddLine('y', [
            midY - patch_length/2 - mesh_res/2*0.66,
            midY - patch_length/2 + mesh_res/2*0.33,
            midY + patch_length/2 + mesh_res/2*0.66,
            midY + patch_length/2 - mesh_res/2*0.33
        ])

# Create smooth mesh
mesh.SmoothMeshLines('all', mesh_res, 1.4)

### Create substrate
substrate = CSX.AddMaterial('substrate', epsilon=substrate_epsR, kappa=substrate_kappa)
start = [-substrate_width/2, -substrate_length/2, 0]
stop = [substrate_width/2, substrate_length/2, substrate_thickness]
substrate.AddBox(priority=0, start=start, stop=stop)

### Create ground (same size as substrate)
gnd = CSX.AddMetal('gnd')
start[2] = 0
stop[2] = 0
gnd.AddBox(start=start, stop=stop, priority=10)

### Create patches and feeds
patch = CSX.AddMetal('patch')
port_number = 1

ports = []
for xn in range(array['xn']):
    for yn in range(array['yn']):
        midX = (array['xn']/2 - xn - 0.5) * array['x_spacing']
        midY = (array['yn']/2 - yn - 0.5) * array['y_spacing']
        
        # Create patch
        start = [midX - patch_width/2, midY - patch_length/2, substrate_thickness]
        stop = [midX + patch_width/2, midY + patch_length/2, substrate_thickness]
        patch.AddBox(priority=10, start=start, stop=stop)
        
        # Add feed port
        port_start = [midX + feed_pos - feed_width/2, midY - feed_width/2, 0]
        port_stop = [midX + feed_pos + feed_width/2, midY + feed_width/2, substrate_thickness]
        port = FDTD.AddLumpedPort(port_number, feed_R, port_start, port_stop, 'z', 1.0, priority=5)
        ports.append(port)
        port_number += 1

### Add the nf2ff recording box
nf2ff = FDTD.CreateNF2FFBox()

### Run the simulation
if not post_proc_only:
    FDTD.Run(Sim_Path, cleanup=True)

### Post-processing and plotting
f = np.linspace(max(1e9, f0-fc), f0+fc, 501)

# Calculate total input power
P_in = 0
for port_nr in range(1, array['xn']*array['yn'] + 1):
    port = ports[port_nr-1]
    port.CalcPort(Sim_Path, f)
    P_in += 0.5 * port.uf_tot * np.conj(port.if_tot)

# Plot S11 for first port
port1 = ports[0]
s11 = port1.uf_ref/port1.uf_inc
s11_dB = 20.0*np.log10(np.abs(s11))

figure()
plot(f/1e9, s11_dB, 'k-', linewidth=2, label='$S_{11}$')
grid()
legend()
ylabel('S-Parameter (dB)')
xlabel('Frequency (GHz)')

# Find resonance frequency
idx = np.where((s11_dB < -10) & (s11_dB == np.min(s11_dB)))[0]
if len(idx) == 1:
    f_res = f[idx[0]]
    
    # Calculate far field at phi=0 and phi=90 degrees
    theta = np.arange(-180, 180, 2)
    phi = [0, 90]
    print('Calculating far field at phi=[0 90] deg...')
    nf2ff_res = nf2ff.CalcNF2FF(Sim_Path, f_res, theta, phi, center=[0, 0, 1e-3])
    
    Dlog = 10*np.log10(nf2ff_res.Dmax)
    
    # Display power and directivity
    print(f'Radiated power: Prad = {nf2ff_res.Prad:.2f} Watt')
    print(f'Directivity: Dmax = {Dlog:.2f} dBi')
    print(f'Efficiency: nu_rad = {100*nf2ff_res.Prad/np.real(P_in[idx[0]]):.2f} %')
    
    # Plot far field pattern
    figure()
    E_norm = 20.0*np.log10(nf2ff_res.E_norm[0]/np.max(nf2ff_res.E_norm[0])) + 10.0*np.log10(nf2ff_res.Dmax[0])
    plot(theta, np.squeeze(E_norm[:,0]), 'k-', linewidth=2, label='xz-plane')
    plot(theta, np.squeeze(E_norm[:,1]), 'r--', linewidth=2, label='yz-plane')
    grid()
    ylabel('Directivity (dBi)')
    xlabel('Theta (deg)')
    title(f'Frequency: {f_res/1e9:.2f} GHz')
    legend()
    
    if draw_3d_pattern:
        # Calculate 3D far field pattern
        phi = np.arange(0, 360, 3)
        theta = np.unique(np.concatenate([np.arange(0, 15.5, 0.5), np.arange(10, 181, 3)]))
        print('Calculating 3D far field...')
        nf2ff_3d = nf2ff.CalcNF2FF(Sim_Path, f_res, theta, phi, center=[0, 0, 1e-3])
        
        figure()
        plotFF3D(nf2ff_3d)

show()