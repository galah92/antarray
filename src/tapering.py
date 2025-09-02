import numpy as np


def uniform_taper(nx=4, ny=4):
    amplitude = np.ones((nx, ny))
    return amplitude


def hamming_taper(nx=4, ny=4):
    amplitude = np.outer(np.hamming(nx), np.hamming(ny))
    return amplitude


def ideal_steering(
    nx=4,  # nyumber of elements along the x-axis
    ny=4,  # nyumber of elements along the y-axis
    dx=0.5,  # Spacing in wavelengths (e.g., 0.5 = λ/2)
    dy=0.5,  # Spacing in wavelengths (e.g., 0.5 = λ/2)
    elev_deg=0.0,  # Steering elevation angle in degrees
    azim_deg=0.0,  # Steering azimuthal angle in degrees
):
    # Phase terms for steering
    elev_rad, azim_rad = np.radians(elev_deg), np.radians(azim_deg)
    sin_elev = np.sin(elev_rad)
    sin_azim, cos_azim = np.sin(azim_rad), np.cos(azim_rad)

    # Element indices centered around zero
    i = np.arange(nx) - (nx - 1) / 2.0
    j = np.arange(ny) - (ny - 1) / 2.0
    ii, jj = np.meshgrid(i, j, indexing="ij")

    # Wavenumber components
    kd_x = 2.0 * np.pi * dx * ii
    kd_y = 2.0 * np.pi * dy * jj

    phase = kd_x * sin_elev * cos_azim + kd_y * sin_elev * sin_azim
    return phase


if __name__ == "__main__":
    amplitude = hamming_taper(nx=4, ny=4)
    phase = ideal_steering(nx=4, ny=4, dx=0.5, dy=0.5, elev_deg=0, azim_deg=0)
    w = amplitude * np.exp(1j * phase)

    np.set_printoptions(precision=3)
    print("Amplitude:\n", np.abs(w))
    print("Phase (deg):\n", np.angle(w, deg=True))
    print("Complex weights:\n", w)
