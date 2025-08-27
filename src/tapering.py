import numpy as np


def planar_hamming_weights(
    M=4,  # Number of elements along the x-axis
    N=4,  # Number of elements along the y-axis
    dx=0.5,  # Spacing in wavelengths (e.g., 0.5 = λ/2)
    dy=0.5,  # Spacing in wavelengths (e.g., 0.5 = λ/2)
    elev_deg=0.0,  # Steering elevation angle in degrees
    azim_deg=0.0,  # Steering azimuthal angle in degrees
    normalize=True,  # Normalize amplitude sum to 1
):
    amplitude = np.outer(np.hamming(M), np.hamming(N))  # 2D Hamming taper
    if normalize:
        amplitude /= amplitude.max()

    # Phase terms for steering
    elev_rad, azim_rad = np.radians(elev_deg), np.radians(azim_deg)
    sin_elev = np.sin(elev_rad)
    sin_azim, cos_azim = np.sin(azim_rad), np.cos(azim_rad)

    # Element indices centered around zero
    i = np.arange(M) - (M - 1) / 2.0
    j = np.arange(N) - (N - 1) / 2.0
    ii, jj = np.meshgrid(i, j, indexing="ij")

    # Wavenumber components
    kd_x = 2.0 * np.pi * dx * ii
    kd_y = 2.0 * np.pi * dy * jj

    phase = kd_x * sin_elev * cos_azim + kd_y * sin_elev * sin_azim

    return amplitude * np.exp(1j * -phase)


if __name__ == "__main__":
    W = planar_hamming_weights(M=4, N=4, dx=0.5, dy=0.5, elev_deg=30, azim_deg=0)
    np.set_printoptions(precision=3)
    print("Amplitude (A):\n", np.abs(W))
    print("Phase (deg):\n", np.angle(W, deg=True))
    print("Complex weights (W):\n", W)
