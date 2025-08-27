import numpy as np


def planar_hamming_weights(
    M=4,
    N=4,
    dx=0.5,
    dy=0.5,  # spacings in wavelengths (e.g., 0.5 = λ/2)
    theta0_deg=0.0,
    phi0_deg=0.0,
    normalize_peak=True,  # peak amplitude = 1
    return_phase_deg=True,
):
    """
    Returns:
    A : (M,N) amplitude matrix (non-negative reals)
    P : (M,N) phase matrix (radians or degrees)
    W : (M,N) complex weight matrix A * exp(1j*P)
    """
    # 1) 1-D Hamming tapers along x and y
    hx = np.hamming(M)  # 0.54 - 0.46*cos(2π n/(M-1))
    hy = np.hamming(N)

    # 2) Separable 2-D amplitude
    A = np.outer(hx, hy)
    if normalize_peak:
        A /= A.max()

    # 3) Phase terms for steering
    th0 = np.deg2rad(theta0_deg)
    ph0 = np.deg2rad(phi0_deg)

    # centered indices (m along x, n along y)
    m = np.arange(M) - (M - 1) / 2.0
    n = np.arange(N) - (N - 1) / 2.0
    mm, nn = np.meshgrid(m, n, indexing="ij")

    # If dx,dy are in wavelengths, k*dx = 2π*dx
    kd_x = 2.0 * np.pi * dx
    kd_y = 2.0 * np.pi * dy

    # Transmit steering phase (negative sign)
    P = -(kd_x * mm * np.sin(th0) * np.cos(ph0) + kd_y * nn * np.sin(th0) * np.sin(ph0))

    # Complex weights
    W = A * np.exp(1j * P)

    if return_phase_deg:
        return A, np.rad2deg(P), W
    else:
        return A, P, W


# --- Example: 4x4, λ/2 spacing, steer to θ0=30°, φ0=0° ---
if __name__ == "__main__":
    A, Pdeg, W = planar_hamming_weights(
        M=4, N=4, dx=0.5, dy=0.5, theta0_deg=30, phi0_deg=0, normalize_peak=True
    )
    np.set_printoptions(precision=3, suppress=True)
    print("Amplitude (A):\n", A)
    print("\nPhase (deg):\n", Pdeg)
    print("\nComplex weights (W):\n", W)
