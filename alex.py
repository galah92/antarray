import matplotlib.pyplot as plt
import numpy as np


def calc(xr1, xp, xa0, za=100.0, A=10.0, lambda1=0.1, u=340.0, N=10, size=2000):
    t = np.arange(size) * 0.01  # time
    xa = xa0 - u * t  # position of the source

    alpha = np.arctan2(za, xa - xr1)
    beta = np.arctan2(za, xp - xa)
    theta = alpha + beta

    k = 2 * np.pi / lambda1  # wave number
    mu = 2 * k * np.sin(theta / 2)
    r1 = np.sqrt(za**2 + (xp - xa) ** 2)  # distance from the source to the receiver

    r = (np.sin(mu * A) - (mu * A) * np.cos(mu * A)) / (mu * A) ** 3
    sp = 1 / r1**2 * (N + 3 * N * (N - 1) * r)  # scattering phase

    return sp, xa


if __name__ == "__main__":
    xr1 = -100.0
    xa0 = 3000.0

    sps1, xas = calc(xr1=xr1, xp=-1300.0, xa0=xa0)
    sps2, xas = calc(xr1=xr1, xp=-1000.0, xa0=xa0)

    plt.plot(xas, sps1 + sps2)
    plt.savefig("alex.png")
