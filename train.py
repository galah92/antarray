from pathlib import Path
import numpy as np
import analyze


def get_elem_data(sim_dir: Path, freq: float, normalize=True) -> tuple:
    elem_antenna_filename = f"farfield_1x1_60x60_{freq / 1e6:n}.h5"
    elem_antenna_nf2ff = analyze.read_nf2ff(sim_dir / elem_antenna_filename)
    elem_E_norm = elem_antenna_nf2ff["E_norm"][0][0]
    elem_Dmax = elem_antenna_nf2ff["Dmax"]
    theta, phi = elem_antenna_nf2ff["theta"], elem_antenna_nf2ff["phi"]

    if not normalize:
        return elem_E_norm, elem_Dmax, theta, phi

    elem_E_norm = elem_E_norm / np.max(np.abs(elem_E_norm))
    elem_E_norm_dbi = 20 * np.log10(np.abs(elem_E_norm)) + 10.0 * np.log10(elem_Dmax)
    return elem_E_norm_dbi, elem_Dmax, theta, phi


def calc_array_E_norm(E_norm, Dmax, theta, phi, freq, xn, yn=1, dx=60, dy=60):
    # Array Factor calculation
    af = analyze.array_factor(theta, phi[0], freq, xn, yn, dx, dy)
    array_E_norm = E_norm * af.T

    # Normalize and calculate dB for Array Factor
    array_E_norm = array_E_norm / np.max(np.abs(array_E_norm))
    array_Dmax = Dmax * (xn * yn)
    array_E_norm_dbi = 20 * np.log10(np.abs(array_E_norm)) + 10.0 * np.log10(array_Dmax)

    return array_E_norm_dbi
