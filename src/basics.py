import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.projections import PolarAxes

# Array configuration
n_elements = 8
freq_hz = 2.4e9
c = 3e8  # Speed of light
wavelength = c / freq_hz
spacing_wavelengths = 0.5  # Half-wavelength spacing
spacing_m = spacing_wavelengths * wavelength
k = 2 * np.pi / wavelength  # Wave number
print(f"{freq_hz / 1e9:.1f}GHz, {spacing_m * 1000:.1f}mm ({spacing_wavelengths:.1f}λ)")

theta = np.linspace(0, 2 * np.pi, 360)  # Azimuth angles (0 to 2π)

# Element positions along x-axis (linear array centered at origin)
element_positions = (np.arange(n_elements) - (n_elements - 1) / 2) * spacing_m

# Create synthetic Antenna Element Patterns (AEPs)
amplitude_base = 0.7 + 0.5 * np.cos(theta)
amplitude = np.outer(np.ones(n_elements), amplitude_base)  # (n_elements, n_angles)

# Mutual coupling phase shift
element_indices = np.arange(n_elements)
mutual_coupling_phase = 0.1 * np.outer(element_indices, np.sin(theta))

# Position-dependent phase variation
position_phase = 0.05 * np.outer(element_indices, np.cos(2 * theta))

# Combine amplitude and phase
total_phase = mutual_coupling_phase + position_phase
aeps = amplitude * np.exp(1j * total_phase)

# Plot individual element patterns
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
ax = typing.cast(PolarAxes, ax)

for i in range(n_elements):
    ax.plot(theta, np.abs(aeps[i]), alpha=0.7, label=f"Element {i + 1}")

ax.set_title("Individual Antenna Element Patterns (AEPs)")
ax.set_theta_offset(np.pi / 2)  # 0° at top
ax.set_theta_direction(-1)  # Clockwise
ax.legend()
fig.savefig("individual_aeps.png", dpi=200, bbox_inches="tight")

# Calculate Array Factor for broadside direction (no steering)
phase_diff = k * np.outer(np.sin(theta), element_positions)
array_factor = np.sum(np.exp(1j * phase_diff), axis=1)

# Plot Array Factor
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
ax = typing.cast(PolarAxes, ax)

# Convert to dB for better visualization
af_db = 20 * np.log10(np.abs(array_factor) + 1e-10)
af_db_norm = af_db - np.max(af_db)  # Normalize to 0 dB max

ax.plot(theta, af_db_norm, "r-", linewidth=2, label="Array Factor")
ax.set_title("Array Factor (Broadside)")
ax.set_ylim(-40, 0)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.legend()
fig.savefig("array_factor.png", dpi=200, bbox_inches="tight")

# Combine AEPs with Array Factor to get total pattern
# Total pattern = Element Pattern × Array Factor
total_patterns = aeps * array_factor

# Sum all element contributions using matrix multiplication
weights = np.ones(n_elements)  # Uniform amplitude weighting
combined_pattern = weights @ total_patterns

# Convert to power pattern and dB
power_pattern = np.abs(combined_pattern) ** 2
power_pattern_db = 10 * np.log10(power_pattern + 1e-10)
power_pattern_db_norm = power_pattern_db - np.max(power_pattern_db)

# Plot the final combined pattern
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
ax = typing.cast(PolarAxes, ax)
ax.plot(theta, power_pattern_db_norm, "b-", linewidth=2)
ax.set_title("Combined Array Pattern")
ax.set_ylim(-40, 0)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.grid(True)

fig.savefig("combined_pattern.png", dpi=200, bbox_inches="tight")

main_lobe_idx = np.argmax(power_pattern_db_norm)
main_lobe_angle = np.degrees(theta[main_lobe_idx])
print(f"Main lobe: {main_lobe_angle:.1f}° {power_pattern_db_norm[main_lobe_idx]:.1f}dB")

# Create a steered beam by adding progressive phase to each element
steer_angle_deg = 30  # Steer beam to 30 degrees
steer_angle_rad = np.radians(steer_angle_deg)

# Calculate progressive phase shifts for beam steering
steering_phases = -k * element_positions * np.sin(steer_angle_rad)

# Apply steering phases to element patterns
# Create steered element patterns (same amplitude pattern for all elements)
steered_amplitude_base = 0.7 + 0.5 * np.cos(theta)  # (n_angles,)
steered_aeps = np.outer(np.ones(n_elements), steered_amplitude_base).astype(
    np.complex64
)

# Calculate steered array factor
phase_diff_steered = k * np.outer(np.sin(theta), element_positions)

# Apply steering phases to each element
steered_weights = np.exp(1j * steering_phases)  # (n_elements,)

# Calculate steered array factor using matrix multiplication
steered_array_factor = (steered_weights @ np.exp(1j * phase_diff_steered.T)).T
# Equivalent to: np.sum(steered_weights_matrix * np.exp(1j * phase_diff_steered), axis=1)

# Combine element patterns with steered array factor
steered_total_patterns = steered_aeps * steered_array_factor

# Sum all element contributions using matrix multiplication
steered_combined_pattern = weights @ steered_total_patterns

# Convert to power pattern and dB
steered_power_pattern = np.abs(steered_combined_pattern) ** 2
steered_power_pattern_db = 10 * np.log10(steered_power_pattern + 1e-10)
steered_power_pattern_db_norm = steered_power_pattern_db - np.max(
    steered_power_pattern_db
)

# Find the actual beam direction
steered_main_lobe_idx = np.argmax(steered_power_pattern_db_norm)
actual_steer_angle = np.degrees(theta[steered_main_lobe_idx])
steered_lobe_db_norm = steered_power_pattern_db_norm[steered_main_lobe_idx]

print(f"Steered lobe: {actual_steer_angle:.1f}° {steered_lobe_db_norm:.1f}dB")

# Plot comparison
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
ax = typing.cast(PolarAxes, ax)
ax.plot(theta, power_pattern_db_norm, "b-", linewidth=2, label="Broadside")
ax.plot(
    theta,
    steered_power_pattern_db_norm,
    "r-",
    linewidth=2,
    label=f"Steered {steer_angle_deg}°",
)
ax.set_title("Beam Steering Comparison")
ax.set_ylim(-40, 0)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.legend()

fig.savefig("beam_steering_demo.png", dpi=200, bbox_inches="tight")
