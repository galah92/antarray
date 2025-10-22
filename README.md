# Phased Array Beamforming Optimization

This project optimizes phased array antenna weights to match radiation patterns between different environments using gradient-based optimization.

## Documentation

- 📊 **[Weight Initialization Study](docs/weight_initialization_study.md)** - Comprehensive investigation of initialization strategies, bug fixes, and performance analysis

## Overview

The optimization uses JAX for automatic differentiation and vectorized batch processing to adjust antenna array weights, achieving pattern matching between:
- **env0**: Free space (reference environment)
- **env1**: Environment with obstacles

### Key Features

- **Vectorized optimization**: Process multiple steering angles in parallel (10-18x speedup)
- **Type-safe configuration**: Literal types for environments, tapers, loss functions
- **Comprehensive metrics**: NMSE, success rates, percentile analysis
- **Flexible initialization**: Support for env0, random, uniform, and zero initializations
- **Grid evaluation**: Systematic testing across elevation/azimuth angles

## Quick Start

```python
from milestone import run_optimization

# Run optimization for a single steering angle
result = run_optimization(
    env0_name="no_env_rotated",
    env1_name="Env1_2_rotated",
    taper="uniform",              # or "hamming"
    weight_init="env0",           # RECOMMENDED: always use env0
    elev_deg=0.0,
    azim_deg=0.0,
    lr=1e-5,
    plot=True
)

print(f"NMSE improvement: {result.nmse_opt - result.nmse_env1:.3f} dB")
```

## Key Results

- **env0 initialization**: -21.6 dB NMSE, 91.7% success rate across angles
- **Vectorized processing**: 360 optimizations in ~40 seconds (vs 1-2 hours sequential)
- **Best environment**: Env1_2 with uniform taper achieves -5.2 dB average improvement

## Research Notes

### Physics-based Loss (Future Work)

Potential multi-objective loss combining:
1. Mainlobe gain in desired direction
2. Sidelobe suppression

```python
def loss_fn(pattern, mu):
    mainlobe_gain = ..
    max_sidelobe_gain = ..
    loss = -mainlobe_gain + mu * (max_gain - second_max_gain)
```

### Implementation Notes

- ✓ Loss computed in linear power space (not dB) for stable gradients
- ✓ NMSE metric: MSE normalized by target power, reported in dB
- ✓ Type safety with Literal types for all configuration options
