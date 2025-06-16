# Physics Flow Consolidation Specification

## Overview
Add OpenEMS data support to the existing unified physics interface in `create_physics_setup()`.

## Current State
- **Legacy flow**: `calc_array_params()` → `rad_pattern_from_geo()` (OpenEMS data, used in data.py)
- **Modern flow**: `create_physics_setup()` → synthesizers (synthetic data, used in training modules)

## Target State
- **Unified flow**: `create_physics_setup()` supports both OpenEMS and synthetic data
- All modules use the same interface
- Full backward compatibility

## Implementation

### 1. Enhanced `create_element_patterns()`
Add `openems_path` parameter to load real OpenEMS data:

```python
def create_element_patterns(
    config: ArrayConfig, 
    key: jax.Array, 
    is_embedded: bool,
    openems_path: Path | None = None  # New parameter
) -> jax.Array:
    if openems_path is not None:
        # Load OpenEMS data and tile across array positions
        openems_data = load_openems_nf2ff(openems_path)
        single_element = openems_data.E_field  # (n_theta, n_phi, 2)
        base_patterns = jnp.tile(single_element[None, None, ...], (*config.array_size, 1, 1, 1))
        
        if is_embedded:
            # Add element variations to simulate coupling
            return add_embedding_effects(base_patterns, key)
        return base_patterns
    else:
        # Existing synthetic generation (unchanged)
        return create_synthetic_patterns(config, key, is_embedded)
```

### 2. Updated `create_physics_setup()` (backward compatible)
```python
def create_physics_setup(
    key: jax.Array, 
    config: ArrayConfig | None = None,
    openems_path: Path | None = None,  # New optional parameter
):
    # Always use synthetic for ideal patterns (clean reference)
    ideal_patterns = create_element_patterns(config, key, is_embedded=False)
    
    # Use OpenEMS or synthetic for embedded patterns based on parameter
    embedded_patterns = create_element_patterns(config, key, is_embedded=True, openems_path=openems_path)
    
    # Rest unchanged...
```

### 3. Consumer Updates
All training modules need one-line change:
```python
# Add openems_path parameter to existing calls
synthesize_ideal, synthesize_embedded, compute_analytical = create_physics_setup(
    key, config=config, openems_path=openems_path
)
```

## Testing Strategy

### Core Tests
```python
def test_openems_integration():
    """Test OpenEMS data loading and format conversion."""
    config = ArrayConfig(array_size=(4, 4))
    key = jax.random.key(42)
    
    patterns = create_element_patterns(config, key, is_embedded=False, openems_path=DEFAULT_SIM_PATH)
    assert patterns.shape == (4, 4, 180, 360, 2)  # Correct 5D format

def test_backward_compatibility():
    """Verify existing calls work unchanged."""
    key = jax.random.key(42)
    synthesize_ideal, synthesize_embedded, compute_analytical = create_physics_setup(key)
    assert all(callable(f) for f in [synthesize_ideal, synthesize_embedded, compute_analytical])

def test_both_modes_produce_valid_patterns():
    """Compare synthetic vs OpenEMS modes."""
    key = jax.random.key(42)
    config = ArrayConfig()
    weights = jnp.ones((16, 16), dtype=jnp.complex64)
    
    # Both modes should produce valid patterns
    _, syn_synthesizer, _ = create_physics_setup(key, config)
    _, ems_synthesizer, _ = create_physics_setup(key, config, openems_path=DEFAULT_SIM_PATH)
    
    syn_pattern = syn_synthesizer(weights)
    ems_pattern = ems_synthesizer(weights)
    
    assert syn_pattern.shape == ems_pattern.shape
    assert jnp.isfinite(syn_pattern).all() and jnp.isfinite(ems_pattern).all()
```

### Integration Tests
- Run short training pipelines with both modes
- Test dataset generation with OpenEMS data
- Verify JIT compilation works for both paths

## Success Criteria
- [ ] All existing code works without changes
- [ ] OpenEMS and synthetic modes produce consistent pattern shapes
- [ ] All training modules support `openems_path` parameter
- [ ] Performance: <5% regression, JIT compilation works
- [ ] Error handling: Clear messages for invalid OpenEMS files

## Timeline
**1 day implementation + 1 day testing** - Simple additive change with comprehensive validation.