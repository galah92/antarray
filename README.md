# antarray

# Tasks

Physics-based loss:

1. Mainlobe gain in direction
2. Sidelobe level

```python
def loss_fn(pattern, mu):
    mainlobe_gain = ..
    max_sidelobe_gain = ..
    # loss = -mainlobe_gain + mu * np.max(mainlobe_gain - max_sidelobe_gain - max_ssl, 0)
    loss = -mainlobe_gain + mu * (max_gain - second_max_gain)
```

Loss against linear power instead of dB power
