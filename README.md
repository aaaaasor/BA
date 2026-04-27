# GP Flow Matching Demo

This project contains a decoupled 1D GP-based flow matching demo.

The demo uses:

- the source distribution is a standard Gaussian;
- the target distribution is a mixture of two Gaussians;
- the conditional path is linear;
- the velocity field is learned with Gaussian process regression.

The core supervised flow matching construction is:

- sample `x0 ~ p0`;
- sample `x1 ~ p1`;
- sample `t ~ Uniform(0, 1)`;
- form `xt = (1 - t) * x0 + t * x1`;
- use target velocity `v*(t, xt) = x1 - x0`.

After fitting a GP model to `(t, xt) -> v*`, we generate samples by solving:

`dx/dt = v(t, x), x(0) ~ p0`

from `t = 0` to `t = 1`.

## Python demo

Run:

```powershell
& 'C:\Users\JieLi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' python/main.py
```

Outputs are saved under `python/outputs/`:

- `gp_flow_matching_demo.png`: 4-panel summary figure;
- `generated_samples.csv`: generated terminal samples at `t=1`;
- `trajectory_samples.csv`: selected trajectories across time.

The Python implementation now uses `matplotlib` for plotting.

## Notes

- This is a clean 1D demo, not a full research-grade training pipeline.
- The GP is implemented as exact kernel regression with an RBF kernel on inputs `(t, x_t)`.
- The conditional linear path follows the standard flow-matching construction in the referenced guide:
  `x_t = (1 - t) x_0 + t x_1`, with supervision `v* = x_1 - x_0`.
