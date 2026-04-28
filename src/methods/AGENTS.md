# Method Guidance

This package owns diffusion or flow-matching algorithms. For HW1, `ddpm.py` is
the central algorithm file.

## DDPM Implementation Rules

- Keep the implementation close to the standard DDPM equations so the student
  can map code back to the homework derivation.
- Register fixed schedules as buffers on the method/model object when they must
  move with the device.
- Use a helper for extracting timestep-indexed schedule values into broadcastable
  image shapes.
- `compute_loss` should make the prediction target explicit. For baseline HW1,
  that target is usually the sampled Gaussian noise `epsilon`.
- `reverse_process` should clearly separate the predicted mean, variance/noise
  term, and the `t == 0` case.
- `sample` should start from Gaussian noise and iterate timesteps in descending
  order.

## Debugging Priorities

- First check schedules and tensor ranges.
- Then check timestep sampling, noise target, and model output shape.
- Then check reverse-process coefficients and whether noise is added at `t = 0`.
- Use tiny `num_timesteps` and tiny image batches for fast sanity checks before
  expensive training.
