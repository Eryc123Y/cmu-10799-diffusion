# Source Code Guidance

This directory contains the implementation that should teach the DDPM pipeline.
Preserve that educational shape.

## Python And PyTorch Style

- Keep public function signatures typed and tensor contracts explicit.
- Prefer direct, formula-shaped code over abstraction layers that obscure DDPM.
- Use the provided helpers and blocks before adding new utilities.
- Keep tensor ranges clear:
  - model/data tensors for training should be in `[-1, 1]`;
  - image saving should convert or clamp to `[0, 1]` as needed.
- Name variables to match the math where helpful: `x_0`, `x_t`, `noise`,
  `alpha`, `alpha_bar`, `beta`, and `pred_noise`.
- Comments should explain non-obvious math, broadcasting, or shape choices, not
  restate simple operations.

## Learning Workflow

- Before filling a TODO, identify the formula and the expected tensor shapes.
- Add small smoke checks for shape/range/device behavior before large training
  runs.
- When debugging, compare both loss behavior and generated samples. A good loss
  curve alone is not proof that the sampling equation is correct.

## Verification

- Minimum local syntax check after code edits:
  `python -m compileall src train.py sample.py`
- For model changes, instantiate a small model and run a forward pass on CPU
  before attempting full training.
- For method changes, test at least one tiny batch through `compute_loss`, then
  test a very small sampling loop.
