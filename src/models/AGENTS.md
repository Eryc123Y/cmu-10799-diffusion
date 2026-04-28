# Model Guidance

This package owns U-Net architecture and neural network building blocks.

## U-Net Rules

- Reuse `TimestepEmbedding`, `ResBlock`, `AttentionBlock`, `Downsample`, and
  `Upsample` from `blocks.py` before introducing new modules.
- The model forward contract is `model(x_t, t) -> prediction` with output shape
  matching the input image shape.
- Preserve skip connections carefully. Track channel counts during construction
  instead of guessing in the forward pass.
- Apply attention only at configured spatial resolutions. Keep this tied to the
  current feature map size, not hardcoded layer numbers.
- Keep timestep conditioning present in every residual block unless an experiment
  explicitly removes it.

## Checks

- After architecture edits, run a CPU forward pass with a small channel count and
  at least one config-like 64x64 shape.
- Count parameters and report them when comparing model configs.
- Watch for GroupNorm divisibility constraints when changing channel counts.
