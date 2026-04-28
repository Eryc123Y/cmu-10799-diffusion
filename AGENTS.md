# Repository Guidance

This repository is a learning workspace for CMU 10-799 Diffusion & Flow
Matching. Treat the primary goal as helping the student understand diffusion
models deeply, not only making the starter code run.

## Collaboration Style

- Start from the course artifact in front of the user: the homework TeX, the
  relevant Python file, the config, or the notebook they named.
- Explain concepts before code when the task is learning-oriented. Connect each
  implementation detail back to the corresponding diffusion formula, tensor
  shape, or experiment requirement.
- Prefer scaffold-first help unless the user explicitly asks for a full
  implementation. Good scaffold-first help includes function contracts, shape
  expectations, pseudocode, sanity checks, and common failure modes.
- When implementing, keep the code readable enough for the student to defend in
  a homework report. Avoid clever abstractions that hide the DDPM mechanics.
- Keep code, configs, notebooks, figures, and LaTeX answers aligned. If a result
  changes, update the report surface or note the stale artifact.

## Course Context

- HW1 implements DDPM from scratch on a filtered CelebA 64x64 subset.
- The intended workflow is: inspect data, build 1D intuition, implement DDPM and
  U-Net, debug with loss/sample evidence, run ablations, then reflect.
- Important implementation surfaces:
  - `src/data/celeba.py`: image transforms and normalization.
  - `src/methods/ddpm.py`: forward noising, training loss, reverse step, sampling.
  - `src/models/unet.py`: timestep-conditioned U-Net using `src/models/blocks.py`.
  - `train.py` and `sample.py`: sampling and image-saving pipeline integration.
  - `configs/`: experiment and infrastructure parameters.
- Homework answers live under `HW1_CMU_10799_Spring_2026/`.

## Evidence And Verification

- For Python changes, run the lightest meaningful check first, then expand:
  `python -m compileall src train.py sample.py`, targeted smoke scripts, and
  overfit/sampling checks when the implementation is ready.
- For LaTeX changes, compile the homework and inspect the rendered PDF when
  visual correctness matters.
- For training claims, report concrete evidence: config path, checkpoint/log
  path, step count, loss curve, sample grid, KID command, KID mean/std, and any
  known compute limitations.
- Do not commit generated datasets, checkpoints, logs, wandb output, or LaTeX
  build intermediates. Keep large model artifacts outside source control.

## Academic And Resource Notes

- The homework explicitly allows AI and external resources, but resource usage
  must be documented in the reflection section.
- Do not invent experimental results. If a run was not performed, say what would
  be run and mark the result as pending.
- Budget-sensitive Modal or GPU work should be explicit about expected cost,
  duration, and whether a smaller sanity run can prove the next step first.
