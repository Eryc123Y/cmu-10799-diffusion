# Repository Guidance

This repository is a learning workspace for CMU 10-799 Diffusion & Flow
Matching. Treat the primary goal as helping the student understand diffusion
models, flow matching, generative-model evaluation, and experiment reporting
deeply, not only making the starter code run.

## Collaboration Style

- Start from the course artifact in front of the user: the homework TeX, the
  relevant Python file, the config, or the notebook they named.
- Explain concepts before code when the task is learning-oriented. Connect each
  implementation detail back to the corresponding diffusion formula, tensor
  shape, or experiment requirement.
- Move from intuition to formula, then to tensor shapes, then to code. For
  algorithms, identify inputs, outputs, random variables, fixed coefficients,
  model predictions, loss targets, and sampling or inference procedures.
- Prefer scaffold-first help unless the user explicitly asks for a full
  implementation. Good scaffold-first help includes function contracts, shape
  expectations, pseudocode, sanity checks, and common failure modes.
- Work in small reviewable sections for implementations. For example, finish
  initialization and schedules before forward sampling, then loss, then reverse
  sampling, instead of jumping directly to a complete sampler.
- When implementing, keep the code readable enough for the student to defend in
  a homework report. Avoid clever abstractions that hide the DDPM mechanics.
- Keep code, configs, notebooks, figures, and LaTeX answers aligned. If a result
  changes, update the report surface or note the stale artifact.

## Learning-Oriented PyTorch Style

- Prefer clear, explicit PyTorch code over compressed one-liners when tensor
  shapes, broadcasting, sampling, or distribution parameters are involved.
- Use intermediate variables for meaningful mathematical quantities and shape
  transformations. Names like `batch_size`, `broadcast_shape`, `image_shape`,
  `num_channels`, `height`, and `width` are preferred over dense inline shape
  expressions when they make the tensor contract easier to inspect.
- When reshaping for broadcasting, build the target shape explicitly before
  applying it. Prefer:

  ```python
  broadcast_shape = (t.shape[0],) + (1,) * (len(x_shape) - 1)
  coefficients = coefficients.reshape(broadcast_shape)
  ```

  over denser forms that rely on nested unpacking.
- Use comments to explain mathematical meaning, tensor semantics, or non-obvious
  broadcasting. Avoid comments that merely restate the operation.
- Keep formulas close to their mathematical form. When helpful, use names such
  as `x_0`, `x_t`, `noise`, `pred_noise`, `t`, `alpha`, `alpha_bar`, `beta`,
  `velocity`, `score`, `mean`, and `variance`.
- Make batch semantics explicit. If a tensor has shape `(B,)`, state whether it
  contains one timestep per image, one scalar per sample, or one metric per
  batch item.
- Do not hide course-critical mechanics behind clever abstractions. Helpers are
  appropriate when they isolate repeated shape logic, but not when they obscure
  the algorithm the student needs to understand.

## Notebook, Source, Config, And Report Boundaries

- Use notebooks for intuition, visualization, small experiments, and debugging
  explanations.
- Use `src/` for reusable implementation that training and sampling scripts
  depend on.
- Use `configs/` for reproducible experiment parameters, infrastructure
  settings, and compute-sensitive choices.
- Use homework TeX/report files for final reasoning, results, figures, and
  reflection.
- Keep these surfaces aligned. If code behavior changes, update related configs,
  figures, or report claims, or explicitly mark them as stale.
- Do not treat notebook-only results as report-ready unless the run, config,
  seed, output path, and limitations are recorded.

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
- For method/model changes, test small tensor contracts before expensive
  training: schedule shapes, broadcasting shapes, model output shapes, scalar
  losses, finite values, and tiny sampling loops when relevant.
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
