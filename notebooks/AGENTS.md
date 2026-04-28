# Notebook Guidance

Notebooks are for intuition, inspection, and visual evidence.

## Rules

- Keep explanatory markdown close to the code cell it explains.
- Prefer small, reproducible cells over long hidden pipelines.
- Set seeds when a visualization or toy experiment is used as evidence.
- Keep important learning content inside the notebook instead of moving it only
  into external markdown.
- Do not store large outputs, datasets, or checkpoints in notebooks.

## Expected Uses

- `01_1d_playground.ipynb`: build DDPM intuition on toy 1D distributions.
- `02_dataset_exploration.ipynb`: inspect CelebA samples and justify transforms.
- `03_sampling_visualization.ipynb`: understand sampling behavior and failure
  modes.
