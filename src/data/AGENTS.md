# Data Guidance

This package owns dataset loading, preprocessing, visualization helpers, and
image range conversion.

## Rules

- Keep `CelebADataset.__getitem__` returning only image tensors for
  unconditional generation unless the assignment scope changes.
- Transforms should produce `torch.Tensor` images with shape `(3, H, W)` in
  `[-1, 1]`.
- Use torchvision transforms rather than ad hoc PIL or tensor manipulation when
  possible.
- Keep augmentation modest and defensible for faces. Horizontal flips are usually
  easy to justify; aggressive crops, rotations, or color shifts need evidence.
- Do not silently resize in a way that changes the assignment image-size
  contract. If resizing is used, make it explicit and config-driven.

## Checks

- Verify one dataset item has the expected shape, dtype, min/max range, and no
  NaNs.
- When changing image saving helpers, verify the produced PNG visually, not only
  by checking that the file exists.
