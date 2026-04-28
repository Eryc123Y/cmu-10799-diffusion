# Script Guidance

Scripts are examples for cluster, Modal, and evaluation workflows.

## Rules

- Keep scripts parameterized by method, config, checkpoint, or output path. Avoid
  hardcoding user-specific absolute paths unless the script name documents that
  scope.
- Do not put secrets, API keys, wandb tokens, or credentials in scripts.
- Echo enough context before a long run: method, config, checkpoint, output
  directory, and important environment variables.
- Prefer dry-run-readable commands so failures can be copied into the homework
  notes or debugging log.
- Be conservative with compute-heavy defaults.

## Verification

- For shell edits, run `bash -n <script>` when possible.
- For evaluation scripts, confirm the output includes the metric names and paths
  needed for the report.
