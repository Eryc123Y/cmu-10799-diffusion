# Config Guidance

Configs are experiment records, not just runtime knobs.

## Rules

- Keep configs runnable and explicit. Avoid leaving required values as YAML nulls
  once a config is meant to be used.
- Name new config files after the experiment purpose, hardware, or ablation.
- Keep model, DDPM, training, sampling, infrastructure, checkpoint, and logging
  sections aligned with the Python code.
- Do not hide major experiment changes in command-line overrides only. Put
  durable choices in a config file.
- For Modal or cluster runs, record conservative sample/save/log intervals so a
  failed run still leaves useful evidence.

## Reporting

- When a config produces report numbers, record the config path, seed, iteration
  count, sampling steps, and checkpoint path in the homework answer or notes.
