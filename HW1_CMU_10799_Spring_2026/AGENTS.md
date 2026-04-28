# HW1 LaTeX Guidance

This directory contains the Homework 1 handout and answer surface for DDPM
implementation, debugging, ablations, and reflection.

## Editing Rules

- Preserve the existing `main.tex` structure and macros unless a formatting
  problem requires changing them.
- Put answers in the appropriate `questions/part_*.tex` file, near the matching
  `\answerbox`.
- Keep answers concise, evidence-backed, and tied to the exact prompt.
- When writing about experiments, include the concrete config, training length,
  sample grid, KID mean/std, or note that the run is still pending.
- Do not edit LaTeX build artifacts such as `.aux`, `.bbl`, `.blg`,
  `.fdb_latexmk`, `.fls`, `.out`, `.log`, or `.synctex.gz`.
- Keep generated figures under `figures/` with descriptive names when they are
  part of the submission.

## Learning And Academic Notes

- The homework allows AI help, but Part V asks for helpful resources. If AI
  materially helps with code, derivations, debugging, or writing, mention that in
  the resources answer.
- Do not fabricate experiment evidence. Draft placeholders as pending when the
  required training or evaluation has not been run.

## Verification

- Compile after meaningful answer or figure edits.
- Visual correctness matters: inspect the rendered PDF for math layout, figure
  sizing, overfull content, and whether answers appear under the right question.
