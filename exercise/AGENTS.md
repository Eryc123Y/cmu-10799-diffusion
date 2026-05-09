# Exercise Workspace Guidance

This folder stores lecture-specific practice material for CMU 10-799 Diffusion
and Flow Matching. Treat it as a durable study surface, not a scratch folder.

## Directory Structure

- Put each lecture in its own folder.
- Use lowercase, hyphenated folder names:
  - `lecture-4-score/`
  - `lecture-5-flow/`
  - `lecture-6-fast-sampling/`
- Keep all files for one lecture inside that lecture folder: Typst source,
  compiled PDFs, small helper scripts, and any local figures.
- Do not put lecture-specific `.typ` or `.pdf` files directly under
  `exercise/`.

## Typsidian Template Contract

All Typst exercise handouts in this folder should use the Typsidian template:

```typst
#import "@preview/typsidian:0.0.3": *

#show: typsidian.with(
  theme: "light",
  title: [Lecture N Topic],
  course: [Short study subtitle],
  author: [Eric],
  text-args: (
    main: (font: "New Computer Modern"),
    mono: (font: "Menlo"),
    math: (font: "New Computer Modern Math"),
    headings: (font: "New Computer Modern"),
  ),
)
```

Use Typsidian-provided containers and helpers only:

- `#box(theme: "frame", icon: none, title: [...], breakable: true)[...]`
- `#box(theme: "example", icon: none, title: [...], breakable: true)[...]`
- `#box(theme: "info", icon: none, title: [...], breakable: true)[...]`
- `#note(icon: none)[...]` only when the note style is appropriate.
- `#qa(question, answer)` for compact question-answer examples.
- `#term(word, definition)` for glossary-like concept entries.
- `#hr()` for template-styled separators.

Do not create custom visual containers with raw `block`, `rect`, `box`,
`stroke`, `fill`, `radius`, or handcrafted color palettes. Ordinary Typst
tables, headings, lists, math, and code fences are fine; let Typsidian's global
show rules provide their visual style.

Avoid custom table colors, borders, or insets unless the user explicitly asks
for a non-template visual exception.

## Typst Math Style

Write Typst math in Typst syntax, not LaTeX muscle memory:

- Use `overline(alpha)_t` for $\bar{\alpha}_t$.
- Do not use `bar(alpha)_t`; in Typst this does not render as the intended
  overline accent.
- Use `hat(epsilon)_theta`, `hat(x)_(0, theta)`, and `hat(s)_theta` for hatted
  predictions.
- Use `nabla_(x_t)`, `sqrt(...)`, `cal(N)`, `epsilon`, `theta`, and `sigma`
  directly in Typst math mode.
- After math-heavy edits, compile and visually inspect the PDF around the
  changed formulas, especially accents, subscripts, and hats.

## Exercise Design Style

Exercises should match the student's current state:

- Start from the lecture PDF and the homework surface currently open or active.
- Connect intuition -> formula -> tensor/code shape -> homework implication.
- Include both theory derivations and coding sanity checks when useful.
- Prefer short, defendable exercises over broad question dumps.
- Include solution sketches, but place them after the practice section so the
  student can attempt first.
- Do not invent experimental results. If a task asks for outputs, make it clear
  that the student should run the code and record their own result.

## Collaboration Contract

When creating or modifying exercises here:

- First identify the lecture, homework anchor, and intended learning outcome.
- Keep lecture-specific material in the matching lecture folder.
- Preserve the Typsidian style contract above before adding new visual elements.
- If the user asks for a template-based handout, use template containers rather
  than custom helper wrappers.
- Compile the Typst file after edits with `typst compile <path>.typ`.
- Report the source path, PDF path, and any compile warnings honestly.

## Verification

For a standard edit, run:

```sh
typst compile exercise/<lecture-folder>/<file>.typ
```

If a PDF already exists, regenerate it in the same lecture folder. Do not leave
stale PDFs at the `exercise/` root after moving files.
