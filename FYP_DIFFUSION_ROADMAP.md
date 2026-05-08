# FYP-Oriented Course Study Roadmap

This note is about **how to study CMU 10-799 Diffusion and Flow Matching with
an FYP purpose in mind**.

The FYP context is:

> 2D-to-3D video conversion for XR applications.

That does not mean every study session should become an FYP implementation
session. The course still has its own structure and should be learned as a
course. The FYP lens only changes **where to slow down, what to connect, and
which follow-up questions to keep asking**.

## Guiding Principle

Study the course in this order of priority:

```text
understand the course concept
-> implement or inspect the homework mechanism
-> ask how this mechanism would appear in 2D-to-3D video / XR
```

Do not skip the course foundations just because the FYP is more applied. For
this project, the most useful knowledge is not a list of 2D-to-3D papers. It is
the ability to read those papers with a clear mental model of diffusion,
conditioning, sampling, and flow-based generation.

## What the FYP Lens Should Change

When learning each lecture, ask three extra questions:

1. What is the input condition?
   For the FYP, conditions may be an RGB frame, depth map, optical flow, camera
   motion, previous frame, or target view.

2. What is the model trying to predict or generate?
   It may be noise, score, velocity, a clean image, depth, a latent, a target
   view, or an inpainted disocclusion region.

3. What would break when this is applied to video/XR?
   Common answers: slow sampling, temporal flicker, inconsistent geometry,
   disocclusion artifacts, or metrics that do not match viewer comfort.

These questions should sit beside the course material, not replace it.

## Course Priority Map

### Tier 1: Study Deeply

These parts should be learned slowly and actively, with formulas, code, and
experiments connected.

#### DDPM and Score-Based Models

Course role:

This is the foundation for understanding diffusion as a generative process.

Learn until you can explain:

- \(x_0\), \(x_t\), \(\epsilon\), \(\beta_t\), \(\alpha_t\), and
  \(\bar{\alpha}_t\),
- the forward noising process,
- the learned reverse process,
- why many models predict noise,
- how score prediction relates to noise prediction,
- why sampling requires many sequential denoising steps,
- why raw training loss and generated quality are not always the same story.

Homework anchor:

- HW1 DDPM implementation,
- HW1 Part IV parameterization ablation,
- HW1 Part IV sampling-step ablation.

FYP connection:

Diffusion-based depth models such as Marigold are much easier to understand
once DDPM and score-based denoising are clear. Even if the FYP uses pretrained
models, you still need to know what the denoiser is doing and what the sampling
process costs.

Study action:

After HW1, write a short explanation of how a diffusion prior could be used to
estimate or refine depth. Keep it conceptual; do not jump to implementation
yet.

#### Flow Matching

Course role:

Flow matching is the next generative-model language after DDPM. It reframes
generation as learning a vector field that transports a simple distribution
toward the data distribution.

Learn until you can explain:

- what path connects noise and data,
- what velocity/vector field the network predicts,
- how flow matching differs from DDPM-style stepwise denoising,
- why ODE-style generation can be attractive for modern image and video models.

Homework anchor:

- HW2 flow matching implementation and experiments.

FYP connection:

Modern image and video generation systems increasingly use flow or
rectified-flow ideas. For the FYP, this matters because future 2D-to-3D methods
may be built on flow-based backbones rather than classic DDPM samplers.

Study action:

When implementing HW2, keep a comparison table against HW1:

- model target,
- training loss,
- sampling procedure,
- speed implication,
- where conditioning would enter.

#### Design Space and Fast Sampling

Course role:

This teaches how diffusion models become practical: schedules,
parameterizations, solvers, and sampling shortcuts.

Learn until you can explain:

- DDIM,
- DPM-Solver,
- EDM-style design choices,
- timestep spacing,
- few-step sampling,
- how solver choice affects speed and quality.

Homework anchor:

- HW1 Q7 is the first concrete version of this idea.
- Later homework or readings may introduce stronger solvers.

FYP connection:

Video and XR are latency-sensitive. A method that is acceptable for one image
may be too slow for a video pipeline. When reading any diffusion-based depth,
view-synthesis, or inpainting paper, ask how many denoising steps it needs and
whether that cost scales across frames.

Study action:

For each solver or sampler, write one sentence answering:

> Would this make a video/XR pipeline faster, more stable, or just more
> complicated?

### Tier 2: Study with FYP Connections in Mind

These parts are important, but you do not need to master every derivation on
the first pass.

#### Guidance and Conditional Generation

Course role:

This explains how diffusion models become controllable rather than purely
unconditional.

Learn until you can explain:

- classifier-free guidance,
- image-conditioned generation,
- inpainting,
- posterior sampling,
- where external constraints enter the denoising process.

FYP connection:

2D-to-3D conversion is a conditional generation problem. The model is not asked
to generate anything from scratch; it is guided by an input frame or video,
geometry cues, camera/view information, and temporal context.

Study action:

For every guidance method, identify what the condition would be in the FYP:

- source RGB frame,
- estimated depth,
- target camera shift,
- optical flow,
- previous frame,
- occlusion mask,
- stereo consistency constraint.

#### Latent Diffusion and DiT

Course role:

This explains the architecture used by many modern systems: latent-space
generation, cross-attention conditioning, and Transformer denoisers.

Learn until you can explain:

- why latent diffusion is cheaper than pixel-space diffusion,
- how cross-attention injects conditioning information,
- why DiT replaces U-Net blocks with Transformer blocks,
- how conditioning and timestep information are represented in the network.

FYP connection:

Many useful depth, view-synthesis, and video-generation systems are built on
latent diffusion or Transformer-style backbones. Understanding the architecture
lets you read those systems without treating them as black boxes.

Study action:

When reading a model paper, draw a simple block diagram:

```text
input condition -> encoder/condition module -> denoiser/flow model -> output
```

Label whether the output lives in pixel space, latent space, depth space, or
view space.

#### Video Diffusion and Temporal Modeling

Course role:

This may not be the central course topic, but it is the closest bridge to the
FYP application.

Learn until you can explain:

- spatial attention versus temporal attention,
- framewise generation versus video-aware generation,
- temporal modules,
- why per-frame quality does not guarantee video quality.

FYP connection:

Temporal consistency is one of the main FYP risks. A depth model can look good
on individual frames and still fail badly when played as video.

Study action:

Whenever a lecture or paper discusses video, ask:

- how information moves across frames,
- whether the model enforces consistency or only hopes for it,
- whether the metric measures flicker or viewer comfort.

### Tier 3: Study Lightly Unless the FYP Changes Direction

These topics are valuable, but they are less central for the current FYP.

#### Discrete Diffusion and Masked Diffusion

Course role:

These extend diffusion ideas to discrete or token-like spaces.

Current FYP relevance:

Lower priority. They matter more if the project shifts toward tokenized video,
language-conditioned planning, multimodal tokens, or discrete scene
representations.

Study action:

Learn the high-level idea and vocabulary, but do not spend the first pass
trying to master every objective.

## Homework Strategy

The homeworks should be used as controlled practice for course concepts, not
only as assignments to finish.

### HW1: DDPM Foundation

Primary learning goal:

Understand the mechanics of diffusion from scratch.

FYP-oriented emphasis:

- parameterization affects optimization,
- sampling steps affect speed and quality,
- evaluation metrics are useful but incomplete,
- debugging diffusion models requires looking beyond raw loss.

What to keep from HW1:

- your DDPM implementation,
- Part IV ablation results,
- the reflection on slow sampling and indirect objectives,
- the study package for explaining DDPM to teammates.

### HW2: Flow Matching

Primary learning goal:

Understand generation as a learned transport/vector-field problem.

FYP-oriented emphasis:

- compare flow sampling cost against DDPM sampling cost,
- understand whether flow-style models may be better suited for faster video
  pipelines,
- track how conditioning would enter the vector field.

### Later Homeworks or Project Tracks

Primary learning goal:

Use them as chances to specialize toward the FYP.

FYP-oriented emphasis:

- choose topics involving conditional generation, image/video conditioning,
  latent models, inverse problems, or fast sampling when options exist,
- avoid spending too much optional effort on discrete/text-only directions
  unless the FYP scope changes.

## How to Read Course Papers

For every paper assigned by the course, answer two sets of questions.

Course understanding:

1. What distribution or path is being modeled?
2. What does the neural network predict?
3. What is the training objective?
4. What is the sampling or inference procedure?
5. What problem was the paper trying to fix compared with earlier methods?

FYP translation:

1. Could this method condition on an input video or geometry cue?
2. Does it help speed, control, temporal consistency, or quality?
3. Would it operate on RGB pixels, latent features, depth maps, or views?
4. What would be expensive if applied frame-by-frame?
5. What failure mode would matter most in XR?

This keeps the FYP present without letting it swallow the course.

## Paper Priority for FYP Context

Use the course papers as the main sequence. Add FYP papers only when they help
you understand why a course concept matters.

### Read Alongside DDPM and Score Models

- Ho et al., "Denoising Diffusion Probabilistic Models", 2020.
- Song et al., "Score-Based Generative Modeling through Stochastic Differential
  Equations", 2021.
- Ke et al., "Marigold: Repurposing Diffusion-Based Image Generators for
  Monocular Depth Estimation", 2023.

### Read Alongside Fast Sampling and Solvers

- Song et al., "Denoising Diffusion Implicit Models", 2020.
- Lu et al., "DPM-Solver", 2022.
- Karras et al., "Elucidating the Design Space of Diffusion-Based Generative
  Models", 2022.

### Read Alongside Conditional and Latent Diffusion

- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion
  Models", 2022.
- Ho and Salimans, "Classifier-Free Diffusion Guidance", 2021.
- Lugmayr et al., "RePaint", 2022.
- Zero-1-to-3 for view-conditioned generation.

### Read Alongside Video Topics

- Video Depth Anything.
- ChronoDepth.
- video diffusion or motion-module papers such as Imagen Video and AnimateDiff.

Do not treat this as a separate FYP reading list. Treat it as an annotation
layer on top of the course.

## Weekly Study Template

For each lecture or homework block, use this lightweight template.

### Before Lecture

- Skim the lecture topic and assigned paper abstract.
- Write one FYP question this topic might help answer.

### During Lecture

- Track definitions and equations first.
- Mark any idea related to conditioning, sampling cost, or temporal stability.

### After Lecture

- Write a five-sentence summary in your own words.
- Add one code or homework anchor if available.
- Add one FYP translation sentence.

Example:

```text
Course idea: DDIM reduces sampling steps by using a deterministic non-Markovian
sampling path.

Homework anchor: HW1 Q7 compared fewer sampling steps against the full
1000-step sampler.

FYP translation: If a depth or inpainting model uses diffusion sampling, fewer
steps may make video conversion faster but could increase temporal artifacts.
```

## What Not to Do

- Do not turn every lecture into a full FYP literature review.
- Do not skip math because the FYP is applied.
- Do not judge methods only by whether they mention 2D-to-3D.
- Do not over-invest in low-priority discrete diffusion topics before the main
  diffusion/flow/conditioning path is clear.
- Do not assume a good image metric means good XR comfort.

## Bottom Line

Study the course as a structured path into modern generative modeling.

Use the FYP lens to choose depth:

```text
DDPM and score models: deep
Flow matching: deep
Fast sampling and solvers: deep
Guidance and conditional generation: medium-deep
Latent diffusion, DiT, video diffusion: medium-deep
Discrete diffusion: light first pass
```

The aim is to finish the course able to read, evaluate, and adapt modern
diffusion/flow models for temporally stable, geometry-aware 2D-to-3D video
conversion, not merely to collect FYP papers.
