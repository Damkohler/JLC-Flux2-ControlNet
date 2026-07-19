# Performance and Memory

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


FLUX.2-dev plus the compact Union ControlNet side model is a demanding workload. Release 1.0.0 focuses on correct ComfyUI lifecycle integration, shared model ownership, and elimination of avoidable static preparation—not on making every high-resolution branch combination inexpensive.

## Main cost drivers

### Output resolution

Higher output resolution increases target token count, activation size, VAE work, and residual tensors.

### Active ControlNet branches

The Orchestrator shares one side-model weight set, but each active branch still executes the side model independently during denoising. Sharing weights prevents model duplication; it does not make four branches cost the same as one.

### Reference images

Reference images add native reference tokens to the FLUX.2 image sequence. Every active ControlNet branch processes the expanded target-plus-reference sequence. Reducing reference-image dimensions can therefore have a large effect.

### Inpainting preparation

Without a warm inpaint-context cache, the masked source image must be VAE-encoded during initial runtime preparation. The experimental cache removes that repeated static encode on a matching hit, but it does not reduce denoising cost.

## What the implementation optimizes

- Deferred checkpoint materialization allows the text encoder and other models to load before the side checkpoint is read.
- ComfyUI `CoreModelPatcher` ownership participates in normal loading and offloading.
- Orchestrator children share one model owner and weight set.
- Exact zero-strength branches do not stage or execute the side model.
- Hint, reference, and inpaint-context caches retain only bounded CPU tensors.
- Reference and hint prewarming can reduce VAE model churn.
- Inpaint-context prewarming removes the former first-step masked-source encode cliff.

## Validation snapshot

A warmed 1024 × 1536 validation run with:

- three reduced-size reference images;
- OpenPose/DWPose as the host;
- conservative luminance and depth auxiliaries;
- three active ControlNet branches;
- warmed hint, reference, and inpaint-context caches;

completed in approximately 119.38 seconds on the development RTX 4090 Laptop GPU with 16 GB VRAM.

That result confirms that the cache architecture removes the former preparation bottleneck. It is not a universal benchmark. The same three-branch setup can still exhibit visible conditioning conflict even when runtime is acceptable.

## Practical scaling strategy

1. Validate one ControlNet at moderate resolution.
2. Add reference images at reduced size.
3. Warm static caches.
4. Increase output resolution.
5. Add a second control.
6. Add a third or fourth control only when visually justified.

Change one dimension at a time so performance regressions can be attributed.

## VRAM versus system RAM

Dynamic model offloading can make a workflow fit in limited VRAM at the cost of transfer time and system-memory pressure. The caches themselves use CPU memory and are bounded, but FLUX.2 model management can use substantially more RAM than the cache totals.

## Diagnostics

Enable diagnostics while validating:

- checkpoint materialization;
- cache hits, misses, and inserts;
- active strengths and ranges;
- reference-token expansion;
- injection blocks;
- inpaint-context source.

Disable diagnostics for routine operation after the workflow is understood.

## Visual conflict is not a performance failure

A dense third branch may be computationally fast after cache warming yet reduce image quality. Distinguish:

- **compute performance** — time, memory, staging, cache hits;
- **conditioning performance** — whether prompt, reference, structural, and inpaint guidance agree.

Release 1.0.0's practical high-end limit is often conditioning balance rather than cache or model-loading performance.


---

[Documentation home](../README.md) · [Project README](../../README.md)
