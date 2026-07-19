# Experimental Inpaint Context Cache

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


> [!WARNING]
> **Experimental feature.** The In/Out-Paint Adapter and Inpaint Context Cache are functional Release 1.0.0 baselines, but are not represented as artifact-free or final mask-transition solutions.


## Purpose

**JLC Flux2 Inpaint Context Cache - Experimental** precomputes the static mask-aware context used by the experimental In/Out-Paint Adapter:

- a packed four-channel hard inverse keep-mask context;
- a VAE-encoded neutral-filled masked-source FLUX.2 latent.

On a matching runtime request, the adapter can reuse those CPU tensors and avoid performing the masked-source VAE encode inside the first sampling step.

## Required placement

Use the cache node in the same mutually exclusive setup branch as:

- JLC Flux2 ControlNet Latents Cache;
- JLC Flux2 Reference Latents Cache.

The setup branch must be executed before normal inference and in the same ComfyUI server process.

## Critical `LATENT` wiring rule

The `latent` input must come from the same **Empty Flux2 Latent** node used by the sampler.

```text
Empty Flux2 Latent
    ├── sampler latent_image
    └── Inpaint Context Cache latent
```

Do not connect the sampler output. The cache needs target geometry before sampling and must match the adapter's active canvas.

## Exact geometry contract

The source `image` and `mask` must already match:

- each other;
- the sampling canvas represented by the connected latent.

Automatic spatial resizing is intentionally rejected. This prevents a visually plausible but incorrectly aligned mask from being silently cached.

The source image must use ComfyUI IMAGE layout. The mask may use supported MASK layouts, but the final spatial dimensions must be exact.

## Cache preparation

The cache node:

1. validates FLUX.2 latent rank and 128-channel geometry;
2. fingerprints the image, thresholded mask, VAE, target geometry, latent format, and hard-mask contract;
3. checks the bounded shared CPU cache;
4. returns immediately on a hit;
5. otherwise prepares mask context and masked-source latent;
6. stores detached contiguous CPU float32 tensors if capacity permits.

Outputs pass through the input image and mask for branch routing and provide `cache_set` plus a human-readable `cache_report`.

## Runtime fallback

Existing workflows remain valid without this node. If the adapter cannot find a matching shared entry, it performs inline preparation and continues.

The cache is an optimization layer, not a separate required correctness path.

## What the cache does not accelerate

It does not reduce:

- base FLUX.2 denoising;
- ControlNet side-model execution;
- reference-token attention;
- residual size;
- additional branch cost;
- conditioning conflict.

A workflow can become computationally fast while still producing poor results from competing dense controls.

## Validated release pattern

A validated warmed configuration includes:

- 1024 × 1536 target resolution;
- three reduced-size reference images;
- OpenPose/DWPose at full range;
- optional conservative luminance guidance;
- warmed hint, reference, and inpaint-context caches.

A third dense auxiliary branch can remain fast but introduce visible conditioning-conflict artifacts.


---

[Documentation home](../README.md) · [Project README](../../README.md)
