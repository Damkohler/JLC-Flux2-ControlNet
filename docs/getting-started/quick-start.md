# Quick Start

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


The safest first run is one stable ControlNet branch without reference images, inpainting, or explicit cache-preparation branches. Add those features one at a time after the baseline completes.

## Stable single-ControlNet path

```text
Text conditioning
    -> JLC Flux2 ControlNet Apply
    -> FLUX.2 guider
    -> sampler
```

1. Load a FLUX.2-dev model, compatible text encoder, and FLUX.2 VAE.
2. Add **JLC Flux2 ControlNet Loader** and choose the compact Union checkpoint.
3. Prepare one control image with an external preprocessor.
4. Add **JLC Flux2 ControlNet Apply**.
5. Connect `controlnet`, `conditioning`, `vae`, and `control_image`.
6. Begin with:
   - `strength`: `0.75`
   - `start_percent`: `0.0`
   - `end_percent`: `1.0`
7. Connect the returned conditioning to the guider and sample.

A real control image is required. `strength = 0` is an exact runtime bypass, but it is not permission to pass an absent `None` image.

## Preferred multi-ControlNet path

```text
Clean text or reference conditioning
    -> JLC Flux2 ControlNet Orchestrator
    -> FLUX.2 guider
    -> sampler
```

1. Replace Apply with **JLC Flux2 ControlNet Orchestrator**.
2. Connect the required `control_image_1`.
3. Set `slot_count` to the number of branches in use.
4. Connect optional images 2–4.
5. Set each branch's strength and start/end percentages independently.
6. Keep dense secondary controls conservative until the primary branch is stable.

The Orchestrator requires clean conditioning with no existing ControlNet. Place it directly after text conditioning or after the Reference Image Orchestrator—not after another Apply node.

## Add reference images

```text
Positive and negative conditioning
    -> JLC Flux2 Reference Image Orchestrator
    -> Apply or Orchestrator
```

The Reference Image Orchestrator:

- accepts up to ten reference images;
- preserves slot order;
- can apply the same reference sequence to positive, negative, or both streams;
- uses native FLUX.2 `reference_latents` conditioning;
- can reuse bounded CPU-cached reference latents.

Complete resizing and cropping upstream. The node encodes the exact image tensor it receives.

## Add experimental in/out-painting

```text
Apply or Orchestrator output
    -> JLC Flux2 ControlNet Inpaint Adapter - Experimental
    -> guider
    -> sampler using the same clean/empty Flux2 latent
```

Mask convention:

- white = editable or regenerate;
- black = preserve or retain.

The image and mask must match each other and the active sampling canvas exactly. The adapter does not replace the sampler latent.

Use OpenPose/DWPose as the recommended first active host control. Additional depth, luminance, color, or other dense controls are full-frame controls and are not spatially gated by the edit mask.

## Prewarm reusable caches

For repeated runs with unchanged inputs, build a mutually exclusive setup branch containing the relevant cache nodes:

```text
Control images  -> JLC Flux2 ControlNet Latents Cache
References      -> JLC Flux2 Reference Latents Cache
Image + mask    -> JLC Flux2 Inpaint Context Cache - Experimental
```

Use one Boolean switch or branch controller so that:

- `TRUE` executes cache preparation;
- `FALSE` executes normal inference.

The experimental Inpaint Context Cache `latent` input must come from the same **Empty Flux2 Latent** node used by the sampler. It must not come from the sampler output.

After a preparation run, switch to inference without restarting ComfyUI. Caches are process-local and disappear when the server process ends.

## Recommended build order

1. One stable ControlNet
2. Additional ControlNet branches
3. Reference images
4. Experimental adapter
5. Explicit cache-preparation branch

This order isolates wiring, geometry, memory, and conditioning conflicts.


---

[Documentation home](../README.md) · [Project README](../../README.md)
