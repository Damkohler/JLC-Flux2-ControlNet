# Experimental In/Out-Painting

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


> [!WARNING]
> **Experimental feature.** The In/Out-Paint Adapter and Inpaint Context Cache are functional Release 1.0.0 baselines, but are not represented as artifact-free or final mask-transition solutions.


## Workflow role

The adapter upgrades an already configured JLC ControlNet path to the FLUX.2-dev Fun ControlNet Union mask-aware 260-channel contract.

```text
Text or reference conditioning
    -> JLC Flux2 ControlNet Apply or Orchestrator
    -> JLC Flux2 ControlNet Inpaint Adapter - Experimental
    -> guider and sampler
```

The sampler continues to use the validated clean or Empty Flux2 Latent. The adapter does not replace the sampler latent with the source image latent.

## Mask convention

The user-facing convention is fixed:

- **white / 1.0** = editable or regenerate;
- **black / 0.0** = preserve or retain.

The mask is thresholded at `>= 0.5` and remains hard and binary.

Editable source-image pixels are replaced with pixel-space `0.5` before VAE encoding. Under ComfyUI's VAE normalization, this corresponds to neutral model-space zero. The packed ControlNet mask lanes use the inverse hard keep-mask.

## Exact canvas requirement

The source image and mask must:

- have identical spatial dimensions;
- exactly match the active sampling canvas;
- be prepared upstream from the same width and height source used by the scheduler and Empty Flux2 Latent.

Mismatches produce a clear error rather than silent resizing.

For outpainting, create the expanded canvas and its mask upstream. The source image should already be placed into the larger canvas, with white over the region to generate and black over content to retain.

## First-active host branch

For a single Apply result, that configured control receives the inpaint context.

For an Orchestrator result, only the first active child receives the shared inpaint context. “Active” means the first child whose strength is not exactly zero. Other children remain ordinary full-frame controls.

This design preserves one shared non-recursive injection hook and follows the Union model's shared inpaint-context restriction.

## Recommended host and auxiliaries

OpenPose/DWPose is the recommended host because it supplies sparse structural guidance without densely reproducing the source image.

Additional branches are not mask-gated. Dense modalities such as:

- depth;
- luminance;
- color;
- tile/detail;
- full-frame edges;

can preserve or imprint source structure inside white editable regions.

Use conservative strengths and short activation ranges for dense auxiliaries. Add them one at a time at a fixed seed.

## Reference images

Native reference-image conditioning remains supported. Apply references before the ControlNet path:

```text
Reference Image Orchestrator
    -> ControlNet Orchestrator
    -> Inpaint Adapter
```

Reference tokens are preserved in the native FLUX.2 sequence. The JLC ControlNet path appends exact-zero raw control padding for reference tokens where required.

## Known experimental limitations

- Hard mask boundaries can produce seed-variable contour or edge artifacts.
- Additional dense controls may overpower prompt, reference, or inpaint guidance.
- Mask expansion and feathering controls are not included; an experimental revision produced visible mask-shaped gray artifacts and was removed.
- Results remain sensitive to mask geometry, model behavior, seed, control modality, strength, and timing.
- The adapter is not described as a general seamless blending engine.

## Troubleshooting

### Editable areas remain too similar to the source

Disable dense auxiliary controls. Validate with only OpenPose/DWPose as the host, then reintroduce auxiliaries conservatively.

### A gray or mask-shaped region appears

Confirm that no external feathering or gray-valued mask preprocessing is being introduced. The adapter thresholds at 0.5, but upstream image preparation can still affect the source canvas.

### A geometry error appears

Drive scheduler, Empty Flux2 Latent, source-image resize, mask creation, and Inpaint Context Cache from the same width and height values.

### Edge artifacts vary by seed

That remains a known experimental limitation of the current hard binary mask path. Compare several seeds and avoid claiming an artifact-free boundary.


---

[Documentation home](../README.md) · [Project README](../../README.md)
