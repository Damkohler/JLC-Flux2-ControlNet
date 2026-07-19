# Reference Images

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


## Native reference-latent path

The **JLC Flux2 Reference Image Orchestrator** uses ComfyUI's native FLUX.2 `reference_latents` conditioning mechanism. It does not implement ControlNet-style reference residuals or a proprietary reference-fusion algorithm.

Each enabled and connected reference image is:

1. accepted as an upstream-prepared BHWC RGB image;
2. VAE-encoded independently, or retrieved from the shared CPU cache;
3. appended to conditioning in original slot order.

The node performs no weighting, averaging, pooling, latent multiplication, or attention fusion.

## Slot behavior

The node supports up to ten reference slots.

- `slot_count` controls how many slots are active and visible.
- Every visible slot has an `enabled_N` Boolean.
- A disabled slot is an exact omission before validation, hashing, cache lookup, or VAE encode.
- An enabled but unconnected optional slot is skipped.
- Slots above `slot_count` are ignored.
- Later references are not promoted to fill earlier gaps.

If no enabled reference is connected, positive and negative conditioning are returned unchanged and no reference-method metadata is attached.

## Apply routing

`apply_to` supports:

- `positive_and_negative`;
- `positive_only`;
- `negative_only`.

The same ordered reference sequence is attached to every selected conditioning stream.

## Native reference method

`reference_latents_method` can be left at `do_not_set`, or can pass one of the current native method labels:

- `offset`;
- `index`;
- `uxo/uno`;
- `index_timestep_zero`.

The JLC node stores this as native conditioning metadata. The method does not alter the VAE encoding of a reference image, so cache identity is deliberately method-agnostic.

## Image preparation

The node does not resize, crop, pad, or reposition reference images. Complete those operations upstream. Cache identity is based on the exact final image tensor supplied to the node and the connected VAE identity.

Reference images do not need to match the sampling canvas. Reducing their dimensions can significantly reduce reference-token count and runtime cost while retaining useful appearance or palette influence.

## Reference-latent cache

When caching is enabled, the Orchestrator can reuse a detached CPU latent instead of repeating VAE encoding.

Relevant widgets include:

- `cache_enabled`;
- `cache_max_entries`;
- `cache_max_cpu_mb`;
- `clear_cache_before_run`;
- `diagnostics`.

The separate **JLC Flux2 Reference Latents Cache** node can prewarm the same shared cache in a mutually exclusive setup branch. Prep and runtime align when they receive the same final reference tensors and VAE.

## Relationship to ControlNet

Reference conditioning should normally be applied before the JLC Apply or Orchestrator:

```text
Text conditioning
    -> Reference Image Orchestrator
    -> ControlNet Apply or Orchestrator
    -> guider and sampler
```

At runtime, JLC expands the raw 260-channel ControlNet context with exact-zero control tokens for the native reference-token suffix. The target control prefix remains intact while reference tokens participate in the FLUX.2 sequence without receiving a fabricated control hint.

## Practical recommendations

- Start with one reference image.
- Resize references deliberately upstream.
- Add more references one at a time.
- Use a fixed seed while testing.
- Separate reference-image count from ControlNet branch count when diagnosing performance.
- Prewarm reference latents when repeated VAE encode churn is visible.
- Keep dense ControlNet auxiliaries conservative when strong reference identity or style influence is important.


---

[Documentation home](../README.md) · [Project README](../../README.md)
