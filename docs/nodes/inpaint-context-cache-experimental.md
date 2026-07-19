# Inpaint Context Cache Node — Experimental

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


> [!WARNING]
> **Experimental feature.** The In/Out-Paint Adapter and Inpaint Context Cache are functional Release 1.0.0 baselines, but are not represented as artifact-free or final mask-transition solutions.


## JLC Flux2 Inpaint Context Cache - Experimental

This utility prewarms one hard-mask inpaint context in the shared bounded CPU cache.

## Inputs

| Input | Type | Default | Notes |
|---|---|---:|---|
| `vae` | VAE | — | Must match the adapter VAE. |
| `image` | IMAGE | — | Source/edit canvas. |
| `mask` | MASK | — | White edits; black preserves. |
| `latent` | LATENT | — | Same Empty Flux2 Latent used by the sampler. |
| `clear_before_prepare` | BOOLEAN | `false` | Clears the shared inpaint cache. |
| `diagnostics` | BOOLEAN | `true` | Emits cache and geometry information. |

## Outputs

| Output | Type | Purpose |
|---|---|---|
| `image` | IMAGE | Passthrough for branch routing. |
| `mask` | MASK | Passthrough for branch routing. |
| `cache_set` | BOOLEAN | True on a hit or successful insert. |
| `cache_report` | STRING | Key, hit/miss, entry count, and byte total. |

## Validation

The connected LATENT must contain rank-4 `samples` with 128 FLUX.2 channels.

Image and mask must exactly match the pixel canvas implied by the latent. Automatic resizing is rejected.

## Prepared tensors

- mask context: four patchified inverse hard keep-mask channels;
- masked-source latent: 128 FLUX.2 channels.

Stored tensors are detached contiguous CPU float32 tensors.

## Branch behavior

The node is not an output node. It must be requested by a downstream setup branch. `IS_CHANGED` returns NaN so the side effect is refreshed every time that branch executes.

A runtime cache miss remains safe because the experimental adapter can prepare inline.


---

[Documentation home](../README.md) · [Project README](../../README.md)
