# ControlNet and Reference Latent Cache Nodes

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


These two stable utility nodes prewarm shared bounded CPU caches without executing the FLUX.2 base model, ControlNet side model, or sampler.

They are branch-driven and intentionally are not output nodes.

## JLC Flux2 ControlNet Latents Cache

**Status:** Stable utility

### Inputs

| Input | Type | Default | Notes |
|---|---|---:|---|
| `vae` | VAE | — | Encodes control hints. |
| `control_image_1` | IMAGE | — | Required first hint. |
| `width` | INT | `1024` | Final output pixel width; step 16. |
| `height` | INT | `1024` | Final output pixel height; step 16. |
| `slot_count` | INT | `4` | Active range `1–4`. |
| `clear_before_prepare` | BOOLEAN | `false` | Clears the shared hint cache. |
| `diagnostics` | BOOLEAN | `true` | Reports preparation state. |
| `control_image_2` … `control_image_4` | IMAGE | — | Optional. |

### Outputs

- `control_image_1` passthrough;
- `cache_set`;
- `cache_report`.

### Preparation contract

```text
IMAGE
 -> BCHW control hint
 -> common_upscale to output canvas, center crop
 -> VAE encode
 -> FLUX.2 latent process_in
 -> bounded CPU cache
```

The cache key includes output geometry, VAE identity, preprocessing callable, interpolation and crop contract, image content, and latent format.

## JLC Flux2 Reference Latents Cache

**Status:** Stable utility

### Inputs

| Input | Type | Default |
|---|---|---:|
| `vae` | VAE | — |
| `reference_image_1` | IMAGE | — |
| `slot_count` | INT | `2` |
| `clear_before_prepare` | BOOLEAN | `false` |
| `diagnostics` | BOOLEAN | `true` |
| `reference_image_2` … `reference_image_10` | IMAGE | — |

### Outputs

- `reference_image_1` passthrough;
- `cache_set`;
- `cache_report`.

### Preparation contract

The exact upstream-prepared BHWC RGB tensor is VAE-encoded and stored as a detached CPU latent. No internal resizing or reference-method processing occurs.

## Execution behavior

Both nodes implement `IS_CHANGED = NaN` so that preparation runs whenever the active setup branch requests the node, even if visible inputs are unchanged.

A downstream lazy switch, group controller, Any Switch, or equivalent sink must request the branch.

`cache_set` is true when all active connected inputs are either cache hits or successful inserts. The report includes hit, miss, insert, skip, entry-count, and total-byte information.

## Shared cache cautions

- These are process-local side effects.
- Restarting ComfyUI clears them.
- `clear_before_prepare` clears every entry in that cache family.
- A large prepared tensor can be skipped if it exceeds capacity.
- Do not assume a prep run executed merely because the node is present in the graph; it must be on the requested branch.


---

[Documentation home](../README.md) · [Project README](../../README.md)
