# Reference Image Orchestrator

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


## JLC Flux2 Reference Image Orchestrator

**Status:** Stable

This node applies one to ten independently prepared reference images through native FLUX.2 reference-latent conditioning.

## Required inputs

| Input | Type | Default | Purpose |
|---|---|---:|---|
| `positive` | CONDITIONING | — | Positive conditioning stream. |
| `negative` | CONDITIONING | — | Negative conditioning stream. |
| `vae` | VAE | — | Encodes reference images. |
| `apply_to` | dropdown | `positive_and_negative` | Routes the reference sequence. |
| `reference_latents_method` | dropdown | `do_not_set` | Optional native reference-method metadata. |
| `slot_count` | INT | `2` | Active range `1–10`. |
| `enabled_1` … `enabled_10` | BOOLEAN | `true` | Exact per-slot omission controls. |
| `cache_enabled` | BOOLEAN | `true` | Enables the shared CPU reference cache. |
| `cache_max_entries` | INT | `32` | Entry capacity, `0–256`. |
| `cache_max_cpu_mb` | INT | `256` | CPU limit, `0–4096` MB. |
| `clear_cache_before_run` | BOOLEAN | `false` | Clears the shared reference cache. |
| `diagnostics` | BOOLEAN | `true` | Console and JSON diagnostics. |

## Optional inputs

`reference_image_1` through `reference_image_10` are optional IMAGE inputs. An enabled but unconnected slot is skipped.

## Outputs

- `positive`
- `negative`
- `vae`
- `reference_image_1` through `reference_image_10`
- `diagnostics_json`

Image passthrough outputs are present only for active, enabled, connected slots.

## `apply_to` values

- `positive_and_negative`
- `positive_only`
- `negative_only`

## Reference methods

- `do_not_set`
- `offset`
- `index`
- `uxo/uno`
- `index_timestep_zero`

The dropdown passes native metadata; it does not change VAE encoding and is not part of cache identity.

## Exact omission

When `enabled_N` is false, that slot is omitted before validation, hashing, cache retrieval, VAE encoding, or conditioning mutation. If no active images remain, conditioning is returned unchanged.

## Diagnostics JSON

The final string output reports:

- active and skipped slots;
- cache hits, misses, inserts, and uncached encodes;
- image and latent shapes;
- normalized reference method;
- current cache information;
- whether conditioning remained unchanged.

## Upstream preparation

The exact input image is encoded. Resize, crop, pad, or place reference images upstream. Use the same prepared tensor for the prep node and runtime node to obtain a cache hit.


---

[Documentation home](../README.md) · [Project README](../../README.md)
