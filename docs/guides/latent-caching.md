# Latent Caching and Prewarming

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


JLC Flux2 ControlNet provides three process-local CPU cache families. Two are stable Release 1.0.0 utilities; one is explicitly experimental.

| Cache | Status | Reused data |
|---|---|---|
| ControlNet hint-latent cache | Stable | Resized, VAE-encoded, FLUX.2-processed control hints |
| Reference-latent cache | Stable | VAE-encoded upstream-prepared reference images |
| Inpaint-context cache | Experimental | Packed hard keep-mask context and masked-source Flux2 latent |

## What caching improves

Caching avoids repeated static VAE preparation when the relevant inputs have not changed. It can substantially reduce first-step delay and model load/offload churn in repeated workflows.

Caching does **not** remove:

- FLUX.2 base-model execution;
- ControlNet side-model execution;
- per-step residual composition and injection;
- reference-token attention cost;
- the effect of high output resolution;
- conditioning conflict between control branches.

## Shared process-local lifetime

Caches live in the current ComfyUI Python process. They are not stored on disk and are not preserved across a server restart.

Entries are detached, contiguous CPU tensors. They do not retain GPU tensors, sampler state, conditioning objects, residual tensors, or model patches.

Each cache is bounded by entry count and CPU-memory capacity. Least-recently-used entries can be evicted when limits are exceeded.

## Runtime caching versus explicit prewarming

### Runtime path

The ordinary nodes can prepare data when needed and insert it into the shared cache. Later identical requests can hit the cache.

### Explicit prewarm path

The three cache-preparation nodes perform static preparation before inference. They are intentionally not output nodes and must be requested by a downstream branch or sink.

A typical pattern is:

```text
Shared Boolean
    TRUE  -> all cache-preparation nodes -> lazy setup sink
    FALSE -> normal inference            -> final image
```

**JLC Conditional Save Image** is designed to select only one expensive image branch lazily and optionally save only the inference state.

## ControlNet hint-latent cache identity

The hint cache covers, among other inputs:

- the final control-hint tensor;
- output pixel and latent geometry;
- connected VAE identity;
- control preprocessing callable;
- resize and crop contract;
- FLUX.2 latent format.

The prep node takes user-facing output `width` and `height`, then follows the same common-upscale, center-crop, VAE encode, and FLUX.2 `process_in` path used at runtime.

## Reference-latent cache identity

The reference cache covers:

- exact final upstream-prepared reference image;
- connected VAE identity;
- external preparation contract;
- latent contract.

It does not include the native reference method because that method is downstream conditioning metadata and does not change VAE encoding.

## Experimental inpaint-context cache identity

The inpaint cache covers:

- exact source-image content;
- thresholded mask content;
- target latent and pixel geometry;
- connected VAE identity;
- FLUX.2 latent format;
- hard-mask preparation contract;
- cache-contract revision.

The cache node derives geometry from a connected Empty Flux2 Latent. See the dedicated [experimental cache guide](inpaint-context-cache-experimental.md).

## Cold/warm procedure

1. Build the setup and inference branches.
2. Enable diagnostics while validating the workflow.
3. Select setup.
4. Optionally clear the relevant caches.
5. Queue one setup run.
6. Confirm cache reports indicate hits or successful inserts.
7. Select inference.
8. Queue normal generation in the same ComfyUI process.
9. Leave cache clearing disabled for subsequent warm runs.

## Cache misses are safe

A cache miss does not invalidate the normal workflow. Runtime preparation proceeds inline and can populate the cache for a later run. A miss commonly occurs after changing:

- an image or mask;
- output width or height;
- VAE;
- reference preparation;
- cache contract or implementation version;
- the ComfyUI server process.


---

[Documentation home](../README.md) · [Project README](../../README.md)
