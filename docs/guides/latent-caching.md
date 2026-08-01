# Latent Caching and Prewarming

JLC Flux2 ControlNet Release 1.0.1 provides three process-local CPU cache families. Two are stable utilities; the inpaint-context cache remains explicitly experimental.

| Cache | Status | Reused data |
|---|---|---|
| ControlNet hint-latent cache | Stable | Resized, VAE-encoded, FLUX.2-processed control hints |
| Reference-latent cache | Stable | VAE-encoded upstream-prepared reference images |
| Inpaint-context cache | Experimental | Packed hard keep-mask context and masked-source Flux2 latent |

The **JLC Flux2 Conditioning Cache Prep** node provides one workflow-facing preparation surface for all three cache families. It is additive: the original specialist cache-preparation nodes remain available, and the three underlying backends continue to use independent key spaces, limits, values, and eviction policies.

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

### Unified explicit prewarm path

For new combined workflows, **JLC Flux2 Conditioning Cache Prep** can prepare:

- `0–10` reference images;
- `0–4` ControlNet hints;
- one optional inpaint IMAGE/MASK pair.

Set `reference_count`, `control_count`, and `use_inpaint`, then press **Apply Input Layout**. The button exposes only the sockets selected by the current layout.

When an active ControlNet hint or active inpaint pair is present, connect `empty_flux2_latent` to the same **Empty Flux2 Latent** used by the sampler. Do not connect the sampler output. The node derives ControlNet and inpaint geometry from that latent, which keeps cache preparation aligned with the active generation canvas.

The node has no VAE passthrough output. This helps prevent the inference graph from accidentally traversing the cache-preparation branch.

### Muted and optional branches

In normal frontend workflows, a hidden revisioned wire contract distinguishes a prewired inactive branch from a missing connection:

- selected and physically wired input with a runtime tensor → active and prepared;
- selected and physically wired input that is muted, pruned, bypassed, or resolves to `None` → intentionally inactive and skipped;
- selected input that was never physically wired → configuration error.

This supports Group Controllers, switchboards, Set/Get pairs, and other optional branch arrangements.

Inpaint is treated as one paired selection:

- image and mask both active → prepare inpaint context;
- image and mask both physically wired but inactive → skip inpaint;
- only one active, or either selected socket unwired → error.

When the frontend wire contract is unavailable, such as in a manually authored headless/API prompt, the node falls back to strict runtime input-presence validation.

### Branch completion and routing

`cache_ready_image` is emitted only after every active requested entry is confirmed as either a cache hit or a successful insertion.

The returned image follows this precedence:

1. active inpaint image;
2. otherwise the last active ControlNet image;
3. otherwise the last active reference image.

`cache_set` remains available as a Boolean diagnostic, and `cache_report` provides the combined preparation result.

If every selected conditioning branch is inactive, preparation completes as `no_cache_required`. No backend is cleared or modified, and the node returns a small CPU image labeled **No Images to Cache** so an IMAGE-routed lazy branch remains valid.

A typical workflow is:

```text
Shared Boolean
    TRUE  -> JLC Flux2 Conditioning Cache Prep -> image_on_true
    FALSE -> normal inference image             -> image_on_false

                    JLC Conditional Save Image
```

A common configuration uses `save_when="FALSE"` so the setup branch executes without saving its routing image, while the inference branch writes the final PNG.

### Original specialist preparation nodes

The following nodes remain available and unchanged for existing or deliberately separated workflows:

- **JLC Flux2 ControlNet Latents Cache**
- **JLC Flux2 Reference Latents Cache**
- **JLC Flux2 Inpaint Context Cache - Experimental**

## ControlNet hint-latent cache identity

The hint cache covers, among other inputs:

- the final control-hint tensor;
- output pixel and latent geometry;
- connected VAE identity;
- control preprocessing callable;
- resize and crop contract;
- FLUX.2 latent format.

The unified node derives target geometry from `empty_flux2_latent`, then follows the same common-upscale, center-crop, VAE encode, and FLUX.2 `process_in` path used at runtime.

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

The unified and specialist inpaint preparation paths both derive geometry from a connected Empty Flux2 Latent. See the dedicated [experimental cache guide](inpaint-context-cache-experimental.md).

## Cold/warm procedure

1. Build the setup and inference branches.
2. Enable diagnostics while validating the workflow.
3. Select setup.
4. Optionally clear the relevant active caches.
5. Queue one setup run.
6. Confirm the report indicates hits or successful inserts.
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
