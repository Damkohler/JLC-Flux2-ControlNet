# Conditioning and Specialist Cache-Preparation Nodes

JLC Flux2 ControlNet Release 1.0.1 provides one unified cache-preparation node and three original specialist preparation nodes.

The unified **JLC Flux2 Conditioning Cache Prep** node is additive. It provides one workflow-facing interface for the existing reference-latent, ControlNet hint-latent, and experimental inpaint-context cache backends. It does not merge or replace those engines, and the original specialist nodes remain available for backward compatibility.

All preparation nodes are branch-driven. They must be requested by a downstream lazy switch, Group Controller, Conditional Save node, or equivalent output path.

## JLC Flux2 Conditioning Cache Prep

**Status:** Experimental utility

The unified node can prepare any selected combination of:

- `0–10` reference images;
- `0–4` ControlNet hints;
- one optional inpaint image-and-mask pair.

It delegates preparation to the same independent cache backends used by the specialist nodes:

| Domain | Existing backend |
|---|---|
| Reference images | Reference latent cache |
| ControlNet hints | ControlNet hint-latent cache |
| Inpaint | Experimental inpaint-context cache |

Their key spaces, limits, stored values, and eviction policies remain independent.

### Layout controls

| Control | Range or default | Purpose |
|---|---:|---|
| `reference_count` | `0–10` | Number of reference-image sockets exposed by the layout. |
| `control_count` | `0–4` | Number of ControlNet-hint sockets exposed by the layout. |
| `use_inpaint` | `false` | Exposes one inpaint IMAGE/MASK pair. |
| `Apply Input Layout` | — | Applies the selectors and exposes only the requested sockets. |
| `clear_before_prepare` | `false` | Clears only the active cache domains before preparation. |
| `diagnostics` | `true` | Enables detailed cache reporting. |

The frontend stores a hidden revisioned connection contract when the layout is applied. Users should not edit that value manually.

### Main inputs

| Input | Type | Requirement |
|---|---|---|
| `vae` | VAE | Always required. |
| `empty_flux2_latent` | LATENT | Required when an active ControlNet hint or active inpaint pair is present. |
| `reference_image_1` … `reference_image_10` | IMAGE | Exposed according to `reference_count`. |
| `control_image_1` … `control_image_4` | IMAGE | Exposed according to `control_count`. |
| `inpaint_image` | IMAGE | Exposed when `use_inpaint` is enabled. |
| `inpaint_mask` | MASK | Exposed when `use_inpaint` is enabled. |

Connect `empty_flux2_latent` to the same **Empty Flux2 Latent** used by the sampler. Do not connect a sampler-output latent.

ControlNet hint and inpaint geometry are derived from this exact latent. This keeps cache preparation aligned with the active generation canvas and avoids manually entered target-dimension mismatches.

The node intentionally has no VAE passthrough output, preventing the inference workflow from accidentally traversing the cache-preparation branch.

### Active and inactive sockets

In a normal frontend workflow, the hidden wire contract interprets selected dynamic sockets as follows:

- **Physically wired and producing a tensor:** active and prepared.
- **Physically wired but muted, bypassed, pruned, or resolving to `None`:** intentionally inactive and skipped.
- **Selected but never physically wired:** configuration error.

This distinction supports Group Controllers, switchboards, optional Set/Get paths, and other workflows where a branch remains prewired but is disabled for a particular run.

Inpaint is handled as one paired input:

- image and mask both active → prepare inpaint context;
- image and mask both physically wired but inactive → skip inpaint;
- only one active, or either selected socket unwired → error.

If the frontend contract is unavailable, the node uses strict runtime input-presence validation. This preserves predictable behavior for headless or manually authored API prompts.

### Outputs

| Output | Type | Purpose |
|---|---|---|
| `cache_ready_image` | IMAGE | Branch-routing image emitted only after all active requested entries are ready. |
| `cache_set` | BOOLEAN | Diagnostic success result. |
| `cache_report` | STRING | Combined hit, miss, insert, skip, geometry, and clearing report. |

The ready image is selected from the final active source using this precedence:

1. active inpaint image;
2. otherwise the last active ControlNet image;
3. otherwise the last active reference image.

This makes the output suitable for the TRUE or setup input of **JLC Conditional Save Image**.

### No-cache-required behavior

If all selected Reference Image, ControlNet, and Inpaint paths are inactive, the node completes successfully as `no_cache_required`.

In that state:

- no cache backend is cleared;
- no cache backend is modified;
- `cache_set` is true;
- `cache_ready_image` is a small CPU image labeled **No Images to Cache**.

The placeholder image keeps IMAGE-routed lazy branch workflows valid even when a switchboard disables all conditioning paths.

### Recommended branch pattern

```text
Shared Boolean
    TRUE  -> JLC Flux2 Conditioning Cache Prep -> image_on_true
    FALSE -> normal inference image             -> image_on_false

                    JLC Conditional Save Image
```

A common configuration uses `save_when="FALSE"` so the setup branch executes without saving its routing image, while the inference branch produces the final PNG.

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

## JLC Flux2 Inpaint Context Cache - Experimental

**Status:** Experimental utility

The specialist inpaint cache node prepares the packed hard keep-mask context and VAE-encoded masked-source Flux2 latent used by the experimental In/Out-Paint Adapter.

Its source image and mask must match one another and the active sampling canvas exactly. Its `LATENT` input must come from the same **Empty Flux2 Latent** used by the sampler, not from a sampler-output latent.

The specialist node remains available for focused or existing workflows. New combined workflows may prepare the same backend through **JLC Flux2 Conditioning Cache Prep**.

## Execution behavior

The preparation nodes force reevaluation when their active setup branch is requested, even if visible inputs are unchanged. The unified node additionally validates its selected layout and frontend wire contract before clearing or populating any cache.

A downstream lazy switch, Group Controller, Any Switch, Conditional Save node, or equivalent sink must request the branch.

`cache_set` is true when all active inputs are either cache hits or successful inserts. Reports include hit, miss, insert, skip, entry-count, and total-byte information.

## Shared cache cautions

- These are process-local side effects.
- Restarting ComfyUI clears them.
- `clear_before_prepare` clears entries only in the active cache domains selected by the node being executed.
- A large prepared tensor can be skipped if it exceeds capacity.
- Do not assume a prep run executed merely because the node is present in the graph; it must be on the requested branch.
- Prewarming removes repeated static preparation but does not remove sampler, ControlNet side-model, or reference-token attention cost.

---

[Documentation home](../README.md) · [Project README](../../README.md)
