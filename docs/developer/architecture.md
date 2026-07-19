# Architecture

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


## Design objective

JLC Flux2 ControlNet integrates a compact FLUX.2-dev Fun ControlNet Union side model without replacing ComfyUI's native FLUX.2 transformer, sampler, or global model classes.

The architecture is based on:

- a `ControlBase` lifecycle object;
- a deferred compact side-model owner;
- ComfyUI `CoreModelPatcher` loading and offloading;
- a per-invocation transformer wrapper;
- local `TransformerOptionsHook` registration;
- explicit flat composition.

## Source layers

### Node interface layer — `nodes/`

Defines ComfyUI inputs, outputs, validation, dynamic slots, passthroughs, and user-facing utilities.

### Runtime layer — `jlc_flux2_controlnet/`

Owns:

- checkpoint architecture inspection and deferred loading;
- ControlBase lifecycle;
- compact side-model execution;
- hook installation and residual injection;
- flat composition;
- reference-token alignment;
- hint, reference, and inpaint caches;
- specialized mask-aware context construction.

### Frontend layer — `web/`

Adds project icons and dynamic slot visibility. Backend `slot_count` validation remains authoritative even when stale hidden workflow values exist.

## Lazy side-model loading

The Loader creates a `LazyFlux2ControlHandle` rather than immediately reading the multi-gigabyte checkpoint. Materialization occurs when ComfyUI gathers additional sampling models through `get_models()`.

The handle:

1. loads the checkpoint safely;
2. inspects required Union architecture keys;
3. validates 260 control input channels;
4. derives hidden size, head dimensions, and control-block count;
5. chooses inference and manual-cast dtypes through ComfyUI;
6. constructs the side model on the meta device;
7. assigns state tensors;
8. wraps the model in `CoreModelPatcher`;
9. publishes the completed owner atomically.

All shallow child controls share this handle.

## Control context

The ordinary ControlNet path builds a 260-channel raw context:

```text
128 control-latent channels
+ 4 zero mask channels
+ 128 zero masked-source-latent channels
= 260 channels
```

The experimental inpaint path replaces the zero mask and masked-source portions with:

```text
128 control-latent channels
+ 4 packed inverse hard keep-mask channels
+ 128 masked-source FLUX.2 latent channels
= 260 channels
```

## Reference-token compatibility

Native FLUX.2 appends reference image tokens after target image tokens. The JLC control path preserves the target control prefix and appends exact-zero 260-channel raw control tokens for the reference suffix.

This aligns the compact side model with the complete native image-token sequence without inventing a reference control hint.

## Residual path

The compact model contains up to four control transformer blocks. The current checkpoint contract maps outputs to native FLUX.2 double blocks:

```text
0, 2, 4, 6
```

Strength is applied at injection time. The wrapper preserves existing ComfyUI block replacements and performs no global monkey patch.

## Flat composition

`JLCFlux2ComposedControl` owns detached child controls.

- Child `previous_controlnet` pointers are cleared.
- Child hook groups are hidden.
- The composed owner installs one injection hook.
- Each child contributes a request to a shared transformer-options list.
- Requests are evaluated in a flat loop.
- Residuals are accumulated by target block and injected once.

Active children share model weights but retain separate hints, ranges, strengths, and cache state.

## Stable caches

### Hint cache

Stores processed control latents keyed by hint content, target geometry, VAE, preprocess callable, resize/crop policy, and latent format.

### Reference cache

Stores detached VAE reference latents keyed by exact prepared image and VAE. It is method-agnostic.

## Experimental inpaint cache

Stores a pair:

- packed mask context;
- masked-source latent.

Identity includes source image, thresholded mask, canvas geometry, VAE, latent format, and contract revision.

## Cleanup

Control and composed objects clear per-run state during cleanup while shared process caches remain bounded and process-local. ComfyUI retains authority over model loading, offloading, and device placement.


---

[Documentation home](../README.md) · [Project README](../../README.md)
