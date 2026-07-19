# Validation and Design History

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


This page summarizes the engineering path that led to Release 1.0.0. It is not a substitute for reproducible testing on a current ComfyUI build.

## 1. Native single-ControlNet baseline

The first validated milestone established:

- compact Union checkpoint loading;
- ComfyUI `ControlBase` ownership;
- `CoreModelPatcher` model management;
- local per-invocation FLUX.2 wrapping;
- residual injection at double blocks 0, 2, 4, and 6;
- exact zero-strength bypass;
- no global FLUX.2 monkey patch.

This path remains the stable Apply regression baseline.

## 2. Flat non-recursive composition

The next milestone replaced recursive multi-branch ownership with an explicit composed owner.

Validation included:

- two independently configured branches;
- shared side-model weights;
- no child `previous_controlnet` recursion;
- one shared injection hook;
- independent timing ranges;
- three and four active branches;
- a pixel-exact composition check in which two identical `0.5` branches matched one `1.0` branch.

The Orchestrator later gained authoritative dynamic `slot_count` behavior while retaining a fixed maximum of four.

## 3. Native reference-image compatibility

Reference support was aligned with ComfyUI's native `reference_latents` append contract.

Validation included:

- target-plus-reference token expansion;
- exact-zero raw control padding for reference tokens;
- preservation of target ControlNet influence;
- native reference-method metadata;
- positive/negative routing;
- up to ten dynamic reference slots.

The Reference Image Orchestrator deliberately avoids non-native weighting or fusion.

## 4. Static latent caches

The project then introduced bounded process-local CPU caches for:

- ControlNet hint latents;
- reference-image VAE latents.

Cold runs insert prepared tensors; warm runs reuse them. Separate prep nodes allow preparation to happen in a mutually exclusive setup branch before inference.

## 5. Experimental mask-aware adapter

The In/Out-Paint Adapter established a downstream mask-aware upgrade without changing the sampler latent.

Important corrections and validations included:

- retaining the native standalone Apply injection hook;
- retaining the composed owner's shared hook;
- one shared inpaint context on the first active branch;
- hard white-edit / black-preserve mask contract;
- neutral pixel-space source fill before VAE encoding;
- strict source image, mask, and canvas validation;
- native reference-image compatibility;
- removal of experimental expansion/feathering controls after gray mask-shaped artifacts.

The remaining edge limitations are understood as part of the current hard Union-model mask contract and conditioning balance, not as an unresolved recursive-composition failure.

## 6. Experimental Inpaint Context Cache

The final Release 1.0.0 performance milestone caches:

- the packed four-channel inverse keep-mask context;
- the VAE-encoded masked-source FLUX.2 latent.

A warmed 1024 × 1536 validation run with three reduced reference images and three active ControlNets completed in approximately 119.38 seconds on an RTX 4090 Laptop GPU with 16 GB VRAM. This confirmed that static masked-source preparation no longer creates the former first-step performance cliff.

The same testing also demonstrated that computational success does not guarantee visual harmony: a third dense auxiliary control can remain fast while introducing visible conditioning-conflict artifacts.

## Stable versus experimental freeze

### Stable Release 1.0.0 paths

- Loader
- Apply and Apply Advanced
- Orchestrator and Orchestrator Advanced
- Reference Image Orchestrator
- ControlNet Latents Cache
- Reference Latents Cache
- Conditional Save Image

### Experimental Release 1.0.0 paths

- Inpaint Adapter
- Inpaint Adapter Advanced
- Inpaint Context Cache

## Historical paper

`docs/JLC_Flux2_ControlNet_Technical_Paper_preview.pdf` records early concepts and should be read as a historical white-paper preview. Current code and documentation supersede incomplete or inaccurate early assumptions.

## Compatibility posture

FLUX.2 and ComfyUI internals can evolve. Release validation establishes a known working implementation, not permanent compatibility with every future ComfyUI commit. Bug reports should include exact ComfyUI, Python, PyTorch, GPU, checkpoint, workflow, and diagnostic information.


---

[Documentation home](../README.md) · [Project README](../../README.md)
