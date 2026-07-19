# ControlNet Loader and Apply Nodes

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


## JLC Flux2 ControlNet Loader

**Status:** Stable

The Loader resolves a checkpoint from `ComfyUI/models/controlnet/` and returns an unconditioned `JLC_FLUX2_CONTROLNET` base object.

### Input

| Input | Type | Purpose |
|---|---|---|
| `controlnet_name` | dropdown | Selects a checkpoint from ComfyUI's ControlNet model folder. |

### Output

| Output | Type | Purpose |
|---|---|---|
| `controlnet` | `JLC_FLUX2_CONTROLNET` | Reusable base object for Apply or Orchestrator nodes. |

### Runtime behavior

Loader execution creates a small deferred checkpoint handle. The checkpoint is read and the compact side model is constructed when ComfyUI gathers sampling models. Shallow configured copies share the same handle and `CoreModelPatcher`.

The loader validates the compact Union architecture, including the 260-channel input and contiguous control-block indices. It does not globally replace the FLUX.2 model.

## JLC Flux2 ControlNet Apply

**Status:** Stable

Apply configures one ControlNet branch and attaches it to one conditioning stream.

### Inputs

| Input | Type | Default | Notes |
|---|---|---:|---|
| `controlnet` | JLC ControlNet | — | Output of the Loader. |
| `conditioning` | CONDITIONING | — | May already contain a conventional previous ControlNet. |
| `vae` | VAE | — | Used to encode the control image. |
| `control_image` | IMAGE | — | Required; `None` is rejected. |
| `strength` | FLOAT | `0.75` | Range `0.0–2.0`. |
| `start_percent` | FLOAT | `0.0` | Range `0.0–1.0`. |
| `end_percent` | FLOAT | `1.0` | Range `0.0–1.0`. |
| `diagnostics` | BOOLEAN | `true` | Enables console validation messages. |

### Output

| Output | Type |
|---|---|
| `conditioning` | CONDITIONING |

Apply copies the base object, attaches the hint and VAE, and preserves any existing control as `previous_controlnet`. This is the conventional chained path. Use the Orchestrator for flat multi-branch composition.

Exact `strength = 0` bypasses side-model execution, but the required image must still be present.

## JLC Flux2 ControlNet Apply Advanced

**Status:** Stable

Advanced Apply configures one shared ControlNet object for positive and negative conditioning.

### Inputs

The inputs match Apply, except that `positive` and `negative` replace the single `conditioning` input.

### Outputs

| Output | Type |
|---|---|
| `positive` | CONDITIONING |
| `negative` | CONDITIONING |

Positive and negative entries with the same previous control share one configured JLC control object and therefore one branch cache, matching native advanced ownership semantics.

## Validation rules

- `start_percent` must be less than or equal to `end_percent`.
- `control_image` must be a real image tensor.
- Control images are expected in ComfyUI IMAGE layout.
- The connected VAE must be compatible with FLUX.2.
- For one-to-four flat JLC branches, use an Orchestrator rather than chaining Apply nodes.


---

[Documentation home](../README.md) · [Project README](../../README.md)
