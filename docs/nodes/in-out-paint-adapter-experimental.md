# In/Out-Paint Adapter Nodes — Experimental

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


> [!WARNING]
> **Experimental feature.** The In/Out-Paint Adapter and Inpaint Context Cache are functional Release 1.0.0 baselines, but are not represented as artifact-free or final mask-transition solutions.


## JLC Flux2 ControlNet Inpaint Adapter - Experimental

### Inputs

| Input | Type | Default | Purpose |
|---|---|---:|---|
| `conditioning` | CONDITIONING | — | Must already contain a JLC Apply or Orchestrator control. |
| `vae` | VAE | — | Authoritative VAE for masked-source encoding. |
| `image` | IMAGE | — | Source/edit canvas. |
| `mask` | MASK | — | White edits; black preserves. |
| `diagnostics` | BOOLEAN | `true` | Enables runtime logging. |

### Outputs

- `conditioning`
- `vae`

## JLC Flux2 ControlNet Inpaint Adapter Advanced - Experimental

The Advanced node replaces the single conditioning input with:

- `positive`;
- `negative`.

It returns:

- `positive`;
- `negative`;
- `vae`.

When positive and negative streams share the same input control object, the upgraded mask-aware object is shared by identity.

## Placement rule

The adapter must be downstream of:

- JLC Flux2 ControlNet Apply; or
- JLC Flux2 ControlNet Orchestrator.

It rejects clean text conditioning and unsupported ControlNet object types.

## Upgrade behavior

For a standalone Apply object, the adapter upgrades that configured control to `JLCFlux2InpaintControl` while retaining its hint, strength, range, VAE-related state, lazy model owner, and native residual-injection hook.

For a composed object:

1. determine the first nonzero-strength child;
2. upgrade that child with the shared inpaint context;
3. copy other children as ordinary controls;
4. create a new flat composed owner with one shared injection hook.

No recursive `previous_controlnet` chain is created.

## Mask/image validation

The node first verifies that IMAGE and MASK agree spatially. Full target-canvas validation occurs later when active sampling geometry is known.

The canonical input name is `image`. The earlier experimental `edit_canvas_image` alias is not retained.

## Cache integration

The upgraded host checks the shared inpaint-context cache. A hit reuses prepared CPU tensors. A miss triggers normal inline preparation and can populate the cache.

The adapter's correctness does not depend on an explicit cache-preparation node.

## Experimental limitations

- hard threshold at 0.5;
- seed-variable edge artifacts;
- no expansion or feathering controls;
- only one host branch receives inpaint context;
- additional controls are full-frame and may imprint into editable areas;
- exact image, mask, and canvas geometry is required.


---

[Documentation home](../README.md) · [Project README](../../README.md)
