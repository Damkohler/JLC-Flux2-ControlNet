# ControlNet Orchestrators

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


## JLC Flux2 ControlNet Orchestrator

**Status:** Stable

The Orchestrator builds one flat composed control from one to four independently configured branches and attaches it to one conditioning stream.

### Required inputs

| Input | Type | Default | Notes |
|---|---|---:|---|
| `controlnet` | JLC ControlNet | — | One Loader output shared by all slots. |
| `conditioning` | CONDITIONING | — | Must contain no existing ControlNet. |
| `vae` | VAE | — | Shared by all branches. |
| `control_image_1` | IMAGE | — | Required. |
| `slot_count` | INT | `4` | Authoritative range `1–4`. |
| `strength_1` | FLOAT | `0.5` | Range `0.0–2.0`. |
| `start_percent_1` | FLOAT | `0.0` | Range `0.0–1.0`. |
| `end_percent_1` | FLOAT | `1.0` | Range `0.0–1.0`. |
| `diagnostics` | BOOLEAN | `true` | Console diagnostics. |

### Optional branch inputs

Slots 2–4 each expose:

- `control_image_N`;
- `strength_N`, default `0.5`;
- `start_percent_N`, default `0.0`;
- `end_percent_N`, default `1.0`.

The frontend hides slots above `slot_count`; the backend also ignores stale hidden values.

### Outputs

| Output | Type | Purpose |
|---|---|---|
| `conditioning` | CONDITIONING | Conditioning carrying the composed control. |
| `vae` | VAE | Passthrough for downstream adapter/cache wiring. |
| `control_image_1` … `control_image_4` | IMAGE | Active image passthroughs; inactive outputs are absent. |

The image passthroughs are useful for cache-preparation branches and workflow routing.

## JLC Flux2 ControlNet Orchestrator Advanced

**Status:** Stable

The Advanced variant accepts `positive` and `negative` and attaches the same composed-control object to both.

### Outputs

- `positive`
- `negative`
- `vae`
- `control_image_1` through `control_image_4`

## Clean-conditioning rule

Both variants reject conditioning that already contains a ControlNet. Valid placement is directly after text/reference conditioning.

This prevents accidental mixing of recursive and flat ownership models.

## Slot and branch rules

- Slot 1 cannot be absent.
- Optional `None` slots are omitted.
- `slot_count` is clamped to `1–4`.
- Every active branch gets its own configured shallow copy.
- Children have `previous_controlnet = None`.
- The composite owns the single injection hook.
- All active children share one model-patcher owner.
- A zero-strength branch does not contribute or stage the side model.

## Inpaint-host implication

When the experimental In/Out-Paint Adapter is placed downstream, the first branch with nonzero strength becomes the single host for shared inpaint context. Branch order should therefore be deliberate.


---

[Documentation home](../README.md) · [Project README](../../README.md)
