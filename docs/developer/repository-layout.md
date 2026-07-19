# Repository Layout

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


## Front-page map

```text
JLC-Flux2-ControlNet/
├── assets/
│   ├── icons/
│   └── workflows/
│       ├── Release_0.1.0/
│       └── Release_1.0.0/
├── docs/
│   ├── developer/
│   ├── getting-started/
│   ├── guides/
│   ├── legal/
│   └── nodes/
├── jlc_flux2_controlnet/
├── nodes/
├── web/
├── __init__.py
├── jlc_flux2_controlnet_versions.py
├── LICENSE
├── pyproject.toml
└── README.md
```

## `assets/`

Repository presentation and example workflows.

- `icons/` contains package artwork.
- `workflows/Release_0.1.0/` retains early regression examples.
- `workflows/Release_1.0.0/` contains current stable and experimental demonstrations.

## `docs/`

Progressive documentation.

- `getting-started/` answers installation and first-run questions.
- `guides/` explains user goals and feature interactions.
- `nodes/` records current UI contracts.
- `developer/` explains implementation and validation history.
- `legal/` contains third-party reference and licensing notes.
- the technical paper PDF is historical and may contain superseded concepts.

## `jlc_flux2_controlnet/`

Core runtime modules.

| File | Responsibility |
|---|---|
| `constants.py` | Shared channel, block, hook, and log constants. |
| `loader.py` | Deferred checkpoint inspection and model-patcher construction. |
| `model.py` | Compact FLUX.2 ControlNet side-model implementation. |
| `control.py` | Stable `ControlBase` lifecycle and ordinary 260-channel context. |
| `composition.py` | Flat non-recursive composed-control owner. |
| `hooks.py` | Per-invocation wrapper and residual injection. |
| `hint_latent_cache.py` | Stable control-hint CPU cache. |
| `reference_latent_cache.py` | Stable reference-latent CPU cache. |
| `inpaint_control.py` | Specialized mask-aware ControlNet context. |
| `inpaint_context_cache.py` | Experimental mask-context and masked-latent cache. |

## `nodes/`

User-facing ComfyUI node classes.

| File | Nodes |
|---|---|
| `jlc_flux2_controlnet_loader_node.py` | Loader |
| `jlc_flux2_controlnet_apply_node.py` | Apply and Apply Advanced |
| `jlc_flux2_controlnet_orchestrator_node.py` | Orchestrator and Advanced |
| `jlc_flux2_reference_image_orchestrator_node.py` | Reference Image Orchestrator |
| `jlc_flux2_hint_latent_cache_prep_node.py` | ControlNet Latents Cache |
| `jlc_flux2_reference_latent_cache_prep_node.py` | Reference Latents Cache |
| `jlc_flux2_inpaint_context_cache_prep_node.py` | Experimental Inpaint Context Cache |
| `jlc_flux2_controlnet_inpaint_adapter_node.py` | Experimental adapter variants |
| `jlc_conditional_save_image_node.py` | Lazy conditional output utility |

## `web/`

- `jlc_flux2_controlnet_icons.js` applies project icons.
- `jlc_flux2_dynamic_slots.js` hides and reveals slot widgets and pins.

The backend does not depend on frontend visibility for correctness.

## Root metadata

- `__init__.py` registers classes, display names, static frontend files, and package version.
- `jlc_flux2_controlnet_versions.py` is the central package version source.
- `pyproject.toml` provides Comfy Registry metadata.
- `LICENSE` applies the MIT license to repository source.
- `README.md` is the public landing page.

## Files not intended for release

Development-only backups, tests, bytecode caches, local archives, and temporary outputs should remain excluded through `.gitignore` and release-archive tooling.


---

[Documentation home](../README.md) · [Project README](../../README.md)
