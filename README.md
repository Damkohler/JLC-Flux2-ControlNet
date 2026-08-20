<p align="center">
  <img src="assets/icons/jlc-comfyui-nodes_Logo-0512.png" alt="JLC ComfyUI Nodes logo" width="180">
</p>

<h1 align="center">JLC Flux2 ControlNet for ComfyUI</h1>

<p align="center">
  ComfyUI-native FLUX.2 ControlNet with flat multi-ControlNet composition,<br>
  multi-reference conditioning, reusable latent caching, and experimental accelerated in/out-paint support.
</p>

<p align="center">
  <img alt="Release" src="https://img.shields.io/badge/release-v1.1.0-blue.svg">
  <img alt="ComfyUI custom nodes" src="https://img.shields.io/badge/ComfyUI-custom%20nodes-2f80ed.svg">
  <img alt="FLUX2 ControlNet" src="https://img.shields.io/badge/FLUX2-ControlNet-6f42c1.svg">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776ab.svg">
  <img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg">
</p>

---

## Overview

**JLC Flux2 ControlNet** integrates the compact FLUX.2-dev Fun ControlNet Union side model into ComfyUI while preserving ComfyUI's native FLUX.2 transformer, sampler, hooks, model loading, offloading, and cleanup paths. **Release 1.1.0 adds transparent support for both dense BF16 checkpoints and compatible mixed FP8/BF16 checkpoints through the same Loader and downstream node interfaces.**

The project provides a conventional single-ControlNet Apply path and a preferred **flat, non-recursive Orchestrator** for combining up to four independently configured ControlNet branches through one shared loaded side model. It also adds multi-reference conditioning, reusable CPU latent caches, a unified cache-preparation workflow, and an experimental mask-aware in/out-paint path with optional precomputed inpaint context.

> **Extend FLUX.2 without replacing it.**

## Highlights

- **ComfyUI-native integration** with no global replacement of the FLUX.2 model or sampler
- **Unified BF16 and FP8 ControlNet loading** — the same Loader automatically supports the original dense FLUX.2 Fun ControlNet checkpoints and the JLC mixed FP8/BF16 Union-2602 checkpoint, with no precision selector or separate FP8 node family
- **Single-ControlNet Apply** and positive/negative Advanced Apply interfaces
- **One-to-four-branch ControlNet Orchestrator** with independent images, strengths, and timestep ranges
- **Flat non-recursive composition** with shared side-model weights
- **Up to ten reference images** with per-slot enable controls and native reference-method metadata
- **Bounded process-local CPU caches** for ControlNet hints, reference images, and experimental inpaint context
- **Unified Conditioning Cache Prep** for reference images, ControlNet hints, and optional inpaint context
- **Original specialist cache-prep nodes retained** for existing workflows and backward compatibility
- **Dynamic slot interfaces** that expose only the configured ControlNet, reference-image, and cache-preparation sockets
- **Conditional Save Image** utility designed for branch-gated cache workflows
- **Experimental In/Out-Paint Adapter** for mask-aware FLUX.2-dev Fun ControlNet Union workflows
- **Experimental Inpaint Context Cache** that precomputes the hard keep-mask context and masked-source Flux2 latent before sampling

## Project Principles

- **Native ComfyUI first** — use ComfyUI's lifecycle instead of replacing it.
- **Local execution hooks** — no global FLUX.2 monkey-patching.
- **Explicit ownership** — configured branches share model weights without duplicating the side model.
- **Flat composition** — Orchestrator branches are evaluated independently rather than recursively chained.
- **Narrow claims** — stable and experimental capabilities are identified separately.

---

## Included Workflows

The repository includes a Release 1.1.0 mixed FP8/BF16 multi-ControlNet example together with the Release 1.0.1 reference-image, multi-ControlNet, experimental In/Out-Paint Adapter, and cache-preparation workflows. The PNG files contain embedded ComfyUI workflow data and can be dragged directly into ComfyUI. JSON files are also provided for standard workflow loading.

### Release 1.1.0 — mixed FP8/BF16 multi-ControlNet example

This workflow demonstrates the unified Loader using the JLC mixed FP8/BF16 `FLUX.2-dev-Fun-Controlnet-Union-2602` checkpoint. The FP8 checkpoint uses the same Apply, Orchestrator, reference-conditioning, and cache infrastructure as the dense checkpoint.

[![JLC Flux2 FP8 multi-ControlNet workflow](assets/workflows/Release_1.1.0/jlc_flux2_fp8_multicontrol_workflow.png)](assets/workflows/Release_1.1.0/jlc_flux2_fp8_multicontrol_workflow.png)

[Download the PNG workflow](assets/workflows/Release_1.1.0/jlc_flux2_fp8_multicontrol_workflow.png) ·
[Download the JSON workflow](assets/workflows/Release_1.1.0/jlc_flux2_fp8_multicontrol_workflow.json)

### Basic reference-image and multi-ControlNet workflow

[![JLC Flux2 ControlNet basic workflow](assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_RefImages_BASIC.png)](assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_RefImages_BASIC.png)

[Download the PNG workflow](assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_RefImages_BASIC.png) ·
[Download the JSON workflow](assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_RefImages_BASIC.json)

### Focused experimental inpainting workflow

[![JLC Flux2 ControlNet experimental inpainting workflow](assets/workflows/Release_1.0.1/jlc_Flux2_Inpainting_workflow.png)](assets/workflows/Release_1.0.1/jlc_Flux2_Inpainting_workflow.png)

[Download the PNG workflow](assets/workflows/Release_1.0.1/jlc_Flux2_Inpainting_workflow.png) ·
[Download the JSON workflow](assets/workflows/Release_1.0.1/jlc_Flux2_Inpainting_workflow.json)

### Full reference, multi-ControlNet, inpainting, and cache workflow with original cache preparation nodes

[![JLC Flux2 ControlNet reference-image, multi-ControlNet, inpainting, and cache workflow](assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_RefImages_Inpaint_workflow.png)](assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_RefImages_Inpaint_workflow.png)

[Download the PNG workflow](assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_RefImages_Inpaint_workflow.png) ·
[Download the JSON workflow](assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_RefImages_Inpaint_workflow.json)

### Comprehensive workflow, making full use of the new ecosystem and the new JLC Flux2 Conditioning Cache Prep node

[![JLC Flux2 ControlNet reference-image, multi-ControlNet, inpainting, and cache workflow](assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_All_In_One.png)](assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_All_In_One.png)

[Download the PNG workflow](assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_All_In_One.png) ·
[Download the JSON workflow](assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_All_In_One.json)

### Cache wiring snapshot

The cache-preparation branch is contained inside a subgraph and becomes visible after double-clicking that subgraph in ComfyUI. The image below provides a quick view of the internal wiring without requiring the subgraph to be opened first.

[![JLC Flux2 ControlNet cache wiring snapshot](assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_Cache_Wiring.jpg)](assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_Cache_Wiring.jpg)

> [!NOTE]
> This cache wiring image is a static snapshot only and is not a standalone workflow file. The editable ComfyUI workflow data is embedded only in the full workflow PNG provided above.

> [!NOTE]
> The package does not include pose, depth, edge, luminance, color, or other image preprocessors. Example workflows may use ComfyUI preprocessors and companion custom nodes that must be installed separately. Users may also choose auxiliary preprocessing and workflow-utility nodes from the optional companion package [JLC ComfyUI Nodes](https://github.com/Damkohler/jlc-comfyui-nodes). That package is optional and is not required for the core JLC Flux2 ControlNet nodes to function.

---

## Installation

Clone the repository into ComfyUI's `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Damkohler/JLC-Flux2-ControlNet.git
```

Or copy the repository manually to:

```text
ComfyUI/custom_nodes/JLC-Flux2-ControlNet/
```

Place a compatible compact FLUX.2-dev Fun ControlNet Union checkpoint in:

```text
ComfyUI/models/controlnet/
```

### Model checkpoints

JLC Flux2 ControlNet supports the original dense BF16 FLUX.2 Fun ControlNet checkpoints and the JLC mixed FP8/BF16 Union-2602 derivative through the same Loader:

- **Original BF16 FLUX.2 Fun ControlNet Union:** [Alibaba-PAI / FLUX.2-dev-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union)
- **JLC mixed FP8/BF16 Union-2602:** [FLUX.2-dev-Fun-Controlnet-Union-2602-JLC-FP8](https://huggingface.co/Damkohler/FLUX.2-dev-Fun-Controlnet-Union-2602-JLC-FP8)

Model weights are not included in this repository. Download the desired checkpoint from its model repository and follow the applicable model license and usage terms.

The Loader automatically identifies compatible FLUX.2 Fun ControlNet checkpoints and handles supported dense and mixed-precision representations without requiring a separate node or precision setting.

Then restart ComfyUI.

---

### Requirements

- A current ComfyUI installation with native FLUX.2 support
- Python 3.10 or newer
- A compatible FLUX.2-dev diffusion model, text encoder, and VAE
- A compatible compact `FLUX.2-dev-Fun-Controlnet-Union` checkpoint
- Sufficient system RAM, VRAM, or model-offloading capacity for FLUX.2-dev and the ControlNet side model

Model weights are not included in this repository. Obtain all model files from their original distribution sources and follow their respective licenses and terms.

---

## Quick Start

1. Load a FLUX.2-dev diffusion model, compatible text encoder, and FLUX.2 VAE.
2. Add **JLC Flux2 ControlNet Loader** and select the compact ControlNet checkpoint.
3. Prepare a control image with the appropriate external preprocessor.
4. Add **JLC Flux2 ControlNet Orchestrator** and connect the loader, conditioning, VAE, and control image to slot 1.
5. Set `slot_count` to the number of ControlNet branches in use. Slot 1 is required; slots 2–4 are optional.
6. Configure each active branch's strength and start/end percentages.
7. Connect the resulting conditioning to the FLUX.2 guider and sampler.

Use **JLC Flux2 ControlNet Apply** for a conventional single-ControlNet path. Use an **Advanced** variant when separate positive and negative conditioning inputs are required.

### Add reference images

Place **JLC Flux2 Reference Image Orchestrator** before the ControlNet Apply or Orchestrator node. It can attach one shared reference-latent sequence to positive conditioning, negative conditioning, or both, with up to ten independently enabled reference-image slots.

### Prewarm reusable conditioning caches

The runtime can reuse unchanged ControlNet hint latents, reference-image latents, and experimental inpaint context from bounded CPU caches during the same ComfyUI server session.

For new combined workflows, use **JLC Flux2 Conditioning Cache Prep**. It provides one setup node for:

- `0–10` reference images;
- `0–4` ControlNet hints;
- an optional inpaint image-and-mask pair.

Choose the required counts, enable or disable inpaint, and press **Apply Input Layout** to expose only the selected sockets.

The unified node is an additive convenience layer. It does not merge or replace the three cache engines. Reference images, ControlNet hints, and inpaint context continue to use their existing independent cache backends, key spaces, limits, and eviction policies.

When ControlNet hints or inpaint are active, connect `empty_flux2_latent` to the same **Empty Flux2 Latent** used by the sampler. Do not connect a sampler-output latent. The node derives ControlNet and inpaint geometry directly from this generation latent.

The `cache_ready_image` output becomes available only after every active requested entry is confirmed as either a cache hit or a successful insertion. It is designed to connect to the setup side of **JLC Conditional Save Image**, while the normal generated image connects to the inference side.

Selected sockets that are physically wired but muted, bypassed, pruned, or resolved to `None` are treated as intentionally inactive and skipped. A selected socket that was never wired remains a configuration error. This allows the node to work with Group Controllers, switchboards, and Set/Get-based optional branches.

If every selected conditioning branch is inactive, the node completes successfully without clearing or modifying any cache. It returns a small CPU image labeled **No Images to Cache** so IMAGE-routed branch control remains valid.

The original specialist nodes remain available for existing workflows:

- **JLC Flux2 ControlNet Latents Cache**
- **JLC Flux2 Reference Latents Cache**
- **JLC Flux2 Inpaint Context Cache - Experimental**

Caches are process-local and are cleared when the ComfyUI process ends or when explicitly reset. The ordinary inline preparation path remains available whenever no matching cache entry exists.

---

## Experimental In/Out-Paint Adapter

JLC Flux2 ControlNet includes an **Experimental In/Out-Paint Adapter** for the FLUX.2-dev Fun ControlNet Union mask-aware path.

The adapter is placed after **JLC Flux2 ControlNet Apply** or **JLC Flux2 ControlNet Orchestrator** and preserves the validated clean/empty Flux2 sampler latent workflow.

Mask convention:

- **White** = editable or regenerate
- **Black** = preserve or retain

Release 1.0.1 uses a hard binary mask thresholded at `0.5`. The source image and mask must already match the active sampling canvas exactly; mismatched dimensions are rejected with a clear error rather than resized silently.

The first active ControlNet branch carries the shared inpaint context. Additional ControlNet branches remain ordinary full-frame controls and are not spatially mask-gated. Dense controls such as luminance, depth, or color may therefore preserve or imprint source structure inside editable regions. OpenPose/DWPose is the recommended host control, with auxiliary controls kept at conservative strengths and short activation ranges.

> [!WARNING]
> The adapter remains **Experimental**. Seed-variable mask-edge or contour artifacts may still occur, and dense or high-strength auxiliary branches can compete with prompt, reference-image, or inpaint guidance. Experimental mask expansion and feathering controls were removed after validation produced visible mask-shaped gray artifacts.

## Experimental Inpaint Context Cache

The **JLC Flux2 Inpaint Context Cache - Experimental** precomputes and stores the static inpaint context before sampling:

- packed four-channel hard keep-mask context
- VAE-encoded masked-source Flux2 latent

Cached tensors are detached, contiguous CPU tensors held in a bounded process-local cache. During inference, the adapter reuses matching prepared context and avoids performing the masked-source VAE encode inside the first sampling step.

For new combined workflows, prepare this cache through **JLC Flux2 Conditioning Cache Prep**. The original specialist Inpaint Context Cache node remains available for existing workflows. In either case, the required latent must come from the same **Empty Flux2 Latent** used by the sampler, not from the sampler output.

The normal inline preparation path remains available as a fallback when no matching cache entry exists.

Validated Release 1.0.1 configurations include:

- 1024 × 1536 target resolution
- reduced-size reference images
- OpenPose/DWPose as the host control
- optional conservative dense auxiliary guidance
- warmed ControlNet, reference-image, and inpaint-context caches
- switchboard-controlled muted and inactive conditioning paths

A dense auxiliary ControlNet may remain computationally fast while still introducing visible conditioning-conflict artifacts. Both the adapter and the Inpaint Context Cache remain explicitly **Experimental** in Release 1.0.1.

---

## Node Overview

| Node | Status | Purpose |
|---|---|---|
| **JLC Flux2 ControlNet Loader** | Stable | Loads compatible dense BF16 or mixed FP8/BF16 FLUX.2 Fun ControlNet checkpoints as the same reusable JLC ControlNet object. |
| **JLC Flux2 ControlNet Apply** | Stable | Attaches one configured ControlNet branch to one conditioning stream. |
| **JLC Flux2 ControlNet Apply Advanced** | Stable | Attaches one shared configuration to positive and negative conditioning. |
| **JLC Flux2 ControlNet Orchestrator** | Stable | Builds a flat one-to-four-branch composition for one conditioning stream. |
| **JLC Flux2 ControlNet Orchestrator Advanced** | Stable | Shares one flat composition across positive and negative conditioning. |
| **JLC Flux2 Reference Image Orchestrator** | Stable | Encodes and attaches up to ten reference images with optional CPU caching. |
| **JLC Flux2 Conditioning Cache Prep** | **Experimental utility** | Unified additive preparation surface for up to ten reference images, four ControlNet hints, and optional inpaint context using the existing independent cache backends. |
| **JLC Flux2 ControlNet Latents Cache** | Stable utility | Prewarms reusable ControlNet hint latents for up to four active images. |
| **JLC Flux2 Reference Latents Cache** | Stable utility | Prewarms reusable reference-image latents for up to ten active images. |
| **JLC Flux2 Inpaint Context Cache - Experimental** | **Experimental utility** | Precomputes the packed hard keep-mask context and masked-source Flux2 latent for reuse by the experimental adapter. |
| **JLC Conditional Save Image** | Stable utility | Selects a lazy true/false image branch and conditionally saves its result. |
| **JLC Flux2 ControlNet Inpaint Adapter - Experimental** | **Experimental** | Adds one shared mask-aware in/out-paint context to the first active branch of an existing JLC control path. |
| **JLC Flux2 ControlNet Inpaint Adapter Advanced - Experimental** | **Experimental** | Applies the same experimental shared mask-aware context to positive and negative streams. |

---

## Documentation

The Release 1.1.0 documentation is organized from practical usage toward implementation detail. The source code and current ComfyUI node interfaces remain authoritative for exact behavior.

- [Documentation home](docs/README.md)

### Getting Started

- [Installation](docs/getting-started/installation.md)
- [Quick Start](docs/getting-started/quick-start.md)
- [Included Workflows](docs/getting-started/workflows.md)

### Feature Guides

- [Multi-ControlNet Composition](docs/guides/multi-controlnet-composition.md)
- [Reference Images](docs/guides/reference-images.md)
- [Latent Caching and Prewarming](docs/guides/latent-caching.md)
- [Experimental Inpaint Context Cache](docs/guides/inpaint-context-cache-experimental.md)
- [Experimental In/Out-Painting](docs/guides/in-out-painting-experimental.md)
- [Performance and Memory](docs/guides/performance-and-memory.md)

### Node Reference

- [ControlNet Loader and Apply Nodes](docs/nodes/loader-and-apply.md)
- [ControlNet Orchestrators](docs/nodes/controlnet-orchestrators.md)
- [Reference Image Orchestrator](docs/nodes/reference-image-orchestrator.md)
- [Conditioning and Specialist Cache-Preparation Nodes](docs/nodes/cache-preparation.md)
- [Experimental Inpaint Context Cache](docs/nodes/inpaint-context-cache-experimental.md)
- [Conditional Save Image](docs/nodes/conditional-save-image.md)
- [Experimental In/Out-Paint Adapter](docs/nodes/in-out-paint-adapter-experimental.md)

### Architecture and Development

- [Architecture](docs/developer/architecture.md)
- [Repository Layout](docs/developer/repository-layout.md)
- [Validation and Design History](docs/developer/validation-and-design-history.md)

### Historical Material

- [Early technical concept paper](docs/JLC_Flux2_ControlNet_Technical_Paper_preview.pdf) — retained as a historical white-paper preview; some concepts are incomplete or superseded by the current implementation and documentation.

---

## Current Scope and Limitations

- The validated target is the compact FLUX.2-dev Fun ControlNet Union architecture.
- The stable Orchestrator supports a fixed maximum of four ControlNet branches.
- The Reference Image Orchestrator and cache-preparation utilities support up to ten reference images.
- The unified Conditioning Cache Prep supports up to four ControlNet hints and one optional inpaint pair.
- Control-image preprocessing is external to this repository.
- The caches retain bounded CPU tensors only and last for the current ComfyUI server process.
- The experimental Inpaint Context Cache requires an exact match for the source image, mask, VAE identity, and active sampling geometry; unmatched runs fall back to inline preparation.
- High-resolution workflows combining several ControlNets and reference images can be slow and memory intensive.
- Single-device execution is the validated target; the project does not claim a separate multi-GPU branch-cloning implementation.
- The In/Out-Paint Adapter and Inpaint Context Cache remain experimental because the current hard binary Union-model mask contract can produce seed-variable edge or contour artifacts, especially when dense auxiliary controls compete with inpaint guidance.
- This is an independent project and is not an official Black Forest Labs or ComfyUI release.

---

## Repository Layout

```text
JLC-Flux2-ControlNet/
├── assets/
│   ├── icons/
│   └── workflows/
│       ├── Release_0.1.0/
│       ├── Release_1.0.1/
│       └── Release_1.1.0/
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

---

## Feedback and Issue Reports

Reproducible bug reports and testing feedback are welcome. Please include, when possible:

- ComfyUI version or commit
- Python and PyTorch versions
- Operating system and GPU
- ControlNet checkpoint filename
- Relevant workflow JSON
- Console diagnostics or traceback
- Whether reference images, any cache-preparation nodes, or the experimental adapter were active

---

## Acknowledgments

This project builds on and was informed by ComfyUI's native FLUX.2 implementation and model lifecycle, Black Forest Labs' FLUX.2 model family, Alibaba VideoX-Fun's FLUX.2 ControlNet work, the public Flux2Fun experiment, and the authors and distributors of the compact FLUX.2 ControlNet checkpoint.

JLC Flux2 ControlNet implements its own ComfyUI-native integration and does not import those reference projects at runtime or bundle their model weights.

Developed by **J. L. Córdova**, with research and implementation assistance from **OpenAI ChatGPT**.

See [Third-Party Reference Notes](docs/legal/third-party-notes.md) for upstream references, attribution, and licensing notes.

---

## License

The source code in this repository is released under the [MIT License](LICENSE).

Model weights are not included and remain subject to the licenses and terms of their original publishers.
