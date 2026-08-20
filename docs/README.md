# `docs/README.md`

# JLC Flux2 ControlNet Documentation

This documentation covers **JLC Flux2 ControlNet Release 1.1.0**.

Release 1.1.0 adds transparent support for compatible **dense BF16** and **mixed FP8/BF16** FLUX.2 Fun ControlNet checkpoints through the same **JLC Flux2 ControlNet Loader** and downstream node interfaces. The established Release 1.0.1 Apply, Orchestrator, reference-image, cache-preparation, and experimental in/out-paint workflows remain applicable unless a page explicitly states otherwise.

The documentation is arranged as a progression: new users can begin with installation and an included workflow, while advanced users can continue into node contracts, cache behavior, architecture, and validation history.

## Recommended reading paths

### First generation

1. [Installation](getting-started/installation.md)
2. [Quick Start](getting-started/quick-start.md)
3. [Included Workflows](getting-started/workflows.md)

### Building a larger workflow

1. [Multi-ControlNet Composition](guides/multi-controlnet-composition.md)
2. [Reference Images](guides/reference-images.md)
3. [Latent Caching and Prewarming](guides/latent-caching.md)
4. [Performance and Memory](guides/performance-and-memory.md)

### Experimental inpainting or outpainting

1. [Experimental In/Out-Painting](guides/in-out-painting-experimental.md)
2. [Experimental Inpaint Context Cache](guides/inpaint-context-cache-experimental.md)
3. [In/Out-Paint Adapter node reference](nodes/in-out-paint-adapter-experimental.md)
4. [Inpaint Context Cache node reference](nodes/inpaint-context-cache-experimental.md)

## Getting started

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quick-start.md)
- [Included Workflows](getting-started/workflows.md)

## Feature guides

- [Multi-ControlNet Composition](guides/multi-controlnet-composition.md)
- [Reference Images](guides/reference-images.md)
- [Latent Caching and Prewarming](guides/latent-caching.md)
- [Experimental Inpaint Context Cache](guides/inpaint-context-cache-experimental.md)
- [Experimental In/Out-Painting](guides/in-out-painting-experimental.md)
- [Performance and Memory](guides/performance-and-memory.md)

## Node reference

- [ControlNet Loader and Apply Nodes](nodes/loader-and-apply.md)
- [ControlNet Orchestrators](nodes/controlnet-orchestrators.md)
- [Reference Image Orchestrator](nodes/reference-image-orchestrator.md)
- [Conditioning and Specialist Cache-Preparation Nodes](nodes/cache-preparation.md)
- [Experimental Inpaint Context Cache](nodes/inpaint-context-cache-experimental.md)
- [Conditional Save Image](nodes/conditional-save-image.md)
- [Experimental In/Out-Paint Adapter](nodes/in-out-paint-adapter-experimental.md)

## Architecture and development

- [Architecture](developer/architecture.md)
- [Repository Layout](developer/repository-layout.md)
- [Validation and Design History](developer/validation-and-design-history.md)

## Legal and historical material

- [Third-Party Reference Notes](legal/third-party-notes.md)
- [Early technical concept paper](JLC_Flux2_ControlNet_Technical_Paper_preview.pdf) — historical; current source code and documentation supersede incomplete or changed early concepts.

## Status vocabulary

| Label | Meaning |
|---|---|
| **Stable** | A validated Release 1.1.0 path intended for normal use within the documented scope. |
| **Stable utility** | A supporting node whose current contract is validated but whose usefulness depends on workflow design. |
| **Experimental utility** | A validated supporting node that remains explicitly subject to interface refinement or narrower compatibility expectations. |
| **Experimental** | Functional and validated as a baseline, but still subject to visible limitations or interface revision. |
| **Historical** | Retained for context; it may describe concepts that were changed or superseded. |

## Release 1.1.0 checkpoint compatibility

Release 1.1.0 uses one **JLC Flux2 ControlNet Loader** for supported FLUX.2 Fun ControlNet checkpoint representations.

The Loader automatically distinguishes between:

- compatible original dense FLUX.2 Fun ControlNet checkpoints; and
- the JLC mixed FP8/BF16 Union-2602 derivative.

No user-facing precision selector or separate FP8 Apply/Orchestrator node family is required. Downstream Apply, Orchestrator, reference-conditioning, cache, and experimental in/out-paint nodes operate through the same `JLC_FLUX2_CONTROLNET` runtime abstraction.

See [Installation](getting-started/installation.md) for model download locations and [ControlNet Loader and Apply Nodes](nodes/loader-and-apply.md) for the runtime contract.

## Source-of-truth order

When documentation and implementation differ, use this order:

1. Current Release 1.1.0 source code and package registration
2. Current user-facing node inputs, outputs, and validation errors
3. Included Release 1.1.0 and compatible Release 1.0.1 workflows
4. This documentation
5. The early technical concept paper

The concept paper is retained as historical material rather than as the current implementation specification.

---

[Project README](../README.md)