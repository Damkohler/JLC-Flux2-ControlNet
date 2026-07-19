# JLC Flux2 ControlNet Documentation

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


This documentation is arranged as a progression. New users can begin with installation and an included workflow, while advanced users can continue into node contracts, cache behavior, architecture, and validation history.

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
- [ControlNet and Reference Latent Cache Nodes](nodes/cache-preparation.md)
- [Experimental Inpaint Context Cache](nodes/inpaint-context-cache-experimental.md)
- [Conditional Save Image](nodes/conditional-save-image.md)
- [Experimental In/Out-Paint Adapter](nodes/in-out-paint-adapter-experimental.md)

## Architecture and development

- [Architecture](developer/architecture.md)
- [Repository Layout](developer/repository-layout.md)
- [Validation and Design History](developer/validation-and-design-history.md)

## Legal and historical material

- [Third-Party Reference Notes](legal/third-party-notes.md)
- [Early technical concept paper](JLC_Flux2_ControlNet_Technical_Paper_preview.pdf) — historical; current code and documentation supersede incomplete early concepts.

## Status vocabulary

| Label | Meaning |
|---|---|
| **Stable** | A validated Release 1.0.0 path intended for normal use within the documented scope. |
| **Stable utility** | A supporting node whose current contract is validated but whose usefulness depends on workflow design. |
| **Experimental** | Functional and validated as a baseline, but still subject to visible limitations or interface revision. |
| **Historical** | Retained for context; it may describe concepts that were changed or superseded. |

## Source-of-truth order

When documentation and implementation differ, use this order:

1. Current Release 1.0.0 source code and package registration
2. Current user-facing node inputs, outputs, and validation errors
3. Included Release 1.0.0 workflows
4. These draft documentation pages
5. The early technical concept paper

The concept paper is retained as historical material rather than as the current implementation specification.

---

[Project README](../README.md)
