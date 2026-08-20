# `docs/getting-started/workflows.md`

# Included Workflows

> [!NOTE]
> **Documentation status: Release 1.1.0.**  
> This page follows the current Release 1.1.0 implementation and supplied workflows. The source code and current ComfyUI node interfaces remain authoritative for exact behavior.

Release 1.1.0 adds a mixed FP8/BF16 multi-ControlNet example while retaining the Release 1.0.1 workflow set for the established reference-image, multi-ControlNet, cache-preparation, and experimental in/out-paint paths.

The unified Release 1.1.0 Loader uses the same downstream Apply, Orchestrator, reference-conditioning, cache, and experimental adapter interfaces for supported dense and mixed-precision ControlNet checkpoints. The Release 1.0.1 workflows therefore remain valid and are not duplicated solely to create FP8-specific versions.

PNG workflow files contain embedded ComfyUI workflow data and can be dragged directly onto the ComfyUI canvas. JSON files are provided for standard loading and source control.

## Release 1.1.0 — mixed FP8/BF16 multi-ControlNet example

Files:

- [PNG](../../assets/workflows/Release_1.1.0/jlc_flux2_fp8_multicontrol_workflow.png)
- [JSON](../../assets/workflows/Release_1.1.0/jlc_flux2_fp8_multicontrol_workflow.json)

This workflow demonstrates the Release 1.1.0 unified Loader using the JLC mixed FP8/BF16 `FLUX.2-dev-Fun-Controlnet-Union-2602` checkpoint.

It demonstrates:

- the same **JLC Flux2 ControlNet Loader** used for dense and mixed-precision checkpoints;
- the JLC mixed FP8/BF16 Union-2602 checkpoint;
- multiple preprocessed ControlNet hints;
- flat multi-ControlNet composition through the Orchestrator;
- the same downstream runtime path used by compatible dense BF16 checkpoints;
- normal FLUX.2 sampling and decode.

No FP8-specific Apply or Orchestrator node family is required.

The JLC mixed FP8/BF16 checkpoint is available from:

[FLUX.2-dev-Fun-Controlnet-Union-2602-JLC-FP8](https://huggingface.co/Damkohler/FLUX.2-dev-Fun-Controlnet-Union-2602-JLC-FP8)

The original dense FLUX.2 Fun ControlNet Union distribution is available from:

[Alibaba-PAI / FLUX.2-dev-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union)

## Release 1.0.1 — basic reference-image and multi-ControlNet workflow

Files:

- [PNG](../../assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_RefImages_BASIC.png)
- [JSON](../../assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_RefImages_BASIC.json)

This is the recommended stable orientation workflow for the established reference-image and multi-ControlNet path.

It demonstrates:

- FLUX.2 model, text encoder, and VAE loading;
- the JLC Flux2 ControlNet Loader;
- native multi-reference conditioning;
- a flat multi-ControlNet Orchestrator;
- externally prepared control images;
- normal FLUX.2 sampling and decode.

Under Release 1.1.0, the Loader may use a supported dense BF16 or mixed FP8/BF16 ControlNet checkpoint without changing the downstream Apply, Orchestrator, or reference-conditioning topology.

## Release 1.0.1 — focused experimental inpainting workflow

Files:

- [PNG](../../assets/workflows/Release_1.0.1/jlc_Flux2_Inpainting_workflow.png)
- [JSON](../../assets/workflows/Release_1.0.1/jlc_Flux2_Inpainting_workflow.json)

This smaller workflow demonstrates the core experimental inpaint path:

```text
ControlNet Orchestrator
    -> Inpaint Adapter - Experimental
    -> clean/empty Flux2 sampler latent
```

It is useful for confirming:

- mask polarity;
- exact canvas geometry;
- first-active ControlNet host behavior;
- the clean/empty Flux2 sampler-latent contract;

before introducing reference images or multiple dense controls.

The In/Out-Paint Adapter remains explicitly **Experimental** in Release 1.1.0.

## Release 1.0.1 — full reference, multi-ControlNet, inpainting, and cache workflow

Files:

- [PNG](../../assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_RefImages_Inpaint_workflow.png)
- [JSON](../../assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_RefImages_Inpaint_workflow.json)

This workflow combines:

- multiple reference images;
- multiple active ControlNet branches;
- the experimental In/Out-Paint Adapter;
- ControlNet hint-latent preparation;
- reference-latent preparation;
- experimental inpaint-context preparation;
- a mutually exclusive setup/inference branch;
- Conditional Save Image as a lazy branch companion.

The workflow is intended as a reference architecture rather than as a universal preset.

Dense auxiliary controls can conflict with the editable region even when cache preparation makes the workflow computationally efficient.

## Release 1.0.1 — comprehensive unified-cache workflow

Files:

- [PNG](../../assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_All_In_One.png)
- [JSON](../../assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_All_In_One.json)

This workflow demonstrates the broader conditioning ecosystem while using **JLC Flux2 Conditioning Cache Prep** as the combined setup surface for:

- reference images;
- ControlNet hints;
- optional inpaint context.

The unified cache-preparation node does not merge the underlying cache engines. Reference images, ControlNet hints, and inpaint context retain independent cache identities, limits, and runtime behavior.

The original specialist cache-preparation nodes remain available for existing workflows and targeted use.

### Cache wiring snapshot

A static image of the cache-preparation wiring is available here:

[Cache wiring snapshot](../../assets/workflows/Release_1.0.1/jlc_Flux2_ControlNet_Cache_Wiring.jpg)

The snapshot is for visual reference only and is not a standalone workflow file. Editable workflow data is contained in the corresponding workflow PNG/JSON files.

## Release compatibility

Release 1.1.0 deliberately does **not** introduce separate FP8 versions of the established Apply, Orchestrator, reference-image, cache, or experimental in/out-paint workflows.

The Loader automatically detects the supported checkpoint representation and publishes the same `JLC_FLUX2_CONTROLNET` runtime abstraction downstream.

As a result:

- the Release 1.1.0 FP8 workflow demonstrates the new mixed-precision loader path;
- existing Release 1.0.1 workflows remain valid for the established feature set;
- a compatible dense checkpoint and the JLC mixed FP8/BF16 checkpoint can use the same downstream node topology.

## External dependencies

The example workflows may include nodes from packages outside this repository, such as:

- ComfyUI ControlNet Auxiliary Preprocessors;
- KJNodes;
- Impact Pack;
- rgthree-comfy;
- JLC ComfyUI Nodes.

Use ComfyUI's missing-node information to identify absent packages.

Equivalent local preprocessors or routing nodes may be substituted where appropriate. These companion packages are not required for the core JLC Flux2 ControlNet runtime unless a particular workflow contains nodes from them.

## Local files to replace

Before running an example:

1. Select locally installed FLUX.2 model, text encoder, VAE, and ControlNet files.
2. Replace reference and source images.
3. Replace optional LoRAs or remove their loaders.
4. Inspect output-folder fields; development paths may not exist on another system.
5. Confirm width and height are shared consistently by the scheduler, Empty Flux2 Latent, cache-preparation nodes, source image, and mask where applicable.
6. Confirm mask polarity before sampling when using the experimental in/out-paint path.

## Cache workflow sequence

For workflows using an explicit setup/inference cache branch:

1. Set the shared setup switch to the cache-preparation state.
2. Optionally clear caches before preparation.
3. Queue one setup run.
4. Read cache reports or diagnostics if enabled.
5. Set the switch to the inference state.
6. Queue the generation without restarting ComfyUI.

If a cached identity no longer matches the current inputs, the runtime path performs its normal cold preparation rather than intentionally reusing the stale entry.

---

[Documentation home](../README.md) · [Project README](../../README.md)