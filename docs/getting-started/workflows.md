# Included Workflows

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


Release 1.0.0 includes three PNG/JSON workflow pairs under:

```text
assets/workflows/Release_1.0.0/
```

PNG files contain embedded ComfyUI workflow data and can be dragged onto the ComfyUI canvas. JSON files are provided for standard loading and source control.

## Basic reference-image and multi-ControlNet workflow

- [PNG](../../assets/workflows/Release_1.0.0/Flux2_ControlNet_RefImages_BASIC_01.png)
- [JSON](../../assets/workflows/Release_1.0.0/Flux2_ControlNet_RefImages_BASIC_01.json)

This is the recommended stable orientation workflow. It demonstrates:

- FLUX.2 model, text encoder, and VAE loading;
- the JLC Flux2 ControlNet Loader;
- native multi-reference conditioning;
- a flat multi-ControlNet Orchestrator;
- externally prepared control images;
- normal FLUX.2 sampling and decode.

Use this workflow first when validating the package.

## Focused experimental inpainting workflow

- [PNG](../../assets/workflows/Release_1.0.0/jlc_Flux2_ControlNet_with_Inpainting_workflow.png)
- [JSON](../../assets/workflows/Release_1.0.0/jlc_Flux2_ControlNet_with_Inpainting_workflow.json)

This smaller workflow demonstrates the core experimental inpaint path:

```text
ControlNet Orchestrator
    -> Inpaint Adapter - Experimental
    -> clean/empty Flux2 sampler latent
```

It is useful for confirming mask polarity, exact canvas geometry, and first-active host behavior before introducing reference images or multiple dense controls.

## Full reference, multi-ControlNet, inpainting, and cache workflow

- [PNG](../../assets/workflows/Release_1.0.0/Flux2_ControlNet_RefImages_Inpaint_workflow.png)
- [JSON](../../assets/workflows/Release_1.0.0/Flux2_ControlNet_RefImages_Inpaint_workflow.json)

This is the feature-complete Release 1.0.0 demonstration. It combines:

- multiple reference images;
- up to three active ControlNet branches;
- the experimental In/Out-Paint Adapter;
- ControlNet hint-latent preparation;
- reference-latent preparation;
- experimental inpaint-context preparation;
- a mutually exclusive setup/inference branch;
- Conditional Save Image as a lazy branch companion.

The workflow is intended as a reference architecture, not as a universal preset. Dense auxiliary controls can conflict with the editable region even when the cache makes the workflow computationally fast.

## External dependencies

The example workflows may include nodes from packages outside this repository, such as:

- ComfyUI ControlNet Auxiliary Preprocessors;
- KJNodes;
- Impact Pack;
- rgthree-comfy;
- JLC ComfyUI Nodes.

Use ComfyUI's missing-node information to identify absent packages. Equivalent local preprocessors or routing nodes may be substituted.

## Local files to replace

Before running an example:

1. Select locally installed FLUX.2 model, text encoder, VAE, and ControlNet files.
2. Replace reference and source images.
3. Replace optional LoRAs or remove their loaders.
4. Inspect output-folder fields; development paths may not exist on another system.
5. Confirm width and height are shared consistently by the scheduler, Empty Flux2 Latent, cache-preparation nodes, source image, and mask.
6. Confirm mask polarity before sampling.

## Cache workflow sequence

For the full workflow:

1. Set the shared setup switch to the cache-preparation state.
2. Optionally clear caches before preparation.
3. Queue one setup run.
4. Read cache reports or diagnostics if enabled.
5. Set the switch to the inference state.
6. Queue the generation without restarting ComfyUI.

If any cached identity no longer matches, the runtime path performs its normal cold preparation rather than using the stale entry.


---

[Documentation home](../README.md) · [Project README](../../README.md)
