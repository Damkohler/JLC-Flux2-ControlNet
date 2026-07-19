# Installation

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


## Requirements

JLC Flux2 ControlNet requires:

- a current ComfyUI installation with native FLUX.2 support;
- Python 3.10 or newer;
- a compatible FLUX.2-dev diffusion model, text encoder, and VAE;
- a compatible compact FLUX.2-dev Fun ControlNet Union checkpoint;
- enough VRAM, system RAM, or model-offloading capacity for FLUX.2-dev and the compact side model.

The repository does not bundle model weights or image preprocessors.

## Install through the Comfy Registry or Manager

After Release 1.0.0 is published in the Comfy Registry, search for **JLC Flux2 ControlNet** in ComfyUI Manager and install the stable version. Restart ComfyUI after installation.

Registry releases are versioned snapshots. A later change to the GitHub `main` branch does not silently rewrite an already published Registry version.

## Install from GitHub

From the ComfyUI installation directory:

```bash
cd custom_nodes
git clone https://github.com/Damkohler/JLC-Flux2-ControlNet.git
```

The resulting path should be:

```text
ComfyUI/
└── custom_nodes/
    └── JLC-Flux2-ControlNet/
```

To update a Git clone later:

```bash
cd ComfyUI/custom_nodes/JLC-Flux2-ControlNet
git pull
```

Restart ComfyUI after updating.

## Manual installation

Download or copy the repository into:

```text
ComfyUI/custom_nodes/JLC-Flux2-ControlNet/
```

Do not place a second nested copy of the repository inside that folder. The root should contain `__init__.py`, `README.md`, `pyproject.toml`, `nodes/`, and `jlc_flux2_controlnet/`.

## Install model files

Place the compact ControlNet checkpoint in:

```text
ComfyUI/models/controlnet/
```

The **JLC Flux2 ControlNet Loader** reads the same ControlNet model folder that ComfyUI exposes to other ControlNet loaders.

FLUX.2 model, text-encoder, and VAE files belong in the model folders expected by the corresponding ComfyUI loader nodes. Exact filenames can vary by distribution. The included workflows contain development filenames and should be updated to match the files installed on the local system.

## Optional companion nodes

This package does not include pose, depth, edge, luminance, color, or other image preprocessors. Included workflows may use:

- ComfyUI core nodes;
- ComfyUI ControlNet Auxiliary Preprocessors;
- KJNodes;
- Impact Pack or other workflow utilities;
- the optional companion [JLC ComfyUI Nodes](https://github.com/Damkohler/jlc-comfyui-nodes) package.

These companion packages are not required for the core JLC Flux2 ControlNet nodes. They are required only when a loaded example workflow contains nodes from them.

## Verify installation

After restarting ComfyUI, search for the following display names:

- JLC Flux2 ControlNet Loader
- JLC Flux2 ControlNet Apply
- JLC Flux2 ControlNet Apply Advanced
- JLC Flux2 ControlNet Orchestrator
- JLC Flux2 ControlNet Orchestrator Advanced
- JLC Flux2 Reference Image Orchestrator
- JLC Flux2 ControlNet Latents Cache
- JLC Flux2 Reference Latents Cache
- JLC Flux2 Inpaint Context Cache - Experimental
- JLC Conditional Save Image
- JLC Flux2 ControlNet Inpaint Adapter - Experimental
- JLC Flux2 ControlNet Inpaint Adapter Advanced - Experimental

## Common installation problems

### Nodes do not appear

Confirm that:

- the repository is not nested one folder too deep;
- ComfyUI was restarted after installation;
- the console shows the JLC Flux2 ControlNet package loading;
- there is no import traceback;
- the installed ComfyUI version has native FLUX.2 support.

### The checkpoint is absent from the Loader dropdown

Confirm that the checkpoint is inside `ComfyUI/models/controlnet/`, then refresh or restart ComfyUI.

### An included workflow reports missing nodes

Install the companion package named in the missing-node dialog, or replace that node with an equivalent local tool. The core package intentionally does not bundle third-party preprocessors or general workflow utilities.

### A workflow loads but points to missing images or models

Example workflows retain development filenames as wiring examples. Replace image paths, model selections, LoRAs, and output folders with local equivalents before running.


---

[Documentation home](../README.md) · [Project README](../../README.md)
