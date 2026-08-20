# `docs/getting-started/installation.md`

# Installation

> [!NOTE]
> **Documentation status: Release 1.1.0.**  
> This page follows the current Release 1.1.0 implementation and supplied workflows. The source code and current ComfyUI node interfaces remain authoritative for exact behavior.

## Requirements

JLC Flux2 ControlNet requires:

- a current ComfyUI installation with native FLUX.2 support;
- Python 3.10 or newer;
- a compatible FLUX.2-dev diffusion model, text encoder, and VAE;
- a compatible compact FLUX.2-dev Fun ControlNet Union checkpoint;
- enough VRAM, system RAM, or model-offloading capacity for FLUX.2-dev and the compact side model.

The repository does not bundle model weights or image preprocessors.

Release 1.1.0 supports compatible dense BF16 and mixed FP8/BF16 FLUX.2 Fun ControlNet checkpoints through the same Loader. No separate FP8 loader or precision selector is required.

## Install through the Comfy Registry or Manager

Search for **JLC Flux2 ControlNet** in ComfyUI Manager and install or update to **Release 1.1.0 or later** when that version is available in your current Registry/Manager index.

If the current Registry index has not yet refreshed to Release 1.1.0, use the GitHub installation method below.

Registry releases are versioned snapshots. A later change to the GitHub `main` branch does not silently rewrite an already published Registry version.

Restart ComfyUI after installation or update.

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

To update an existing Git clone later:

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

Place a compatible compact FLUX.2-dev Fun ControlNet checkpoint in:

```text
ComfyUI/models/controlnet/
```

### Model checkpoints

JLC Flux2 ControlNet supports the original dense BF16 FLUX.2 Fun ControlNet checkpoints and the JLC mixed FP8/BF16 Union-2602 derivative through the same Loader:

- **Original BF16 FLUX.2 Fun ControlNet Union:** [Alibaba-PAI / FLUX.2-dev-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union)
- **JLC mixed FP8/BF16 Union-2602:** [FLUX.2-dev-Fun-Controlnet-Union-2602-JLC-FP8](https://huggingface.co/Damkohler/FLUX.2-dev-Fun-Controlnet-Union-2602-JLC-FP8)

Model weights are not included in this repository. Download the desired checkpoint from its model repository and follow the applicable model license and usage terms.

The **JLC Flux2 ControlNet Loader** reads the same ControlNet model folder that ComfyUI exposes to other ControlNet loaders.

Release 1.1.0 filters the Loader dropdown to checkpoints whose safetensors structure matches the supported FLUX.2 Fun ControlNet architecture. Compatible dense and mixed-precision representations are detected automatically; users do not select a precision mode manually.

For supported mixed FP8/BF16 checkpoints, the runtime additionally validates the quantization metadata, FP8 weights, scale tensors, and ComfyUI-native quantized-module descriptors before publishing the loaded model.

FLUX.2 diffusion-model, text-encoder, and VAE files belong in the model folders expected by their corresponding ComfyUI loader nodes. Exact filenames can vary by distribution. Included workflows may contain development filenames and should be updated to match the files installed on the local system.

## Optional companion nodes

This package does not include pose, depth, edge, luminance, color, or other image preprocessors.

Included workflows may use:

- ComfyUI core nodes;
- ComfyUI ControlNet Auxiliary Preprocessors;
- KJNodes;
- Impact Pack or other workflow utilities;
- the optional companion [JLC ComfyUI Nodes](https://github.com/Damkohler/jlc-comfyui-nodes) package.

These companion packages are not required for the core JLC Flux2 ControlNet nodes. They are required only when a loaded example workflow contains nodes from them.

Equivalent local preprocessors or routing utilities may be substituted where appropriate.

## Verify installation

After restarting ComfyUI, search for the following display names:

- JLC Flux2 ControlNet Loader
- JLC Flux2 ControlNet Apply
- JLC Flux2 ControlNet Apply Advanced
- JLC Flux2 ControlNet Orchestrator
- JLC Flux2 ControlNet Orchestrator Advanced
- JLC Flux2 Reference Image Orchestrator
- JLC Flux2 Conditioning Cache Prep
- JLC Flux2 ControlNet Latents Cache
- JLC Flux2 Reference Latents Cache
- JLC Flux2 Inpaint Context Cache - Experimental
- JLC Conditional Save Image
- JLC Flux2 ControlNet Inpaint Adapter - Experimental
- JLC Flux2 ControlNet Inpaint Adapter Advanced - Experimental

To verify the Release 1.1.0 unified-loader path, select either a compatible original dense checkpoint or the JLC mixed FP8/BF16 Union-2602 checkpoint in **JLC Flux2 ControlNet Loader**. Both use the same downstream Apply and Orchestrator nodes.

## Common installation problems

### Nodes do not appear

Confirm that:

- the repository is not nested one folder too deep;
- ComfyUI was restarted after installation;
- the console shows the JLC Flux2 ControlNet package loading;
- there is no import traceback;
- the installed ComfyUI version has native FLUX.2 support.

### The checkpoint is absent from the Loader dropdown

Confirm that:

- the checkpoint is inside `ComfyUI/models/controlnet/`;
- the file is a compatible FLUX.2 Fun ControlNet checkpoint;
- the checkpoint safetensors header matches the architecture expected by the Release 1.1.0 Loader.

Then refresh or restart ComfyUI.

The Release 1.1.0 dropdown is intentionally filtered and does not expose arbitrary files from the ControlNet model directory.

### A mixed FP8/BF16 checkpoint is rejected

Release 1.1.0 validates the mixed-precision checkpoint contract rather than treating every FP8 tensor file as compatible.

For the JLC mixed FP8/BF16 Union-2602 release, confirm that the checkpoint was downloaded intact from:

[FLUX.2-dev-Fun-Controlnet-Union-2602-JLC-FP8](https://huggingface.co/Damkohler/FLUX.2-dev-Fun-Controlnet-Union-2602-JLC-FP8)

The model page publishes the release checksum and validation information.

### An included workflow reports missing nodes

Install the companion package named in the missing-node dialog, or replace that node with an equivalent local tool.

The core package intentionally does not bundle third-party preprocessors or general workflow utilities.

### A workflow loads but points to missing images or models

Example workflows may retain development filenames as wiring examples.

Before running, replace:

- image paths;
- model selections;
- optional LoRAs;
- output folders;

with locally installed equivalents.

For workflows using multiple conditioning paths, also confirm that canvas dimensions are shared consistently by the scheduler, Empty Flux2 Latent, cache-preparation nodes, source image, and mask where applicable.

---

[Documentation home](../README.md) · [Project README](../../README.md)