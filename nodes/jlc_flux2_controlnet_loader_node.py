"""
JLC Flux2 ControlNet Loader
---------------------------

- JLC Flux2 ControlNet
  - This node is part of the **JLC Flux2 ControlNet** package developed
    by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/JLC-Flux2-ControlNet

- Node Purpose
  - The **JLC Flux2 ControlNet Loader** loads a supported FLUX.2
    ControlNet checkpoint and exposes it as a reusable
    `JLC_FLUX2_CONTROLNET` object.

  - The loader:
        • resolves checkpoints from ComfyUI's `models/controlnet` folder
        • constructs the compact FLUX.2 ControlNet side model
        • loads and validates the checkpoint tensors
        • wraps the side model in ComfyUI's native model-patcher lifecycle
        • returns an unconditioned base ControlNet object that can be copied
          and configured by Apply or Orchestrator nodes

  - Runtime Integration
    - The loaded side model participates in normal ComfyUI model loading,
      offloading, device placement, and DynamicVRAM behavior.

    - The loader does not globally patch or replace the native FLUX.2 model.

    - Conditioning images, strengths, and timestep ranges are attached later
      to isolated ControlNet copies rather than mutating the loaded base object.

- Supported Architecture
  - The initial implementation targets the compact
    `FLUX.2-dev-Fun-Controlnet-Union` side-model architecture.

  - The checkpoint contains four ControlNet transformer blocks whose
    residuals are injected into native FLUX.2 double blocks 0, 2, 4, and 6.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Built for interoperability with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

from ..jlc_flux2_controlnet_versions import JLC_FLUX2_CONTROLNET_VERSION


MANIFEST = {
    "name": "JLC Flux2 ControlNet Loader",
    "version": JLC_FLUX2_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Loads a supported compact FLUX.2 ControlNet checkpoint into a "
        "ComfyUI-native model-patcher lifecycle and returns a reusable, "
        "unconditioned JLC_FLUX2_CONTROLNET base object for Apply and "
        "Orchestrator workflows."
    ),
}


import folder_paths

from ..jlc_flux2_controlnet.loader import load_jlc_flux2_controlnet


class JLCFlux2ControlNetLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet_name": (folder_paths.get_filename_list("controlnet"),),
            }
        }

    RETURN_TYPES = ("JLC_FLUX2_CONTROLNET",)
    RETURN_NAMES = ("controlnet",)
    FUNCTION = "load_controlnet"
    CATEGORY = "JLC/ControlNet/Flux2"

    def load_controlnet(self, controlnet_name):
        checkpoint_path = folder_paths.get_full_path("controlnet", controlnet_name)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Unable to resolve ControlNet checkpoint '{controlnet_name}'."
            )
        control = load_jlc_flux2_controlnet(
            checkpoint_path, checkpoint_name=controlnet_name
        )
        return (control,)
