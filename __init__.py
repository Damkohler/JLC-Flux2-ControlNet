"""
JLC Flux2 ControlNet
--------------------

- JLC Flux2 ControlNet
  - A ComfyUI-native FLUX.2 ControlNet implementation developed by
    **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/JLC-Flux2-ControlNet

- Package Purpose
  - This package provides a clean FLUX.2 ControlNet integration for ComfyUI
    without globally monkey-patching the native diffusion model or replacing
    ComfyUI's sampler and model-management systems.

  - The package currently includes:
        • JLC Flux2 ControlNet Loader
        • JLC Flux2 ControlNet Apply
        • JLC Flux2 ControlNet Orchestrator

  - Core capabilities include:
        • compact FLUX.2 ControlNet side-model loading
        • ComfyUI-native ControlBase and CoreModelPatcher lifecycle support
        • lazy loading and DynamicVRAM-compatible model staging
        • stateless per-forward diffusion-model integration
        • residual injection after native FLUX.2 double blocks 0, 2, 4, and 6
        • exact zero-strength bypass
        • independent control-image, strength, and timestep-range handling
        • non-recursive two-branch ControlNet composition
        • shared side-model ownership across configured ControlNet branches

- Package Registration
  - This module:
        • imports and registers the public ComfyUI node classes
        • defines user-facing node display names
        • exposes the package version
        • declares the frontend web directory
        • mounts static frontend assets used by the JLC node-logo extension

- Runtime Design
  - The implementation is designed to cooperate with current ComfyUI:
        • model loading and offloading
        • sampler lifecycle
        • transformer-option hooks
        • block-replacement composition
        • DynamicVRAM behavior

  - It does not:
        • replace ComfyUI core files
        • globally modify FLUX.2 model methods
        • duplicate the shared ControlNet side model per configured branch
        • use recursive execution for Orchestrator composition

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Built for interoperability with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

import os
from server import PromptServer  # used for static route mounting

from .nodes.jlc_flux2_controlnet_loader_node import JLCFlux2ControlNetLoader
from .nodes.jlc_flux2_controlnet_apply_node import JLCFlux2ControlNetApplyDiagnostic
from .nodes.jlc_flux2_controlnet_orchestrator_node import JLCFlux2ControlNetOrchestrator

__version__ = "0.0.3-first-injection"

NODE_CLASS_MAPPINGS = {
    "JLCFlux2ControlNetLoader": JLCFlux2ControlNetLoader,
    "JLCFlux2ControlNetApplyDiagnostic": JLCFlux2ControlNetApplyDiagnostic,
    "JLCFlux2ControlNetOrchestrator": JLCFlux2ControlNetOrchestrator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLCFlux2ControlNetLoader": "\u2003JLC Flux2 ControlNet Loader",
    "JLCFlux2ControlNetApplyDiagnostic": "\u2003JLC Flux2 ControlNet Apply",
    "JLCFlux2ControlNetOrchestrator": "\u2003JLC Flux2 ControlNet Orchestrator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Path to web folder
WEB_DIRECTORY = "./web"
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

# Mount it into ComfyUI frontend
ps = PromptServer.instance

if os.path.exists(WEB_DIR):
    ps.app.router.add_static(
        "/extensions/JLC-Flux2-ControlNet",
        WEB_DIR,
    )