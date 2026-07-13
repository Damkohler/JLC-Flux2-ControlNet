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
        • JLC Flux2 ControlNet Apply Advanced
        • JLC Flux2 ControlNet Orchestrator
        • JLC Flux2 ControlNet Orchestrator Advanced
        • JLC Flux2 Reference Image Orchestrator
        • JLC Flux2 Hint Latent Cache Prep

- Package Registration
  - ComfyUI loads the package located in `custom_nodes/<repo_name>/` and reads
    the following mappings:

        NODE_CLASS_MAPPINGS
        NODE_DISPLAY_NAME_MAPPINGS

  - Each NODE_CLASS_MAPPINGS entry maps one internal node name directly to the
    class implementing that node. Do not map a node name to another mapping dict.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Copyright (c) 2026 J. L. Córdova
  - Released under the **MIT License**.
"""

from __future__ import annotations

import os
from server import PromptServer  # used for static route mounting

from .jlc_flux2_controlnet_versions import JLC_FLUX2_CONTROLNET_VERSION

# Flux2 ControlNet nodes
from .nodes.jlc_flux2_controlnet_loader_node import JLCFlux2ControlNetLoader
from .nodes.jlc_flux2_controlnet_apply_node import (
    JLCFlux2ControlNetApplyAdvanced,
    JLCFlux2ControlNetApplyDiagnostic,
)
from .nodes.jlc_flux2_controlnet_orchestrator_node import (
    JLCFlux2ControlNetOrchestrator,
    JLCFlux2ControlNetOrchestratorAdvanced,
)

# Flux2 Reference Image and Inpaint Adapter nodes
# Import the class directly. Do not import or nest this module's local
# NODE_CLASS_MAPPINGS dict into the package-level NODE_CLASS_MAPPINGS.
from .nodes.jlc_flux2_reference_image_orchestrator_node import (
    JLCFlux2ReferenceImageOrchestrator,
)

# Add these imports near the other node imports in __init__.py:
from .nodes.jlc_flux2_controlnet_inpaint_adapter_node import (
    JLCFlux2ControlNetInpaintAdapter,
    JLCFlux2ControlNetInpaintAdapterAdvanced,
)


# Flux2 utility nodes
from .nodes.jlc_flux2_hint_latent_cache_prep_node import (
     JLCFlux2HintLatentCachePrep,
 )
from .nodes.jlc_flux2_reference_latent_cache_prep_node import (
    JLCFlux2ReferenceLatentCachePrep,
)

from .nodes.jlc_conditional_save_image_node import (
    JLCConditionalSaveImage,
)


__version__ = JLC_FLUX2_CONTROLNET_VERSION


NODE_CLASS_MAPPINGS = {
    # Flux2 ControlNet nodes
    "JLCFlux2ControlNetLoader": JLCFlux2ControlNetLoader,
    "JLCFlux2ControlNetApplyDiagnostic": JLCFlux2ControlNetApplyDiagnostic,
    "JLCFlux2ControlNetApplyAdvanced": JLCFlux2ControlNetApplyAdvanced,
    "JLCFlux2ControlNetOrchestrator": JLCFlux2ControlNetOrchestrator,
    "JLCFlux2ControlNetOrchestratorAdvanced": JLCFlux2ControlNetOrchestratorAdvanced,

    # Flux2 Reference Image and Inpaint Adapter nodes
    "JLCFlux2ReferenceImageOrchestrator": JLCFlux2ReferenceImageOrchestrator,

    "JLCFlux2ControlNetInpaintAdapter": JLCFlux2ControlNetInpaintAdapter,
    "JLCFlux2ControlNetInpaintAdapterAdvanced": JLCFlux2ControlNetInpaintAdapterAdvanced,

    # Flux2 utility nodes
    "JLCFlux2HintLatentCachePrep": JLCFlux2HintLatentCachePrep,
    "JLCFlux2ReferenceLatentCachePrep": JLCFlux2ReferenceLatentCachePrep,
    "JLCConditionalSaveImage": JLCConditionalSaveImage,
}


# Keep \u2003 leading em-space in names to avoid logo overlap.
NODE_DISPLAY_NAME_MAPPINGS = {
    # Flux2 ControlNet nodes
    "JLCFlux2ControlNetLoader": "\u2003JLC Flux2 ControlNet Loader",
    "JLCFlux2ControlNetApplyDiagnostic": "\u2003JLC Flux2 ControlNet Apply",
    "JLCFlux2ControlNetApplyAdvanced": "\u2003JLC Flux2 ControlNet Apply Advanced",
    "JLCFlux2ControlNetOrchestrator": "\u2003JLC Flux2 ControlNet Orchestrator",
    "JLCFlux2ControlNetOrchestratorAdvanced": (
        "\u2003JLC Flux2 ControlNet Orchestrator Advanced"
    ),

    # Flux2 Reference Image and Inpaint Adapter nodes
    "JLCFlux2ReferenceImageOrchestrator": (
        "\u2003JLC Flux2 Reference Image Orchestrator"
    ),

    "JLCFlux2ControlNetInpaintAdapter": "\u2003JLC Flux2 ControlNet Inpaint Adapter",
    "JLCFlux2ControlNetInpaintAdapterAdvanced": "\u2003JLC Flux2 ControlNet Inpaint Adapter Advanced",

    # Flux2 utility nodes
    "JLCFlux2HintLatentCachePrep": "\u2003JLC Flux2 Hint Latent Cache Prep",
    "JLCFlux2ReferenceLatentCachePrep": "\u2003JLC Flux2 Reference Latent Cache Prep",
    "JLCConditionalSaveImage": "\u2003JLC Conditional Save Image",
}


def _validate_node_class_mappings() -> None:
    """Fail early with the exact bad key if a mapping value is not a class.

    ComfyUI expects every value in NODE_CLASS_MAPPINGS to be a node class. A
    common mistake is accidentally assigning a whole imported mapping dict as a
    value, which later produces:

        AttributeError: 'dict' object has no attribute 'RELATIVE_PYTHON_MODULE'
    """

    bad_entries = {
        node_name: type(node_cls).__name__
        for node_name, node_cls in NODE_CLASS_MAPPINGS.items()
        if isinstance(node_cls, dict)
    }
    if bad_entries:
        raise TypeError(
            "Invalid JLC Flux2 NODE_CLASS_MAPPINGS entries; expected node "
            f"classes, got mapping/dict values: {bad_entries}"
        )


_validate_node_class_mappings()


JLC_FLUX2_CONTROLNET_ICON = "🕸️🔗🕸️🔗🕸️"
JLC_FLUX2_CONTROLNET_NAME = (
    f"{JLC_FLUX2_CONTROLNET_ICON} JLC Flux2 ControlNet"
)
print(f"{JLC_FLUX2_CONTROLNET_NAME} loading...")
# for node_name in sorted(NODE_CLASS_MAPPINGS.keys()):
#     node_cls = NODE_CLASS_MAPPINGS[node_name]
#     print(f"  {node_name} -> {node_cls.__module__}.{node_cls.__name__}")
print(f"{JLC_FLUX2_CONTROLNET_NAME} loaded {len(NODE_CLASS_MAPPINGS)} nodes.")


__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]


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
