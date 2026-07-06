"""
JLC Flux2 ControlNet Orchestrator
---------------------------------

- JLC Flux2 ControlNet
  - This node is part of the **JLC Flux2 ControlNet** package developed
    by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/JLC-Flux2-ControlNet

- Node Purpose
  - The **JLC Flux2 ControlNet Orchestrator** creates a two-branch,
    non-recursive FLUX.2 ControlNet composition from one shared loaded
    side model.

  - The node accepts:
        • one clean conditioning input
        • one loaded JLC FLUX.2 ControlNet base object
        • one VAE
        • two independent control images
        • independent strength values for both branches
        • independent start and end percentages for both branches
        • shared diagnostic control

  - Each branch:
        • receives an isolated ControlNet copy
        • receives its own control hint
        • receives its own strength and timestep range
        • maintains its own encoded-hint and runtime state
        • shares the same underlying side-model weights and CoreModelPatcher
        • is detached from `previous_controlnet` recursion

- Non-Recursive Composition
  - Both branches are represented by one ControlNet-compatible composition
    object presented to the ComfyUI sampler.

  - During each active denoising call:
        • each branch is evaluated independently against the same native
          FLUX.2 sampler state
        • each branch produces four residual tensors
        • corresponding residuals are combined linearly
        • the combined residuals are injected after native FLUX.2 double
          blocks 0, 2, 4, and 6

  - The composition path:
        • does not build a recursive chain between the two active branches
        • does not duplicate the underlying 4.116-billion-parameter side model
        • does not mutate the loader's reusable base object
        • does not use `deepcopy`
        • preserves independent branch strengths and activation windows
        • supports exact bypass of inactive or zero-strength branches

- Runtime Integration
  - The composed wrapper exposes the lifecycle, model, and hook interfaces
    expected by current ComfyUI sampling and model-management paths.

  - It cooperates with ComfyUI DynamicVRAM behavior and does not replace
    ComfyUI's loading, offloading, patching, or sampler policy.

  - The Orchestrator requires clean input conditioning with no previously
    attached ControlNet. Connect it directly after text conditioning rather
    than after a separate ControlNet Apply node.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - The non-recursive composition design builds on the validated JLC
    ControlNet linearization architecture developed for ComfyUI.

  - Built for interoperability with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

from ..jlc_flux2_controlnet_versions import JLC_FLUX2_CONTROLNET_VERSION


MANIFEST = {
    "name": "JLC Flux2 ControlNet Orchestrator",
    "version": JLC_FLUX2_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Creates a two-branch non-recursive FLUX.2 ControlNet composition "
        "from one shared side model. Each branch retains an independent "
        "control image, strength, timestep range, and runtime state while "
        "sharing the underlying model weights and ComfyUI model patcher."
    ),
}


from ..jlc_flux2_controlnet.composition import JLCFlux2ComposedControl


def _configured_child(
    controlnet,
    control_image,
    vae,
    strength,
    start_percent,
    end_percent,
    diagnostics,
):
    if start_percent > end_percent:
        raise ValueError("start_percent must be less than or equal to end_percent")

    child = controlnet.copy().set_cond_hint(
        control_image.movedim(-1, 1),
        strength=float(strength),
        timestep_percent_range=(float(start_percent), float(end_percent)),
        vae=vae,
    )
    child.diagnostics_enabled = bool(diagnostics)
    child.set_previous_controlnet(None)
    child.extra_hooks = None
    return child


class JLCFlux2ControlNetOrchestrator:
    """Apply two independently configured branches from one shared side model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "controlnet": ("JLC_FLUX2_CONTROLNET",),
                "vae": ("VAE",),
                "control_image_1": ("IMAGE",),
                "strength_1": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "start_percent_1": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_percent_1": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "control_image_2": ("IMAGE",),
                "strength_2": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "start_percent_2": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_percent_2": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "diagnostics": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "orchestrate"
    CATEGORY = "JLC/ControlNet/Flux2"

    def orchestrate(
        self,
        conditioning,
        controlnet,
        vae,
        control_image_1,
        strength_1,
        start_percent_1,
        end_percent_1,
        control_image_2,
        strength_2,
        start_percent_2,
        end_percent_2,
        diagnostics,
    ):
        output = []
        for tensor, metadata in conditioning:
            metadata_copy = metadata.copy()
            if metadata_copy.get("control") is not None:
                raise ValueError(
                    "JLC Flux2 ControlNet Orchestrator requires clean conditioning with no existing control. "
                    "Connect it directly after text encoding rather than after an Apply node."
                )

            # Both detached children share the exact same underlying model and
            # CoreModelPatcher. Only their hints, strengths, ranges, and runtime
            # caches are independent.
            child_1 = _configured_child(
                controlnet,
                control_image_1,
                vae,
                strength_1,
                start_percent_1,
                end_percent_1,
                diagnostics,
            )
            child_2 = _configured_child(
                controlnet,
                control_image_2,
                vae,
                strength_2,
                start_percent_2,
                end_percent_2,
                diagnostics,
            )

            orchestrated_control = JLCFlux2ComposedControl(
                (child_1, child_2),
                diagnostics_enabled=bool(diagnostics),
            )

            metadata_copy["control"] = orchestrated_control
            metadata_copy["control_apply_to_uncond"] = False
            output.append([tensor, metadata_copy])

        return (output,)
