"""
JLC Flux2 ControlNet Orchestrator
---------------------------------

Creates a fixed-maximum, non-recursive FLUX.2 ControlNet composition from one
shared loaded side model.

Slots 1 and 2 remain required for backward compatibility with the validated
two-branch workflow. Slots 3 and 4 are optional. Every configured branch owns
its own control hint, strength, timestep range, encoded-hint cache, and runtime
state while sharing the same underlying side-model weights and CoreModelPatcher.

All children are detached from `previous_controlnet` recursion and are evaluated
sequentially by the existing flat composition runtime. Residual ownership,
weighted in-place accumulation, native-block injection, lazy materialization,
and exact zero-strength bypass remain unchanged.

Concept and implementation by J. L. Córdova with development assistance from
ChatGPT (OpenAI). Released under the MIT License.
"""

from __future__ import annotations

from ..jlc_flux2_controlnet_versions import JLC_FLUX2_CONTROLNET_VERSION
from ..jlc_flux2_controlnet.composition import JLCFlux2ComposedControl


MAX_CONTROL_SLOTS = 4


MANIFEST = {
    "name": "JLC Flux2 ControlNet Orchestrator",
    "version": JLC_FLUX2_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Creates a two-to-four-branch non-recursive FLUX.2 ControlNet "
        "composition from one shared side model. Each configured branch "
        "retains an independent control image, strength, timestep range, "
        "and runtime state while sharing the underlying model weights and "
        "ComfyUI model patcher."
    ),
}


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


def _optional_configured_child(
    controlnet,
    control_image,
    vae,
    strength,
    start_percent,
    end_percent,
    diagnostics,
):
    if control_image is None:
        return None

    return _configured_child(
        controlnet,
        control_image,
        vae,
        strength,
        start_percent,
        end_percent,
        diagnostics,
    )


class JLCFlux2ControlNetOrchestrator:
    """Apply two to four detached branches from one shared side model."""

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
            },
            "optional": {
                "control_image_3": ("IMAGE",),
                "strength_3": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "start_percent_3": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_percent_3": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "control_image_4": ("IMAGE",),
                "strength_4": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "start_percent_4": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_percent_4": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
            },
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
        control_image_3=None,
        strength_3=0.5,
        start_percent_3=0.0,
        end_percent_3=1.0,
        control_image_4=None,
        strength_4=0.5,
        start_percent_4=0.0,
        end_percent_4=1.0,
    ):
        output = []

        for tensor, metadata in conditioning:
            metadata_copy = metadata.copy()
            if metadata_copy.get("control") is not None:
                raise ValueError(
                    "JLC Flux2 ControlNet Orchestrator requires clean "
                    "conditioning with no existing control. Connect it directly "
                    "after text encoding rather than after an Apply node."
                )

            # Slots 1 and 2 preserve the validated required two-branch surface.
            # Every copy shares the exact same underlying model and
            # CoreModelPatcher; only branch-local configuration and caches differ.
            children = [
                _configured_child(
                    controlnet,
                    control_image_1,
                    vae,
                    strength_1,
                    start_percent_1,
                    end_percent_1,
                    diagnostics,
                ),
                _configured_child(
                    controlnet,
                    control_image_2,
                    vae,
                    strength_2,
                    start_percent_2,
                    end_percent_2,
                    diagnostics,
                ),
            ]

            # Slots 3 and 4 are fixed optional branches. Absence means that no
            # child object is created. A connected image with strength 0.0 still
            # creates the branch and uses the existing exact bypass path.
            optional_children = (
                _optional_configured_child(
                    controlnet,
                    control_image_3,
                    vae,
                    strength_3,
                    start_percent_3,
                    end_percent_3,
                    diagnostics,
                ),
                _optional_configured_child(
                    controlnet,
                    control_image_4,
                    vae,
                    strength_4,
                    start_percent_4,
                    end_percent_4,
                    diagnostics,
                ),
            )
            children.extend(
                child for child in optional_children if child is not None
            )

            if len(children) > MAX_CONTROL_SLOTS:
                raise RuntimeError(
                    f"Flux2 Orchestrator supports at most "
                    f"{MAX_CONTROL_SLOTS} configured branches."
                )

            orchestrated_control = JLCFlux2ComposedControl(
                tuple(children),
                diagnostics_enabled=bool(diagnostics),
            )

            metadata_copy["control"] = orchestrated_control
            metadata_copy["control_apply_to_uncond"] = False
            output.append([tensor, metadata_copy])

        return (output,)
def _attach_clean_composed_control(conditioning, composed_control):
    output = []
    for tensor, metadata in conditioning:
        metadata_copy = metadata.copy()
        if metadata_copy.get("control") is not None:
            raise ValueError(
                "JLC Flux2 ControlNet Orchestrator Advanced requires clean "
                "positive and negative conditioning with no existing control. "
                "Connect it directly after text encoding or reference "
                "conditioning."
            )
        metadata_copy["control"] = composed_control
        metadata_copy["control_apply_to_uncond"] = False
        output.append([tensor, metadata_copy])
    return output


class JLCFlux2ControlNetOrchestratorAdvanced:
    """Apply two to four detached branches to positive and negative."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
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
            },
            "optional": {
                "control_image_3": ("IMAGE",),
                "strength_3": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "start_percent_3": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_percent_3": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "control_image_4": ("IMAGE",),
                "strength_4": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "start_percent_4": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_percent_4": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "orchestrate"
    CATEGORY = "JLC/ControlNet/Flux2"

    def orchestrate(
        self,
        positive,
        negative,
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
        control_image_3=None,
        strength_3=0.5,
        start_percent_3=0.0,
        end_percent_3=1.0,
        control_image_4=None,
        strength_4=0.5,
        start_percent_4=0.0,
        end_percent_4=1.0,
    ):
        # Validate both streams before allocating any child branch.
        for conditioning in (positive, negative):
            for _, metadata in conditioning:
                if metadata.get("control") is not None:
                    raise ValueError(
                        "JLC Flux2 ControlNet Orchestrator Advanced requires "
                        "clean positive and negative conditioning with no "
                        "existing control."
                    )

        children = [
            _configured_child(
                controlnet,
                control_image_1,
                vae,
                strength_1,
                start_percent_1,
                end_percent_1,
                diagnostics,
            ),
            _configured_child(
                controlnet,
                control_image_2,
                vae,
                strength_2,
                start_percent_2,
                end_percent_2,
                diagnostics,
            ),
        ]

        optional_children = (
            _optional_configured_child(
                controlnet,
                control_image_3,
                vae,
                strength_3,
                start_percent_3,
                end_percent_3,
                diagnostics,
            ),
            _optional_configured_child(
                controlnet,
                control_image_4,
                vae,
                strength_4,
                start_percent_4,
                end_percent_4,
                diagnostics,
            ),
        )
        children.extend(child for child in optional_children if child is not None)

        if len(children) > MAX_CONTROL_SLOTS:
            raise RuntimeError(
                f"Flux2 Orchestrator supports at most "
                f"{MAX_CONTROL_SLOTS} configured branches."
            )

        composed_control = JLCFlux2ComposedControl(
            tuple(children),
            diagnostics_enabled=bool(diagnostics),
        )

        return (
            _attach_clean_composed_control(positive, composed_control),
            _attach_clean_composed_control(negative, composed_control),
        )
