"""
JLC Flux2 ControlNet Apply
--------------------------

- JLC Flux2 ControlNet
  - This node is part of the **JLC Flux2 ControlNet** package developed
    by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/JLC-Flux2-ControlNet

- Node Purpose
  - The **JLC Flux2 ControlNet Apply** node attaches one control image to
    FLUX.2 conditioning through the JLC ComfyUI-native ControlNet runtime.

  - The node:
        • copies the loaded ControlNet base object
        • attaches a control image and VAE
        • applies an independent strength value
        • applies an independent start and end percentage
        • optionally enables runtime diagnostics
        • preserves an existing ControlNet as `previous_controlnet` when
          used in a conventional chained workflow

  - The configured ControlNet copy owns its own:
        • control hint
        • encoded control latent cache
        • strength
        • timestep activation range
        • diagnostic state

  - The loaded side-model weights and CoreModelPatcher remain shared rather
    than being duplicated for each configured copy.

- Null-Input Contract
  - Apply nodes require a real control image. A `None` value from a disabled or
    hidden JLC auxiliary-wrapper slot is rejected explicitly before any tensor
    operation, ControlNet copy, cache lookup, or conditioning mutation occurs.
  - Strength zero remains a runtime side-model bypass, not permission to attach
    an absent control image.

- Runtime Integration
  - The ControlNet side branch is executed through a stateless,
    per-invocation FLUX.2 diffusion-model wrapper.

  - The implementation:
        • does not globally monkey-patch the FLUX.2 model
        • does not replace ComfyUI's sampler
        • does not use `deepcopy`
        • preserves existing block replacements
        • injects residuals after native FLUX.2 double blocks 0, 2, 4, and 6
        • applies strength at residual-injection time
        • cooperates with normal ComfyUI model-management and DynamicVRAM paths

  - A strength of zero uses the exact bypass path and does not execute or
    stage the ControlNet side model.

- Workflow Role
  - Use this node for a conventional single-ControlNet workflow.

  - Multiple Apply nodes may follow native chained-ControlNet semantics.

  - For validated non-recursive two-ControlNet composition using a shared
    side model, use the **JLC Flux2 ControlNet Orchestrator**.

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
    "name": "JLC Flux2 ControlNet Apply",
    "version": JLC_FLUX2_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Configures and attaches one FLUX.2 ControlNet branch to ComfyUI "
        "conditioning with an independent control image, VAE, strength, "
        "timestep range, and diagnostics. A null control image is rejected "
        "explicitly so disabled auxiliary-wrapper slots cannot silently pass "
        "through or configure an invalid branch. Residuals are injected through "
        "a stateless native-model wrapper after FLUX.2 double blocks 0, 2, 4, "
        "and 6."
    ),
}


def _require_control_image(control_image, *, node_name: str):
    """Reject the deliberate absent-control signal before tensor handling."""
    if control_image is None:
        raise ValueError(
            f"{node_name} requires an actual control image, but received None. "
            "The connected auxiliary-wrapper slot is DISABLED, hidden above "
            "slot_count, or otherwise absent. Enable that slot or disconnect it."
        )
    return control_image


class JLCFlux2ControlNetApplyDiagnostic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet": ("JLC_FLUX2_CONTROLNET",),
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "control_image": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "diagnostics": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "Flux2 Controlnet"

    def apply_controlnet(
        self,
        controlnet,
        conditioning,
        vae,
        control_image,
        strength,
        start_percent,
        end_percent,
        diagnostics,
    ):
        if start_percent > end_percent:
            raise ValueError("start_percent must be less than or equal to end_percent")

        control_image = _require_control_image(
            control_image,
            node_name="JLC Flux2 ControlNet Apply",
        )
        output = []
        hint_bchw = control_image.movedim(-1, 1)
        for tensor, metadata in conditioning:
            metadata_copy = metadata.copy()
            control_copy = controlnet.copy().set_cond_hint(
                hint_bchw,
                strength=float(strength),
                timestep_percent_range=(float(start_percent), float(end_percent)),
                vae=vae,
            )
            control_copy.diagnostics_enabled = bool(diagnostics)
            control_copy.set_previous_controlnet(metadata_copy.get("control"))
            metadata_copy["control"] = control_copy
            metadata_copy["control_apply_to_uncond"] = False
            output.append([tensor, metadata_copy])
        return (output,)


class JLCFlux2ControlNetApplyAdvanced:
    """Attach one shared ControlNet configuration to positive and negative."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet": ("JLC_FLUX2_CONTROLNET",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "control_image": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "diagnostics": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"
    CATEGORY = "Flux2 Controlnet"

    def apply_controlnet(
        self,
        controlnet,
        positive,
        negative,
        vae,
        control_image,
        strength,
        start_percent,
        end_percent,
        diagnostics,
    ):
        if start_percent > end_percent:
            raise ValueError("start_percent must be less than or equal to end_percent")

        control_image = _require_control_image(
            control_image,
            node_name="JLC Flux2 ControlNet Apply Advanced",
        )
        hint_bchw = control_image.movedim(-1, 1)
        configured_by_previous = {}
        outputs = []

        for conditioning in (positive, negative):
            output = []
            for tensor, metadata in conditioning:
                metadata_copy = metadata.copy()
                previous_control = metadata_copy.get("control")

                # Match native ControlNetApplyAdvanced ownership: positive and
                # negative entries with the same previous control share one
                # configured ControlNet object and therefore one branch cache.
                if previous_control not in configured_by_previous:
                    control_copy = controlnet.copy().set_cond_hint(
                        hint_bchw,
                        strength=float(strength),
                        timestep_percent_range=(
                            float(start_percent),
                            float(end_percent),
                        ),
                        vae=vae,
                    )
                    control_copy.diagnostics_enabled = bool(diagnostics)
                    control_copy.set_previous_controlnet(previous_control)
                    configured_by_previous[previous_control] = control_copy

                metadata_copy["control"] = configured_by_previous[previous_control]
                metadata_copy["control_apply_to_uncond"] = False
                output.append([tensor, metadata_copy])
            outputs.append(output)

        return (outputs[0], outputs[1])
