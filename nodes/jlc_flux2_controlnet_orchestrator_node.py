"""
JLC Flux2 ControlNet Orchestrator
---------------------------------

- JLC Flux2 ControlNet
  - This node is part of the **JLC Flux2 ControlNet** package developed
    by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/JLC-Flux2-ControlNet

  - The package provides ComfyUI-native FLUX.2 ControlNet support focused on:
        • independent ControlNet branch preparation
        • non-recursive multi-ControlNet composition
        • shared side-model execution
        • reference-image and inpaint/outpaint integration
        • encoded hint-latent caching
        • compatibility with normal ComfyUI model-management paths

- Node Purpose
  - The **JLC Flux2 ControlNet Orchestrator** builds a one-to-four-branch
    FLUX.2 ControlNet composition from one externally loaded JLC FLUX.2
    ControlNet side model.

  - The file provides two workflow interfaces:
        • **JLC Flux2 ControlNet Orchestrator**
          attaches the composed control to one CONDITIONING input

        • **JLC Flux2 ControlNet Orchestrator Advanced**
          attaches one shared composed-control object to both positive
          and negative CONDITIONING inputs

  - Slot 1 is required.

  - Slots 2 through 4 are optional and are omitted when no control image
    is supplied.

  - Slot order is preserved deliberately. A missing slot 1 is an error;
    later slots are never promoted because branch order may affect
    composition policy, diagnostics, caching behavior, and inpaint-host
    selection in related workflow paths.

- Per-Branch Configuration
  - Every active branch receives its own:
        • control hint image
        • strength
        • start percentage
        • end percentage
        • encoded hint-latent cache state
        • diagnostic state
        • isolated configured ControlNet copy

  - All branches share:
        • the loaded FLUX.2 ControlNet side-model weights
        • the underlying ComfyUI CoreModelPatcher
        • the supplied VAE
        • normal ComfyUI model-management infrastructure

  - The loaded ControlNet base object is not mutated.

  - Branch preparation uses shallow ControlNet copies rather than
    duplicating the side-model weights or using `deepcopy`.

- Null and Inactive-Slot Contract
  - A genuine absent-control signal is represented by `None`.

  - Optional slots receiving `None` are omitted cleanly before composition.

  - Required slot 1 rejects `None` with a clear validation error.

  - The node does not interpret white images, black images, tiny tensors,
    blank-looking images, or source-image passthroughs as disabled controls.

  - This contract allows dynamic auxiliary-preprocessor wrappers and
    cache-preparation workflows to disable slots without accidentally feeding
    a dense RGB source image into ControlNet.

- Non-Recursive Composition Architecture
  - Active branches are detached from native recursive
    `previous_controlnet` chaining.

  - Each prepared child has:
        • `previous_controlnet` cleared
        • independent hint and activation state
        • independent runtime preparation and cleanup
        • no ownership of another active branch

  - The children are presented to the sampler through one
    `JLCFlux2ComposedControl` wrapper.

  - The composition runtime:
        • evaluates each child independently against the same sampler state
        • preserves the validated flat, non-recursive execution architecture
        • clones tensors only when taking ownership of returned data
        • accumulates later residual contributions in-place
        • exposes one ControlNet-compatible object to ComfyUI
        • avoids recursive branch execution and recursive model staging

  - This file does not alter the validated residual-fusion implementation.
    It is responsible only for branch activation, configuration, ordering,
    validation, and attachment to conditioning.

- FLUX.2 Runtime Integration
  - ControlNet execution uses the JLC stateless, per-invocation FLUX.2
    diffusion-model wrapper.

  - The runtime:
        • does not globally monkey-patch the FLUX.2 model
        • does not replace ComfyUI's sampler
        • does not use `deepcopy`
        • preserves existing block replacements
        • injects ControlNet residuals after native FLUX.2 double blocks
          0, 2, 4, and 6
        • applies branch strength at residual-injection time
        • cooperates with normal ComfyUI loading, offloading, patching,
          and DynamicVRAM behavior

  - A branch with strength zero follows the exact ControlNet bypass path and
    does not execute or stage the side model for that branch.

- Conditioning Ownership
  - The Orchestrator requires clean conditioning with no previously attached
    ControlNet.

  - It must be connected directly after text encoding, reference conditioning,
    or another conditioning stage that has not already installed a `control`
    object.

  - Existing controls are rejected explicitly rather than being silently
    chained into the composed runtime.

  - The standard node creates a composed control for each conditioning entry.

  - The Advanced node creates one composed-control object and shares it across
    corresponding positive and negative conditioning paths, preserving shared
    branch caches and runtime ownership.

- Relationship to JLC Flux2 ControlNet Apply
  - **JLC Flux2 ControlNet Apply** provides the conventional single-branch
    interface and supports native `previous_controlnet` chaining.

  - **JLC Flux2 ControlNet Apply Advanced** attaches one shared configured
    branch to positive and negative conditioning.

  - **JLC Flux2 ControlNet Orchestrator** is the preferred interface for
    validated flat composition of multiple branches that share one loaded
    FLUX.2 ControlNet side model.

  - The Apply and Orchestrator paths use the same underlying JLC FLUX.2
    ControlNet implementation, but differ deliberately in branch ownership:

        • Apply preserves conventional recursive chaining semantics
        • Orchestrator detaches children and uses non-recursive composition

- Inpaint / Outpaint Scope
  - This stable Orchestrator exposes ordinary ControlNet hint images only.

  - Mask-aware inpaint/outpaint conditioning is implemented by the separate
    **JLC Flux2 ControlNet In/Out-Paint** path.

  - Keeping the interfaces separate prevents experimental mask-specific
    behavior from changing the validated ordinary ControlNet composition
    architecture.

- Cache and Preparation Behavior
  - Each configured child owns its encoded control-hint cache state.

  - Shared side-model weights remain common across all children.

  - Optional null slots are removed before child construction and therefore
    are not encoded, cached, staged, or executed.

  - Cache-preparation workflows should apply the same null contract and skip
    disabled hints rather than manufacturing placeholder images.

- Model Management and MultiGPU Scope
  - The composed wrapper exposes the models, hooks, lifecycle methods, and
    inference-memory requirements expected by current ComfyUI sampler and
    model-management paths.

  - The node does not replace ComfyUI loading, offloading, device-residency,
    patching, or memory-management policy.

  - Compatibility attributes required by current ComfyUI interfaces do not
    constitute a separate MultiGPU ControlNet cloning implementation.

  - Use this node as a validated single-device orchestration path unless
    explicit MultiGPU branch-cloning support is added separately.

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
from ..jlc_flux2_controlnet.composition import JLCFlux2ComposedControl


MAX_CONTROL_SLOTS = 4


MANIFEST = {
    "name": "JLC Flux2 ControlNet Orchestrator",
    "version": JLC_FLUX2_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Builds a validated one-to-four-branch, non-recursive FLUX.2 "
        "ControlNet composition from one shared loaded side model. Each active "
        "branch uses an isolated configured ControlNet copy with its own hint "
        "image, strength, timestep range, encoded hint-latent cache, and "
        "diagnostic state while sharing the underlying side-model weights and "
        "ComfyUI model patcher. Slot 1 is required; null optional slots are "
        "omitted without promotion or placeholder images. Children are detached "
        "from previous_controlnet recursion and presented through one flat "
        "JLCFlux2ComposedControl runtime. Standard and Advanced interfaces "
        "support single-conditioning or shared positive/negative attachment. "
        "Stable ordinary-hint path; mask-aware inpaint/outpaint behavior remains "
        "in the separate JLC Flux2 ControlNet In/Out-Paint implementation."
    ),
}


def _coerce_slot_count(value) -> int:
    """Clamp saved, API-supplied, or frontend slot counts to the fixed maximum."""
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = MAX_CONTROL_SLOTS
    return max(1, min(MAX_CONTROL_SLOTS, count))


def _require_primary_control_image(control_image):
    """Reject absence in required slot 1 without promoting later slots."""
    if control_image is None:
        raise ValueError(
            "JLC Flux2 ControlNet Orchestrator requires a real control image in "
            "slot 1, but received None. The connected auxiliary-wrapper slot is "
            "DISABLED, hidden above slot_count, or otherwise absent. Later slots "
            "are not promoted because slot order is semantically significant."
        )
    return control_image


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


def _build_children(
    controlnet,
    vae,
    diagnostics,
    control_image_1,
    strength_1,
    start_percent_1,
    end_percent_1,
    control_image_2=None,
    strength_2=0.5,
    start_percent_2=0.0,
    end_percent_2=1.0,
    control_image_3=None,
    strength_3=0.5,
    start_percent_3=0.0,
    end_percent_3=1.0,
    control_image_4=None,
    strength_4=0.5,
    start_percent_4=0.0,
    end_percent_4=1.0,
    slot_count=MAX_CONTROL_SLOTS,
):
    count = _coerce_slot_count(slot_count)
    control_image_1 = _require_primary_control_image(control_image_1)

    # slot_count is authoritative at runtime, not merely a frontend layout hint.
    # This protects API prompts and stale workflow payloads as well as the GUI.
    if count < 2:
        control_image_2 = None
    if count < 3:
        control_image_3 = None
    if count < 4:
        control_image_4 = None

    children = [
        _configured_child(
            controlnet,
            control_image_1,
            vae,
            strength_1,
            start_percent_1,
            end_percent_1,
            diagnostics,
        )
    ]

    optional_children = (
        _optional_configured_child(
            controlnet,
            control_image_2,
            vae,
            strength_2,
            start_percent_2,
            end_percent_2,
            diagnostics,
        ),
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

    return children


class JLCFlux2ControlNetOrchestrator:
    """Apply one to four detached branches from one shared side model."""

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
                "diagnostics": ("BOOLEAN", {"default": True}),
            },
            "optional": {
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
                "slot_count": (
                    "INT",
                    {
                        "default": MAX_CONTROL_SLOTS,
                        "min": 1,
                        "max": MAX_CONTROL_SLOTS,
                        "step": 1,
                        "tooltip": (
                            "Number of exposed and active ControlNet slots. "
                            "Slots above this count are ignored by the backend "
                            "even if stale workflow data remains connected."
                        ),
                    },
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
        diagnostics,
        control_image_2=None,
        strength_2=0.5,
        start_percent_2=0.0,
        end_percent_2=1.0,
        control_image_3=None,
        strength_3=0.5,
        start_percent_3=0.0,
        end_percent_3=1.0,
        control_image_4=None,
        strength_4=0.5,
        start_percent_4=0.0,
        end_percent_4=1.0,
        slot_count=MAX_CONTROL_SLOTS,
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

            children = _build_children(
                controlnet,
                vae,
                diagnostics,
                control_image_1,
                strength_1,
                start_percent_1,
                end_percent_1,
                control_image_2=control_image_2,
                strength_2=strength_2,
                start_percent_2=start_percent_2,
                end_percent_2=end_percent_2,
                control_image_3=control_image_3,
                strength_3=strength_3,
                start_percent_3=start_percent_3,
                end_percent_3=end_percent_3,
                control_image_4=control_image_4,
                strength_4=strength_4,
                start_percent_4=start_percent_4,
                end_percent_4=end_percent_4,
                slot_count=slot_count,
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
    """Apply one to four detached branches to positive and negative."""

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
                "diagnostics": ("BOOLEAN", {"default": True}),
            },
            "optional": {
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
                "slot_count": (
                    "INT",
                    {
                        "default": MAX_CONTROL_SLOTS,
                        "min": 1,
                        "max": MAX_CONTROL_SLOTS,
                        "step": 1,
                        "tooltip": (
                            "Number of exposed and active ControlNet slots. "
                            "Slots above this count are ignored by the backend "
                            "even if stale workflow data remains connected."
                        ),
                    },
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
        diagnostics,
        control_image_2=None,
        strength_2=0.5,
        start_percent_2=0.0,
        end_percent_2=1.0,
        control_image_3=None,
        strength_3=0.5,
        start_percent_3=0.0,
        end_percent_3=1.0,
        control_image_4=None,
        strength_4=0.5,
        start_percent_4=0.0,
        end_percent_4=1.0,
        slot_count=MAX_CONTROL_SLOTS,
    ):
        for conditioning in (positive, negative):
            for _, metadata in conditioning:
                if metadata.get("control") is not None:
                    raise ValueError(
                        "JLC Flux2 ControlNet Orchestrator Advanced requires "
                        "clean positive and negative conditioning with no "
                        "existing control."
                    )

        children = _build_children(
            controlnet,
            vae,
            diagnostics,
            control_image_1,
            strength_1,
            start_percent_1,
            end_percent_1,
            control_image_2=control_image_2,
            strength_2=strength_2,
            start_percent_2=start_percent_2,
            end_percent_2=end_percent_2,
            control_image_3=control_image_3,
            strength_3=strength_3,
            start_percent_3=start_percent_3,
            end_percent_3=end_percent_3,
            control_image_4=control_image_4,
            strength_4=strength_4,
            start_percent_4=start_percent_4,
            end_percent_4=end_percent_4,
            slot_count=slot_count,
        )

        composed_control = JLCFlux2ComposedControl(
            tuple(children),
            diagnostics_enabled=bool(diagnostics),
        )

        return (
            _attach_clean_composed_control(positive, composed_control),
            _attach_clean_composed_control(negative, composed_control),
        )
