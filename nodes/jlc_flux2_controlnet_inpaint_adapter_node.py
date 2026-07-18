"""
JLC Flux2 ControlNet In/Out-Paint Adapter Experimental
------------------------------------------------------

- JLC Flux2 ControlNet
  - This node is part of the **JLC Flux2 ControlNet** custom-node project
    developed by **J. L. Córdova**.

  - The project provides a native FLUX.2 ControlNet implementation for
    ComfyUI, including:
        • standalone ControlNet Apply workflows
        • multi-ControlNet orchestration
        • validated non-recursive residual composition
        • native reference-image conditioning
        • reusable hint-latent and reference-latent caches
        • mask-aware inpainting and outpainting support

- Node Purpose
  - The **JLC Flux2 ControlNet In/Out-Paint Adapter Experimental** converts an
    existing JLC Flux2 ControlNet conditioning chain into the mask-aware
    Flux2Fun-style ControlNet contract.

  - It augments the selected ControlNet path with:
        • the edit-canvas image
        • the user-provided editable-region mask
        • VAE-encoded masked-source context
        • the expanded 260-channel mask-aware ControlNet input representation

  - The node does not replace the sampler latent path.

    The sampler latent may still be prepared independently with:
        • VAE Encode (for Inpainting)
        • another mask-aware latent encoder
        • an empty latent
        • a custom latent-preparation workflow

- Supported Workflow Placement
  - The adapter supports both the standalone Apply path and the composed
    Orchestrator path.

  - Standalone Apply ordering:

        text / reference conditioning
        -> JLC Flux2 ControlNet Apply
        -> JLC Flux2 ControlNet In/Out-Paint Adapter Experimental
        -> guider / sampler

  - Multi-ControlNet ordering:

        text / reference conditioning
        -> JLC Flux2 ControlNet Orchestrator
        -> JLC Flux2 ControlNet In/Out-Paint Adapter Experimental
        -> guider / sampler

  - When the incoming conditioning contains one ordinary JLC Flux2 ControlNet
    instance, the adapter preserves that control object's native execution and
    injection-hook behavior while adding the mask-aware inpaint context.

  - When the incoming conditioning contains a non-recursive composed
    ControlNet wrapper, the adapter preserves the composition wrapper's shared
    execution and injection-hook ownership.

- Single-ControlNet Behavior
  - For a conditioning chain produced by **JLC Flux2 ControlNet Apply**:
        • the active ControlNet remains the sampler-visible control object
        • its native Flux2 residual-injection hook remains active
        • its strength and timestep range remain unchanged
        • the edit image and editable-region mask are attached to that control
        • the control image remains the original structural ControlNet hint

  - This makes the adapter a valid downstream extension of the native Apply
    workflow rather than requiring the Orchestrator.

- Multi-ControlNet Behavior
  - For a composed multi-ControlNet object, one shared edit canvas and mask
    apply to the composition as a whole.

  - The mask-aware context is attached exactly once, to the first active
    ControlNet child.

  - Remaining children continue as ordinary control-only branches.

  - This prevents the same masked-source context from being independently
    encoded and injected by every active child, which would otherwise multiply
    the inpaint contribution during non-recursive residual composition.

  - The composed wrapper continues to:
        • evaluate each active ControlNet independently
        • combine weighted residuals linearly
        • expose one ControlNet-compatible object to the ComfyUI sampler
        • own one shared Flux2 residual-injection hook

- Mask and Image Contract
  - User-facing mask convention:

        white / 1.0 = editable or regenerate region
        black / 0.0 = preserve or retain region

  - Before VAE encoding, editable pixels in the source image are replaced with
    the neutral model-space fill expected by the FLUX.2 mask-aware ControlNet
    path.

  - The mask and edit-canvas image are normalized to the dimensions required
    by the active Flux2 latent and ControlNet context.

  - The ordinary ControlNet hint remains separate from the masked edit canvas:
        • the hint provides pose, edge, depth, luminance, color, or other
          structural guidance
        • the edit canvas and mask provide the source-image preservation and
          regeneration context

- Execution and Model-Management Scope
  - The adapter does not load a separate base diffusion model.

  - It does not replace ComfyUI model loading, offloading, patching, sampler
    execution, or DynamicVRAM policy.

  - It preserves the incoming ControlNet strength and activation range.

  - It does not build a recursive `previous_controlnet` chain.

  - It does not duplicate the shared inpaint context across every child of a
    composed ControlNet.

  - The node is intended to remain downstream of Apply or Orchestrator and
    upstream of the guider or sampler.

- Experimental Status
  - The mask-aware FLUX.2 input contract and single-context composition policy
    are implemented and operational, but the node remains marked Experimental
    while broader validation continues across:
        • different mask shapes and feathering policies
        • multiple ControlNet modalities
        • mixed ControlNet strengths and timestep ranges
        • reference-image conditioning
        • batch execution
        • inpainting and outpainting layouts
        • changing ComfyUI model-management behavior

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Designed to interoperate with the ControlNet and sampler execution
    interfaces provided by ComfyUI:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

from ..jlc_flux2_controlnet_versions import JLC_FLUX2_CONTROLNET_VERSION
from ..jlc_flux2_controlnet.control import JLCFlux2Control
from ..jlc_flux2_controlnet.composition import JLCFlux2ComposedControl
from ..jlc_flux2_controlnet.inpaint_control import JLCFlux2InpaintControl


EXPERIMENTAL_VERSION = "0.2.0"


MANIFEST = {
    "name": "JLC Flux2 ControlNet In/Out-Paint Adapter Experimental",
    "version": EXPERIMENTAL_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Downstream FLUX.2 mask-aware ControlNet adapter for both standalone "
        "JLC Flux2 ControlNet Apply workflows and non-recursive multi-ControlNet "
        "Orchestrator workflows. The adapter combines an edit-canvas image, an "
        "editable-region mask, and VAE-encoded masked-source context with the "
        "existing structural ControlNet hint while preserving the incoming "
        "ControlNet strength, timestep range, and sampler integration. For a "
        "single Apply result, the control object's native residual-injection "
        "hook is preserved. For a composed ControlNet, the shared inpaint "
        "context is attached exactly once to the first active child while the "
        "composition wrapper retains ownership of the single shared injection "
        "hook, preventing duplicate masked-source residuals across branches."
    ),
    "capabilities": (
        "single_controlnet_apply",
        "multi_controlnet_orchestrator",
        "mask_aware_controlnet",
        "inpainting",
        "outpainting",
        "shared_inpaint_context",
        "non_recursive_composition",
        "native_flux2_hook_preservation",
    ),
    "mask_convention": {
        "white": "editable_or_regenerate",
        "black": "preserve_or_keep",
    },
    "workflow_position": (
        "JLC Flux2 ControlNet Apply or Orchestrator",
        "JLC Flux2 ControlNet In/Out-Paint Adapter Experimental",
        "Guider or Sampler",
    ),
    "status": "experimental",
    "base_package_version": JLC_FLUX2_CONTROLNET_VERSION,
    "license": "MIT",
}


def _finalize_detached_child(child, diagnostics):
    child.diagnostics_enabled = bool(diagnostics)
    child.set_previous_controlnet(None)
    return child


def _as_inpaint_child(control, *, vae, mask, edit_canvas_image, diagnostics):
    if isinstance(control, JLCFlux2InpaintControl):
        child = control.copy()
    elif isinstance(control, JLCFlux2Control):
        child = JLCFlux2InpaintControl.from_control(control)
    else:
        raise TypeError(
            "JLC Flux2 ControlNet In/Out-Paint Adapter can only upgrade "
            f"JLCFlux2Control children, got {type(control)!r}."
        )

    # The adapter VAE is authoritative for the masked-source latent. The
    # configured control hint, strength and timestep range were retained by
    # JLCFlux2InpaintControl.from_control().
    child.vae = vae
    child.set_inpaint_conditioning(
        mask=mask,
        edit_canvas_image=edit_canvas_image,
    )
    return _finalize_detached_child(child, diagnostics)


def _as_plain_child(control, *, diagnostics):
    """Copy one ordinary composed branch without adding inpaint context."""
    if not isinstance(control, JLCFlux2Control):
        raise TypeError(
            "JLC Flux2 ControlNet In/Out-Paint Adapter expected a JLCFlux2Control "
            f"child, got {type(control)!r}."
        )
    return _finalize_detached_child(control.copy(), diagnostics)


def _inpaint_host_index(children) -> int:
    """Choose one deterministic active branch to carry the shared mask."""
    for index, child in enumerate(children):
        if float(getattr(child, "strength", 0.0)) != 0.0:
            return index
    return 0


def _upgrade_control_object(control, *, vae, mask, edit_canvas_image, diagnostics):
    if isinstance(control, JLCFlux2ComposedControl):
        if not control.children:
            raise ValueError(
                "JLC Flux2 ControlNet In/Out-Paint Adapter received an empty "
                "composed control object."
            )

        host_index = _inpaint_host_index(control.children)
        children = tuple(
            _as_inpaint_child(
                child,
                vae=vae,
                mask=mask,
                edit_canvas_image=edit_canvas_image,
                diagnostics=diagnostics,
            )
            if index == host_index
            else _as_plain_child(child, diagnostics=diagnostics)
            for index, child in enumerate(control.children)
        )
        return JLCFlux2ComposedControl(
            children,
            diagnostics_enabled=bool(diagnostics),
        )

    if isinstance(control, (JLCFlux2Control, JLCFlux2InpaintControl)):
        return _as_inpaint_child(
            control,
            vae=vae,
            mask=mask,
            edit_canvas_image=edit_canvas_image,
            diagnostics=diagnostics,
        )

    raise TypeError(
        "JLC Flux2 ControlNet In/Out-Paint Adapter requires conditioning that "
        "already contains a JLC Flux2 ControlNet Apply or Orchestrator control. "
        f"Found unsupported control object: {type(control)!r}."
    )


def _upgrade_conditioning(
    conditioning,
    *,
    vae,
    mask,
    edit_canvas_image,
    diagnostics,
    upgraded_by_identity,
    node_name,
):
    output = []
    for tensor, metadata in conditioning:
        metadata_copy = metadata.copy()
        control = metadata_copy.get("control")
        if control is None:
            raise ValueError(
                f"{node_name} must be placed after JLC Flux2 ControlNet Apply "
                "or Orchestrator. It cannot operate on clean text conditioning."
            )

        key = id(control)
        if key not in upgraded_by_identity:
            upgraded_by_identity[key] = _upgrade_control_object(
                control,
                vae=vae,
                mask=mask,
                edit_canvas_image=edit_canvas_image,
                diagnostics=diagnostics,
            )

        metadata_copy["control"] = upgraded_by_identity[key]
        metadata_copy["control_apply_to_uncond"] = False
        output.append([tensor, metadata_copy])
    return output


class JLCFlux2ControlNetInpaintAdapter:
    """Attach one mask-aware context to existing JLC control conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "mask": ("MASK",),
                "edit_canvas_image": ("IMAGE",),
                "diagnostics": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "VAE")
    RETURN_NAMES = ("conditioning", "vae")
    FUNCTION = "apply_inpaint_context"
    CATEGORY = "Flux2 Controlnet"

    def apply_inpaint_context(
        self,
        conditioning,
        vae,
        mask,
        edit_canvas_image,
        diagnostics,
    ):
        output = _upgrade_conditioning(
            conditioning,
            vae=vae,
            mask=mask,
            edit_canvas_image=edit_canvas_image,
            diagnostics=diagnostics,
            upgraded_by_identity={},
            node_name="JLC Flux2 ControlNet In/Out-Paint Adapter",
        )
        return (output, vae)


class JLCFlux2ControlNetInpaintAdapterAdvanced:
    """Attach one shared mask-aware context to positive and negative streams."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "mask": ("MASK",),
                "edit_canvas_image": ("IMAGE",),
                "diagnostics": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "VAE")
    RETURN_NAMES = ("positive", "negative", "vae")
    FUNCTION = "apply_inpaint_context"
    CATEGORY = "Flux2 Controlnet"

    def apply_inpaint_context(
        self,
        positive,
        negative,
        vae,
        mask,
        edit_canvas_image,
        diagnostics,
    ):
        upgraded_by_identity = {}
        positive_out = _upgrade_conditioning(
            positive,
            vae=vae,
            mask=mask,
            edit_canvas_image=edit_canvas_image,
            diagnostics=diagnostics,
            upgraded_by_identity=upgraded_by_identity,
            node_name="JLC Flux2 ControlNet In/Out-Paint Adapter Advanced",
        )
        negative_out = _upgrade_conditioning(
            negative,
            vae=vae,
            mask=mask,
            edit_canvas_image=edit_canvas_image,
            diagnostics=diagnostics,
            upgraded_by_identity=upgraded_by_identity,
            node_name="JLC Flux2 ControlNet In/Out-Paint Adapter Advanced",
        )
        return (positive_out, negative_out, vae)
