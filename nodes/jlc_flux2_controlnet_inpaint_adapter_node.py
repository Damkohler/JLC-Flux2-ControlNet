"""
JLC Flux2 ControlNet In/Out-Paint Adapter Experimental
------------------------------------------------------

- Project and Release Status
  - This node is part of **JLC Flux2 ControlNet / Non-Recursive ControlNet
    Release 2.0.0**, developed by **J. L. Córdova**.

  - The adapter is included as an **Experimental** feature. Its core FLUX.2
    mask-aware conditioning path is operational and has been validated with
    standalone Apply conditioning, non-recursive Orchestrator composition,
    OpenPose/DWPose guidance, and native reference-image conditioning.

  - The node keeps its own experimental revision while reporting the base
    package version through ``JLC_FLUX2_CONTROLNET_VERSION``.

- Node Purpose
  - The adapter upgrades an existing JLC Flux2 ControlNet conditioning chain
    to the FLUX.2-dev Fun ControlNet Union mask-aware 260-channel contract.

  - It combines:
        • the existing structural ControlNet hint
        • one source/edit-canvas ``image``
        • one user-facing editable-region ``mask``
        • one VAE-encoded hard-masked source-image latent
        • one packed four-channel inverse keep-mask context

  - It does not replace or modify the sampler latent. The validated sampler
    workflow may continue to use the clean/empty Flux2 latent path.

- Workflow Placement
  - Standalone Apply path:

        text / reference conditioning
        -> JLC Flux2 ControlNet Apply
        -> JLC Flux2 ControlNet In/Out-Paint Adapter Experimental
        -> guider / sampler

  - Multi-ControlNet path:

        text / reference conditioning
        -> JLC Flux2 ControlNet Orchestrator
        -> JLC Flux2 ControlNet In/Out-Paint Adapter Experimental
        -> guider / sampler

- Single and Composed Control Behavior
  - For a standalone Apply result, the upgraded control remains the
    sampler-visible object and retains its native Flux2 residual-injection hook.

  - For a non-recursive composed control, the adapter attaches one shared
    inpaint context only to the first active child. Other active children remain
    ordinary control-only branches, and the composition wrapper retains
    ownership of the single shared injection hook.

  - No recursive ``previous_controlnet`` chain is created.

- Mask and Image Contract
  - User-facing mask polarity is fixed:

        white / 1.0 = editable or regenerate
        black / 0.0 = preserve or retain

  - The mask is thresholded at ``>= 0.5`` and remains hard and binary.

  - Editable source-image pixels are replaced with pixel-space ``0.5`` before
    VAE encoding, which maps to neutral model-space zero under ComfyUI's VAE
    normalization.

  - The packed ControlNet mask lanes use the inverse hard keep-mask, resized
    with nearest-exact sampling and patchified into four 2x2 lanes.

  - The ordinary ControlNet hint remains separate from the edit canvas and
    mask. Reference-image tokens remain supported and receive exact-zero
    260-channel ControlNet padding where required by the runtime sequence.

- Cache and Model-Management Behavior
  - Prepared inpaint context can be pre-warmed in the shared bounded CPU
    cache. Runtime falls back to inline preparation on a cache miss.

  - IMAGE and MASK must already match the active sampling canvas exactly.
    Silent spatial resizing is rejected with a clear error.

  - PyTorch inference tensors are treated as versionless without accessing an
    unavailable autograd version counter.

  - DynamicVRAM/model-loading state is restored after masked-source VAE
    encoding. Existing hint-latent and reference-latent caches are unchanged.

- Canonical Inputs
  - The visible source-image input is named ``image`` and appears above
    ``mask``. No compatibility alias is retained for the earlier experimental
    ``edit_canvas_image`` name. Recreate or reconnect older experimental nodes.

- Experimental Limitations
  - The release deliberately retains the hard-mask contract. Experimental mask
    expansion and feathering were removed after testing produced visible
    mask-shaped gray artifacts.

  - Seed-variable boundary or contour artifacts may still occur. Dense
    appearance-derived auxiliary controls, excessive strengths, and long
    activation ranges may overpower prompt, reference, or inpaint behavior.

  - Continue validating mask geometry, ControlNet modality balance, reference
    conditioning, batch execution, inpainting/outpainting layouts, and changing
    ComfyUI model-management behavior.

- Attribution and License
  - Concept and implementation by **J. L. Córdova**, with development
    assistance from **ChatGPT (OpenAI)**.

  - Designed to interoperate with ComfyUI ControlNet and sampler interfaces:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova
  - Released under the **MIT License**.
"""

from __future__ import annotations

from ..jlc_flux2_controlnet_versions import JLC_FLUX2_CONTROLNET_VERSION
from ..jlc_flux2_controlnet.control import JLCFlux2Control
from ..jlc_flux2_controlnet.composition import JLCFlux2ComposedControl
from ..jlc_flux2_controlnet.inpaint_control import JLCFlux2InpaintControl


EXPERIMENTAL_VERSION = "0.4.0"


MANIFEST = {
    "name": "JLC Flux2 ControlNet In/Out-Paint Adapter Experimental",
    "version": EXPERIMENTAL_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Experimental downstream FLUX.2 mask-aware ControlNet adapter for "
        "standalone JLC Flux2 ControlNet Apply workflows and validated "
        "non-recursive multi-ControlNet Orchestrator workflows. It attaches "
        "one hard binary editable-region mask, one source image, and one "
        "VAE-encoded neutral-filled masked-source latent to the first active "
        "ControlNet branch while preserving structural hints, native reference "
        "conditioning, strength/timestep policy, DynamicVRAM restoration, and "
        "the clean sampler-latent workflow."
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
        "reference_image_compatibility",
        "hard_binary_mask_contract",
        "inference_tensor_safe_context_cache",
        "shared_inpaint_context_cpu_cache",
        "strict_canvas_geometry_validation",
        "dynamic_vram_restore",
    ),
    "mask_contract": {
        "user_white": "editable_or_regenerate",
        "user_black": "preserve_or_keep",
        "threshold": 0.5,
        "masked_source_pixel_fill": 0.5,
        "packed_mask": "inverse_hard_keep_mask_2x2_patchified",
    },
    "workflow_position": (
        "JLC Flux2 ControlNet Apply or Orchestrator",
        "JLC Flux2 ControlNet In/Out-Paint Adapter Experimental",
        "Guider or Sampler",
    ),
    "canonical_inputs": ("image", "mask"),
    "known_limitations": (
        "hard_mask_boundary_artifacts_may_be_seed_variable",
        "dense_or_high_strength_controls_may_overpower_inpaint_or_reference_guidance",
        "mask_expansion_and_feathering_not_included_due_to_visible_artifacts",
        "image_and_mask_must_exactly_match_the_sampling_canvas",
    ),
    "status": "experimental",
    "release_track": "Non-Recursive ControlNet 2.0.0",
    "base_package_version": JLC_FLUX2_CONTROLNET_VERSION,
    "license": "MIT",
}


def _validate_image_mask_pair(image, mask, *, node_name: str) -> None:
    """Fail at node execution when IMAGE and MASK disagree with each other.

    Full target-canvas validation occurs in the inpaint context/cache layer,
    where the active sampling latent geometry is known.
    """

    if not hasattr(image, "shape") or len(image.shape) != 4 or image.shape[-1] < 3:
        raise ValueError(
            f"{node_name} expected IMAGE in BHWC layout, got {getattr(image, 'shape', None)}."
        )
    image_hw = (int(image.shape[1]), int(image.shape[2]))

    if not hasattr(mask, "shape"):
        raise ValueError(f"{node_name} expected a MASK tensor.")
    if len(mask.shape) == 2:
        mask_hw = (int(mask.shape[0]), int(mask.shape[1]))
    elif len(mask.shape) == 3:
        mask_hw = (int(mask.shape[-2]), int(mask.shape[-1]))
    elif len(mask.shape) == 4:
        if mask.shape[-1] == 1:
            mask_hw = (int(mask.shape[1]), int(mask.shape[2]))
        else:
            mask_hw = (int(mask.shape[-2]), int(mask.shape[-1]))
    else:
        raise ValueError(
            f"{node_name} received unsupported MASK shape {tuple(mask.shape)}."
        )

    if image_hw != mask_hw:
        raise ValueError(
            f"{node_name} requires IMAGE and MASK to have identical spatial "
            f"dimensions. Received image={image_hw[1]}x{image_hw[0]}, "
            f"mask={mask_hw[1]}x{mask_hw[0]}."
        )


def _finalize_detached_child(child, diagnostics):
    child.diagnostics_enabled = bool(diagnostics)
    child.set_previous_controlnet(None)
    return child


def _as_inpaint_child(control, *, vae, mask, image, diagnostics):
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
        image=image,
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


def _upgrade_control_object(control, *, vae, mask, image, diagnostics):
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
                image=image,
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
            image=image,
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
    image,
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
                image=image,
                diagnostics=diagnostics,
            )

        metadata_copy["control"] = upgraded_by_identity[key]
        metadata_copy["control_apply_to_uncond"] = False
        output.append([tensor, metadata_copy])
    return output


class JLCFlux2ControlNetInpaintAdapter:
    EXPERIMENTAL = True
    """Attach one mask-aware context to existing JLC control conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
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
        image,
        mask,
        diagnostics,
    ):
        _validate_image_mask_pair(
            image, mask, node_name="JLC Flux2 ControlNet In/Out-Paint Adapter"
        )
        output = _upgrade_conditioning(
            conditioning,
            vae=vae,
            mask=mask,
            image=image,
            diagnostics=diagnostics,
            upgraded_by_identity={},
            node_name="JLC Flux2 ControlNet In/Out-Paint Adapter",
        )
        return (output, vae)


class JLCFlux2ControlNetInpaintAdapterAdvanced:
    EXPERIMENTAL = True
    """Attach one shared mask-aware context to positive and negative streams."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
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
        image,
        mask,
        diagnostics,
    ):
        _validate_image_mask_pair(
            image, mask, node_name="JLC Flux2 ControlNet In/Out-Paint Adapter Advanced"
        )
        upgraded_by_identity = {}
        positive_out = _upgrade_conditioning(
            positive,
            vae=vae,
            mask=mask,
            image=image,
            diagnostics=diagnostics,
            upgraded_by_identity=upgraded_by_identity,
            node_name="JLC Flux2 ControlNet In/Out-Paint Adapter Advanced",
        )
        negative_out = _upgrade_conditioning(
            negative,
            vae=vae,
            mask=mask,
            image=image,
            diagnostics=diagnostics,
            upgraded_by_identity=upgraded_by_identity,
            node_name="JLC Flux2 ControlNet In/Out-Paint Adapter Advanced",
        )
        return (positive_out, negative_out, vae)
