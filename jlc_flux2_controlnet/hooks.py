"""Stateless, per-invocation Flux.2 ControlNet integration.

No ComfyUI method or class is replaced globally. The wrapper is delivered by a
native TransformerOptionsHook. Single controls and explicit non-recursive
composites share the same native double-block injection seam.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch

import comfy.hooks
import comfy.patcher_extension

from .constants import (
    CONTROL_LAYERS,
    EXPECTED_FLUX2_LATENT_CHANNELS,
    EXPECTED_FLUX2_PATCH_SIZE,
    PROJECT_LOG_PREFIX,
    REQUESTS_KEY,
    WRAPPER_KEY,
)


def _request_owner(request: dict):
    return request.get("owner") or request.get("control")


def _unique_owners(requests: list[dict]):
    owners = []
    seen = set()
    for request in requests:
        owner = _request_owner(request)
        if owner is None or id(owner) in seen:
            continue
        seen.add(id(owner))
        owners.append(owner)
    return owners


def _validate_flux2_module(module, requests: list[dict]) -> None:
    """Fail early when the attached base model cannot accept this side branch."""
    params = getattr(module, "params", None)
    if params is None or not getattr(params, "global_modulation", False):
        raise RuntimeError(
            "JLC Flux2 ControlNet requires a native Flux.2 diffusion model with global modulation."
        )

    if getattr(module, "patch_size", None) != EXPECTED_FLUX2_PATCH_SIZE:
        raise RuntimeError(
            f"JLC Flux2 ControlNet expected Flux.2 patch_size={EXPECTED_FLUX2_PATCH_SIZE}, "
            f"got {getattr(module, 'patch_size', None)}."
        )

    if getattr(module, "in_channels", None) != EXPECTED_FLUX2_LATENT_CHANNELS:
        raise RuntimeError(
            f"JLC Flux2 ControlNet expected {EXPECTED_FLUX2_LATENT_CHANNELS} model input channels, "
            f"got {getattr(module, 'in_channels', None)}."
        )

    base_hidden = getattr(module, "hidden_size", None)
    base_heads = getattr(module, "num_heads", None)
    double_blocks = getattr(module, "double_blocks", None)
    if double_blocks is None or len(double_blocks) <= max(CONTROL_LAYERS):
        raise RuntimeError(
            f"JLC Flux2 ControlNet requires native double blocks through index {max(CONTROL_LAYERS)}."
        )

    for request in requests:
        control = request.get("control")
        control_model = getattr(control, "control_model", None)
        if control_model is None:
            continue

        control_hidden = getattr(control_model, "hidden_size", None)
        control_heads = getattr(control_model, "num_attention_heads", None)
        if control_heads is None:
            blocks = getattr(control_model, "control_transformer_blocks", None)
            if blocks is not None and len(blocks) > 0:
                control_heads = getattr(blocks[0].attn, "heads", None)
        if base_hidden != control_hidden:
            raise RuntimeError(
                "JLC Flux2 ControlNet/base-model hidden-size mismatch: "
                f"base={base_hidden}, control={control_hidden}. "
                "Use the FLUX.2-dev family expected by this checkpoint."
            )
        if base_heads != control_heads:
            raise RuntimeError(
                "JLC Flux2 ControlNet/base-model attention-head mismatch: "
                f"base={base_heads}, control={control_heads}. "
                "Use the FLUX.2-dev family expected by this checkpoint."
            )


def _run_side_branches_once(request_states, args):
    """Evaluate each active branch once, explicitly and non-recursively."""
    active_requests = [
        state["request"]
        for state in request_states
        if state["request"].get("control") is not None
        and float(state["request"].get("strength", 1.0)) != 0.0
    ]

    by_owner: dict[int, list[dict]] = {}
    owner_objects = {}
    for request in active_requests:
        owner = _request_owner(request)
        if owner is None:
            continue
        owner_id = id(owner)
        owner_objects[owner_id] = owner
        by_owner.setdefault(owner_id, []).append(request)
    for owner_id, owner_requests in by_owner.items():
        if len(owner_requests) > 1:
            note = getattr(owner_objects[owner_id], "note_composition", None)
            if note is not None:
                note(owner_requests)

    multiple = len(active_requests) > 1
    for state in request_states:
        if state["executed"]:
            continue

        request = state["request"]
        control = request.get("control")
        strength = float(request.get("strength", 1.0))
        if control is None or strength == 0.0:
            state["executed"] = True
            continue

        residuals = control.execute_side_branch(
            request,
            img=args["img"],
            txt=args["txt"],
            vec=args["vec"],
            pe=args["pe"],
            attn_mask=args.get("attn_mask"),
        )
        if len(residuals) != len(CONTROL_LAYERS):
            raise RuntimeError(
                f"Expected {len(CONTROL_LAYERS)} Flux2 control residuals, got {len(residuals)}."
            )

        state["residuals"] = residuals
        state["executed"] = True
        control.note_diagnostic_sidebranch(request, args["img"], args["txt"], residuals)

        # Match the established JLC non-recursive ownership discipline: one
        # child finishes before the next child is evaluated.
        if multiple and args["img"].is_cuda:
            torch.cuda.synchronize(device=args["img"].device)


def _validated_residual(block_index: int, image, request: dict, residual):
    if residual.ndim != image.ndim:
        raise RuntimeError(
            f"Flux2 control residual rank mismatch at block {block_index}: "
            f"residual={tuple(residual.shape)}, image={tuple(image.shape)}."
        )
    if residual.shape[0] != image.shape[0] or residual.shape[-1] != image.shape[-1]:
        raise RuntimeError(
            f"Flux2 control residual shape mismatch at block {block_index}: "
            f"residual={tuple(residual.shape)}, image={tuple(image.shape)}."
        )
    if residual.shape[1] > image.shape[1]:
        raise RuntimeError(
            f"Flux2 control residual has more image tokens than the native block at block {block_index}: "
            f"residual={tuple(residual.shape)}, image={tuple(image.shape)}."
        )
    if residual.device != image.device or residual.dtype != image.dtype:
        residual = residual.to(device=image.device, dtype=image.dtype)
    return residual


def _apply_block_residuals(block_index: int, out: dict, request_states):
    """Combine child residuals explicitly, then inject once into the native block."""
    residual_index = CONTROL_LAYERS.index(block_index)
    image = out.get("img")
    if image is None:
        raise RuntimeError(
            f"Flux2 double-block replacement for block {block_index} did not return an 'img' tensor."
        )

    participants = []
    for state in request_states:
        request = state["request"]
        control = request.get("control")
        residuals = state.get("residuals")
        if control is None or residuals is None:
            continue

        residual = residuals[residual_index]
        residual = _validated_residual(block_index, image, request, residual)
        participants.append(
            {
                "request": request,
                "control": control,
                "owner": _request_owner(request),
                "strength": float(request.get("strength", 1.0)),
                "residual": residual,
            }
        )

    if len(participants) == 1:
        # Preserve the already-validated single-ControlNet path exactly.
        participant = participants[0]
        residual = participant["residual"]
        strength = participant["strength"]
        image[:, : residual.shape[1]].add_(residual, alpha=strength)
        participant["control"].note_diagnostic_injection(
            block_index=block_index,
            strength=strength,
            residual=residual,
        )
    elif len(participants) > 1:
        # Take ownership only once: clone the first child residual, scale it,
        # then accumulate later children in-place before one native injection.
        first = participants[0]
        combined = first["residual"].clone()
        if first["strength"] != 1.0:
            combined.mul_(first["strength"])

        for participant in participants[1:]:
            residual = participant["residual"]
            if residual.shape != combined.shape:
                raise RuntimeError(
                    f"Flux2 composed residual shape mismatch at block {block_index}: "
                    f"first={tuple(combined.shape)}, next={tuple(residual.shape)}."
                )
            combined.add_(residual, alpha=participant["strength"])

        image[:, : combined.shape[1]].add_(combined)

        owners = []
        seen = set()
        for participant in participants:
            owner = participant["owner"]
            if owner is None or id(owner) in seen:
                continue
            seen.add(id(owner))
            owners.append(owner)

        strengths = [participant["strength"] for participant in participants]
        composed_notified = False
        for owner in owners:
            note = getattr(owner, "note_composed_injection", None)
            if note is not None:
                note(
                    block_index=block_index,
                    strengths=strengths,
                    residual=combined,
                )
                composed_notified = True

        # Defensive fallback for unsupported manually chained single controls.
        if not composed_notified:
            for participant in participants:
                participant["control"].note_diagnostic_injection(
                    block_index=block_index,
                    strength=participant["strength"],
                    residual=participant["residual"],
                )

    # Release each block's child residual after its final use.
    for state in request_states:
        residuals = state.get("residuals")
        if residuals is not None:
            residuals[residual_index] = None

    out["img"] = image
    return out


def _make_injection_replacement(
    block_index: int,
    existing_replacement: Callable | None,
    request_states,
):
    """Compose with any existing replacement, then inject the JLC residual."""

    def injection_replacement(args, extra_options):
        if block_index == CONTROL_LAYERS[0]:
            _run_side_branches_once(request_states, args)

        if existing_replacement is not None:
            out = existing_replacement(args, extra_options)
        else:
            out = extra_options["original_block"](args)

        return _apply_block_residuals(block_index, out, request_states)

    return injection_replacement


def flux2_injection_wrapper(
    executor,
    x,
    timestep,
    context,
    y=None,
    guidance=None,
    ref_latents=None,
    control=None,
    transformer_options=None,
    **kwargs,
):
    """Inject compact-sidecar residuals at double blocks 0, 2, 4 and 6."""
    if transformer_options is None:
        transformer_options = {}

    requests = transformer_options.get(REQUESTS_KEY, [])
    if not requests:
        return executor(
            x,
            timestep,
            context,
            y,
            guidance,
            ref_latents,
            control,
            transformer_options,
            **kwargs,
        )

    _validate_flux2_module(executor.class_obj, requests)

    owners = _unique_owners(requests)
    for owner in owners:
        note = getattr(owner, "note_diagnostic_wrapper", None)
        if note is not None:
            note(x, context, ref_latents)

    # Native Flux.2 appends reference image tokens before entering the double
    # blocks. Each JLC side branch expands its raw 260-channel context with the
    # matching exact-zero suffix and returns full-sequence residuals.

    local_options = transformer_options.copy()
    patches_replace = comfy.patcher_extension.copy_nested_dicts(
        transformer_options.get("patches_replace", {})
    )
    dit_replacements = patches_replace.setdefault("dit", {})

    # Mutable child residuals live only inside this denoising forward.
    request_states = [
        {"request": request, "executed": False, "residuals": None}
        for request in requests
    ]

    for block_index in CONTROL_LAYERS:
        key = ("double_block", block_index)
        existing = dit_replacements.get(key)
        dit_replacements[key] = _make_injection_replacement(
            block_index,
            existing,
            request_states,
        )

    local_options["patches_replace"] = patches_replace
    return executor(
        x,
        timestep,
        context,
        y,
        guidance,
        ref_latents,
        control,
        local_options,
        **kwargs,
    )


def make_injection_hook_group() -> comfy.hooks.HookGroup:
    """Create a clone-safe hook group containing one stateless wrapper."""
    transformer_dict: dict = {}
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        WRAPPER_KEY,
        flux2_injection_wrapper,
        transformer_dict,
    )

    group = comfy.hooks.HookGroup()
    group.add(
        comfy.hooks.TransformerOptionsHook(
            transformer_dict,
            hook_scope=comfy.hooks.EnumHookScope.AllConditioning,
        )
    )
    logging.debug("%s Residual-injection wrapper hook created.", PROJECT_LOG_PREFIX)
    return group
