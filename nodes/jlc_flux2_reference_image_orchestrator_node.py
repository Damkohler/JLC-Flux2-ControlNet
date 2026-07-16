"""
JLC Flux2 Reference Image Orchestrator
--------------------------------------

- JLC Flux2 ControlNet
  - This node is part of the **JLC Flux2 ControlNet** package developed
    by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/JLC-Flux2-ControlNet

- Node Purpose
  - The **JLC Flux2 Reference Image Orchestrator** applies one to ten
    independently prepared reference images through ComfyUI's native FLUX.2
    `reference_latents` conditioning mechanism.

  - The node:
        • accepts up to ten dynamically exposed reference-image slots
        • provides one explicit enabled/disabled toggle per visible slot
        • assumes all resizing, cropping, and image preparation occur upstream
        • VAE-encodes each enabled and connected image independently
        • appends reference latents in original slot order
        • can attach references to positive conditioning, negative conditioning,
          or both
        • optionally sets the native FLUX.2 reference-latent method
        • reports concrete cache and slot diagnostics

- Native Conditioning Contract
  - Each enabled reference is appended with the same conditioning operation used
    by ComfyUI's stock ReferenceLatent node:

        conditioning_set_values(
            conditioning,
            {"reference_latents": [latent]},
            append=True,
        )

  - The implementation performs no reference-strength scaling, VAE-latent
    multiplication, averaging, pooling, attention weighting, token fusion, or
    ControlNet-style residual composition.

  - `enabled_N = False` is an exact slot omission. The image is not validated,
    hashed, retrieved from cache, VAE-encoded, or appended to conditioning.

  - `slot_count` is authoritative. Slots above it are ignored even when an older
    workflow contains stale hidden values or connections. Empty slots within the
    visible range are skipped without promoting later references forward.

  - When no enabled reference image is connected, the input conditioning objects
    are returned unchanged and no reference method is attached.

- Reference-Latent Cache
  - The only non-native conditioning enhancement is the bounded JLC
    reference-latent CPU cache.

  - On a cold cache request, the node:
        • keys the final upstream-prepared BHWC RGB image and VAE identity
        • VAE-encodes the image
        • stores a detached CPU latent when cache capacity permits

  - On a warm request, the node clones the cached CPU latent and avoids repeated
    VAE encoding. Conditioning outputs retain detached CPU latents so ComfyUI's
    node-output cache does not unnecessarily retain VAE GPU tensors.

  - This contract aligns with **JLC Flux2 Reference Latent Cache Prep** whenever
    both nodes receive the same final upstream-prepared image tensors and VAE.
    Cache identity is method-agnostic because native reference-method selection
    is applied later as conditioning metadata and does not alter VAE encoding.

- Workflow Role
  - Use this node as the stable native multi-reference path for FLUX.2.

  - Future weighted, fused, pooled, gated, or otherwise non-native reference
    methods are intentionally outside this version's implementation and should
    be introduced only as separately named, explicitly experimental modes after
    architectural validation.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Built for interoperability with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova

  - Released under the **MIT License**.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import torch

import node_helpers

from ..jlc_flux2_controlnet_versions import JLC_FLUX2_CONTROLNET_VERSION

try:
    from ..jlc_flux2_controlnet.constants import PROJECT_LOG_PREFIX
except Exception:  # pragma: no cover - permits standalone syntax inspection.
    PROJECT_LOG_PREFIX = "[JLC Flux2]"

try:
    from ..jlc_flux2_controlnet.reference_latent_cache import (
        REFERENCE_LATENT_CACHE,
        make_reference_latent_cache_key,
        reference_latent_cache_info,
    )
except Exception:  # pragma: no cover - permits helper-local inspection.
    from reference_latent_cache import (  # type: ignore
        REFERENCE_LATENT_CACHE,
        make_reference_latent_cache_key,
        reference_latent_cache_info,
    )


REFERENCE_IMAGE_ORCHESTRATOR_VERSION = "1.0.1"

MANIFEST = {
    "name": "JLC Flux2 Reference Image Orchestrator",
    "version": REFERENCE_IMAGE_ORCHESTRATOR_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Stable native FLUX.2 multi-reference orchestrator with up to ten "
        "dynamic reference-image slots, exact per-slot enable/disable behavior, "
        "native slot-ordered reference_latents appending, positive/negative "
        "routing, optional native reference-method selection, and a bounded "
        "method-agnostic CPU reference-latent cache that avoids repeated VAE "
        "encoding across native reference-method changes. No reference "
        "weighting, latent scaling, fusion, pooling, or non-native "
        "composition is performed."
    ),
    "base_package_version": JLC_FLUX2_CONTROLNET_VERSION,
    "status": "stable",
    "license": "MIT",
}


MAX_REFERENCE_SLOTS = 10
APPLY_TO_MODES = ["positive_and_negative", "positive_only", "negative_only"]
REFERENCE_METHODS = [
    "do_not_set",
    "offset",
    "index",
    "uxo/uno",
    "index_timestep_zero",
]
_REFERENCE_CACHE_RESIZE_MODE = "none"
_REFERENCE_CACHE_UPSCALE_METHOD = "external"
_REFERENCE_CACHE_CROP_MODE = "external"
_REFERENCE_PREP_CONTRACT = "upstream_prepared_image"
_NATIVE_CONDITIONING_CONTRACT = "stock_reference_latents_append"


def _empty_preview_image() -> torch.Tensor:
    return torch.zeros((1, 1, 1, 3), dtype=torch.float32)


def _safe_reference_image(image: torch.Tensor) -> torch.Tensor:
    """Return the final upstream-prepared BHWC RGB tensor for ``vae.encode``."""

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected IMAGE tensor, got {type(image)!r}.")
    if image.ndim != 4:
        raise ValueError(
            f"Expected IMAGE tensor in BHWC format, got shape {tuple(image.shape)}."
        )
    if image.shape[-1] < 3:
        raise ValueError(
            "Expected IMAGE tensor with at least 3 channels, "
            f"got shape {tuple(image.shape)}."
        )
    return image[:, :, :, :3].contiguous()


def _normalize_reference_method(reference_latents_method: str) -> Optional[str]:
    """Translate the user-facing dropdown label to ComfyUI's native value."""

    if reference_latents_method == "do_not_set":
        return None
    if any(alias in reference_latents_method for alias in ("uxo", "uso", "uno")):
        return "uxo"
    return str(reference_latents_method)


def _conditioning_set_reference_method(
    conditioning: Any,
    reference_latents_method: str,
) -> Any:
    normalized = _normalize_reference_method(reference_latents_method)
    if normalized is None:
        return conditioning
    return node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents_method": normalized},
    )


def _append_reference_latent(conditioning: Any, latent: torch.Tensor) -> Any:
    """Mirror ComfyUI's stock ReferenceLatent append operation exactly."""

    return node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents": [latent]},
        append=True,
    )


def _cpu_reference_latent_for_conditioning(latent: torch.Tensor) -> torch.Tensor:
    """Return a detached CPU-owned latent safe for conditioning and caching."""

    cpu_latent = latent.detach().to(device="cpu").contiguous()
    if latent.device.type == "cpu":
        cpu_latent = cpu_latent.clone()
    cpu_latent.requires_grad_(False)
    return cpu_latent


def _enabled_value(kwargs: dict[str, Any], index: int) -> bool:
    return bool(kwargs.get(f"enabled_{index}", True))


def _image_value(kwargs: dict[str, Any], index: int) -> Optional[torch.Tensor]:
    return kwargs.get(f"reference_image_{index}")


class JLCFlux2ReferenceImageOrchestrator:
    """Native FLUX.2 multi-reference conditioning with CPU latent caching."""

    @classmethod
    def INPUT_TYPES(cls):
        required: dict[str, Any] = {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE",),
            "apply_to": (
                APPLY_TO_MODES,
                {
                    "default": "positive_and_negative",
                    "tooltip": (
                        "Attach the same native reference-latent sequence to "
                        "positive conditioning, negative conditioning, or both."
                    ),
                },
            ),
            "reference_latents_method": (
                REFERENCE_METHODS,
                {
                    "default": "do_not_set",
                    "tooltip": (
                        "Optional equivalent of ComfyUI's native Edit Model "
                        "Reference Method node."
                    ),
                },
            ),
            "slot_count": (
                "INT",
                {
                    "default": 2,
                    "min": 1,
                    "max": MAX_REFERENCE_SLOTS,
                    "step": 1,
                    "tooltip": (
                        "Number of visible reference-image slots. The backend "
                        "ignores every slot above this count."
                    ),
                },
            ),
        }

        for index in range(1, MAX_REFERENCE_SLOTS + 1):
            required[f"enabled_{index}"] = (
                "BOOLEAN",
                {
                    "default": True,
                    "tooltip": (
                        "When disabled, this slot is omitted before image "
                        "validation, cache lookup, VAE encoding, or conditioning."
                    ),
                },
            )

        required.update(
            {
                "cache_enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Use the bounded CPU reference-latent cache to avoid "
                            "repeated VAE encoding of unchanged prepared images."
                        ),
                    },
                ),
                "cache_max_entries": (
                    "INT",
                    {
                        "default": 32,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                    },
                ),
                "cache_max_cpu_mb": (
                    "INT",
                    {
                        "default": 256,
                        "min": 0,
                        "max": 4096,
                        "step": 16,
                    },
                ),
                "clear_cache_before_run": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Clear the shared reference-latent cache first.",
                    },
                ),
                "diagnostics": ("BOOLEAN", {"default": True}),
            }
        )

        optional: dict[str, Any] = {}
        for index in range(1, MAX_REFERENCE_SLOTS + 1):
            optional[f"reference_image_{index}"] = ("IMAGE",)

        return {"required": required, "optional": optional}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "IMAGE", "STRING")
    RETURN_NAMES = (
        "positive",
        "negative",
        "first_reference_image",
        "diagnostics_json",
    )
    FUNCTION = "apply"
    CATEGORY = "Flux2 Conditioning"
    DESCRIPTION = (
        "Stable native FLUX.2 multi-reference orchestrator. Enabled images are "
        "VAE-encoded or loaded from the JLC CPU reference-latent cache, then "
        "appended in slot order through native reference_latents conditioning."
    )

    def _encode_one_reference(
        self,
        *,
        image: torch.Tensor,
        vae: Any,
        slot_index: int,
        cache_active: bool,
        diagnostics: bool,
        stats: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        final_image = _safe_reference_image(image)
        target_height = int(final_image.shape[1])
        target_width = int(final_image.shape[2])

        request = None
        cached = None
        if cache_active:
            request = make_reference_latent_cache_key(
                image=final_image,
                vae=vae,
                resize_mode=_REFERENCE_CACHE_RESIZE_MODE,
                upscale_method=_REFERENCE_CACHE_UPSCALE_METHOD,
                target_width=target_width,
                target_height=target_height,
                target_megapixels=None,
                crop_mode=_REFERENCE_CACHE_CROP_MODE,
            )
            cached = REFERENCE_LATENT_CACHE.get(
                request,
                diagnostics=diagnostics,
            )

        if cached is not None:
            stats["cache_hits"] += 1
            source = "cache_hit"
            latent_for_conditioning = cached.clone()
        else:
            if cache_active:
                stats["cache_misses"] += 1
                source = "vae_encode_cache_miss"
            else:
                stats["uncached_vae_encodes"] += 1
                source = "vae_encode_cache_inactive"

            if diagnostics:
                logging.info(
                    "%s Reference slot %d VAE encode: image_shape=%s, key=%s.",
                    PROJECT_LOG_PREFIX,
                    slot_index,
                    tuple(final_image.shape),
                    None if request is None else request.short_key,
                )

            encoded = vae.encode(final_image)

            if cache_active and request is not None:
                inserted = REFERENCE_LATENT_CACHE.put(
                    request,
                    encoded,
                    diagnostics=diagnostics,
                )
                if inserted:
                    stats["cache_inserts"] += 1
                else:
                    stats["cache_insert_skips"] += 1

            latent_for_conditioning = _cpu_reference_latent_for_conditioning(encoded)

        slot_info = {
            "slot": slot_index,
            "enabled": True,
            "source": source,
            "cache_key": None if request is None else request.short_key,
            "image_shape": list(final_image.shape),
            "latent_shape": list(latent_for_conditioning.shape),
            "latent_dtype": str(latent_for_conditioning.dtype),
            "latent_device": str(latent_for_conditioning.device),
            "reference_preprocess_contract": _REFERENCE_PREP_CONTRACT,
            "resize_mode": _REFERENCE_CACHE_RESIZE_MODE,
            "upscale_method": _REFERENCE_CACHE_UPSCALE_METHOD,
            "target_width": target_width,
            "target_height": target_height,
            "target_megapixels": None,
            "crop_mode": _REFERENCE_CACHE_CROP_MODE,
        }
        return latent_for_conditioning, final_image, slot_info

    def apply(
        self,
        positive,
        negative,
        vae,
        apply_to,
        reference_latents_method,
        slot_count=2,
        cache_enabled=True,
        cache_max_entries=32,
        cache_max_cpu_mb=256,
        clear_cache_before_run=False,
        diagnostics=True,
        **kwargs,
    ):
        if vae is None:
            raise ValueError(
                "JLC Flux2 Reference Image Orchestrator requires a VAE."
            )

        slot_count = max(1, min(MAX_REFERENCE_SLOTS, int(slot_count)))
        cache_max_entries = max(0, int(cache_max_entries))
        cache_max_cpu_bytes = max(0, int(cache_max_cpu_mb)) * 1024 * 1024

        if clear_cache_before_run:
            REFERENCE_LATENT_CACHE.clear(
                reason="node_clear_before_run",
                diagnostics=bool(diagnostics),
            )

        REFERENCE_LATENT_CACHE.configure(
            max_entries=cache_max_entries,
            max_cpu_bytes=cache_max_cpu_bytes,
            enabled=bool(cache_enabled),
            diagnostics=bool(diagnostics),
        )
        cache_active = bool(cache_enabled) and bool(
            REFERENCE_LATENT_CACHE.is_enabled()
        )

        stats: dict[str, Any] = {
            "node_version": REFERENCE_IMAGE_ORCHESTRATOR_VERSION,
            "base_package_version": JLC_FLUX2_CONTROLNET_VERSION,
            "conditioning_contract": _NATIVE_CONDITIONING_CONTRACT,
            "native_reference_latents": True,
            "reference_weighting": False,
            "reference_fusion": False,
            "apply_to": apply_to,
            "reference_latents_method": reference_latents_method,
            "reference_latents_method_normalized": _normalize_reference_method(
                reference_latents_method
            ),
            "reference_preprocess_contract": _REFERENCE_PREP_CONTRACT,
            "slot_count": slot_count,
            "cache_enabled_requested": bool(cache_enabled),
            "cache_active": cache_active,
            "cache_key_method_agnostic": True,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_inserts": 0,
            "cache_insert_skips": 0,
            "uncached_vae_encodes": 0,
            "skipped_disabled_slots": 0,
            "skipped_empty_slots": 0,
            "ignored_slots_above_slot_count": MAX_REFERENCE_SLOTS - slot_count,
            "slots": [],
        }

        active_latents: list[torch.Tensor] = []
        first_reference_image: Optional[torch.Tensor] = None

        for slot_index in range(1, slot_count + 1):
            enabled = _enabled_value(kwargs, slot_index)
            image = _image_value(kwargs, slot_index)

            if not enabled:
                stats["skipped_disabled_slots"] += 1
                stats["slots"].append(
                    {
                        "slot": slot_index,
                        "enabled": False,
                        "source": "disabled_exact_omission",
                    }
                )
                continue

            if image is None:
                stats["skipped_empty_slots"] += 1
                stats["slots"].append(
                    {
                        "slot": slot_index,
                        "enabled": True,
                        "source": "empty_slot",
                    }
                )
                continue

            latent, final_image, slot_info = self._encode_one_reference(
                image=image,
                vae=vae,
                slot_index=slot_index,
                cache_active=cache_active,
                diagnostics=bool(diagnostics),
                stats=stats,
            )
            stats["slots"].append(slot_info)
            active_latents.append(latent)

            if first_reference_image is None:
                first_reference_image = final_image

        output_positive = positive
        output_negative = negative

        # Exact all-slots-bypassed behavior: leave both conditioning objects
        # untouched, including the reference_latents_method field.
        if active_latents:
            if apply_to in {"positive_and_negative", "positive_only"}:
                output_positive = _conditioning_set_reference_method(
                    output_positive,
                    reference_latents_method,
                )
                for latent in active_latents:
                    output_positive = _append_reference_latent(
                        output_positive,
                        latent,
                    )

            if apply_to in {"positive_and_negative", "negative_only"}:
                output_negative = _conditioning_set_reference_method(
                    output_negative,
                    reference_latents_method,
                )
                for latent in active_latents:
                    output_negative = _append_reference_latent(
                        output_negative,
                        latent,
                    )

        stats["active_reference_count"] = len(active_latents)
        stats["conditioning_unchanged"] = len(active_latents) == 0
        stats["cache_info"] = reference_latent_cache_info()

        if diagnostics:
            logging.info(
                "%s Reference Image Orchestrator: active_refs=%d, disabled=%d, "
                "empty=%d, hits=%d, misses=%d, inserts=%d, uncached_encodes=%d.",
                PROJECT_LOG_PREFIX,
                len(active_latents),
                stats["skipped_disabled_slots"],
                stats["skipped_empty_slots"],
                stats["cache_hits"],
                stats["cache_misses"],
                stats["cache_inserts"],
                stats["uncached_vae_encodes"],
            )

        diagnostics_json = json.dumps(stats, sort_keys=True, indent=2)
        return (
            output_positive,
            output_negative,
            (
                first_reference_image
                if first_reference_image is not None
                else _empty_preview_image()
            ),
            diagnostics_json,
        )


NODE_CLASS_MAPPINGS = {
    "JLCFlux2ReferenceImageOrchestrator": JLCFlux2ReferenceImageOrchestrator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JLCFlux2ReferenceImageOrchestrator": (
        "\u2003JLC Flux2 Reference Image Orchestrator"
    ),
}
