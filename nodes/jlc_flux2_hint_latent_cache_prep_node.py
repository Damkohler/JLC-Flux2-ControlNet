"""
JLC Flux2 ControlNet Latents Cache
--------------------------------

- JLC Flux2 ControlNet
  - This node is part of the **JLC Flux2 ControlNet** package developed
    by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/JLC-Flux2-ControlNet

- Node Purpose
  - The **JLC Flux2 ControlNet Latents Cache** node pre-populates the bounded,
    process-local CPU cache used for reusable FLUX.2 ControlNet hint latents.

  - It performs only the validated hint-preparation path:

        IMAGE -> BCHW control hint -> common_upscale(center) -> VAE encode
              -> FLUX.2 latent-format process_in -> bounded CPU cache

  - It does not execute the FLUX.2 base model, the ControlNet side model,
    residual composition, injection hooks, or sampling.

- Workflow Role
  - Use this node in a mutually exclusive cache-preparation branch before
    running the normal inference branch in the same ComfyUI server session.

  - The node is intentionally **not** an output node. A downstream lazy Switch,
    Group Controller, or equivalent execution gate must request this branch.
    When the prep branch is inactive, this node does not independently pull its
    image or VAE dependencies into the execution graph.

  - `IS_CHANGED` returns NaN so the cache side effect is refreshed whenever the
    prep branch is actually requested, even when the visible inputs are unchanged.

- Cache Contract
  - Cached tensors are detached, contiguous CPU tensors keyed from the final
    control-hint tensor, target latent geometry, VAE identity, preprocessing
    callable, interpolation/crop contract, and FLUX.2 latent format.

  - `width` and `height` are user-facing output pixel dimensions. They replace
    the previous LATENT input and are converted internally to target latent
    geometry using the active VAE compression ratio.

  - The first IMAGE is returned unchanged for workflow routing.
    `cache_set` is True when all active connected slots are either cache hits
    or successful inserts. `cache_report` summarizes hits, misses, inserts,
    skips, entry count, and total cached CPU bytes.

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

import logging
from typing import Iterable

import torch

import comfy.controlnet
import comfy.latent_formats
import comfy.model_management
import comfy.utils

from ..jlc_flux2_controlnet.constants import PROJECT_LOG_PREFIX
from ..jlc_flux2_controlnet.hint_latent_cache import (
    HINT_LATENT_CACHE,
    clear_hint_latent_cache,
    hint_latent_cache_info,
    make_hint_latent_cache_key,
)


MANIFEST = {
    "name": "JLC Flux2 ControlNet Latents Cache",
    "version": JLC_FLUX2_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Pre-populates the bounded CPU cache for reusable FLUX.2 ControlNet "
        "hint latents without executing the base model, ControlNet side model, "
        "or sampler. The node is branch-driven rather than an independent "
        "output sink, allowing a lazy Switch or Group Controller to suppress "
        "the inactive preparation path."
    ),
}

_FLUX2_CONTROL_COMPRESSION_RATIO = 1
_MAX_HINT_SLOTS = 4


def _image_to_control_hint(image: torch.Tensor) -> torch.Tensor:
    """Convert ComfyUI IMAGE layout to the ControlBase cond_hint layout.

    ComfyUI IMAGE tensors are normally BHWC. ControlBase.set_cond_hint receives
    BCHW. The runtime apply path uses the BCHW form as cond_hint_original, so the
    prep node must do the same before building the cache key.
    """

    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError(
            "JLC Flux2 Hint Latent Cache Prep expected each IMAGE input to be a rank-4 tensor."
        )

    # Standard ComfyUI IMAGE layout: B,H,W,C.
    if image.shape[-1] in (1, 3, 4):
        return image.movedim(-1, 1).contiguous()

    # Defensive support for already-channel-first tensors supplied by custom nodes.
    if image.shape[1] in (1, 3, 4):
        return image.contiguous()

    # Prefer ComfyUI IMAGE semantics when the layout is ambiguous.
    return image.movedim(-1, 1).contiguous()


def _default_control_preprocess_callable():
    """Return the same identity preprocess callable shape used by ControlBase."""

    return comfy.controlnet.ControlBase().preprocess_image


def _default_control_upscale_algorithm() -> str:
    return str(comfy.controlnet.ControlBase().upscale_algorithm)


def _validate_target_geometry(width: int, height: int, compression_ratio: int) -> tuple[int, int, int, int]:
    pixel_width = int(width)
    pixel_height = int(height)
    ratio = max(1, int(compression_ratio))

    if pixel_width <= 0 or pixel_height <= 0:
        raise ValueError("JLC Flux2 Hint Latent Cache Prep requires width and height > 0.")

    if pixel_width % ratio != 0 or pixel_height % ratio != 0:
        raise ValueError(
            "JLC Flux2 Hint Latent Cache Prep width/height must be divisible by "
            f"the active VAE compression ratio ({ratio}). Got width={pixel_width}, height={pixel_height}."
        )

    expected_w = pixel_width // ratio
    expected_h = pixel_height // ratio
    return expected_w, expected_h, pixel_width, pixel_height


def _iter_slots(
    slot_count: int,
    control_image_1: torch.Tensor,
    control_image_2: torch.Tensor | None,
    control_image_3: torch.Tensor | None,
    control_image_4: torch.Tensor | None,
) -> Iterable[tuple[int, torch.Tensor]]:
    images = (control_image_1, control_image_2, control_image_3, control_image_4)
    active = max(1, min(_MAX_HINT_SLOTS, int(slot_count)))
    for index, image in enumerate(images[:active], start=1):
        if image is not None:
            yield index, image


class JLCFlux2HintLatentCachePrep:
    """Pre-warm the shared CPU hint-latent cache for one to four hints.

    This node always prepares when its workflow branch executes. Use a
    downstream lazy Switch / Group Controller branch to decide whether the prep
    path or the inference path should run.
    """

    CATEGORY = "Flux2 Latents Cache/utils"
    FUNCTION = "prepare"
    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("control_image_1", "cache_set", "cache_report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "control_image_1": ("IMAGE",),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": 16384,
                        "step": 16,
                        "tooltip": "Final output/image width used by the inference latent. Hint images may differ and will be resized to this width before VAE encoding.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": 16384,
                        "step": 16,
                        "tooltip": "Final output/image height used by the inference latent. Hint images may differ and will be resized to this height before VAE encoding.",
                    },
                ),
                "slot_count": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": _MAX_HINT_SLOTS,
                        "step": 1,
                    },
                ),
                "clear_before_prepare": ("BOOLEAN", {"default": False}),
                "diagnostics": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "control_image_2": ("IMAGE",),
                "control_image_3": ("IMAGE",),
                "control_image_4": ("IMAGE",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def prepare(
        self,
        vae,
        control_image_1,
        width=1024,
        height=1024,
        slot_count=4,
        clear_before_prepare=False,
        diagnostics=True,
        control_image_2=None,
        control_image_3=None,
        control_image_4=None,
    ):
        if clear_before_prepare:
            clear_hint_latent_cache(diagnostics=bool(diagnostics))

        if vae is None:
            raise ValueError(
                "JLC Flux2 Hint Latent Cache Prep requires a VAE so it can encode control hints."
            )

        if not HINT_LATENT_CACHE.is_enabled():
            report = "JLC Flux2 hint-latent cache is disabled or has zero capacity; prep skipped."
            if diagnostics:
                logging.info("%s %s", PROJECT_LOG_PREFIX, report)
            return (control_image_1, False, report)

        compression_ratio = _FLUX2_CONTROL_COMPRESSION_RATIO * int(
            vae.spacial_compression_encode()
        )
        expected_w, expected_h, target_pixel_width, target_pixel_height = _validate_target_geometry(
            int(width),
            int(height),
            compression_ratio,
        )

        latent_format = comfy.latent_formats.Flux2()
        preprocess_image = _default_control_preprocess_callable()
        upscale_algorithm = _default_control_upscale_algorithm()

        prepared = 0
        hits = 0
        misses = 0
        inserted = 0
        skipped = 0

        for slot_index, image in _iter_slots(
            slot_count,
            control_image_1,
            control_image_2,
            control_image_3,
            control_image_4,
        ):
            control_hint = _image_to_control_hint(image)
            cache_request = make_hint_latent_cache_key(
                image=control_hint,
                target_latent_width=expected_w,
                target_latent_height=expected_h,
                target_pixel_width=target_pixel_width,
                target_pixel_height=target_pixel_height,
                vae=vae,
                preprocess_image=preprocess_image,
                interpolation=upscale_algorithm,
                resize_mode="common_upscale",
                crop_mode="center",
                latent_format=latent_format,
            )

            cached = HINT_LATENT_CACHE.get(
                cache_request,
                diagnostics=bool(diagnostics),
            )
            if cached is not None:
                hits += 1
                prepared += 1
                continue

            misses += 1
            if diagnostics:
                logging.info(
                    "%s Hint-latent prep slot %d cache miss: encoding control image; key=%s.",
                    PROJECT_LOG_PREFIX,
                    slot_index,
                    cache_request.short_key,
                )

            hint = comfy.utils.common_upscale(
                control_hint,
                target_pixel_width,
                target_pixel_height,
                upscale_algorithm,
                "center",
            )
            hint = preprocess_image(hint)

            loaded_models = comfy.model_management.loaded_models(
                only_currently_used=True
            )
            try:
                hint = vae.encode(hint.movedim(1, -1))
            finally:
                comfy.model_management.load_models_gpu(loaded_models)

            hint = latent_format.process_in(hint)

            if HINT_LATENT_CACHE.put(
                cache_request,
                hint,
                diagnostics=bool(diagnostics),
            ):
                inserted += 1
                prepared += 1
            else:
                skipped += 1
                if diagnostics:
                    logging.info(
                        "%s Hint-latent prep slot %d cache insert skipped; key=%s.",
                        PROJECT_LOG_PREFIX,
                        slot_index,
                        cache_request.short_key,
                    )

        info = hint_latent_cache_info()
        report = (
            "JLC Flux2 hint-latent prep complete: "
            f"prepared={prepared}, hits={hits}, misses={misses}, inserted={inserted}, "
            f"skipped={skipped}, cache_entries={info['entry_count']}, "
            f"total_bytes={info['total_bytes']}."
        )
        if diagnostics:
            logging.info("%s %s", PROJECT_LOG_PREFIX, report)
        cache_set = prepared > 0 and skipped == 0
        return (control_image_1, bool(cache_set), report)
