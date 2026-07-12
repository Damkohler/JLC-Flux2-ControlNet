"""
JLC Flux2 Hint Latent Cache Prep
--------------------------------

- JLC Flux2 ControlNet
  - This node is part of the **JLC Flux2 ControlNet** package developed
    by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/JLC-Flux2-ControlNet

- Node Purpose
  - The **JLC Flux2 Hint Latent Cache Prep** node pre-populates the bounded,
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
    image, latent, or VAE dependencies into the execution graph.

  - `IS_CHANGED` returns NaN so the cache side effect is refreshed whenever the
    prep branch is actually requested, even when the visible inputs are unchanged.

- Cache Contract
  - Cached tensors are detached, contiguous CPU tensors keyed from the final
    control-hint tensor, target latent geometry, VAE identity, preprocessing
    callable, interpolation/crop contract, and FLUX.2 latent format.

  - The LATENT and VAE input/output are passthrough used for branch wiring and target
    geometry. The first IMAGE is also returned unchanged for workflow routing.

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
from typing import Any, Iterable

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
    "name": "JLC Flux2 Hint Latent Cache Prep",
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


def _latent_samples(latent: dict[str, Any]) -> torch.Tensor:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError(
            "JLC Flux2 Hint Latent Cache Prep requires a LATENT input with a 'samples' tensor."
        )
    samples = latent["samples"]
    if not isinstance(samples, torch.Tensor) or samples.ndim != 4:
        raise ValueError(
            "JLC Flux2 Hint Latent Cache Prep expected latent['samples'] to be a rank-4 tensor."
        )
    return samples


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
    """Return the same identity preprocess callable shape used by ControlBase.

    The hint-latent cache key includes the preprocess callable description. A
    fresh ControlBase instance gives the same default callable contract used by
    JLCFlux2Control unless future code deliberately overrides it.
    """

    return comfy.controlnet.ControlBase().preprocess_image


def _default_control_upscale_algorithm() -> str:
    return str(comfy.controlnet.ControlBase().upscale_algorithm)


def _iter_slots(
    slot_count: int,
    image_1: torch.Tensor,
    image_2: torch.Tensor | None,
    image_3: torch.Tensor | None,
    image_4: torch.Tensor | None,
) -> Iterable[tuple[int, torch.Tensor]]:
    images = (image_1, image_2, image_3, image_4)
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

    CATEGORY = "Flux2 ControlNet/Utilities"
    FUNCTION = "prepare"
    RETURN_TYPES = ("IMAGE", "LATENT", "STRING", "VAE")
    RETURN_NAMES = ("image_1", "latent", "cache_report", "vae" )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
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
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # This node's primary product is a process-local cache side effect. Run
        # whenever the prep branch is active so a cleared/reloaded cache is safely
        # repopulated even if graph inputs did not otherwise change.
        return float("nan")

    def prepare(
        self,
        image_1,
        latent,
        vae,
        slot_count=4,
        clear_before_prepare=False,
        diagnostics=True,
        image_2=None,
        image_3=None,
        image_4=None,
    ):
        samples = _latent_samples(latent)
        expected_h = int(samples.shape[-2])
        expected_w = int(samples.shape[-1])

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
            return (image_1, latent, report, vae)

        compression_ratio = _FLUX2_CONTROL_COMPRESSION_RATIO * int(
            vae.spacial_compression_encode()
        )
        target_pixel_width = int(expected_w * compression_ratio)
        target_pixel_height = int(expected_h * compression_ratio)
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
            image_1,
            image_2,
            image_3,
            image_4,
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
        return (image_1, latent, report, vae)
