"""
JLC Flux2 Reference Latent Cache Prep
-------------------------------------

- JLC Flux2 ControlNet
  - This node is part of the **JLC Flux2 ControlNet** package developed
    by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/JLC-Flux2-ControlNet

- Node Purpose
  - The **JLC Flux2 Reference Latent Cache Prep** node pre-populates the
    bounded, process-local CPU cache used by the JLC Flux2 Reference Image
    Orchestrator.

  - It performs only the reusable reference-latent preparation path:

        upstream-prepared IMAGE
            -> final contiguous BHWC RGB tensor
            -> VAE encode
            -> detached CPU latent
            -> bounded shared cache

  - Desired resizing, cropping, padding, placement, or other image preparation
    must be completed upstream. The exact final image tensor supplied here is
    the tensor fingerprinted for cache identity and passed to `vae.encode`.

- Method-Agnostic Cache Contract
  - Native FLUX.2 `reference_latents_method` selection is deliberately absent
    from this node.

  - Reference methods affect downstream reference-token positioning through
    conditioning metadata. They do not alter the result of VAE-encoding the
    prepared image.

  - Cache identity therefore depends on the final prepared image, VAE identity,
    preprocessing contract, and latent contract—but not on the later reference
    method selected by the Orchestrator.

- Slot Contract
  - The node accepts one required image and up to nine optional images.

  - `slot_count` is authoritative. Only connected images within the active slot
    range are prepared; empty optional slots are skipped without promotion.

  - Images are encoded independently and cached independently. Slot order does
    not alter latent values or cache identity.

- Workflow and Execution Contract
  - Use this node in a mutually exclusive cache-preparation branch before the
    normal inference branch, within the same ComfyUI server process.

  - The node is intentionally **not** an output node. A downstream lazy Switch,
    Group Controller, or equivalent execution gate must request the preparation
    branch. An inactive branch does not pull image or VAE dependencies into
    execution.

  - `IS_CHANGED` returns NaN so the cache side effect is refreshed whenever the
    preparation branch is actually requested, even when visible inputs have not
    changed.

- Cache Safety and Churn Reduction
  - Cached latents are detached, contiguous CPU tensors. The cache retains no
    GPU tensors, sampler state, residuals, token blocks, conditioning objects,
    or model patches.

  - Warm cache hits avoid repeated reference-image VAE encoding and reduce the
    VAE load/offload churn that otherwise occurs at inference startup.

  - The cache remains bounded by the shared reference-cache entry and CPU-memory
    limits. `clear_before_prepare` can explicitly reset all shared reference
    entries before repopulation.

- Passthrough Outputs
  - `image_1` is returned unchanged for branch routing.

  - `cache_set` is True when all active connected slots are either cache hits
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
from typing import Iterable, Optional

import torch

import comfy.model_management

from ..jlc_flux2_controlnet.constants import PROJECT_LOG_PREFIX
from ..jlc_flux2_controlnet.reference_latent_cache import (
    REFERENCE_LATENT_CACHE,
    clear_reference_latent_cache,
    make_reference_latent_cache_key,
    reference_latent_cache_info,
)


REFERENCE_LATENT_CACHE_PREP_VERSION = "1.1.0"

MANIFEST = {
    "name": "JLC Flux2 Reference Latent Cache Prep",
    "version": REFERENCE_LATENT_CACHE_PREP_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Pre-populates the bounded, process-local CPU cache for reusable "
        "FLUX.2 reference-image VAE latents. Cache identity is method-agnostic: "
        "the final prepared image and VAE determine the latent, while native "
        "reference-method selection remains downstream conditioning metadata. "
        "The branch-driven node supports up to ten slots and avoids repeated "
        "VAE encoding and associated model churn during inference."
    ),
    "base_package_version": JLC_FLUX2_CONTROLNET_VERSION,
    "cache_contract_revision": "jlc-flux2-reference-latent-v2",
    "reference_method_scope": "conditioning_only",
    "status": "stable",
    "license": "MIT",
}

_MAX_REFERENCE_SLOTS = 10


def _safe_reference_image(image: torch.Tensor) -> torch.Tensor:
    """Match the runtime reference-image VAE encode contract: final BHWC RGB."""

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected IMAGE tensor, got {type(image)!r}.")
    if image.ndim != 4:
        raise ValueError(
            f"Expected IMAGE tensor in BHWC format, got shape {tuple(image.shape)}."
        )
    if image.shape[-1] < 3:
        raise ValueError(
            f"Expected IMAGE tensor with at least 3 channels, got shape {tuple(image.shape)}."
        )
    return image[:, :, :, :3].contiguous()


def _iter_slots(
    slot_count: int,
    reference_image_1: torch.Tensor,
    reference_image_2: Optional[torch.Tensor],
    reference_image_3: Optional[torch.Tensor],
    reference_image_4: Optional[torch.Tensor],
    reference_image_5: Optional[torch.Tensor],
    reference_image_6: Optional[torch.Tensor],
    reference_image_7: Optional[torch.Tensor],
    reference_image_8: Optional[torch.Tensor],
    reference_image_9: Optional[torch.Tensor],
    reference_image_10: Optional[torch.Tensor],
) -> Iterable[tuple[int, torch.Tensor]]:
    images = (
        reference_image_1,
        reference_image_2,
        reference_image_3,
        reference_image_4,
        reference_image_5,
        reference_image_6,
        reference_image_7,
        reference_image_8,
        reference_image_9,
        reference_image_10,
    )
    active = max(1, min(_MAX_REFERENCE_SLOTS, int(slot_count)))
    for index, image in enumerate(images[:active], start=1):
        if image is not None:
            yield index, image


class JLCFlux2ReferenceLatentCachePrep:
    """Pre-warm the shared CPU reference-latent cache for one to ten images.

    This node always prepares when its workflow branch executes. Use a
    downstream lazy Switch / Group Controller branch to decide whether the prep
    path or the inference path should run.
    """

    CATEGORY = "Flux2 Conditioning/Utilities"
    FUNCTION = "prepare"
    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("reference_image_1", "cache_set", "cache_report")
    DESCRIPTION = (
        "Pre-warms the method-agnostic CPU cache for reusable FLUX.2 "
        "reference-image VAE latents without executing the inference branch."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "reference_image_1": ("IMAGE",),
                "slot_count": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": _MAX_REFERENCE_SLOTS,
                        "step": 1,
                    },
                ),
                "clear_before_prepare": ("BOOLEAN", {"default": False}),
                "diagnostics": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "reference_image_2": ("IMAGE",),
                "reference_image_3": ("IMAGE",),
                "reference_image_4": ("IMAGE",),
                "reference_image_5": ("IMAGE",),
                "reference_image_6": ("IMAGE",),
                "reference_image_7": ("IMAGE",),
                "reference_image_8": ("IMAGE",),
                "reference_image_9": ("IMAGE",),
                "reference_image_10": ("IMAGE",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def prepare(
        self,
        vae,
        reference_image_1,
        slot_count=2,
        clear_before_prepare=False,
        diagnostics=True,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
        reference_image_5=None,
        reference_image_6=None,
        reference_image_7=None,
        reference_image_8=None,
        reference_image_9=None,
        reference_image_10=None,
    ):
        if clear_before_prepare:
            clear_reference_latent_cache(diagnostics=bool(diagnostics))

        if vae is None:
            raise ValueError(
                "JLC Flux2 Reference Latent Cache Prep requires a VAE so it can encode reference images."
            )

        if not REFERENCE_LATENT_CACHE.is_enabled():
            report = "JLC Flux2 reference-latent cache is disabled or has zero capacity; prep skipped."
            if diagnostics:
                logging.info("%s %s", PROJECT_LOG_PREFIX, report)
            return (reference_image_1, False, report)

        prepared = 0
        hits = 0
        misses = 0
        inserted = 0
        skipped = 0

        for slot_index, image in _iter_slots(
            slot_count,
            reference_image_1,
            reference_image_2,
            reference_image_3,
            reference_image_4,
            reference_image_5,
            reference_image_6,
            reference_image_7,
            reference_image_8,
            reference_image_9,
            reference_image_10,
        ):
            final_image = _safe_reference_image(image)
            target_height = int(final_image.shape[1])
            target_width = int(final_image.shape[2])

            cache_request = make_reference_latent_cache_key(
                image=final_image,
                vae=vae,
                resize_mode="none",
                upscale_method="external",
                target_width=target_width,
                target_height=target_height,
                target_megapixels=None,
                crop_mode="external",
            )

            cached = REFERENCE_LATENT_CACHE.get(
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
                    "%s Reference-latent prep slot %d cache miss: encoding reference image; image_shape=%s, key=%s.",
                    PROJECT_LOG_PREFIX,
                    slot_index,
                    tuple(final_image.shape),
                    cache_request.short_key,
                )

            loaded_models = comfy.model_management.loaded_models(
                only_currently_used=True
            )
            try:
                reference_latent = vae.encode(final_image[:, :, :, :3])
            finally:
                comfy.model_management.load_models_gpu(loaded_models)

            if REFERENCE_LATENT_CACHE.put(
                cache_request,
                reference_latent,
                diagnostics=bool(diagnostics),
            ):
                inserted += 1
                prepared += 1
            else:
                skipped += 1
                if diagnostics:
                    logging.info(
                        "%s Reference-latent prep slot %d cache insert skipped; key=%s.",
                        PROJECT_LOG_PREFIX,
                        slot_index,
                        cache_request.short_key,
                    )

        info = reference_latent_cache_info()
        report = (
            "JLC Flux2 reference-latent prep complete: "
            f"prepared={prepared}, hits={hits}, misses={misses}, inserted={inserted}, "
            f"skipped={skipped}, cache_entries={info['entry_count']}, "
            f"total_bytes={info['total_bytes']}."
        )
        if diagnostics:
            logging.info("%s %s", PROJECT_LOG_PREFIX, report)
        cache_set = prepared > 0 and skipped == 0
        return (reference_image_1, bool(cache_set), report)
