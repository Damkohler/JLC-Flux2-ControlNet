"""Shared preparation services for the unified FLUX.2 conditioning cache node.

This module is additive. It does not replace or alter the three proven cache
backends. It prepares values using the same contracts as the released specialist
nodes and inserts them into:

    REFERENCE_LATENT_CACHE
    HINT_LATENT_CACHE
    INPAINT_CONTEXT_CACHE

The released specialist nodes can be migrated to these helpers later, after the
unified node has been validated in real workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterable, Sequence

import torch

import comfy.controlnet
import comfy.latent_formats
import comfy.model_management
import comfy.utils

from .constants import EXPECTED_FLUX2_LATENT_CHANNELS, PROJECT_LOG_PREFIX
from .hint_latent_cache import (
    HINT_LATENT_CACHE,
    clear_hint_latent_cache,
    hint_latent_cache_info,
    make_hint_latent_cache_key,
)
from .inpaint_context_cache import (
    INPAINT_CONTEXT_CACHE,
    clear_inpaint_context_cache,
    inpaint_context_cache_info,
    make_inpaint_context_cache_key,
    prepare_inpaint_context_tensors,
    validate_inpaint_source_geometry,
)
from .reference_latent_cache import (
    REFERENCE_LATENT_CACHE,
    clear_reference_latent_cache,
    make_reference_latent_cache_key,
    reference_latent_cache_info,
)


CONTROL_COMPRESSION_RATIO = 1
MAX_REFERENCE_IMAGES = 10
MAX_CONTROL_HINTS = 4


@dataclass(frozen=True)
class Flux2TargetGeometry:
    """Geometry derived from the exact FLUX.2 latent used by inference."""

    latent_width: int
    latent_height: int
    pixel_width: int
    pixel_height: int
    vae_compression_ratio: int


@dataclass(frozen=True)
class CachePrepResult:
    """Domain-level result used by the unified node's combined report."""

    domain: str
    requested: int
    hits: int
    misses: int
    inserted: int
    skipped: int
    cache_entries: int
    total_bytes: int

    @property
    def prepared(self) -> int:
        return int(self.hits) + int(self.inserted)

    @property
    def complete(self) -> bool:
        return (
            self.requested > 0
            and self.prepared == self.requested
            and self.skipped == 0
        )

    def summary(self) -> str:
        return (
            f"{self.domain}: requested={self.requested}, prepared={self.prepared}, "
            f"hits={self.hits}, misses={self.misses}, inserted={self.inserted}, "
            f"skipped={self.skipped}, cache_entries={self.cache_entries}, "
            f"total_bytes={self.total_bytes}"
        )


def derive_flux2_target_geometry(vae: Any, latent: Any) -> Flux2TargetGeometry:
    """Derive hint/inpaint geometry from the connected target FLUX.2 latent."""

    if vae is None:
        raise ValueError("JLC Flux2 Conditioning Cache Prep requires a VAE.")
    if not isinstance(latent, dict):
        raise ValueError(
            "JLC Flux2 Conditioning Cache Prep requires empty_flux2_latent to be a LATENT mapping."
        )

    samples = latent.get("samples")
    if not isinstance(samples, torch.Tensor):
        raise ValueError(
            "JLC Flux2 Conditioning Cache Prep requires empty_flux2_latent['samples'] to be a tensor."
        )
    if samples.ndim != 4:
        raise ValueError(
            "JLC Flux2 Conditioning Cache Prep expected empty_flux2_latent samples "
            f"[B,C,H,W], got {tuple(samples.shape)}."
        )
    if int(samples.shape[1]) != EXPECTED_FLUX2_LATENT_CHANNELS:
        raise ValueError(
            "JLC Flux2 Conditioning Cache Prep expected "
            f"{EXPECTED_FLUX2_LATENT_CHANNELS} FLUX.2 latent channels, "
            f"got {int(samples.shape[1])}."
        )

    latent_height = int(samples.shape[-2])
    latent_width = int(samples.shape[-1])
    if latent_width <= 0 or latent_height <= 0:
        raise ValueError(
            "JLC Flux2 Conditioning Cache Prep requires positive target latent geometry."
        )

    compression_ratio = int(vae.spacial_compression_encode())
    if compression_ratio <= 0:
        raise ValueError(
            "JLC Flux2 Conditioning Cache Prep received an invalid VAE compression ratio "
            f"({compression_ratio})."
        )

    return Flux2TargetGeometry(
        latent_width=latent_width,
        latent_height=latent_height,
        pixel_width=latent_width * compression_ratio * CONTROL_COMPRESSION_RATIO,
        pixel_height=latent_height * compression_ratio * CONTROL_COMPRESSION_RATIO,
        vae_compression_ratio=compression_ratio,
    )


def safe_reference_image(image: torch.Tensor) -> torch.Tensor:
    """Match the released reference-cache encode contract: contiguous BHWC RGB."""

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected reference IMAGE tensor, got {type(image)!r}.")
    if image.ndim != 4:
        raise ValueError(
            f"Expected reference IMAGE tensor in BHWC format, got {tuple(image.shape)}."
        )
    if int(image.shape[-1]) < 3:
        raise ValueError(
            "Expected reference IMAGE tensor with at least 3 channels, "
            f"got {tuple(image.shape)}."
        )
    return image[:, :, :, :3].contiguous()


def image_to_control_hint(image: torch.Tensor) -> torch.Tensor:
    """Match ControlBase cond_hint_original layout used by the runtime path."""

    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError(
            "Expected each ControlNet IMAGE input to be a rank-4 tensor."
        )

    if image.shape[-1] in (1, 3, 4):
        return image.movedim(-1, 1).contiguous()
    if image.shape[1] in (1, 3, 4):
        return image.contiguous()
    return image.movedim(-1, 1).contiguous()


def validate_selected_images(
    *,
    reference_images: Sequence[torch.Tensor],
    control_images: Sequence[torch.Tensor],
) -> None:
    """Perform inexpensive validation before any cache is cleared or populated."""

    for image in reference_images:
        safe_reference_image(image)
    for image in control_images:
        image_to_control_hint(image)


def validate_inpaint_selection(
    *,
    image: torch.Tensor,
    mask: torch.Tensor,
    geometry: Flux2TargetGeometry,
) -> None:
    """Validate exact inpaint canvas geometry before any cache is cleared."""

    validate_inpaint_source_geometry(
        image=image,
        mask=mask,
        target_pixel_width=geometry.pixel_width,
        target_pixel_height=geometry.pixel_height,
        caller_name="JLC Flux2 Conditioning Cache Prep",
    )


def clear_selected_caches(
    *,
    references: bool,
    hints: bool,
    inpaint: bool,
    diagnostics: bool,
) -> dict[str, int]:
    """Clear only the active cache domains requested by the unified node."""

    cleared: dict[str, int] = {}
    if references:
        cleared["references"] = clear_reference_latent_cache(
            diagnostics=bool(diagnostics)
        )
    if hints:
        cleared["hints"] = clear_hint_latent_cache(diagnostics=bool(diagnostics))
    if inpaint:
        cleared["inpaint"] = clear_inpaint_context_cache(
            diagnostics=bool(diagnostics)
        )
    return cleared


def _restore_loaded_models_after_encode(vae: Any, image_bhwc: torch.Tensor) -> torch.Tensor:
    """Use the same VAE load/restore pattern as the released prep nodes."""

    loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
    try:
        return vae.encode(image_bhwc)
    finally:
        comfy.model_management.load_models_gpu(loaded_models)


def prepare_reference_cache_entries(
    *,
    vae: Any,
    images: Sequence[torch.Tensor],
    diagnostics: bool,
) -> CachePrepResult:
    """Prepare selected reference images in the existing reference backend."""

    requested = len(images)
    if requested == 0:
        info = reference_latent_cache_info()
        return CachePrepResult(
            "references", 0, 0, 0, 0, 0, info["entry_count"], info["total_bytes"]
        )

    if not REFERENCE_LATENT_CACHE.is_enabled():
        info = reference_latent_cache_info()
        return CachePrepResult(
            "references",
            requested,
            0,
            0,
            0,
            requested,
            info["entry_count"],
            info["total_bytes"],
        )

    hits = misses = inserted = skipped = 0
    for slot_index, image in enumerate(images, start=1):
        final_image = safe_reference_image(image)
        target_height = int(final_image.shape[1])
        target_width = int(final_image.shape[2])

        request = make_reference_latent_cache_key(
            image=final_image,
            vae=vae,
            resize_mode="none",
            upscale_method="external",
            target_width=target_width,
            target_height=target_height,
            target_megapixels=None,
            crop_mode="external",
        )
        if REFERENCE_LATENT_CACHE.get(request, diagnostics=bool(diagnostics)) is not None:
            hits += 1
            continue

        misses += 1
        if diagnostics:
            logging.info(
                "%s Unified cache prep reference slot %d miss: encoding image=%s, key=%s.",
                PROJECT_LOG_PREFIX,
                slot_index,
                tuple(final_image.shape),
                request.short_key,
            )

        latent = _restore_loaded_models_after_encode(vae, final_image)
        if REFERENCE_LATENT_CACHE.put(request, latent, diagnostics=bool(diagnostics)):
            inserted += 1
        else:
            skipped += 1

    info = reference_latent_cache_info()
    return CachePrepResult(
        "references",
        requested,
        hits,
        misses,
        inserted,
        skipped,
        info["entry_count"],
        info["total_bytes"],
    )


def prepare_hint_cache_entries(
    *,
    vae: Any,
    images: Sequence[torch.Tensor],
    geometry: Flux2TargetGeometry,
    diagnostics: bool,
) -> CachePrepResult:
    """Prepare selected hints in the existing ControlNet hint backend."""

    requested = len(images)
    if requested == 0:
        info = hint_latent_cache_info()
        return CachePrepResult(
            "hints", 0, 0, 0, 0, 0, info["entry_count"], info["total_bytes"]
        )

    if not HINT_LATENT_CACHE.is_enabled():
        info = hint_latent_cache_info()
        return CachePrepResult(
            "hints",
            requested,
            0,
            0,
            0,
            requested,
            info["entry_count"],
            info["total_bytes"],
        )

    control_base = comfy.controlnet.ControlBase()
    preprocess_image = control_base.preprocess_image
    upscale_algorithm = str(control_base.upscale_algorithm)
    latent_format = comfy.latent_formats.Flux2()

    hits = misses = inserted = skipped = 0
    for slot_index, image in enumerate(images, start=1):
        control_hint = image_to_control_hint(image)
        request = make_hint_latent_cache_key(
            image=control_hint,
            target_latent_width=geometry.latent_width,
            target_latent_height=geometry.latent_height,
            target_pixel_width=geometry.pixel_width,
            target_pixel_height=geometry.pixel_height,
            vae=vae,
            preprocess_image=preprocess_image,
            interpolation=upscale_algorithm,
            resize_mode="common_upscale",
            crop_mode="center",
            latent_format=latent_format,
        )
        if HINT_LATENT_CACHE.get(request, diagnostics=bool(diagnostics)) is not None:
            hits += 1
            continue

        misses += 1
        if diagnostics:
            logging.info(
                "%s Unified cache prep hint slot %d miss: source=%s, target=%dx%d, key=%s.",
                PROJECT_LOG_PREFIX,
                slot_index,
                tuple(control_hint.shape),
                geometry.pixel_width,
                geometry.pixel_height,
                request.short_key,
            )

        hint = comfy.utils.common_upscale(
            control_hint,
            geometry.pixel_width,
            geometry.pixel_height,
            upscale_algorithm,
            "center",
        )
        hint = preprocess_image(hint)
        hint = _restore_loaded_models_after_encode(vae, hint.movedim(1, -1))
        hint = latent_format.process_in(hint)

        if HINT_LATENT_CACHE.put(request, hint, diagnostics=bool(diagnostics)):
            inserted += 1
        else:
            skipped += 1

    info = hint_latent_cache_info()
    return CachePrepResult(
        "hints",
        requested,
        hits,
        misses,
        inserted,
        skipped,
        info["entry_count"],
        info["total_bytes"],
    )


def prepare_inpaint_cache_entry(
    *,
    vae: Any,
    image: torch.Tensor,
    mask: torch.Tensor,
    geometry: Flux2TargetGeometry,
    diagnostics: bool,
) -> CachePrepResult:
    """Prepare one image/mask pair in the existing inpaint backend."""

    info = inpaint_context_cache_info()
    if not INPAINT_CONTEXT_CACHE.is_enabled():
        return CachePrepResult(
            "inpaint", 1, 0, 0, 0, 1, info["entry_count"], info["total_bytes"]
        )

    latent_format = comfy.latent_formats.Flux2()
    request = make_inpaint_context_cache_key(
        image=image,
        mask=mask,
        vae=vae,
        latent_format=latent_format,
        target_latent_width=geometry.latent_width,
        target_latent_height=geometry.latent_height,
        control_compression_ratio=CONTROL_COMPRESSION_RATIO,
    )

    if INPAINT_CONTEXT_CACHE.get(request, diagnostics=bool(diagnostics)) is not None:
        info = inpaint_context_cache_info()
        return CachePrepResult(
            "inpaint", 1, 1, 0, 0, 0, info["entry_count"], info["total_bytes"]
        )

    if diagnostics:
        logging.info(
            "%s Unified cache prep inpaint miss: image=%s, mask=%s, target=%dx%d, key=%s.",
            PROJECT_LOG_PREFIX,
            tuple(image.shape),
            tuple(mask.shape),
            geometry.pixel_width,
            geometry.pixel_height,
            request.short_key,
        )

    mask_context, masked_latent = prepare_inpaint_context_tensors(
        image=image,
        mask=mask,
        vae=vae,
        latent_format=latent_format,
        target_latent_width=geometry.latent_width,
        target_latent_height=geometry.latent_height,
        control_compression_ratio=CONTROL_COMPRESSION_RATIO,
        batched_number=1,
        caller_name="JLC Flux2 Conditioning Cache Prep",
    )
    inserted = INPAINT_CONTEXT_CACHE.put(
        request,
        mask_context,
        masked_latent,
        diagnostics=bool(diagnostics),
    )
    info = inpaint_context_cache_info()
    return CachePrepResult(
        "inpaint",
        1,
        0,
        1,
        int(bool(inserted)),
        int(not inserted),
        info["entry_count"],
        info["total_bytes"],
    )
