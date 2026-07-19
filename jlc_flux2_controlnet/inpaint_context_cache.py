"""Bounded CPU cache for reusable FLUX.2 inpaint contexts.

Each entry stores the two static tensors required by the mask-aware 260-channel
ControlNet contract:

    packed_keep_mask_4 + masked_source_latent_128

The cache is process-local, content-addressed, bounded, and CPU-only. It never
retains CUDA tensors, ControlNet residuals, sampler state, token sequences, or
model patches.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

import comfy.controlnet
import comfy.model_management
import comfy.utils

from .constants import (
    EXPECTED_FLUX2_LATENT_CHANNELS,
    EXPECTED_MASK_CHANNELS,
    PROJECT_LOG_PREFIX,
)
from .hint_latent_cache import (
    describe_latent_format_identity,
    describe_vae_identity,
    tensor_fingerprint,
    tensor_nbytes,
)


INPAINT_CONTEXT_CONTRACT_REVISION = "jlc-flux2-inpaint-context-v1"
_NEUTRAL_MASKED_PIXEL_VALUE = 0.5
_MASK_THRESHOLD = 0.5
_DEFAULT_MAX_ENTRIES = 32
_DEFAULT_MAX_CPU_BYTES = 256 * 1024 * 1024


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _env_max_cpu_bytes() -> int:
    if "JLC_FLUX2_INPAINT_CACHE_MAX_CPU_BYTES" in os.environ:
        return max(
            0,
            _env_int(
                "JLC_FLUX2_INPAINT_CACHE_MAX_CPU_BYTES",
                _DEFAULT_MAX_CPU_BYTES,
            ),
        )
    max_mb = max(0, _env_int("JLC_FLUX2_INPAINT_CACHE_MAX_CPU_MB", 256))
    return max_mb * 1024 * 1024


def image_to_bchw(image: torch.Tensor) -> torch.Tensor:
    """Normalize a ComfyUI IMAGE tensor to contiguous float32 BCHW RGB."""

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected IMAGE tensor, got {type(image)!r}.")
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(
            f"Expected IMAGE tensor in BHWC format with at least 3 channels, got {tuple(image.shape)}."
        )
    return (
        image[:, :, :, :3]
        .movedim(-1, 1)
        .to(dtype=torch.float32)
        .clamp(0.0, 1.0)
        .contiguous()
    )


def mask_to_bchw(mask: torch.Tensor) -> torch.Tensor:
    """Normalize a ComfyUI MASK tensor to contiguous float32 B1HW."""

    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Expected MASK tensor, got {type(mask)!r}.")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4:
        if mask.shape[1] != 1 and mask.shape[-1] == 1:
            mask = mask.movedim(-1, 1)
        if mask.shape[1] != 1:
            mask = mask[:, :1]
    else:
        raise ValueError(f"Unsupported MASK shape: {tuple(mask.shape)}.")
    return mask.to(dtype=torch.float32).clamp(0.0, 1.0).contiguous()


def validate_inpaint_source_geometry(
    *,
    image: torch.Tensor,
    mask: torch.Tensor,
    target_pixel_width: int,
    target_pixel_height: int,
    caller_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Require image and mask to exactly match the sampling canvas.

    Silent spatial resizing is deliberately rejected. It can hide workflow
    wiring mistakes and makes the source-image latent disagree with the target
    canvas selected by the scheduler/empty latent.
    """

    image_bchw = image_to_bchw(image)
    mask_bchw = mask_to_bchw(mask)

    image_hw = (int(image_bchw.shape[-2]), int(image_bchw.shape[-1]))
    mask_hw = (int(mask_bchw.shape[-2]), int(mask_bchw.shape[-1]))
    target_hw = (int(target_pixel_height), int(target_pixel_width))

    if image_hw != mask_hw or image_hw != target_hw:
        raise ValueError(
            f"{caller_name} requires image and mask to exactly match the active "
            "sampling canvas. Automatic spatial resizing is intentionally "
            "disabled. Received "
            f"image={image_hw[1]}x{image_hw[0]}, "
            f"mask={mask_hw[1]}x{mask_hw[0]}, "
            f"target={target_hw[1]}x{target_hw[0]}. "
            "Resize both image and mask upstream from the same width/height "
            "source used by the Flux2 scheduler and empty latent."
        )

    image_batch = int(image_bchw.shape[0])
    mask_batch = int(mask_bchw.shape[0])
    if image_batch != mask_batch and image_batch != 1 and mask_batch != 1:
        raise ValueError(
            f"{caller_name} cannot broadcast image batch {image_batch} against "
            f"mask batch {mask_batch}; one batch must be 1 or both must match."
        )

    return image_bchw, mask_bchw


def _align_batches(
    mask: torch.Tensor,
    image: torch.Tensor,
    batched_number: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_batch = max(int(mask.shape[0]), int(image.shape[0]))
    if mask.shape[0] != target_batch:
        mask = comfy.controlnet.broadcast_image_to(
            mask,
            target_batch,
            int(batched_number),
        )
    if image.shape[0] != target_batch:
        image = comfy.controlnet.broadcast_image_to(
            image,
            target_batch,
            int(batched_number),
        )
    return mask, image


def _patchify_mask_2x2(mask: torch.Tensor) -> torch.Tensor:
    """Convert [B,1,2H,2W] samples into [B,4,H,W] Flux2 mask lanes."""

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask tensor [B,1,H,W], got {tuple(mask.shape)}.")
    if mask.shape[-2] % 2 != 0 or mask.shape[-1] % 2 != 0:
        raise ValueError(f"Mask patchify size must be even, got {tuple(mask.shape)}.")
    b, c, h, w = mask.shape
    mask = mask.view(b, c, h // 2, 2, w // 2, 2)
    mask = mask.permute(0, 1, 3, 5, 2, 4)
    return mask.reshape(b, c * 4, h // 2, w // 2).contiguous()


def prepare_inpaint_context_tensors(
    *,
    image: torch.Tensor,
    mask: torch.Tensor,
    vae: Any,
    latent_format: Any,
    target_latent_width: int,
    target_latent_height: int,
    control_compression_ratio: int = 1,
    batched_number: int = 1,
    caller_name: str = "JLC Flux2 Inpaint Context",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the validated hard-mask context and masked-source latent."""

    if vae is None:
        raise ValueError(f"{caller_name} requires a VAE.")

    expected_w = int(target_latent_width)
    expected_h = int(target_latent_height)
    if expected_w <= 0 or expected_h <= 0:
        raise ValueError(
            f"{caller_name} requires positive latent dimensions, got {expected_w}x{expected_h}."
        )

    compression_ratio = max(1, int(control_compression_ratio))
    compression_ratio *= int(vae.spacial_compression_encode())
    target_pixel_width = int(expected_w * compression_ratio)
    target_pixel_height = int(expected_h * compression_ratio)

    image_bchw, mask_bchw = validate_inpaint_source_geometry(
        image=image,
        mask=mask,
        target_pixel_width=target_pixel_width,
        target_pixel_height=target_pixel_height,
        caller_name=caller_name,
    )

    mask_binary = (mask_bchw >= _MASK_THRESHOLD).to(dtype=torch.float32)
    mask_for_image = comfy.utils.common_upscale(
        mask_binary,
        target_pixel_width,
        target_pixel_height,
        "nearest-exact",
        "center",
    )
    image_bchw = comfy.utils.common_upscale(
        image_bchw,
        target_pixel_width,
        target_pixel_height,
        "bilinear",
        "center",
    )
    mask_for_image, image_bchw = _align_batches(
        mask_for_image,
        image_bchw,
        int(batched_number),
    )

    keep_mask_image = (mask_for_image < _MASK_THRESHOLD).to(dtype=image_bchw.dtype)
    masked_image = (
        image_bchw * keep_mask_image
        + _NEUTRAL_MASKED_PIXEL_VALUE * (1.0 - keep_mask_image)
    )

    loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
    try:
        masked_latent = vae.encode(masked_image.movedim(1, -1))
    finally:
        comfy.model_management.load_models_gpu(loaded_models)

    if latent_format is not None:
        masked_latent = latent_format.process_in(masked_latent)

    mask_for_context = comfy.utils.common_upscale(
        mask_binary,
        expected_w * 2,
        expected_h * 2,
        "nearest-exact",
        "center",
    )
    mask_context = _patchify_mask_2x2(1.0 - mask_for_context)

    if mask_context.shape[1] != EXPECTED_MASK_CHANNELS:
        raise RuntimeError(
            f"Expected {EXPECTED_MASK_CHANNELS} Flux2 mask channels, got {mask_context.shape[1]}."
        )
    if masked_latent.shape[1] != EXPECTED_FLUX2_LATENT_CHANNELS:
        raise RuntimeError(
            f"Expected {EXPECTED_FLUX2_LATENT_CHANNELS} masked-image latent channels, "
            f"got {masked_latent.shape[1]}."
        )
    if mask_context.shape[-2:] != (expected_h, expected_w):
        raise RuntimeError(
            f"Prepared mask context has shape {tuple(mask_context.shape[-2:])}, "
            f"expected {(expected_h, expected_w)}."
        )
    if masked_latent.shape[-2:] != (expected_h, expected_w):
        raise RuntimeError(
            f"Prepared masked latent has shape {tuple(masked_latent.shape[-2:])}, "
            f"expected {(expected_h, expected_w)}."
        )

    return mask_context, masked_latent


@dataclass(frozen=True)
class InpaintContextCacheKey:
    key: str
    image_fingerprint: str
    mask_fingerprint: str
    vae_identity: str
    latent_format_identity: str

    @property
    def short_key(self) -> str:
        return self.key.rsplit(":", 1)[-1][:12]


def make_inpaint_context_cache_key(
    *,
    image: torch.Tensor,
    mask: torch.Tensor,
    vae: Any,
    latent_format: Any,
    target_latent_width: int,
    target_latent_height: int,
    control_compression_ratio: int = 1,
    preprocessing_revision: str = INPAINT_CONTEXT_CONTRACT_REVISION,
) -> InpaintContextCacheKey:
    """Build a strict content key for the hard-mask inpaint context."""

    compression_ratio = max(1, int(control_compression_ratio))
    compression_ratio *= int(vae.spacial_compression_encode())
    target_pixel_width = int(target_latent_width) * compression_ratio
    target_pixel_height = int(target_latent_height) * compression_ratio

    image_bchw, mask_bchw = validate_inpaint_source_geometry(
        image=image,
        mask=mask,
        target_pixel_width=target_pixel_width,
        target_pixel_height=target_pixel_height,
        caller_name="JLC Flux2 Inpaint Context Cache",
    )
    final_image = image_bchw.movedim(1, -1).contiguous()
    binary_mask = (mask_bchw >= _MASK_THRESHOLD).to(dtype=torch.float32).contiguous()

    image_digest = tensor_fingerprint(final_image)
    mask_digest = tensor_fingerprint(binary_mask)
    vae_identity = describe_vae_identity(vae)
    latent_format_identity = describe_latent_format_identity(latent_format)
    payload = {
        "preprocessing_revision": str(preprocessing_revision),
        "image_sha256": image_digest,
        "binary_mask_sha256": mask_digest,
        "image_shape": list(final_image.shape),
        "mask_shape": list(binary_mask.shape),
        "target_latent_width": int(target_latent_width),
        "target_latent_height": int(target_latent_height),
        "target_pixel_width": target_pixel_width,
        "target_pixel_height": target_pixel_height,
        "control_compression_ratio": int(control_compression_ratio),
        "vae_identity": vae_identity,
        "latent_format_identity": latent_format_identity,
        "mask_threshold": _MASK_THRESHOLD,
        "neutral_pixel_fill": _NEUTRAL_MASKED_PIXEL_VALUE,
        "mask_resize": "nearest-exact:center",
        "image_resize": "bilinear:center",
        "packed_mask": "inverse-hard-mask:nearest-exact:2x2-patchify",
        "geometry_policy": "strict_exact_match",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return InpaintContextCacheKey(
        key=f"{preprocessing_revision}:{digest}",
        image_fingerprint=image_digest,
        mask_fingerprint=mask_digest,
        vae_identity=vae_identity,
        latent_format_identity=latent_format_identity,
    )


@dataclass
class InpaintContextCacheEntry:
    key: str
    mask_context: torch.Tensor
    masked_latent: torch.Tensor
    nbytes: int
    vae_identity: str
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    hit_count: int = 0

    def touch(self) -> None:
        self.last_used_at = time.time()
        self.hit_count += 1


class InpaintContextCache:
    """Thread-safe bounded LRU cache containing CPU tensors only."""

    def __init__(
        self,
        *,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        max_cpu_bytes: int = _DEFAULT_MAX_CPU_BYTES,
        enabled: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, InpaintContextCacheEntry] = OrderedDict()
        self._total_bytes = 0
        self._max_entries = max(0, int(max_entries))
        self._max_cpu_bytes = max(0, int(max_cpu_bytes))
        self._enabled = bool(enabled)

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled and self._max_entries > 0 and self._max_cpu_bytes > 0

    def configure(
        self,
        *,
        max_entries: Optional[int] = None,
        max_cpu_bytes: Optional[int] = None,
        enabled: Optional[bool] = None,
        diagnostics: bool = False,
    ) -> None:
        with self._lock:
            if max_entries is not None:
                self._max_entries = max(0, int(max_entries))
            if max_cpu_bytes is not None:
                self._max_cpu_bytes = max(0, int(max_cpu_bytes))
            if enabled is not None:
                self._enabled = bool(enabled)
            self._evict_to_limits_locked(reason="configuration", diagnostics=diagnostics)

    def get(
        self,
        request: InpaintContextCacheKey,
        *,
        diagnostics: bool = False,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        with self._lock:
            if not self.is_enabled():
                return None
            entry = self._entries.get(request.key)
            if entry is None:
                return None
            if (
                entry.mask_context.device.type != "cpu"
                or entry.masked_latent.device.type != "cpu"
            ):
                self._remove_locked(
                    request.key,
                    reason="non_cpu_invariant",
                    diagnostics=diagnostics,
                )
                return None

            entry.touch()
            self._entries.move_to_end(request.key, last=True)
            if diagnostics:
                logging.info(
                    "%s Inpaint-context cache hit: key=%s, mask=%s, masked_latent=%s, bytes=%d.",
                    PROJECT_LOG_PREFIX,
                    request.short_key,
                    tuple(entry.mask_context.shape),
                    tuple(entry.masked_latent.shape),
                    entry.nbytes,
                )
            return entry.mask_context, entry.masked_latent

    def put(
        self,
        request: InpaintContextCacheKey,
        mask_context: torch.Tensor,
        masked_latent: torch.Tensor,
        *,
        diagnostics: bool = False,
    ) -> bool:
        if not isinstance(mask_context, torch.Tensor) or not isinstance(masked_latent, torch.Tensor):
            raise TypeError("Inpaint context cache requires tensor mask_context and masked_latent values.")

        estimated = tensor_nbytes(mask_context) + tensor_nbytes(masked_latent)
        with self._lock:
            if not self.is_enabled() or estimated > self._max_cpu_bytes:
                if diagnostics and self._enabled and estimated > self._max_cpu_bytes:
                    logging.info(
                        "%s Inpaint-context cache insert skipped: key=%s, bytes=%d exceeds max_cpu_bytes=%d.",
                        PROJECT_LOG_PREFIX,
                        request.short_key,
                        estimated,
                        self._max_cpu_bytes,
                    )
                return False

        cpu_mask = mask_context.detach().to(device="cpu", dtype=torch.float32).contiguous()
        cpu_latent = masked_latent.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if mask_context.device.type == "cpu":
            cpu_mask = cpu_mask.clone()
        if masked_latent.device.type == "cpu":
            cpu_latent = cpu_latent.clone()
        cpu_mask.requires_grad_(False)
        cpu_latent.requires_grad_(False)
        nbytes = tensor_nbytes(cpu_mask) + tensor_nbytes(cpu_latent)

        with self._lock:
            if not self.is_enabled() or nbytes > self._max_cpu_bytes:
                return False

            previous = self._entries.pop(request.key, None)
            if previous is not None:
                self._total_bytes -= previous.nbytes

            entry = InpaintContextCacheEntry(
                key=request.key,
                mask_context=cpu_mask,
                masked_latent=cpu_latent,
                nbytes=nbytes,
                vae_identity=request.vae_identity,
            )
            self._entries[request.key] = entry
            self._total_bytes += nbytes

            if diagnostics:
                logging.info(
                    "%s Inpaint-context cache insert: key=%s, mask=%s, masked_latent=%s, bytes=%d, entries=%d, total_bytes=%d.",
                    PROJECT_LOG_PREFIX,
                    request.short_key,
                    tuple(cpu_mask.shape),
                    tuple(cpu_latent.shape),
                    nbytes,
                    len(self._entries),
                    self._total_bytes,
                )

            self._evict_to_limits_locked(reason="lru_capacity", diagnostics=diagnostics)
            return request.key in self._entries

    def clear(self, *, reason: str = "manual", diagnostics: bool = False) -> int:
        with self._lock:
            count_removed = len(self._entries)
            bytes_removed = self._total_bytes
            self._entries.clear()
            self._total_bytes = 0
            if diagnostics and count_removed:
                logging.info(
                    "%s Inpaint-context cache clear: reason=%s, entries=%d, bytes=%d.",
                    PROJECT_LOG_PREFIX,
                    reason,
                    count_removed,
                    bytes_removed,
                )
            return count_removed

    def info(self) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            return {
                "enabled": self._enabled,
                "max_entries": self._max_entries,
                "max_cpu_bytes": self._max_cpu_bytes,
                "entry_count": len(self._entries),
                "total_bytes": self._total_bytes,
                "entries": [
                    {
                        "key": entry.key,
                        "mask_shape": tuple(entry.mask_context.shape),
                        "masked_latent_shape": tuple(entry.masked_latent.shape),
                        "dtype": str(entry.masked_latent.dtype),
                        "device": str(entry.masked_latent.device),
                        "bytes": entry.nbytes,
                        "hit_count": entry.hit_count,
                        "age_sec": round(now - entry.created_at, 3),
                        "idle_sec": round(now - entry.last_used_at, 3),
                    }
                    for entry in self._entries.values()
                ],
            }

    def _evict_to_limits_locked(self, *, reason: str, diagnostics: bool) -> None:
        while self._entries and (
            len(self._entries) > self._max_entries
            or self._total_bytes > self._max_cpu_bytes
            or not self._enabled
        ):
            oldest_key = next(iter(self._entries))
            self._remove_locked(oldest_key, reason=reason, diagnostics=diagnostics)

    def _remove_locked(self, key: str, *, reason: str, diagnostics: bool) -> bool:
        entry = self._entries.pop(key, None)
        if entry is None:
            return False
        self._total_bytes -= entry.nbytes
        if diagnostics:
            logging.info(
                "%s Inpaint-context cache eviction: key=%s, bytes=%d, reason=%s.",
                PROJECT_LOG_PREFIX,
                key.rsplit(":", 1)[-1][:12],
                entry.nbytes,
                reason,
            )
        entry.mask_context = torch.empty(0, device="cpu")
        entry.masked_latent = torch.empty(0, device="cpu")
        return True


_previous_cache = globals().get("INPAINT_CONTEXT_CACHE")
if _previous_cache is not None and hasattr(_previous_cache, "clear"):
    _previous_cache.clear(reason="module_reload", diagnostics=False)

INPAINT_CONTEXT_CACHE = InpaintContextCache(
    max_entries=max(
        0,
        _env_int("JLC_FLUX2_INPAINT_CACHE_MAX_ENTRIES", _DEFAULT_MAX_ENTRIES),
    ),
    max_cpu_bytes=_env_max_cpu_bytes(),
    enabled=_env_bool("JLC_FLUX2_INPAINT_CACHE_ENABLED", True),
)


def clear_inpaint_context_cache(*, diagnostics: bool = True) -> int:
    return INPAINT_CONTEXT_CACHE.clear(
        reason="explicit_clear",
        diagnostics=diagnostics,
    )


def inpaint_context_cache_info() -> dict[str, Any]:
    return INPAINT_CONTEXT_CACHE.info()
