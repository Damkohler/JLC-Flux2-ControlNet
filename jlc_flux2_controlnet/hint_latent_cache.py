"""Bounded CPU cache for reusable FLUX.2 ControlNet hint latents.

The cache stores only the final, unbroadcast control-image latent after the
validated resize/crop, VAE encode, and Flux2 latent-format conversion path.
Cached tensors are detached, contiguous CPU tensors. GPU tensors, residuals,
per-step state, and side-model outputs are never retained here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Optional

import torch

from .constants import PROJECT_LOG_PREFIX


HINT_PREPROCESS_CONTRACT_REVISION = "jlc-flux2-hint-latent-v1"
_DEFAULT_MAX_ENTRIES = 32
_DEFAULT_MAX_CPU_BYTES = 512 * 1024 * 1024


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
    if "JLC_FLUX2_HINT_CACHE_MAX_CPU_BYTES" in os.environ:
        return max(0, _env_int("JLC_FLUX2_HINT_CACHE_MAX_CPU_BYTES", _DEFAULT_MAX_CPU_BYTES))
    max_mb = max(0, _env_int("JLC_FLUX2_HINT_CACHE_MAX_CPU_MB", 256))
    return max_mb * 1024 * 1024


_TOKEN_LOCK = threading.RLock()
_TOKEN_COUNTER = count(1)
_OBJECT_TOKENS: dict[int, tuple[weakref.ReferenceType[Any], int]] = {}


def _object_token(obj: Any) -> str:
    """Return a non-retaining process-local identity token for an object."""

    if obj is None:
        return "none"

    object_id = id(obj)
    with _TOKEN_LOCK:
        existing = _OBJECT_TOKENS.get(object_id)
        if existing is not None and existing[0]() is obj:
            return str(existing[1])

        token = next(_TOKEN_COUNTER)
        try:
            def _remove(reference, *, expected_id=object_id, expected_token=token):
                with _TOKEN_LOCK:
                    current = _OBJECT_TOKENS.get(expected_id)
                    if current is not None and current[0] is reference and current[1] == expected_token:
                        _OBJECT_TOKENS.pop(expected_id, None)

            reference = weakref.ref(obj, _remove)
            _OBJECT_TOKENS[object_id] = (reference, token)
            return str(token)
        except TypeError:
            # A few extension objects are not weak-referenceable. The id is still
            # combined with class/configuration data and is never the sole key.
            return f"id-{object_id:x}"


def _class_name(obj: Any) -> str:
    if obj is None:
        return "none"
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (tuple, list)) and len(value) <= 32:
        converted = [_json_safe(item) for item in value]
        if all(item is not _UNSUPPORTED for item in converted):
            return converted
    return _UNSUPPORTED


_UNSUPPORTED = object()


def _selected_config(obj: Any, names: tuple[str, ...]) -> dict[str, Any]:
    if obj is None:
        return {}
    result: dict[str, Any] = {}
    for name in names:
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        converted = _json_safe(value)
        if converted is not _UNSUPPORTED:
            result[name] = converted
    return result


def describe_vae_identity(vae: Any) -> str:
    """Describe the active VAE without reading or hashing heavyweight weights.

    The process-local object tokens prevent two different live VAE instances
    from colliding. Class and scalar configuration fields ensure the key is not
    based solely on Python object identity.
    """

    first_stage = getattr(vae, "first_stage_model", None)
    patcher = getattr(vae, "patcher", None)
    patcher_model = getattr(patcher, "model", None)
    payload = {
        "wrapper_class": _class_name(vae),
        "wrapper_config": _selected_config(
            vae,
            (
                "downscale_ratio",
                "upscale_ratio",
                "latent_channels",
                "output_channels",
                "vae_dtype",
                "working_dtypes",
                "downscale_index_formula",
            ),
        ),
        "first_stage_class": _class_name(first_stage),
        "first_stage_token": _object_token(first_stage),
        "patcher_class": _class_name(patcher),
        "patcher_model_class": _class_name(patcher_model),
        "patcher_model_token": _object_token(patcher_model),
    }
    if patcher_model is None:
        payload["patcher_token"] = _object_token(patcher)
    if first_stage is None and patcher_model is None:
        payload["wrapper_token"] = _object_token(vae)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _public_simple_config(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    try:
        values = vars(obj)
    except Exception:
        return {}

    result: dict[str, Any] = {}
    for name, value in values.items():
        if name.startswith("_"):
            continue
        converted = _json_safe(value)
        if converted is not _UNSUPPORTED:
            result[name] = converted
    return result


def describe_latent_format_identity(latent_format: Any) -> str:
    # Latent-format objects are lightweight mathematical configuration. Use
    # class plus scalar state so equivalent recreated instances can share.
    config = _selected_config(
        latent_format,
        (
            "latent_channels",
            "latent_dimensions",
            "scale_factor",
            "shift_factor",
            "taesd_decoder_name",
            "spatial_compression_ratio",
        ),
    )
    config.update(_public_simple_config(latent_format))
    payload = {
        "class": _class_name(latent_format),
        "config": config,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def describe_preprocess_callable(preprocess_image: Any) -> str:
    code = getattr(preprocess_image, "__code__", None)
    code_digest = "none"
    closure_values: list[Any] = []
    if code is not None:
        hasher = hashlib.sha256()
        hasher.update(code.co_code)
        hasher.update(repr(code.co_consts).encode("utf-8", errors="backslashreplace"))
        code_digest = hasher.hexdigest()
        for cell in getattr(preprocess_image, "__closure__", None) or ():
            try:
                converted = _json_safe(cell.cell_contents)
            except Exception:
                converted = _UNSUPPORTED
            if converted is _UNSUPPORTED:
                value = getattr(cell, "cell_contents", None)
                converted = {
                    "class": _class_name(value),
                    "token": _object_token(value),
                }
            closure_values.append(converted)

    payload = {
        "module": getattr(preprocess_image, "__module__", "none"),
        "qualname": getattr(preprocess_image, "__qualname__", _class_name(preprocess_image)),
        "code_sha256": code_digest,
        "closure": closure_values,
    }
    if code is None:
        payload["callable_token"] = _object_token(preprocess_image)
        payload["callable_config"] = _public_simple_config(preprocess_image)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def tensor_fingerprint(tensor: torch.Tensor) -> str:
    """Return a stable full-content SHA-256 fingerprint for a tensor."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(tensor)!r}.")

    detached = tensor.detach()
    cpu_tensor = detached.to(device="cpu").contiguous()
    byte_view = cpu_tensor.view(torch.uint8).reshape(-1)

    hasher = hashlib.sha256()
    hasher.update(str(tuple(cpu_tensor.shape)).encode("ascii"))
    hasher.update(str(cpu_tensor.dtype).encode("ascii"))
    if byte_view.numel() > 0:
        hasher.update(memoryview(byte_view.numpy()))
    return hasher.hexdigest()


@dataclass(frozen=True)
class HintLatentCacheKey:
    key: str
    image_fingerprint: str
    vae_identity: str
    latent_format_identity: str

    @property
    def short_key(self) -> str:
        return self.key.rsplit(":", 1)[-1][:12]


def make_hint_latent_cache_key(
    *,
    image: torch.Tensor,
    target_latent_width: int,
    target_latent_height: int,
    target_pixel_width: int,
    target_pixel_height: int,
    vae: Any,
    preprocess_image: Any,
    interpolation: str,
    resize_mode: str = "common_upscale",
    crop_mode: str = "center",
    latent_format: Any = None,
    preprocessing_revision: str = HINT_PREPROCESS_CONTRACT_REVISION,
) -> HintLatentCacheKey:
    image_digest = tensor_fingerprint(image)
    vae_identity = describe_vae_identity(vae)
    latent_format_identity = describe_latent_format_identity(latent_format)
    payload = {
        "preprocessing_revision": str(preprocessing_revision),
        "image_sha256": image_digest,
        "source_shape": list(image.shape),
        "source_dtype": str(image.dtype),
        "target_latent_width": int(target_latent_width),
        "target_latent_height": int(target_latent_height),
        "target_pixel_width": int(target_pixel_width),
        "target_pixel_height": int(target_pixel_height),
        "vae_identity": vae_identity,
        "preprocess_callable": describe_preprocess_callable(preprocess_image),
        "resize_mode": str(resize_mode),
        "crop_mode": str(crop_mode),
        "interpolation": str(interpolation),
        "latent_format_identity": latent_format_identity,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return HintLatentCacheKey(
        key=f"{preprocessing_revision}:{digest}",
        image_fingerprint=image_digest,
        vae_identity=vae_identity,
        latent_format_identity=latent_format_identity,
    )


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


@dataclass
class HintLatentCacheEntry:
    key: str
    latent: torch.Tensor
    nbytes: int
    vae_identity: str
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    hit_count: int = 0

    def touch(self) -> None:
        self.last_used_at = time.time()
        self.hit_count += 1


class HintLatentCache:
    """Thread-safe, bounded LRU cache containing CPU tensors only."""

    def __init__(
        self,
        *,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        max_cpu_bytes: int = _DEFAULT_MAX_CPU_BYTES,
        enabled: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, HintLatentCacheEntry] = OrderedDict()
        self._total_bytes = 0
        self._max_entries = max(0, int(max_entries))
        self._max_cpu_bytes = max(0, int(max_cpu_bytes))
        self._enabled = bool(enabled)

    def is_enabled(self) -> bool:
        with self._lock:
            return (
                self._enabled
                and self._max_entries > 0
                and self._max_cpu_bytes > 0
            )

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
            self._evict_to_limits_locked(
                reason="configuration",
                diagnostics=diagnostics,
            )

    def get(
        self,
        request: HintLatentCacheKey,
        *,
        diagnostics: bool = False,
    ) -> Optional[torch.Tensor]:
        with self._lock:
            if not self.is_enabled():
                return None
            entry = self._entries.get(request.key)
            if entry is None:
                return None
            if entry.latent.device.type != "cpu":
                # Defensive invariant enforcement; a corrupted entry is removed
                # instead of ever returning a retained CUDA tensor.
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
                    "%s Hint-latent cache hit: reused CPU latent; key=%s, shape=%s, bytes=%d.",
                    PROJECT_LOG_PREFIX,
                    request.short_key,
                    tuple(entry.latent.shape),
                    entry.nbytes,
                )
            # This is an internal read-only reference. The integration point must
            # copy it to the runtime device before any downstream use.
            return entry.latent

    def put(
        self,
        request: HintLatentCacheKey,
        latent: torch.Tensor,
        *,
        diagnostics: bool = False,
    ) -> bool:
        if not isinstance(latent, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor latent, got {type(latent)!r}.")

        nbytes = tensor_nbytes(latent)
        with self._lock:
            if not self.is_enabled() or nbytes > self._max_cpu_bytes:
                if diagnostics and self._enabled and nbytes > self._max_cpu_bytes:
                    logging.info(
                        "%s Hint-latent cache insert skipped: key=%s, bytes=%d exceeds max_cpu_bytes=%d.",
                        PROJECT_LOG_PREFIX,
                        request.short_key,
                        nbytes,
                        self._max_cpu_bytes,
                    )
                return False

        # Copy outside the lock because a GPU-to-CPU transfer may synchronize.
        cpu_latent = latent.detach().to(device="cpu").contiguous()
        if latent.device.type == "cpu":
            cpu_latent = cpu_latent.clone()
        cpu_latent.requires_grad_(False)
        nbytes = tensor_nbytes(cpu_latent)

        with self._lock:
            if not self.is_enabled() or nbytes > self._max_cpu_bytes:
                return False

            previous = self._entries.pop(request.key, None)
            if previous is not None:
                self._total_bytes -= previous.nbytes

            entry = HintLatentCacheEntry(
                key=request.key,
                latent=cpu_latent,
                nbytes=nbytes,
                vae_identity=request.vae_identity,
            )
            self._entries[request.key] = entry
            self._total_bytes += nbytes

            if diagnostics:
                logging.info(
                    "%s Hint-latent cache insert: key=%s, shape=%s, dtype=%s, bytes=%d, entries=%d, total_bytes=%d.",
                    PROJECT_LOG_PREFIX,
                    request.short_key,
                    tuple(cpu_latent.shape),
                    cpu_latent.dtype,
                    nbytes,
                    len(self._entries),
                    self._total_bytes,
                )

            self._evict_to_limits_locked(
                reason="lru_capacity",
                diagnostics=diagnostics,
            )
            return request.key in self._entries

    def clear(
        self,
        *,
        reason: str = "manual",
        diagnostics: bool = False,
    ) -> int:
        with self._lock:
            count_removed = len(self._entries)
            bytes_removed = self._total_bytes
            self._entries.clear()
            self._total_bytes = 0
            if diagnostics and count_removed:
                logging.info(
                    "%s Hint-latent cache clear: reason=%s, entries=%d, bytes=%d.",
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
                        "shape": tuple(entry.latent.shape),
                        "dtype": str(entry.latent.dtype),
                        "device": str(entry.latent.device),
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
            self._remove_locked(
                oldest_key,
                reason=reason,
                diagnostics=diagnostics,
            )

    def _remove_locked(self, key: str, *, reason: str, diagnostics: bool) -> bool:
        entry = self._entries.pop(key, None)
        if entry is None:
            return False
        self._total_bytes -= entry.nbytes
        if diagnostics:
            logging.info(
                "%s Hint-latent cache eviction: key=%s, shape=%s, bytes=%d, reason=%s.",
                PROJECT_LOG_PREFIX,
                key.rsplit(":", 1)[-1][:12],
                tuple(entry.latent.shape),
                entry.nbytes,
                reason,
            )
        entry.latent = torch.empty(0, device="cpu")
        return True


_previous_cache = globals().get("HINT_LATENT_CACHE")
if _previous_cache is not None and hasattr(_previous_cache, "clear"):
    _previous_cache.clear(reason="module_reload", diagnostics=False)

HINT_LATENT_CACHE = HintLatentCache(
    max_entries=max(0, _env_int("JLC_FLUX2_HINT_CACHE_MAX_ENTRIES", _DEFAULT_MAX_ENTRIES)),
    max_cpu_bytes=_env_max_cpu_bytes(),
    enabled=_env_bool("JLC_FLUX2_HINT_CACHE_ENABLED", True),
)


def clear_hint_latent_cache(*, diagnostics: bool = True) -> int:
    """Explicitly clear all cached CPU hint latents."""

    return HINT_LATENT_CACHE.clear(reason="explicit_clear", diagnostics=diagnostics)


def hint_latent_cache_info() -> dict[str, Any]:
    """Return process-local cache diagnostics without exposing tensors."""

    return HINT_LATENT_CACHE.info()
