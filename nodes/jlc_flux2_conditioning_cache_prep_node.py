"""
JLC Flux2 Conditioning Cache Prep
--------------------------------

- JLC Flux2 ControlNet
  - This node is part of the **JLC Flux2 ControlNet** package developed
    by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/JLC-Flux2-ControlNet

- Node Purpose
  - The **JLC Flux2 Conditioning Cache Prep** node provides one workflow-facing
    preparation surface for the three existing bounded, process-local CPU cache
    domains used by JLC Flux2 ControlNet:

        • reusable reference-image VAE latents
        • reusable ControlNet hint latents
        • reusable hard-mask inpaint context tensors

  - It does **not** merge those backends or alter their independent key spaces,
    limits, value contracts, or eviction policies.

- Workflow Contract
  - Use this node in a mutually exclusive cache-preparation branch before the
    normal inference branch in the same ComfyUI server process.

  - The node is intentionally **not** an output sink. A downstream lazy Switch,
    Group Controller, Any Switch, Conditional Save, or equivalent sink must
    request the preparation branch.

  - The `cache_ready_image` output is emitted only when every **active**
    conditioning input received by this execution is either already cached or is
    inserted successfully during this run.

  - If no conditioning domain is active, preparation completes successfully as
    `no_cache_required`. No cache is cleared or modified. A small readable CPU IMAGE token
    is emitted so existing image-routed branch control remains valid.

- Active vs Inactive Input Semantics
  - A dedicated hidden frontend contract records which selected sockets are
    physically wired before ComfyUI prunes muted or bypassed upstream paths.

  - For selected reference and ControlNet slots:

        • physically wired + IMAGE at runtime -> prepare/cache it
        • physically wired + absent/None      -> intentionally inactive; skip it
        • not physically wired                -> configuration error

  - For inpaint:

        • both sockets wired and both absent/None -> intentionally inactive
        • both sockets wired and both active      -> prepare/cache context
        • either socket not wired or partial runtime activity -> error

  - If the frontend contract is unavailable, the node deliberately falls back
    to the earlier strict behavior where an absent runtime input is treated as a
    missing wire. This preserves safe headless/API execution.

  - `empty_flux2_latent` is required only when an **active** ControlNet hint or
    active inpaint pair is present. Connect the same Empty Flux2 Latent used by
    the sampler, not a sampler-output latent.

- Cache Safety
  - This node consumes the three released cache backends through shared helper
    functions and does not replace them.

  - Existing workflows using the specialist cache-prep nodes remain valid and
    unchanged.

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
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ..jlc_flux2_controlnet_versions import JLC_FLUX2_CONTROLNET_VERSION
from ..jlc_flux2_controlnet.conditioning_cache_prep_helpers import (
    MAX_CONTROL_HINTS,
    MAX_REFERENCE_IMAGES,
    CachePrepResult,
    clear_selected_caches,
    derive_flux2_target_geometry,
    prepare_hint_cache_entries,
    prepare_inpaint_cache_entry,
    prepare_reference_cache_entries,
    validate_inpaint_selection,
    validate_selected_images,
)
from ..jlc_flux2_controlnet.constants import PROJECT_LOG_PREFIX


CONDITIONING_CACHE_PREP_NODE_VERSION = JLC_FLUX2_CONTROLNET_VERSION
INPUT_CONNECTION_CONTRACT_REVISION = "jlc-flux2-conditioning-cache-input-wires-v1"

MANIFEST = {
    "name": "JLC Flux2 Conditioning Cache Prep",
    "version": CONDITIONING_CACHE_PREP_NODE_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Unified branch-driven preparation surface for the existing reference, "
        "ControlNet hint, and experimental inpaint CPU cache backends. A "
        "revisioned frontend wire contract distinguishes physically wired inputs "
        "that are muted/pruned at runtime from genuinely missing selected wires. "
        "If every selected domain is inactive, the node completes as "
        "no_cache_required without touching any backend. The IMAGE output is "
        "emitted only after every active selected cache entry is confirmed ready."
    ),
    "base_package_version": JLC_FLUX2_CONTROLNET_VERSION,
    "cache_backends": (
        "REFERENCE_LATENT_CACHE",
        "HINT_LATENT_CACHE",
        "INPAINT_CONTEXT_CACHE",
    ),
    "input_connection_contract_revision": INPUT_CONNECTION_CONTRACT_REVISION,
    "status": "experimental",
    "license": "MIT",
}


class JLCFlux2ConditioningCachePrep:
    """Pre-warm any selected combination of the three existing cache domains."""

    CATEGORY = "Flux2 Latents Cache/utils"
    FUNCTION = "prepare"
    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("cache_ready_image", "cache_set", "cache_report")
    DESCRIPTION = (
        "Pre-warms reference, ControlNet hint, and/or hard-mask inpaint caches. "
        "Physically wired sockets may be muted/pruned and therefore absent at "
        "runtime; genuinely missing selected wires remain errors. With no active "
        "conditioning inputs, the node completes successfully as no_cache_required. "
        "Otherwise, the first output becomes available only after all active "
        "selected entries are cache hits or successful inserts."
    )

    @classmethod
    def INPUT_TYPES(cls):
        optional: dict[str, tuple] = {
            "empty_flux2_latent": (
                "LATENT",
                {
                    "tooltip": (
                        "Connect the same Empty Flux2 Latent used by the sampler. "
                        "Do not connect the sampler output or any sampled latent. "
                        "This is required only when an active ControlNet hint or "
                        "active inpaint pair is present."
                    ),
                },
            ),
        }
        optional.update(
            {
                f"reference_image_{index}": ("IMAGE",)
                for index in range(1, MAX_REFERENCE_IMAGES + 1)
            }
        )
        optional.update(
            {
                f"control_image_{index}": ("IMAGE",)
                for index in range(1, MAX_CONTROL_HINTS + 1)
            }
        )
        optional.update(
            {
                "inpaint_image": ("IMAGE",),
                "inpaint_mask": ("MASK",),
            }
        )

        return {
            "required": {
                "vae": ("VAE",),
                "reference_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_REFERENCE_IMAGES,
                        "step": 1,
                        "tooltip": "Number of reference-image cache inputs exposed by the layout button.",
                    },
                ),
                "control_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_CONTROL_HINTS,
                        "step": 1,
                        "tooltip": "Number of ControlNet hint cache inputs exposed by the layout button.",
                    },
                ),
                "use_inpaint": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Expose one inpaint IMAGE/MASK pair. If both connected "
                            "inputs resolve to None at execution time, inpaint is "
                            "treated as intentionally inactive and skipped."
                        ),
                    },
                ),
                "clear_before_prepare": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Clear only the active cache domains before preparing them.",
                    },
                ),
                "diagnostics": ("BOOLEAN", {"default": True}),
                "input_connection_contract": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": False,
                        "tooltip": (
                            "Hidden frontend-generated record of which dynamic input "
                            "sockets are physically wired. Do not edit manually."
                        ),
                    },
                ),
            },
            "optional": optional,
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @staticmethod
    def _validated_count(name: str, value: Any, maximum: int) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, not BOOLEAN.")
        try:
            count = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer.") from exc
        if count < 0 or count > maximum:
            raise ValueError(f"{name} must be between 0 and {maximum}; got {count}.")
        return count

    @staticmethod
    def _parse_connection_contract(
        raw_contract: Any,
        *,
        reference_count: int,
        control_count: int,
        use_inpaint: bool,
    ) -> frozenset[str] | None:
        """Return physically wired input names, or None for strict fallback.

        An empty/default contract means the dedicated frontend did not serialize
        wire state. In that case the node preserves the v0.1.3 behavior and uses
        runtime keyword presence as the only available indication of wiring.
        """

        if raw_contract is None:
            return None
        if not isinstance(raw_contract, str):
            raise ValueError(
                "JLC Flux2 Conditioning Cache Prep received a non-string input "
                "connection contract. Re-add the node after updating the frontend."
            )

        raw_contract = raw_contract.strip()
        if not raw_contract or raw_contract == "{}":
            return None

        try:
            payload = json.loads(raw_contract)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "JLC Flux2 Conditioning Cache Prep received a malformed input "
                "connection contract. Hard-refresh the ComfyUI frontend and re-add "
                "the node."
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "JLC Flux2 Conditioning Cache Prep input connection contract must "
                "be a JSON object."
            )
        if payload.get("revision") != INPUT_CONNECTION_CONTRACT_REVISION:
            raise ValueError(
                "JLC Flux2 Conditioning Cache Prep received an unsupported input "
                "connection contract revision. Hard-refresh the ComfyUI frontend "
                "and re-add the node."
            )

        layout = payload.get("layout")
        if not isinstance(layout, dict):
            raise ValueError(
                "JLC Flux2 Conditioning Cache Prep input connection contract is "
                "missing its layout record."
            )

        expected_layout = {
            "reference_count": int(reference_count),
            "control_count": int(control_count),
            "use_inpaint": bool(use_inpaint),
        }
        actual_layout = {
            "reference_count": layout.get("reference_count"),
            "control_count": layout.get("control_count"),
            "use_inpaint": layout.get("use_inpaint"),
        }
        if actual_layout != expected_layout:
            raise ValueError(
                "JLC Flux2 Conditioning Cache Prep input connection contract does "
                "not match the current selectors. Press Apply Input Layout and "
                "queue the workflow again."
            )

        wired = payload.get("wired")
        if not isinstance(wired, dict) or any(
            not isinstance(name, str) or not isinstance(state, bool)
            for name, state in wired.items()
        ):
            raise ValueError(
                "JLC Flux2 Conditioning Cache Prep input connection contract has "
                "an invalid wired-input record."
            )

        return frozenset(name for name, state in wired.items() if state)

    @staticmethod
    def _is_physically_wired(
        name: str,
        *,
        kwargs: dict[str, Any],
        wired_inputs: frozenset[str] | None,
    ) -> bool:
        if wired_inputs is None:
            return name in kwargs
        return name in wired_inputs

    @classmethod
    def _selected_images(
        cls,
        kwargs: dict[str, Any],
        *,
        prefix: str,
        count: int,
        label: str,
        wired_inputs: frozenset[str] | None,
    ) -> list[torch.Tensor]:
        active: list[torch.Tensor] = []
        missing: list[str] = []
        for index in range(1, count + 1):
            name = f"{prefix}{index}"
            if not cls._is_physically_wired(
                name,
                kwargs=kwargs,
                wired_inputs=wired_inputs,
            ):
                missing.append(name)
                continue

            # A physically wired input can be absent from kwargs when ComfyUI
            # prunes a muted/bypassed upstream branch. Treat absence and explicit
            # None equivalently as intentionally inactive.
            value = kwargs.get(name)
            if value is None:
                continue
            active.append(value)

        if missing:
            raise ValueError(
                f"JLC Flux2 Conditioning Cache Prep selected {count} {label} input(s), "
                f"but these sockets are not physically wired: {', '.join(missing)}. "
                "A wired slot may be muted/pruned and absent at runtime, but a "
                "genuinely missing selected wire remains a configuration error."
            )
        return active

    @classmethod
    def _selected_inpaint_state(
        cls,
        kwargs: dict[str, Any],
        *,
        use_inpaint: bool,
        wired_inputs: frozenset[str] | None,
    ) -> tuple[bool, Any, Any]:
        if not use_inpaint:
            return False, None, None

        has_image_wire = cls._is_physically_wired(
            "inpaint_image",
            kwargs=kwargs,
            wired_inputs=wired_inputs,
        )
        has_mask_wire = cls._is_physically_wired(
            "inpaint_mask",
            kwargs=kwargs,
            wired_inputs=wired_inputs,
        )
        if not has_image_wire or not has_mask_wire:
            missing = []
            if not has_image_wire:
                missing.append("inpaint_image")
            if not has_mask_wire:
                missing.append("inpaint_mask")
            raise ValueError(
                "JLC Flux2 Conditioning Cache Prep has use_inpaint=True, but these "
                f"selected sockets are not physically wired: {', '.join(missing)}. "
                "A fully wired inpaint path may be muted/pruned and absent at "
                "runtime, but missing selected wires remain configuration errors."
            )

        image = kwargs.get("inpaint_image")
        mask = kwargs.get("inpaint_mask")
        image_active = image is not None
        mask_active = mask is not None
        if not image_active and not mask_active:
            return False, None, None
        if image_active != mask_active:
            raise ValueError(
                "JLC Flux2 Conditioning Cache Prep received a partial active inpaint "
                "selection. Physically wired Inpaint IMAGE and MASK inputs must "
                "either both produce active values or both be muted/pruned."
            )
        return True, image, mask

    @classmethod
    def _reject_unselected_connections(
        cls,
        kwargs: dict[str, Any],
        *,
        reference_count: int,
        control_count: int,
        use_inpaint: bool,
        wired_inputs: frozenset[str] | None,
    ) -> None:
        def wired(name: str) -> bool:
            return cls._is_physically_wired(
                name,
                kwargs=kwargs,
                wired_inputs=wired_inputs,
            )

        stale: list[str] = []
        for index in range(reference_count + 1, MAX_REFERENCE_IMAGES + 1):
            name = f"reference_image_{index}"
            if wired(name):
                stale.append(name)
        for index in range(control_count + 1, MAX_CONTROL_HINTS + 1):
            name = f"control_image_{index}"
            if wired(name):
                stale.append(name)
        if not use_inpaint:
            if wired("inpaint_image"):
                stale.append("inpaint_image")
            if wired("inpaint_mask"):
                stale.append("inpaint_mask")
        if control_count == 0 and not use_inpaint and wired("empty_flux2_latent"):
            stale.append("empty_flux2_latent")
        for legacy_name in ("target_latent", "passthrough_image"):
            if wired(legacy_name):
                stale.append(legacy_name)
        if stale:
            raise ValueError(
                "JLC Flux2 Conditioning Cache Prep received physically wired "
                "sockets outside the selected layout: "
                + ", ".join(stale)
                + ". Disconnect them or press Apply Input Layout after changing "
                "the selectors."
            )

    @staticmethod
    def _no_cache_required_image() -> torch.Tensor:
        """Return a small readable CPU IMAGE token for the no-cache-required path."""

        width = 320
        height = 64
        background_rgb = (56, 56, 56)
        text_rgb = (245, 245, 245)
        message = "No Images to Cache"

        image = Image.new("RGB", (width, height), background_rgb)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        try:
            bbox = draw.textbbox((0, 0), message, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(message, font=font)

        x = max((width - text_width) // 2, 8)
        y = max((height - text_height) // 2, 8)
        draw.text((x, y), message, fill=text_rgb, font=font)

        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0).to(device="cpu")

    @staticmethod
    def _cache_ready_image(
        *,
        active_reference_images: list[torch.Tensor],
        active_control_images: list[torch.Tensor],
        active_inpaint: bool,
        inpaint_image: torch.Tensor | None,
    ) -> torch.Tensor:
        if active_inpaint:
            if inpaint_image is None:
                raise RuntimeError("Inpaint was active without an inpaint image.")
            return inpaint_image
        if active_control_images:
            return active_control_images[-1]
        if active_reference_images:
            return active_reference_images[-1]
        raise RuntimeError("No active image is available for cache-ready routing.")

    @staticmethod
    def _combined_report(
        *,
        results: list[CachePrepResult],
        geometry,
        cleared: dict[str, int],
        status: str,
    ) -> str:
        report_parts = [result.summary() for result in results]
        if geometry is not None:
            report_parts.insert(
                0,
                "target_geometry="
                f"{geometry.pixel_width}x{geometry.pixel_height}px/"
                f"{geometry.latent_width}x{geometry.latent_height} latent",
            )
        if cleared:
            cleared_text = ", ".join(
                f"{domain}={count}" for domain, count in cleared.items()
            )
            report_parts.insert(0, f"cleared[{cleared_text}]")
        return f"JLC Flux2 conditioning cache prep {status}: " + " | ".join(
            report_parts
        )

    def prepare(
        self,
        vae,
        reference_count=0,
        control_count=0,
        use_inpaint=False,
        clear_before_prepare=False,
        diagnostics=True,
        input_connection_contract="{}",
        **kwargs,
    ):
        reference_count = self._validated_count(
            "reference_count", reference_count, MAX_REFERENCE_IMAGES
        )
        control_count = self._validated_count(
            "control_count", control_count, MAX_CONTROL_HINTS
        )
        use_inpaint = bool(use_inpaint)
        diagnostics = bool(diagnostics)

        if vae is None:
            raise ValueError("JLC Flux2 Conditioning Cache Prep requires a VAE.")

        wired_inputs = self._parse_connection_contract(
            input_connection_contract,
            reference_count=reference_count,
            control_count=control_count,
            use_inpaint=use_inpaint,
        )

        self._reject_unselected_connections(
            kwargs,
            reference_count=reference_count,
            control_count=control_count,
            use_inpaint=use_inpaint,
            wired_inputs=wired_inputs,
        )

        active_reference_images = self._selected_images(
            kwargs,
            prefix="reference_image_",
            count=reference_count,
            label="reference image",
            wired_inputs=wired_inputs,
        )
        active_control_images = self._selected_images(
            kwargs,
            prefix="control_image_",
            count=control_count,
            label="ControlNet hint",
            wired_inputs=wired_inputs,
        )

        active_inpaint, inpaint_image, inpaint_mask = self._selected_inpaint_state(
            kwargs,
            use_inpaint=use_inpaint,
            wired_inputs=wired_inputs,
        )

        if not active_reference_images and not active_control_images and not active_inpaint:
            report = (
                "JLC Flux2 conditioning cache prep no_cache_required: "
                "no active reference images, ControlNet hints, or inpaint pair were "
                "received. No cache domain was cleared or modified."
            )
            if diagnostics:
                logging.info("%s %s", PROJECT_LOG_PREFIX, report)
            return self._no_cache_required_image(), True, report

        geometry = None
        if active_control_images or active_inpaint:
            if "empty_flux2_latent" not in kwargs or kwargs.get("empty_flux2_latent") is None:
                raise ValueError(
                    "JLC Flux2 Conditioning Cache Prep requires empty_flux2_latent "
                    "whenever an active ControlNet hint or active inpaint pair is "
                    "present. Connect the same Empty Flux2 Latent used by the sampler; "
                    "do not connect the sampler output latent."
                )
            geometry = derive_flux2_target_geometry(vae, kwargs.get("empty_flux2_latent"))

        validate_selected_images(
            reference_images=active_reference_images,
            control_images=active_control_images,
        )
        if active_inpaint:
            assert geometry is not None
            validate_inpaint_selection(
                image=inpaint_image,
                mask=inpaint_mask,
                geometry=geometry,
            )

        active_references = bool(active_reference_images)
        active_hints = bool(active_control_images)
        cleared: dict[str, int] = {}
        if clear_before_prepare:
            cleared = clear_selected_caches(
                references=active_references,
                hints=active_hints,
                inpaint=active_inpaint,
                diagnostics=diagnostics,
            )

        results: list[CachePrepResult] = []
        if active_references:
            results.append(
                prepare_reference_cache_entries(
                    vae=vae,
                    images=active_reference_images,
                    diagnostics=diagnostics,
                )
            )
        if active_hints:
            assert geometry is not None
            results.append(
                prepare_hint_cache_entries(
                    vae=vae,
                    images=active_control_images,
                    geometry=geometry,
                    diagnostics=diagnostics,
                )
            )
        if active_inpaint:
            assert geometry is not None
            results.append(
                prepare_inpaint_cache_entry(
                    vae=vae,
                    image=inpaint_image,
                    mask=inpaint_mask,
                    geometry=geometry,
                    diagnostics=diagnostics,
                )
            )

        cache_set = bool(results) and all(result.complete for result in results)
        status = "complete" if cache_set else "incomplete"
        report = self._combined_report(
            results=results,
            geometry=geometry,
            cleared=cleared,
            status=status,
        )

        if not cache_set:
            logging.error("%s %s", PROJECT_LOG_PREFIX, report)
            raise RuntimeError(
                report
                + ". The cache-ready image was withheld, so the inference branch "
                "cannot proceed from this node."
            )

        if diagnostics:
            logging.info("%s %s", PROJECT_LOG_PREFIX, report)

        cache_ready_image = self._cache_ready_image(
            active_reference_images=active_reference_images,
            active_control_images=active_control_images,
            active_inpaint=active_inpaint,
            inpaint_image=inpaint_image,
        )
        return cache_ready_image, True, report
