"""
JLC Flux2 Inpaint Context Cache Experimental
--------------------------------------------

Pre-warms the bounded CPU cache for the static hard-mask inpaint context used
by the JLC Flux2 ControlNet In/Out-Paint Adapter Experimental.

The node derives target geometry from the connected LATENT instead of exposing
independent width/height widgets. IMAGE and MASK must already match that target
canvas exactly. Silent resizing is rejected.

Cached values:
    packed inverse keep-mask context [B,4,H,W]
    masked-source Flux2 latent       [B,128,H,W]

Use this node in a mutually exclusive preparation branch before inference in
the same ComfyUI server process. Existing workflows remain valid without it;
the runtime adapter falls back to inline preparation on a cache miss.
"""

from __future__ import annotations

from ..jlc_flux2_controlnet_versions import JLC_FLUX2_CONTROLNET_VERSION

import logging

import torch

import comfy.latent_formats

from ..jlc_flux2_controlnet.constants import (
    EXPECTED_FLUX2_LATENT_CHANNELS,
    PROJECT_LOG_PREFIX,
)
from ..jlc_flux2_controlnet.inpaint_context_cache import (
    INPAINT_CONTEXT_CACHE,
    INPAINT_CONTEXT_CONTRACT_REVISION,
    clear_inpaint_context_cache,
    inpaint_context_cache_info,
    make_inpaint_context_cache_key,
    prepare_inpaint_context_tensors,
)


INPAINT_CONTEXT_CACHE_PREP_VERSION = "0.1.0"

MANIFEST = {
    "name": "JLC Flux2 Inpaint Context Cache Experimental",
    "version": INPAINT_CONTEXT_CACHE_PREP_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Pre-populates the bounded process-local CPU cache for the hard-mask "
        "FLUX.2 inpaint context. Target geometry is derived directly from a "
        "connected LATENT, and mismatched IMAGE/MASK dimensions are rejected "
        "instead of silently resized."
    ),
    "base_package_version": JLC_FLUX2_CONTROLNET_VERSION,
    "cache_contract_revision": INPAINT_CONTEXT_CONTRACT_REVISION,
    "status": "experimental",
    "license": "MIT",
}


class JLCFlux2InpaintContextCachePrep:
    """Pre-warm one hard-mask inpaint context in the shared CPU cache."""

    EXPERIMENTAL = True
    CATEGORY = "Flux2 Latents Cache/utils"
    FUNCTION = "prepare"
    RETURN_TYPES = ("IMAGE", "MASK", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image", "mask", "cache_set", "cache_report")
    DESCRIPTION = (
        "Pre-warms the CPU cache for one FLUX.2 hard-mask inpaint context. "
        "IMAGE and MASK must exactly match the connected LATENT canvas."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "latent": ("LATENT",),
                "clear_before_prepare": ("BOOLEAN", {"default": False}),
                "diagnostics": ("BOOLEAN", {"default": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def prepare(
        self,
        vae,
        image,
        mask,
        latent,
        clear_before_prepare=False,
        diagnostics=True,
    ):
        if clear_before_prepare:
            clear_inpaint_context_cache(diagnostics=bool(diagnostics))

        if vae is None:
            raise ValueError("JLC Flux2 Inpaint Context Cache requires a VAE.")
        if not isinstance(latent, dict) or not isinstance(latent.get("samples"), torch.Tensor):
            raise ValueError(
                "JLC Flux2 Inpaint Context Cache requires a LATENT containing a tensor under 'samples'."
            )

        samples = latent["samples"]
        if samples.ndim != 4:
            raise ValueError(
                f"Expected LATENT samples [B,C,H,W], got {tuple(samples.shape)}."
            )
        if int(samples.shape[1]) != EXPECTED_FLUX2_LATENT_CHANNELS:
            raise ValueError(
                f"Expected {EXPECTED_FLUX2_LATENT_CHANNELS} Flux2 latent channels, "
                f"got {samples.shape[1]}."
            )

        expected_h = int(samples.shape[-2])
        expected_w = int(samples.shape[-1])
        latent_format = comfy.latent_formats.Flux2()

        request = make_inpaint_context_cache_key(
            image=image,
            mask=mask,
            vae=vae,
            latent_format=latent_format,
            target_latent_width=expected_w,
            target_latent_height=expected_h,
            control_compression_ratio=1,
        )

        cached = INPAINT_CONTEXT_CACHE.get(
            request,
            diagnostics=bool(diagnostics),
        )
        if cached is not None:
            info = inpaint_context_cache_info()
            report = (
                "JLC Flux2 inpaint-context prep complete: hit=1, miss=0, "
                f"inserted=0, key={request.short_key}, "
                f"cache_entries={info['entry_count']}, total_bytes={info['total_bytes']}."
            )
            if diagnostics:
                logging.info("%s %s", PROJECT_LOG_PREFIX, report)
            return image, mask, True, report

        if diagnostics:
            logging.info(
                "%s Inpaint-context prep cache miss: preparing image=%s, mask=%s, target_latent=%dx%d, key=%s.",
                PROJECT_LOG_PREFIX,
                tuple(image.shape),
                tuple(mask.shape),
                expected_w,
                expected_h,
                request.short_key,
            )

        mask_context, masked_latent = prepare_inpaint_context_tensors(
            image=image,
            mask=mask,
            vae=vae,
            latent_format=latent_format,
            target_latent_width=expected_w,
            target_latent_height=expected_h,
            control_compression_ratio=1,
            batched_number=1,
            caller_name="JLC Flux2 Inpaint Context Cache",
        )

        inserted = INPAINT_CONTEXT_CACHE.put(
            request,
            mask_context,
            masked_latent,
            diagnostics=bool(diagnostics),
        )
        info = inpaint_context_cache_info()
        report = (
            "JLC Flux2 inpaint-context prep complete: hit=0, miss=1, "
            f"inserted={int(bool(inserted))}, key={request.short_key}, "
            f"cache_entries={info['entry_count']}, total_bytes={info['total_bytes']}."
        )
        if diagnostics:
            logging.info("%s %s", PROJECT_LOG_PREFIX, report)
        return image, mask, bool(inserted), report
