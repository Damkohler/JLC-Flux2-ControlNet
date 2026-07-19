"""
JLC Flux2 Inpaint Context Cache - Experimental
-----------------------------------------------

- Project and Release Status
  - This node is part of **JLC Flux2 ControlNet Release 1.0.0**, developed by
    **J. L. Córdova**.

  - It is included as an **Experimental** companion to the JLC Flux2 ControlNet
    In/Out-Paint Adapter and reports the package version through
    ``JLC_FLUX2_CONTROLNET_VERSION``.

- Node Purpose
  - Pre-warms the bounded, process-local CPU cache for the static hard-mask
    inpaint context used by the Experimental In/Out-Paint Adapter.

  - Cached values:
        • packed inverse keep-mask context [B,4,H,W]
        • masked-source Flux2 latent       [B,128,H,W]

  - The cache retains no GPU tensors, sampler state, residuals, token blocks,
    conditioning objects, or model patches.

- Workflow Contract
  - Use this node in a mutually exclusive cache-preparation branch before the
    normal inference branch in the same ComfyUI server process.

  - The node is intentionally not an output node. A downstream lazy Switch,
    Group Controller, Any Switch, or equivalent sink must request the
    preparation branch.

  - Existing workflows remain valid without this node. On a shared-cache miss,
    the runtime adapter falls back to inline inpaint-context preparation.

- Geometry Contract
  - Target geometry is derived from the connected Flux2 ``LATENT``.

  - ``IMAGE`` and ``MASK`` must already match the latent canvas exactly.
    Automatic spatial resizing is deliberately rejected to prevent silent
    mask/canvas misalignment and artifact-producing runs.

  - The connected LATENT should come from the same Empty Flux2 Latent source
    used by the sampler, not from the sampler output.

- Cache Identity and Safety
  - Cache identity covers exact image content, thresholded mask content,
    target latent and pixel dimensions, VAE identity, Flux2 latent format,
    hard-mask preprocessing contract, and cache-contract revision.

  - Cached tensors are detached, contiguous CPU float32 tensors held in a
    bounded LRU cache.

- Experimental Limitations
  - This node accelerates reusable masked-source VAE preparation. It does not
    reduce ControlNet side-model execution, reference-token count, residual
    tensor size, or conflicts caused by additional dense ControlNet branches.

  - The hard-mask adapter remains experimental and may show seed-variable
    boundary artifacts or source-structure imprinting from non-host controls.

- Attribution and License
  - Concept and implementation by **J. L. Córdova**, with development
    assistance from **ChatGPT (OpenAI)**.

  - Repository:
    https://github.com/Damkohler/JLC-Flux2-ControlNet

  - Designed to interoperate with ComfyUI:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova
  - Released under the **MIT License**.
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
    "name": "JLC Flux2 Inpaint Context Cache - Experimental",
    "version": INPAINT_CONTEXT_CACHE_PREP_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Pre-populates the bounded process-local CPU cache for the hard-mask "
        "FLUX.2 inpaint context. Target geometry is derived directly from a "
        "connected LATENT, and mismatched IMAGE/MASK dimensions are rejected "
        "instead of silently resized."
    ),
    "base_package_version": JLC_FLUX2_CONTROLNET_VERSION,
    "release_track": f"JLC Flux2 ControlNet {JLC_FLUX2_CONTROLNET_VERSION}",
    "capabilities": (
        "shared_inpaint_context_cpu_cache",
        "strict_canvas_geometry_validation",
        "inline_runtime_fallback_compatibility",
        "bounded_cpu_lru_cache",
    ),
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
