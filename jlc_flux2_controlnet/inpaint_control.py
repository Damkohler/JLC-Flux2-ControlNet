"""Mask-aware Flux.2 ControlNet context builder.

This module keeps the 260-channel inpaint/mask-aware packing path separate from
``JLCFlux2Control`` in ``control.py``.

Shared machinery remains in the validated core:
- lazy checkpoint materialization
- side-model execution
- residual injection hooks
- reference-token zero padding
- hint-latent caching for the control image

Only the specialized context construction path lives here:
    control_latent_128 + packed_keep_mask_4 + masked_source_latent_128
"""

from __future__ import annotations

import logging

import torch
from einops import rearrange

import comfy.controlnet
import comfy.model_management
import comfy.utils

from .constants import (
    EXPECTED_CONTROL_INPUT_CHANNELS,
    EXPECTED_FLUX2_LATENT_CHANNELS,
    EXPECTED_MASK_CHANNELS,
    PROJECT_LOG_PREFIX,
    REQUESTS_KEY,
)
from .control import JLCFlux2Control
from .inpaint_context_cache import (
    INPAINT_CONTEXT_CACHE,
    make_inpaint_context_cache_key,
    prepare_inpaint_context_tensors,
)


# ComfyUI VAE.encode accepts pixel-space [0, 1] tensors and internally maps
# them to model space with image * 2 - 1. Alibaba/VideoX-Fun instead masks an
# already normalized [-1, 1] image to 0.0. Pixel-space 0.5 therefore reproduces
# that exact neutral model-space fill in ComfyUI.
_NEUTRAL_MASKED_PIXEL_VALUE = 0.5


def _tensor_min_max_mean(tensor: torch.Tensor) -> tuple[float, float, float]:
    stats = tensor.detach().to(device="cpu", dtype=torch.float32)
    return (
        float(stats.min().item()),
        float(stats.max().item()),
        float(stats.mean().item()),
    )


class JLCFlux2InpaintControl(JLCFlux2Control):
    """Flux2 ControlNet context builder with one shared inpaint mask/canvas.

    User-facing ComfyUI MASK convention:
        white / 1.0 = editable or regenerated region
        black / 0.0 = preserved or known region

    The source image is replaced by neutral model-space zero in the editable
    region before VAE encoding. The 4 mask lanes carry the inverse keep-region
    mask, matching Alibaba/VideoX-Fun's 260-channel contract.
    """

    @classmethod
    def from_control(cls, control: JLCFlux2Control) -> "JLCFlux2InpaintControl":
        """Upgrade a configured JLC control without losing branch state.

        ``ControlBase.copy_to`` carries the already configured hint, strength,
        timestep range, VAE, preprocessing policy, latent format, and branch
        cache state. The loaded/lazy side-model owner is then reasserted so the
        upgraded object continues to share the proven underlying model.
        """
        copied = cls(
            None,
            load_device=control.load_device,
            manual_cast_dtype=control.manual_cast_dtype,
            checkpoint_name=control.checkpoint_name,
            lazy_handle=control.lazy_handle,
        )

        # This call is essential. Without it, an adapter upgrade silently drops
        # cond_hint_original, strength, timestep range, VAE and related state.
        control.copy_to(copied)

        copied.control_model = control.control_model
        copied.control_model_wrapped = control.control_model_wrapped
        copied.load_device = control.load_device
        copied.manual_cast_dtype = control.manual_cast_dtype
        copied.checkpoint_name = control.checkpoint_name
        copied.lazy_handle = control.lazy_handle
        copied.diagnostics_enabled = getattr(control, "diagnostics_enabled", True)
        copied.inpaint_mask_original = getattr(
            control, "inpaint_mask_original", None
        )
        copied.image_original = getattr(control, "image_original", None)
        copied.inpaint_mask_context = None
        copied.masked_image_latent = None
        copied._inpaint_cache_key = None
        copied._diagnostic_inpaint_logged = False
        return copied

    def __init__(
        self,
        control_model=None,
        *,
        load_device=None,
        manual_cast_dtype=None,
        checkpoint_name: str = "",
        lazy_handle=None,
    ):
        super().__init__(
            control_model,
            load_device=load_device,
            manual_cast_dtype=manual_cast_dtype,
            checkpoint_name=checkpoint_name,
            lazy_handle=lazy_handle,
        )
        self.inpaint_mask_original = None
        self.image_original = None
        self.inpaint_mask_context = None
        self.masked_image_latent = None
        self._inpaint_cache_key = None
        self._diagnostic_inpaint_logged = False

    def copy(self):
        copied = JLCFlux2InpaintControl(
            None,
            load_device=self.load_device,
            manual_cast_dtype=self.manual_cast_dtype,
            checkpoint_name=self.checkpoint_name,
            lazy_handle=self.lazy_handle,
        )
        self.copy_to(copied)
        copied.control_model = self.control_model
        copied.control_model_wrapped = self.control_model_wrapped
        copied.load_device = self.load_device
        copied.manual_cast_dtype = self.manual_cast_dtype
        copied.checkpoint_name = self.checkpoint_name
        copied.lazy_handle = self.lazy_handle
        copied.diagnostics_enabled = self.diagnostics_enabled
        copied.inpaint_mask_original = self.inpaint_mask_original
        copied.image_original = self.image_original
        copied.inpaint_mask_context = None
        copied.masked_image_latent = None
        copied._inpaint_cache_key = None
        copied._diagnostic_inpaint_logged = False
        return copied

    def set_inpaint_conditioning(self, *, mask, image):
        if mask is None or image is None:
            raise ValueError("JLCFlux2InpaintControl requires both mask and image.")
        self.inpaint_mask_original = mask
        self.image_original = image
        self.inpaint_mask_context = None
        self.masked_image_latent = None
        self._inpaint_cache_key = None
        return self

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        self._diagnostic_inpaint_logged = False
        if self.diagnostics_enabled and self.strength != 0.0:
            logging.info(
                "%s Mask-aware Flux2 ControlNet path active for '%s'; "
                "strength=%.4g, range=%.3f..%.3f. Mask white is editable; "
                "hard-thresholded editable pixels use neutral model-space fill "
                "before VAE encoding.",
                PROJECT_LOG_PREFIX,
                self.checkpoint_name or "unnamed checkpoint",
                self.strength,
                self.timestep_percent_range[0],
                self.timestep_percent_range[1],
            )

    def cleanup(self):
        self.inpaint_mask_context = None
        self.masked_image_latent = None
        self._inpaint_cache_key = None
        self._diagnostic_inpaint_logged = False
        super().cleanup()

    def _prepare_inpaint_context(
        self,
        x_noisy: torch.Tensor,
        batched_number: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float, float], str]:
        if self.inpaint_mask_original is None or self.image_original is None:
            raise ValueError("Flux2 inpaint control requires mask and image.")
        if self.vae is None:
            raise ValueError(
                "Flux2 inpaint/mask-aware ControlNet needs a VAE but none was provided."
            )

        expected_h = int(x_noisy.shape[-2])
        expected_w = int(x_noisy.shape[-1])

        # One control object is immutable during a sampling run. Once its local
        # CPU context has been resolved, reuse it without re-hashing source
        # tensors on every denoising step.
        if (
            self.inpaint_mask_context is not None
            and self.masked_image_latent is not None
            and self._inpaint_cache_key is not None
            and self.inpaint_mask_context.shape[-2:] == (expected_h, expected_w)
            and self.masked_image_latent.shape[-2:] == (expected_h, expected_w)
        ):
            mask_context = self.inpaint_mask_context
            masked_latent = self.masked_image_latent
            cache_status = "local_reuse"
        else:
            request = make_inpaint_context_cache_key(
                image=self.image_original,
                mask=self.inpaint_mask_original,
                vae=self.vae,
                latent_format=self.latent_format,
                target_latent_width=expected_w,
                target_latent_height=expected_h,
                control_compression_ratio=self.compression_ratio,
            )

            cached = INPAINT_CONTEXT_CACHE.get(
                request,
                diagnostics=bool(self.diagnostics_enabled),
            )
            if cached is not None:
                mask_context, masked_latent = cached
                cache_status = "shared_hit"
            else:
                mask_context, masked_latent = prepare_inpaint_context_tensors(
                    image=self.image_original,
                    mask=self.inpaint_mask_original,
                    vae=self.vae,
                    latent_format=self.latent_format,
                    target_latent_width=expected_w,
                    target_latent_height=expected_h,
                    control_compression_ratio=self.compression_ratio,
                    batched_number=batched_number,
                    caller_name="JLC Flux2 ControlNet In/Out-Paint Adapter",
                )
                inserted = INPAINT_CONTEXT_CACHE.put(
                    request,
                    mask_context,
                    masked_latent,
                    diagnostics=bool(self.diagnostics_enabled),
                )
                cache_status = "inline_miss_inserted" if inserted else "inline_miss_uncached"
                if inserted:
                    stored = INPAINT_CONTEXT_CACHE.get(request, diagnostics=False)
                    if stored is not None:
                        mask_context, masked_latent = stored

            self.inpaint_mask_context = mask_context.to(
                device="cpu",
                dtype=torch.float32,
                copy=(mask_context.device.type != "cpu"),
            ).contiguous()
            self.masked_image_latent = masked_latent.to(
                device="cpu",
                dtype=torch.float32,
                copy=(masked_latent.device.type != "cpu"),
            ).contiguous()
            self.inpaint_mask_context.requires_grad_(False)
            self.masked_image_latent.requires_grad_(False)
            self._inpaint_cache_key = request.key
            mask_context = self.inpaint_mask_context
            masked_latent = self.masked_image_latent

        mask_context = mask_context.to(
            device=x_noisy.device,
            dtype=dtype,
            copy=True,
        )
        masked_latent = masked_latent.to(
            device=x_noisy.device,
            dtype=dtype,
            copy=True,
        )

        if mask_context.shape[0] != x_noisy.shape[0]:
            mask_context = comfy.controlnet.broadcast_image_to(
                mask_context,
                x_noisy.shape[0],
                batched_number,
            )
        if masked_latent.shape[0] != x_noisy.shape[0]:
            masked_latent = comfy.controlnet.broadcast_image_to(
                masked_latent,
                x_noisy.shape[0],
                batched_number,
            )

        if mask_context.shape[1] != EXPECTED_MASK_CHANNELS:
            raise RuntimeError(
                f"Expected {EXPECTED_MASK_CHANNELS} Flux2 mask channels, "
                f"got {mask_context.shape[1]}."
            )
        if masked_latent.shape[1] != EXPECTED_FLUX2_LATENT_CHANNELS:
            raise RuntimeError(
                f"Expected {EXPECTED_FLUX2_LATENT_CHANNELS} masked-image latent "
                f"channels, got {masked_latent.shape[1]}."
            )

        return (
            mask_context,
            masked_latent,
            _tensor_min_max_mean(mask_context),
            cache_status,
        )

    def _build_inpaint_control_context(
        self,
        control_latent: torch.Tensor,
        mask_context: torch.Tensor,
        masked_latent: torch.Tensor,
    ) -> torch.Tensor:
        if control_latent.shape[1] != EXPECTED_FLUX2_LATENT_CHANNELS:
            raise ValueError(
                f"Expected {EXPECTED_FLUX2_LATENT_CHANNELS} latent channels, "
                f"got {control_latent.shape[1]}."
            )
        expected_spatial = control_latent.shape[-2:]
        if mask_context.shape[-2:] != expected_spatial:
            raise RuntimeError(
                f"Flux2 mask spatial mismatch: control={expected_spatial}, "
                f"mask={mask_context.shape[-2:]}"
            )
        if masked_latent.shape[-2:] != expected_spatial:
            raise RuntimeError(
                f"Flux2 masked latent spatial mismatch: control={expected_spatial}, "
                f"masked={masked_latent.shape[-2:]}"
            )

        stacked = torch.cat(
            [control_latent, mask_context, masked_latent],
            dim=1,
        )
        if stacked.shape[1] != EXPECTED_CONTROL_INPUT_CHANNELS:
            raise RuntimeError(
                f"Expected packed Flux2 control context with "
                f"{EXPECTED_CONTROL_INPUT_CHANNELS} channels, got {stacked.shape[1]}."
            )
        return rearrange(stacked, "b c h w -> b (h w) c")

    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        previous = None
        if self.previous_controlnet is not None:
            previous = self.previous_controlnet.get_control(
                x_noisy,
                t,
                cond,
                batched_number,
                transformer_options,
            )

        if self.strength == 0.0:
            return previous

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if self.diagnostics_enabled and not self._diagnostic_range_skip_logged:
                    logging.info(
                        "%s Control is outside its active timestep range for this "
                        "denoising call; native Flux.2 output is preserved.",
                        PROJECT_LOG_PREFIX,
                    )
                    self._diagnostic_range_skip_logged = True
                return previous

        if self.control_model_wrapped is None:
            raise RuntimeError(
                "Flux2 side model was not materialized during sampling-model "
                "discovery. This indicates an unexpected ComfyUI lifecycle order."
            )

        dtype = self.control_model.dtype
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        requests = transformer_options.setdefault(REQUESTS_KEY, [])
        if any(request.get("control") is self for request in requests):
            return previous if previous is not None else {
                "input": [],
                "middle": [],
                "output": [],
            }

        control_latent = self._prepare_control_latent(
            x_noisy,
            batched_number,
            dtype,
        )
        mask_context, masked_latent, mask_stats, inpaint_cache_status = self._prepare_inpaint_context(
            x_noisy,
            batched_number,
            dtype,
        )
        control_context = self._build_inpaint_control_context(
            control_latent,
            mask_context,
            masked_latent,
        )

        requests.append(
            {
                "control": self,
                "strength": float(self.strength),
                "checkpoint_name": self.checkpoint_name,
                "hint_shape": (
                    tuple(self.cond_hint_original.shape)
                    if self.cond_hint_original is not None
                    else None
                ),
                "control_latent_shape": tuple(control_latent.shape),
                "mask_context_shape": tuple(mask_context.shape),
                "mask_context_stats": mask_stats,
                "masked_latent_shape": tuple(masked_latent.shape),
                "inpaint_context_cache_status": inpaint_cache_status,
                "control_context_mode": "control+single_inpaint_context",
                "control_context": control_context,
            }
        )

        if previous is not None:
            return previous
        return {"input": [], "middle": [], "output": []}

    def note_diagnostic_sidebranch(self, request, img, txt, residuals):
        if not self.diagnostics_enabled or self._diagnostic_sidebranch_logged:
            return
        self._diagnostic_sidebranch_logged = True
        residual_shapes = [tuple(residual.shape) for residual in residuals]
        residual_norms = [
            round(float(residual.float().norm().item()), 4)
            for residual in residuals
        ]
        logging.info(
            "%s Inpaint side-branch execution confirmed: mode=%s, "
            "control_latent=%s, mask=%s, mask_stats=%s, masked_latent=%s, cache=%s, "
            "target_control_context=%s, runtime_control_context=%s, "
            "target_tokens=%s, reference_tokens=%s, img_tokens=%s, "
            "txt_tokens=%s, residuals=%s, norms=%s.",
            PROJECT_LOG_PREFIX,
            request.get("control_context_mode", "control+single_inpaint_context"),
            request.get("control_latent_shape"),
            request.get("mask_context_shape"),
            request.get("mask_context_stats"),
            request.get("masked_latent_shape"),
            request.get("inpaint_context_cache_status", "unknown"),
            tuple(request["control_context"].shape),
            request.get("runtime_control_context_shape"),
            request.get("target_control_tokens"),
            request.get("reference_tokens", 0),
            tuple(img.shape),
            tuple(txt.shape),
            residual_shapes,
            residual_norms,
        )
        if (
            request.get("reference_tokens", 0) > 0
            and not self._diagnostic_reference_logged
        ):
            self._diagnostic_reference_logged = True
            logging.info(
                "%s Reference-latent compatibility active for inpaint control: "
                "appended %d exact-zero 260-channel control tokens.",
                PROJECT_LOG_PREFIX,
                request["reference_tokens"],
            )
