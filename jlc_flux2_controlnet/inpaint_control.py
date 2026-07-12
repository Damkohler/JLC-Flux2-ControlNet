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


# ComfyUI VAE.encode accepts pixel-space [0, 1] tensors and internally maps
# them to model space with image * 2 - 1. Alibaba/VideoX-Fun instead masks an
# already normalized [-1, 1] image to 0.0. Pixel-space 0.5 therefore reproduces
# that exact neutral model-space fill in ComfyUI.
_NEUTRAL_MASKED_PIXEL_VALUE = 0.5


def _patchify_mask_2x2(mask: torch.Tensor) -> torch.Tensor:
    """Convert [B,1,2H,2W] mask samples into [B,4,H,W] Flux2 patch lanes."""
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask tensor [B,1,H,W], got {tuple(mask.shape)}.")
    if mask.shape[-2] % 2 != 0 or mask.shape[-1] % 2 != 0:
        raise ValueError(f"Mask patchify size must be even, got {tuple(mask.shape)}.")
    b, c, h, w = mask.shape
    mask = mask.view(b, c, h // 2, 2, w // 2, 2)
    mask = mask.permute(0, 1, 3, 5, 2, 4)
    return mask.reshape(b, c * 4, h // 2, w // 2).contiguous()


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
        copied.inpaint_mask_context = None
        copied.masked_image_latent = None
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
        self.edit_canvas_image_original = None
        self.inpaint_mask_context = None
        self.masked_image_latent = None
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
        copied.edit_canvas_image_original = self.edit_canvas_image_original
        copied.inpaint_mask_context = None
        copied.masked_image_latent = None
        copied._diagnostic_inpaint_logged = False
        return copied

    def set_inpaint_conditioning(self, *, mask, edit_canvas_image):
        if mask is None or edit_canvas_image is None:
            raise ValueError(
                "JLCFlux2InpaintControl requires both mask and edit_canvas_image."
            )
        self.inpaint_mask_original = mask
        self.edit_canvas_image_original = edit_canvas_image
        self.inpaint_mask_context = None
        self.masked_image_latent = None
        return self

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        self._diagnostic_inpaint_logged = False
        if self.diagnostics_enabled and self.strength != 0.0:
            logging.info(
                "%s Mask-aware Flux2 ControlNet path active for '%s'; "
                "strength=%.4g, range=%.3f..%.3f. Mask white is editable; "
                "editable pixels use neutral model-space fill before VAE encoding.",
                PROJECT_LOG_PREFIX,
                self.checkpoint_name or "unnamed checkpoint",
                self.strength,
                self.timestep_percent_range[0],
                self.timestep_percent_range[1],
            )

    def cleanup(self):
        self.inpaint_mask_context = None
        self.masked_image_latent = None
        self._diagnostic_inpaint_logged = False
        super().cleanup()

    def _mask_to_bchw(self, mask: torch.Tensor) -> torch.Tensor:
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

    def _image_to_bchw(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected IMAGE tensor, got {type(image)!r}.")
        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError(
                f"Expected IMAGE tensor in BHWC format, got {tuple(image.shape)}."
            )
        return (
            image[:, :, :, :3]
            .movedim(-1, 1)
            .to(dtype=torch.float32)
            .clamp(0.0, 1.0)
            .contiguous()
        )

    @staticmethod
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
                batched_number,
            )
        if image.shape[0] != target_batch:
            image = comfy.controlnet.broadcast_image_to(
                image,
                target_batch,
                batched_number,
            )
        return mask, image

    def _prepare_inpaint_context(
        self,
        x_noisy: torch.Tensor,
        batched_number: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float, float]]:
        if self.inpaint_mask_original is None or self.edit_canvas_image_original is None:
            raise ValueError(
                "Flux2 inpaint control requires mask and edit_canvas_image."
            )

        expected_h = int(x_noisy.shape[-2])
        expected_w = int(x_noisy.shape[-1])

        if (
            self.inpaint_mask_context is not None
            and self.masked_image_latent is not None
            and self.inpaint_mask_context.shape[-2:] == (expected_h, expected_w)
            and self.masked_image_latent.shape[-2:] == (expected_h, expected_w)
        ):
            mask_context = self.inpaint_mask_context
            masked_latent = self.masked_image_latent
        else:
            if self.vae is None:
                raise ValueError(
                    "Flux2 inpaint/mask-aware ControlNet needs a VAE but none was provided."
                )

            compression_ratio = self.compression_ratio
            compression_ratio *= self.vae.spacial_compression_encode()
            target_pixel_width = int(expected_w * compression_ratio)
            target_pixel_height = int(expected_h * compression_ratio)

            mask_bchw = self._mask_to_bchw(self.inpaint_mask_original)
            mask_binary = (mask_bchw >= 0.5).to(dtype=torch.float32)
            mask_for_image = comfy.utils.common_upscale(
                mask_binary,
                target_pixel_width,
                target_pixel_height,
                "nearest-exact",
                "center",
            )

            edit_canvas_bchw = self._image_to_bchw(self.edit_canvas_image_original)
            edit_canvas_bchw = comfy.utils.common_upscale(
                edit_canvas_bchw,
                target_pixel_width,
                target_pixel_height,
                "bilinear",
                "center",
            )
            mask_for_image, edit_canvas_bchw = self._align_batches(
                mask_for_image,
                edit_canvas_bchw,
                batched_number,
            )

            keep_mask_image = (mask_for_image < 0.5).to(
                dtype=edit_canvas_bchw.dtype
            )

            # VideoX-Fun first normalizes the RGB image to [-1, 1] and then
            # multiplies the editable region by zero. ComfyUI normalizes inside
            # VAE.encode, so a 0.5 pixel value is the exact equivalent input.
            masked_image = (
                edit_canvas_bchw * keep_mask_image
                + _NEUTRAL_MASKED_PIXEL_VALUE * (1.0 - keep_mask_image)
            )

            loaded_models = comfy.model_management.loaded_models(
                only_currently_used=True
            )
            try:
                masked_latent = self.vae.encode(masked_image.movedim(1, -1))
            finally:
                comfy.model_management.load_models_gpu(loaded_models)

            if self.latent_format is not None:
                masked_latent = self.latent_format.process_in(masked_latent)

            mask_for_context = comfy.utils.common_upscale(
                mask_binary,
                expected_w * 2,
                expected_h * 2,
                "nearest-exact",
                "center",
            )
            mask_for_context = 1.0 - mask_for_context
            mask_context = _patchify_mask_2x2(mask_for_context)

            self.inpaint_mask_context = mask_context.to(
                device="cpu",
                dtype=torch.float32,
                copy=True,
            ).contiguous()
            self.masked_image_latent = masked_latent.to(
                device="cpu",
                dtype=torch.float32,
                copy=True,
            ).contiguous()
            self.inpaint_mask_context.requires_grad_(False)
            self.masked_image_latent.requires_grad_(False)
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

        return mask_context, masked_latent, _tensor_min_max_mean(mask_context)

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
        mask_context, masked_latent, mask_stats = self._prepare_inpaint_context(
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
            "control_latent=%s, mask=%s, mask_stats=%s, masked_latent=%s, "
            "target_control_context=%s, runtime_control_context=%s, "
            "target_tokens=%s, reference_tokens=%s, img_tokens=%s, "
            "txt_tokens=%s, residuals=%s, norms=%s.",
            PROJECT_LOG_PREFIX,
            request.get("control_context_mode", "control+single_inpaint_context"),
            request.get("control_latent_shape"),
            request.get("mask_context_shape"),
            request.get("mask_context_stats"),
            request.get("masked_latent_shape"),
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
