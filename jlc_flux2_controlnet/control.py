"""ComfyUI-native ControlBase lifecycle for the Flux.2 side model."""

from __future__ import annotations

import logging

import torch
from einops import rearrange

import comfy.controlnet
import comfy.latent_formats
import comfy.model_management
import comfy.model_patcher
import comfy.utils

from .constants import (
    CONTROL_LAYERS,
    EXPECTED_CONTROL_INPUT_CHANNELS,
    EXPECTED_FLUX2_LATENT_CHANNELS,
    EXPECTED_MASK_CHANNELS,
    PROJECT_LOG_PREFIX,
    REQUESTS_KEY,
)
from .hooks import make_injection_hook_group
from .hint_latent_cache import (
    HINT_LATENT_CACHE,
    make_hint_latent_cache_key,
)



def _expand_control_context_for_reference_tokens(
    control_context: torch.Tensor,
    image_tokens: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Pad raw 260-channel control tokens across native Flux.2 references.

    VideoX-Fun appends reference image tokens to the target image sequence and
    appends an equal number of exact-zero control tokens before `control_img_in`.
    The target-control prefix is therefore preserved exactly while references
    participate in the compact side model with a neutral raw control input.
    """
    if control_context.ndim != 3 or image_tokens.ndim != 3:
        raise RuntimeError(
            "Flux2 reference compatibility expects rank-3 token tensors; "
            f"control_context={tuple(control_context.shape)}, "
            f"image_tokens={tuple(image_tokens.shape)}."
        )
    if control_context.shape[0] != image_tokens.shape[0]:
        raise RuntimeError(
            "Flux2 reference compatibility batch mismatch: "
            f"control_context={tuple(control_context.shape)}, "
            f"image_tokens={tuple(image_tokens.shape)}."
        )
    if control_context.shape[-1] != EXPECTED_CONTROL_INPUT_CHANNELS:
        raise RuntimeError(
            f"Expected {EXPECTED_CONTROL_INPUT_CHANNELS} raw control channels, "
            f"got {control_context.shape[-1]}."
        )

    target_tokens = control_context.shape[1]
    native_tokens = image_tokens.shape[1]
    if target_tokens > native_tokens:
        raise RuntimeError(
            "Flux2 control context has more target tokens than the native "
            f"target-plus-reference image sequence: control={target_tokens}, "
            f"native={native_tokens}."
        )

    reference_tokens = native_tokens - target_tokens
    if reference_tokens == 0:
        return control_context, 0

    reference_padding = torch.zeros(
        (
            control_context.shape[0],
            reference_tokens,
            control_context.shape[-1],
        ),
        device=control_context.device,
        dtype=control_context.dtype,
    )
    return torch.cat((control_context, reference_padding), dim=1), reference_tokens


def _reference_modulation_dims(
    temb_mod_params_img,
    *,
    target_tokens: int,
    native_tokens: int,
) -> list[tuple[int, int, int]] | None:
    """Mirror native Flux.2 segmented image modulation when it is active."""
    img_mod_msa, _ = temb_mod_params_img
    shift = img_mod_msa.shift
    modulation_slots = shift.shape[1] if shift.ndim >= 3 else 1

    if modulation_slots == 1:
        return None

    if modulation_slots != 2:
        raise RuntimeError(
            "Unsupported Flux2 image-modulation layout for reference latents: "
            f"shape={tuple(shift.shape)}."
        )
    if native_tokens <= target_tokens:
        raise RuntimeError(
            "Flux2 supplied segmented reference modulation without reference "
            f"tokens: target={target_tokens}, native={native_tokens}."
        )

    return [
        (0, target_tokens, 0),
        (target_tokens, native_tokens, 1),
    ]


class JLCFlux2Control(comfy.controlnet.ControlBase):
    """Hardened single-control implementation with real residual injection."""

    def __init__(
        self,
        control_model=None,
        *,
        load_device=None,
        manual_cast_dtype=None,
        checkpoint_name: str = "",
        lazy_handle=None,
    ):
        super().__init__()
        self.control_model = control_model
        self.load_device = load_device or comfy.model_management.get_torch_device()
        self.manual_cast_dtype = manual_cast_dtype
        self.checkpoint_name = checkpoint_name
        self.lazy_handle = lazy_handle
        self.control_model_wrapped = None
        self.latent_format = comfy.latent_formats.Flux2()
        self.compression_ratio = 1
        if control_model is not None:
            self.control_model_wrapped = comfy.model_patcher.CoreModelPatcher(
                control_model,
                load_device=self.load_device,
                offload_device=comfy.model_management.unet_offload_device(),
            )

        self.extra_hooks = make_injection_hook_group()
        self.model_sampling_current = None
        self.diagnostics_enabled = True
        self._diagnostic_wrapper_seen = False
        self._diagnostic_sidebranch_logged = False
        self._diagnostic_reference_logged = False
        self._diagnostic_injected_blocks: set[int] = set()
        self._diagnostic_injection_logged = False
        self._diagnostic_zero_bypass_logged = False
        self._diagnostic_range_skip_logged = False

    def copy(self):
        copied = JLCFlux2Control(
            None,
            load_device=self.load_device,
            manual_cast_dtype=self.manual_cast_dtype,
            checkpoint_name=self.checkpoint_name,
            lazy_handle=self.lazy_handle,
        )
        copied.control_model = self.control_model
        copied.control_model_wrapped = self.control_model_wrapped
        copied.diagnostics_enabled = self.diagnostics_enabled
        self.copy_to(copied)
        return copied

    def ensure_materialized(self):
        """Bind this shallow control copy to the shared lazy model owner."""
        if self.control_model_wrapped is not None:
            return self.control_model_wrapped
        if self.lazy_handle is None:
            raise RuntimeError(
                "Flux2 control model is unavailable and no lazy checkpoint handle is attached."
            )

        handle = self.lazy_handle.materialize()
        self.control_model = handle.control_model
        self.control_model_wrapped = handle.control_model_wrapped
        self.load_device = handle.load_device
        self.manual_cast_dtype = handle.manual_cast_dtype
        return self.control_model_wrapped

    def get_models(self):
        models = super().get_models()
        # Strength zero is a true bypass: do not materialize or stage the 7.7 GiB side model.
        if self.strength != 0.0:
            models.append(self.ensure_materialized())
        return models

    def inference_memory_requirements(self, dtype):
        return super().inference_memory_requirements(dtype)

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        self.model_sampling_current = model.model_sampling
        self._diagnostic_wrapper_seen = False
        self._diagnostic_sidebranch_logged = False
        self._diagnostic_reference_logged = False
        self._diagnostic_injected_blocks.clear()
        self._diagnostic_injection_logged = False
        self._diagnostic_zero_bypass_logged = False
        self._diagnostic_range_skip_logged = False
        if self.diagnostics_enabled:
            if self.strength == 0.0:
                logging.info(
                    "%s Exact bypass active for '%s' because strength=0; VAE encoding, side-model staging, execution, and injection are skipped.",
                    PROJECT_LOG_PREFIX,
                    self.checkpoint_name or "unnamed checkpoint",
                )
                self._diagnostic_zero_bypass_logged = True
            else:
                logging.info(
                    "%s Hardened control-image path active for '%s'; strength=%.4g, range=%.3f..%.3f.",
                    PROJECT_LOG_PREFIX,
                    self.checkpoint_name or "unnamed checkpoint",
                    self.strength,
                    self.timestep_percent_range[0],
                    self.timestep_percent_range[1],
                )

    def cleanup(self):
        self.model_sampling_current = None
        self._diagnostic_wrapper_seen = False
        self._diagnostic_sidebranch_logged = False
        self._diagnostic_reference_logged = False
        self._diagnostic_injected_blocks.clear()
        self._diagnostic_injection_logged = False
        self._diagnostic_zero_bypass_logged = False
        self._diagnostic_range_skip_logged = False
        super().cleanup()

    def _prepare_control_latent(
        self,
        x_noisy: torch.Tensor,
        batched_number: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.cond_hint_original is None:
            raise ValueError("No control image was provided to JLC Flux2 ControlNet.")

        expected_h = x_noisy.shape[-2]
        expected_w = x_noisy.shape[-1]

        if (
            self.cond_hint is None
            or self.cond_hint.shape[-2] != expected_h
            or self.cond_hint.shape[-1] != expected_w
        ):
            self.cond_hint = None
            compression_ratio = self.compression_ratio
            if self.vae is not None:
                compression_ratio *= self.vae.spacial_compression_encode()
            else:
                raise ValueError(
                    "This Flux2 ControlNet needs a VAE but none was provided; "
                    "please use the JLC apply node with a VAE connection."
                )

            target_pixel_width = x_noisy.shape[-1] * compression_ratio
            target_pixel_height = x_noisy.shape[-2] * compression_ratio
            cache_request = None
            cached_cpu_latent = None

            if HINT_LATENT_CACHE.is_enabled():
                cache_request = make_hint_latent_cache_key(
                    image=self.cond_hint_original,
                    target_latent_width=expected_w,
                    target_latent_height=expected_h,
                    target_pixel_width=target_pixel_width,
                    target_pixel_height=target_pixel_height,
                    vae=self.vae,
                    preprocess_image=self.preprocess_image,
                    interpolation=self.upscale_algorithm,
                    resize_mode="common_upscale",
                    crop_mode="center",
                    latent_format=self.latent_format,
                )
                cached_cpu_latent = HINT_LATENT_CACHE.get(
                    cache_request,
                    diagnostics=self.diagnostics_enabled,
                )

            if cached_cpu_latent is None:
                if self.diagnostics_enabled and cache_request is not None:
                    logging.info(
                        "%s Hint-latent cache miss: encoding control image; key=%s.",
                        PROJECT_LOG_PREFIX,
                        cache_request.short_key,
                    )

                hint = comfy.utils.common_upscale(
                    self.cond_hint_original,
                    target_pixel_width,
                    target_pixel_height,
                    self.upscale_algorithm,
                    "center",
                )
                hint = self.preprocess_image(hint)

                loaded_models = comfy.model_management.loaded_models(
                    only_currently_used=True
                )
                try:
                    hint = self.vae.encode(hint.movedim(1, -1))
                finally:
                    comfy.model_management.load_models_gpu(loaded_models)

                if self.latent_format is not None:
                    hint = self.latent_format.process_in(hint)

                if cache_request is not None:
                    HINT_LATENT_CACHE.put(
                        cache_request,
                        hint,
                        diagnostics=self.diagnostics_enabled,
                    )
                runtime_latent = hint
            else:
                runtime_latent = cached_cpu_latent

            # Always make a runtime-owned copy. A warm-cache hit must never expose
            # the cache's CPU tensor to downstream mutation, even in CPU-only tests.
            self.cond_hint = runtime_latent.to(
                device=x_noisy.device,
                dtype=dtype,
                copy=True,
            )

        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = comfy.controlnet.broadcast_image_to(
                self.cond_hint,
                x_noisy.shape[0],
                batched_number,
            )

        return self.cond_hint

    def _build_control_context(self, control_latent: torch.Tensor) -> torch.Tensor:
        if control_latent.shape[1] != EXPECTED_FLUX2_LATENT_CHANNELS:
            raise ValueError(
                f"Expected {EXPECTED_FLUX2_LATENT_CHANNELS} latent channels, "
                f"got {control_latent.shape[1]}."
            )

        zeros_mask = torch.zeros(
            (
                control_latent.shape[0],
                EXPECTED_MASK_CHANNELS,
                control_latent.shape[2],
                control_latent.shape[3],
            ),
            device=control_latent.device,
            dtype=control_latent.dtype,
        )
        zeros_masked_latent = torch.zeros_like(control_latent)
        stacked = torch.cat(
            [control_latent, zeros_mask, zeros_masked_latent],
            dim=1,
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

        # Exact zero-strength bypass occurs before VAE encoding or request creation.
        if self.strength == 0.0:
            return previous

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if self.diagnostics_enabled and not self._diagnostic_range_skip_logged:
                    logging.info(
                        "%s Control is outside its active timestep range for this denoising call; native Flux.2 output is preserved.",
                        PROJECT_LOG_PREFIX,
                    )
                    self._diagnostic_range_skip_logged = True
                return previous

        if self.control_model_wrapped is None:
            raise RuntimeError(
                "Flux2 side model was not materialized during sampling-model discovery. "
                "This indicates an unexpected ComfyUI lifecycle order."
            )

        dtype = self.control_model.dtype
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        requests = transformer_options.setdefault(REQUESTS_KEY, [])
        # Defensive de-duplication in case ComfyUI revisits the same control
        # object while assembling one model invocation.
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
        control_context = self._build_control_context(control_latent)

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
                "control_context": control_context,
            }
        )

        if previous is not None:
            return previous
        return {"input": [], "middle": [], "output": []}

    def execute_side_branch(self, request: dict, *, img, txt, vec, pe, attn_mask=None):
        if self.control_model_wrapped is None:
            raise RuntimeError(
                "Flux2 control model wrapper is not available. The lazy model must be "
                "materialized through get_models() before side-branch execution."
            )

        target_control_context = request["control_context"].to(
            device=img.device,
            dtype=img.dtype,
        )
        target_tokens = target_control_context.shape[1]
        control_context, reference_tokens = (
            _expand_control_context_for_reference_tokens(
                target_control_context,
                img,
            )
        )
        temb_mod_params_img, temb_mod_params_txt = vec
        modulation_dims_img = _reference_modulation_dims(
            temb_mod_params_img,
            target_tokens=target_tokens,
            native_tokens=img.shape[1],
        )

        request["target_control_tokens"] = target_tokens
        request["reference_tokens"] = reference_tokens
        request["runtime_control_context_shape"] = tuple(control_context.shape)
        request["modulation_dims_img"] = modulation_dims_img

        with torch.no_grad():
            residuals = self.control_model_wrapped.model.forward_control(
                x=img,
                control_context=control_context,
                encoder_hidden_states=txt,
                temb_mod_params_img=temb_mod_params_img,
                temb_mod_params_txt=temb_mod_params_txt,
                image_rotary_emb=pe,
                attention_mask=attn_mask,
                modulation_dims_img=modulation_dims_img,
            )
        return residuals

    def note_diagnostic_wrapper(self, x, context, ref_latents):
        if not self.diagnostics_enabled or self._diagnostic_wrapper_seen:
            return
        self._diagnostic_wrapper_seen = True
        ref_count = 0 if ref_latents is None else len(ref_latents)
        logging.info(
            "%s Native Flux.2 wrapper reached: latent=%s, context=%s, reference_latents=%d.",
            PROJECT_LOG_PREFIX,
            tuple(x.shape),
            None if context is None else tuple(context.shape),
            ref_count,
        )

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
            "%s Side-branch execution confirmed: control_latent=%s, target_control_context=%s, runtime_control_context=%s, target_tokens=%s, reference_tokens=%s, img_tokens=%s, txt_tokens=%s, residuals=%s, norms=%s.",
            PROJECT_LOG_PREFIX,
            request.get("control_latent_shape"),
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
                "%s Reference-latent compatibility active: appended %d exact-zero 260-channel control tokens; compact side-model residuals cover the full target-plus-reference image sequence.",
                PROJECT_LOG_PREFIX,
                request["reference_tokens"],
            )

    def note_diagnostic_injection(self, *, block_index: int, strength: float, residual):
        if not self.diagnostics_enabled:
            return
        self._diagnostic_injected_blocks.add(block_index)
        if (
            not self._diagnostic_injection_logged
            and self._diagnostic_injected_blocks == set(CONTROL_LAYERS)
        ):
            logging.info(
                "%s Residual injection confirmed at double blocks %s with strength=%.4g; final residual dtype=%s, device=%s.",
                PROJECT_LOG_PREFIX,
                CONTROL_LAYERS,
                strength,
                residual.dtype,
                residual.device,
            )
            self._diagnostic_injection_logged = True
