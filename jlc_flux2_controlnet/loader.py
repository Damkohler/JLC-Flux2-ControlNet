"""Final lazy checkpoint registration and deferred ComfyUI-managed model construction."""

from __future__ import annotations

LAZY_LOADER_REVISION = "final-lazy-after-text-encoder-v1"

import gc
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils

from .constants import EXPECTED_CONTROL_INPUT_CHANNELS, PROJECT_LOG_PREFIX
from .control import JLCFlux2Control
from .model import JLCFlux2ControlModel


@dataclass(frozen=True)
class Flux2ControlArchitecture:
    hidden_size: int
    control_in_dim: int
    num_blocks: int
    attention_head_dim: int
    num_attention_heads: int
    parameter_count: int


def _inspect_architecture(state_dict: dict[str, torch.Tensor]) -> Flux2ControlArchitecture:
    required = {
        "control_img_in.weight",
        "control_img_in.bias",
        "control_transformer_blocks.0.before_proj.weight",
        "control_transformer_blocks.0.after_proj.weight",
        "control_transformer_blocks.0.attn.to_q.weight",
        "control_transformer_blocks.0.attn.norm_q.weight",
        "control_transformer_blocks.0.ff.linear_in.weight",
    }
    missing = sorted(required.difference(state_dict))
    if missing:
        raise ValueError(
            "Checkpoint is not a Flux.2 Fun ControlNet side model; missing keys: "
            + ", ".join(missing)
        )

    input_weight = state_dict["control_img_in.weight"]
    if input_weight.ndim != 2:
        raise ValueError("control_img_in.weight must be a rank-2 tensor")
    hidden_size, control_in_dim = map(int, input_weight.shape)
    if control_in_dim != EXPECTED_CONTROL_INPUT_CHANNELS:
        raise ValueError(
            f"Expected {EXPECTED_CONTROL_INPUT_CHANNELS} control input channels, got {control_in_dim}."
        )

    block_indices = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("control_transformer_blocks.")
    }
    if not block_indices or block_indices != set(range(max(block_indices) + 1)):
        raise ValueError(f"Control block indices are not contiguous: {sorted(block_indices)}")
    num_blocks = len(block_indices)
    if num_blocks > 4:
        raise ValueError(f"This build supports at most four control blocks; got {num_blocks}.")

    attention_head_dim = int(
        state_dict["control_transformer_blocks.0.attn.norm_q.weight"].shape[0]
    )
    if hidden_size % attention_head_dim != 0:
        raise ValueError(
            f"Hidden size {hidden_size} is not divisible by head dimension {attention_head_dim}."
        )
    num_attention_heads = hidden_size // attention_head_dim
    parameter_count = sum(int(tensor.numel()) for tensor in state_dict.values())

    return Flux2ControlArchitecture(
        hidden_size=hidden_size,
        control_in_dim=control_in_dim,
        num_blocks=num_blocks,
        attention_head_dim=attention_head_dim,
        num_attention_heads=num_attention_heads,
        parameter_count=parameter_count,
    )


class LazyFlux2ControlHandle:
    """Shared, thread-safe owner for one deferred Flux.2 side model.

    Loader-node execution creates only this small holder. The 7.7 GiB checkpoint
    is not read until ComfyUI gathers additional sampling models through
    ``ControlBase.get_models()``. All shallow control copies and Orchestrator
    branches share the same holder and, after materialization, the same
    ``CoreModelPatcher``.
    """

    def __init__(self, checkpoint_path: str, checkpoint_name: str = ""):
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"ControlNet checkpoint not found: {path}")

        self.checkpoint_path = path
        self.checkpoint_name = checkpoint_name or path.name
        self._lock = threading.RLock()

        self.control_model = None
        self.control_model_wrapped = None
        self.load_device = None
        self.manual_cast_dtype = None
        self.inference_dtype = None
        self.architecture = None

    @property
    def is_materialized(self) -> bool:
        return self.control_model_wrapped is not None

    def materialize(self) -> "LazyFlux2ControlHandle":
        if self.is_materialized:
            return self

        with self._lock:
            if self.is_materialized:
                return self

            logging.info(
                "%s Deferred materialization now reading checkpoint for sampling [%s]: %s",
                PROJECT_LOG_PREFIX,
                LAZY_LOADER_REVISION,
                self.checkpoint_path.name,
            )

            state_dict = None
            model = None
            try:
                state_dict = comfy.utils.load_torch_file(
                    str(self.checkpoint_path),
                    safe_load=True,
                )
                architecture = _inspect_architecture(state_dict)
                weight_dtype = comfy.utils.weight_dtype(state_dict)

                load_device = comfy.model_management.get_torch_device()
                supported_dtypes = [torch.bfloat16, torch.float16, torch.float32]
                inference_dtype = comfy.model_management.unet_dtype(
                    device=load_device,
                    model_params=architecture.parameter_count,
                    supported_dtypes=supported_dtypes,
                    weight_dtype=weight_dtype,
                )
                manual_cast_dtype = comfy.model_management.unet_manual_cast(
                    inference_dtype,
                    load_device,
                    supported_dtypes=supported_dtypes,
                )
                operations = comfy.ops.pick_operations(
                    inference_dtype,
                    manual_cast_dtype,
                    disable_fast_fp8=True,
                )

                if weight_dtype != inference_dtype:
                    logging.info(
                        "%s Converting checkpoint tensors from %s to %s before assignment.",
                        PROJECT_LOG_PREFIX,
                        weight_dtype,
                        inference_dtype,
                    )
                    for key in tuple(state_dict.keys()):
                        tensor = state_dict[key]
                        if (
                            torch.is_floating_point(tensor)
                            and tensor.dtype != inference_dtype
                        ):
                            state_dict[key] = tensor.to(dtype=inference_dtype)

                model = JLCFlux2ControlModel(
                    hidden_size=architecture.hidden_size,
                    control_in_dim=architecture.control_in_dim,
                    num_blocks=architecture.num_blocks,
                    num_attention_heads=architecture.num_attention_heads,
                    attention_head_dim=architecture.attention_head_dim,
                    operations=operations,
                    dtype=inference_dtype,
                    device=torch.device("meta"),
                )

                missing, unexpected = model.load_state_dict(
                    state_dict,
                    strict=False,
                    assign=True,
                )
                if missing or unexpected:
                    raise RuntimeError(
                        "Checkpoint/model mismatch. "
                        f"Missing keys: {missing}; unexpected keys: {unexpected}"
                    )

                del state_dict
                state_dict = None
                gc.collect()

                offload_device = comfy.model_management.unet_offload_device()
                if offload_device.type != "cpu":
                    model.to(offload_device)
                model.eval()
                comfy.model_management.archive_model_dtypes(model)

                wrapped = comfy.model_patcher.CoreModelPatcher(
                    model,
                    load_device=load_device,
                    offload_device=offload_device,
                )

                # Publish only after every construction step succeeds. This
                # prevents another shallow copy from observing partial state.
                self.control_model = model
                self.control_model_wrapped = wrapped
                self.load_device = load_device
                self.manual_cast_dtype = manual_cast_dtype
                self.inference_dtype = inference_dtype
                self.architecture = architecture

                logging.info(
                    "%s Lazily materialized compact side model: %.3fB params, %d blocks, hidden=%d, heads=%d, dtype=%s. Shared model ownership is active.",
                    PROJECT_LOG_PREFIX,
                    architecture.parameter_count / 1_000_000_000,
                    architecture.num_blocks,
                    architecture.hidden_size,
                    architecture.num_attention_heads,
                    inference_dtype,
                )
                return self

            except Exception:
                # Leave the handle unmaterialized so a clear exception can be
                # reported and a later corrected run can retry cleanly.
                if state_dict is not None:
                    del state_dict
                if model is not None:
                    del model
                gc.collect()
                raise


def load_jlc_flux2_controlnet(checkpoint_path: str, *, checkpoint_name: str = ""):
    """Return a lightweight control object without reading checkpoint tensors."""
    handle = LazyFlux2ControlHandle(
        checkpoint_path,
        checkpoint_name=checkpoint_name,
    )
    logging.info(
        "%s Registered lazy checkpoint handle [%s]: %s. No checkpoint tensors have been read; materialization is deferred until sampling-model discovery.",
        PROJECT_LOG_PREFIX,
        LAZY_LOADER_REVISION,
        handle.checkpoint_name,
    )
    return JLCFlux2Control(
        None,
        checkpoint_name=handle.checkpoint_name,
        lazy_handle=handle,
    )
