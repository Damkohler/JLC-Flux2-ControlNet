"""
JLC Flux2 ControlNet Loader Backend
-----------------------------------

- JLC Flux2 ControlNet
  - Shared runtime loader backend for the public **JLC Flux2 ControlNet**
    package.

- Backend Purpose
  - Provides one lazy-loading path for supported
    `FLUX.2-dev-Fun-Controlnet-Union` checkpoints regardless of whether the
    checkpoint stores ordinary dense BF16 weights or ComfyUI-native mixed
    FP8/BF16 quantized weights.

  - The backend:
        • registers a lightweight `JLC_FLUX2_CONTROLNET` handle without reading
          checkpoint tensors at node execution time
        • defers checkpoint materialization until ComfyUI requests the side
          model for sampling
        • validates the compact FLUX.2 Fun ControlNet architecture
        • automatically inspects checkpoint tensors and safetensors metadata
          to determine whether native layer quantization is present
        • preserves the proven meta-device + assign loading path for ordinary
          dense checkpoints
        • uses ComfyUI's native quantization metadata, MixedPrecisionOps, scale
          tensors, and quantized-module loading path for supported mixed
          FP8/BF16 checkpoints
        • wraps either representation in the same ComfyUI CoreModelPatcher
          lifecycle and returns the same runtime ControlNet contract

- Quantization Contract
  - Quantization is detected from checkpoint contents; there is no user-facing
    precision selector.

  - Quantized checkpoints must provide a self-consistent ComfyUI quantization
    payload. The backend verifies agreement between metadata, FP8 weight
    tensors, scale tensors, generated `comfy_quant` descriptors, and bound
    quantized modules before publishing the loaded model.

  - Dense checkpoints continue through the ordinary BF16/FP16/FP32 inference
    dtype selection path and are not converted into quantized modules.

- Runtime Contract
  - Downstream Apply, Apply Advanced, Orchestrator, and Inpaint Adapter nodes
    consume the same `JLC_FLUX2_CONTROLNET` abstraction for both checkpoint
    representations. Weight storage precision is therefore an implementation
    detail of this loader backend, not a separate workflow-node family.

  - Model loading, offloading, device placement, and DynamicVRAM remain under
    ComfyUI's native model-management lifecycle.

- Scope
  - This is an inference/runtime backend. FP8 calibration and quantization
    tooling intentionally remain outside the public loader path.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Built for interoperability with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova
  - Released under the **MIT License**.
"""

from __future__ import annotations

LAZY_LOADER_REVISION = "unified-dense-native-quant-v1"

import gc
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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


def _parse_quantization_metadata(raw_value, *, source_label: str) -> dict:
    """Parse and validate one ComfyUI quantization metadata payload."""
    if isinstance(raw_value, str):
        try:
            payload = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid JSON in {source_label}: {exc}"
            ) from exc
    elif isinstance(raw_value, dict):
        payload = raw_value
    else:
        raise RuntimeError(
            f"{source_label} must be a JSON string or object, got "
            f"{type(raw_value).__name__}."
        )

    if not isinstance(payload, dict):
        raise RuntimeError(f"{source_label} must decode to a JSON object.")

    layers = payload.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise RuntimeError(
            f"{source_label} does not contain a non-empty 'layers' object."
        )

    malformed = sorted(
        name
        for name, config in layers.items()
        if not isinstance(name, str)
        or not name
        or not isinstance(config, dict)
        or not isinstance(config.get("format"), str)
        or not config.get("format")
    )
    if malformed:
        raise RuntimeError(
            f"{source_label} contains malformed layer records: {malformed[:8]}"
        )

    return payload


def _normalize_quantization_metadata(
    metadata: dict[str, str] | None,
) -> tuple[dict[str, str], dict | None, str | None]:
    """Promote JLC's nested metadata into ComfyUI's native top-level field.

    Phase 3A.1 stored `_quantization_metadata` inside the JSON object held by
    the safetensors `metadata.json` entry. Current ComfyUI expects the same
    quantization object serialized directly in the top-level safetensors
    `_quantization_metadata` entry. This normalization is in-memory only and
    does not modify the checkpoint file.
    """
    normalized = dict(metadata or {})

    top_level = normalized.get("_quantization_metadata")
    if top_level is not None:
        payload = _parse_quantization_metadata(
            top_level,
            source_label="top-level safetensors '_quantization_metadata'",
        )
        normalized["_quantization_metadata"] = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        return normalized, payload, "top-level"

    nested_raw = normalized.get("metadata.json")
    if nested_raw is None:
        return normalized, None, None

    try:
        nested = json.loads(nested_raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Safetensors metadata entry 'metadata.json' is not valid JSON: "
            f"{exc}"
        ) from exc

    if not isinstance(nested, dict):
        raise RuntimeError(
            "Safetensors metadata entry 'metadata.json' must decode to an object."
        )

    nested_quant = nested.get("_quantization_metadata")
    if nested_quant is None:
        return normalized, None, None

    payload = _parse_quantization_metadata(
        nested_quant,
        source_label="metadata.json['_quantization_metadata']",
    )
    normalized["_quantization_metadata"] = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return normalized, payload, "metadata.json"


def _layer_names_from_suffix(
    state_dict: dict[str, torch.Tensor],
    suffix: str,
) -> set[str]:
    return {
        key[: -len(suffix)]
        for key in state_dict
        if key.endswith(suffix)
    }


def _fp8_weight_layers(state_dict: dict[str, torch.Tensor]) -> set[str]:
    fp8_dtypes = {
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
        )
        if dtype is not None
    }
    return {
        key[: -len(".weight")]
        for key, tensor in state_dict.items()
        if key.endswith(".weight") and tensor.dtype in fp8_dtypes
    }


def _summarize_name_mismatch(
    expected: set[str],
    actual: set[str],
) -> str:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    return (
        f"missing={missing[:8]}"
        + ("..." if len(missing) > 8 else "")
        + "; "
        + f"unexpected={unexpected[:8]}"
        + ("..." if len(unexpected) > 8 else "")
    )


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
    parameter_count = sum(
        int(tensor.numel())
        for key, tensor in state_dict.items()
        if not (
            key.endswith(".weight_scale")
            or key.endswith(".weight_scale_2")
            or key.endswith(".input_scale")
            or key.endswith(".comfy_quant")
        )
    )

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

    Loader-node execution creates only this small holder. The selected checkpoint
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
                state_dict, metadata = comfy.utils.load_torch_file(
                    str(self.checkpoint_path),
                    safe_load=True,
                    return_metadata=True,
                )

                weight_scale_layers = _layer_names_from_suffix(
                    state_dict,
                    ".weight_scale",
                )
                input_scale_layers = _layer_names_from_suffix(
                    state_dict,
                    ".input_scale",
                )
                fp8_weight_layers = _fp8_weight_layers(state_dict)

                metadata, quant_metadata, metadata_source = (
                    _normalize_quantization_metadata(metadata)
                )
                metadata_layers = (
                    set(quant_metadata["layers"])
                    if quant_metadata is not None
                    else set()
                )

                has_quant_payload = bool(
                    weight_scale_layers
                    or input_scale_layers
                    or fp8_weight_layers
                    or metadata_layers
                )

                if has_quant_payload and quant_metadata is None:
                    raise RuntimeError(
                        "Checkpoint contains quantized weights or scale tensors, but "
                        "no usable quantization metadata was found in either the "
                        "top-level '_quantization_metadata' field or "
                        "metadata.json['_quantization_metadata']. "
                        f"FP8 weights={len(fp8_weight_layers)}, "
                        f"weight scales={len(weight_scale_layers)}, "
                        f"input scales={len(input_scale_layers)}."
                    )

                if quant_metadata is not None:
                    if metadata_layers != weight_scale_layers:
                        raise RuntimeError(
                            "Quantization metadata layers do not match weight-scale "
                            "layers: "
                            + _summarize_name_mismatch(
                                metadata_layers,
                                weight_scale_layers,
                            )
                        )
                    if input_scale_layers and metadata_layers != input_scale_layers:
                        raise RuntimeError(
                            "Quantization metadata layers do not match input-scale "
                            "layers: "
                            + _summarize_name_mismatch(
                                metadata_layers,
                                input_scale_layers,
                            )
                        )
                    if fp8_weight_layers and metadata_layers != fp8_weight_layers:
                        raise RuntimeError(
                            "Quantization metadata layers do not match FP8 weight "
                            "layers: "
                            + _summarize_name_mismatch(
                                metadata_layers,
                                fp8_weight_layers,
                            )
                        )

                    logging.info(
                        "%s Normalized quantization metadata from %s: %d layers, "
                        "%d FP8 weights, %d weight scales, %d input scales.",
                        PROJECT_LOG_PREFIX,
                        metadata_source,
                        len(metadata_layers),
                        len(fp8_weight_layers),
                        len(weight_scale_layers),
                        len(input_scale_layers),
                    )

                # Current ComfyUI expects `_quantization_metadata` at the
                # top-level safetensors metadata map. It converts that payload
                # into one `<layer>.comfy_quant` tensor per quantized module.
                state_dict, metadata = comfy.utils.convert_old_quants(
                    state_dict,
                    model_prefix="",
                    metadata=metadata,
                )
                quant_config = comfy.utils.detect_layer_quantization(
                    state_dict,
                    "",
                )

                comfy_quant_layers = _layer_names_from_suffix(
                    state_dict,
                    ".comfy_quant",
                )
                if quant_metadata is not None:
                    if not quant_config:
                        raise RuntimeError(
                            "Metadata normalization succeeded, but ComfyUI did not "
                            "detect native layer quantization after "
                            "convert_old_quants()."
                        )
                    if comfy_quant_layers != metadata_layers:
                        raise RuntimeError(
                            "ComfyUI generated an unexpected set of comfy_quant "
                            "descriptors: "
                            + _summarize_name_mismatch(
                                metadata_layers,
                                comfy_quant_layers,
                            )
                        )
                    logging.info(
                        "%s ComfyUI generated %d native comfy_quant descriptors.",
                        PROJECT_LOG_PREFIX,
                        len(comfy_quant_layers),
                    )
                elif quant_config:
                    raise RuntimeError(
                        "ComfyUI detected quantization without validated checkpoint "
                        "quantization metadata."
                    )

                logging.info(
                    "%s Detected %s ControlNet checkpoint representation.",
                    PROJECT_LOG_PREFIX,
                    "native quantized" if quant_config else "dense",
                )

                architecture = _inspect_architecture(state_dict)
                weight_dtype = (
                    None
                    if quant_config
                    else comfy.utils.weight_dtype(state_dict)
                )

                load_device = comfy.model_management.get_torch_device()
                offload_device = comfy.model_management.unet_offload_device()
                supported_dtypes = [torch.bfloat16, torch.float16, torch.float32]
                inference_dtype = comfy.model_management.unet_dtype(
                    device=load_device,
                    model_params=architecture.parameter_count,
                    supported_dtypes=supported_dtypes,
                    weight_dtype=weight_dtype,
                )
                manual_cast_dtype = comfy.model_management.unet_manual_cast(
                    None if quant_config else inference_dtype,
                    load_device,
                    supported_dtypes=supported_dtypes,
                )

                if quant_config:
                    # Native quantized checkpoints use the same quant_config contract
                    # consumed by ComfyUI's BaseModel path.
                    operation_config = SimpleNamespace(quant_config=quant_config)
                    operations = comfy.ops.pick_operations(
                        inference_dtype,
                        manual_cast_dtype,
                        load_device=load_device,
                        disable_fast_fp8=True,
                        model_config=operation_config,
                    )
                else:
                    # Preserve the released dense loader's proven operation-selection
                    # call for ordinary checkpoints.
                    operations = comfy.ops.pick_operations(
                        inference_dtype,
                        manual_cast_dtype,
                        disable_fast_fp8=True,
                    )

                # Native quantized modules consume FP8 weights, comfy_quant,
                # weight_scale, and input_scale in _load_from_state_dict.
                # Never pre-cast those checkpoint tensors.
                if not quant_config and weight_dtype != inference_dtype:
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

                # MixedPrecisionOps._load_from_state_dict moves FP8 weights,
                # weight scales, and input scales to the device recorded when
                # each module is constructed. A meta construction device would
                # therefore create successfully-bound tensors with no storage.
                #
                # Use CPU for native quantized checkpoints, matching ComfyUI's
                # normal diffusion-model construction behavior. Keep the proven
                # meta+assign path for ordinary BF16 checkpoints.
                initial_model_device = (
                    torch.device("cpu")
                    if quant_config
                    else torch.device("meta")
                )
                logging.info(
                    "%s Constructing ControlNet modules on %s for %s checkpoint loading.",
                    PROJECT_LOG_PREFIX,
                    initial_model_device,
                    "native quantized" if quant_config else "dense",
                )

                model = JLCFlux2ControlModel(
                    hidden_size=architecture.hidden_size,
                    control_in_dim=architecture.control_in_dim,
                    num_blocks=architecture.num_blocks,
                    num_attention_heads=architecture.num_attention_heads,
                    attention_head_dim=architecture.attention_head_dim,
                    operations=operations,
                    dtype=inference_dtype,
                    device=initial_model_device,
                )

                if quant_config:
                    invalid_quant_modules = []
                    for layer_name in sorted(comfy_quant_layers):
                        try:
                            module = model.get_submodule(layer_name)
                        except AttributeError:
                            invalid_quant_modules.append(
                                f"{layer_name} (module missing)"
                            )
                            continue
                        if not isinstance(module, operations.Linear):
                            invalid_quant_modules.append(
                                f"{layer_name} ({type(module).__qualname__})"
                            )

                    if invalid_quant_modules:
                        raise RuntimeError(
                            "The custom ControlNet architecture did not instantiate "
                            "ComfyUI MixedPrecisionOps.Linear for all quantized "
                            "layers: "
                            + ", ".join(invalid_quant_modules[:8])
                            + ("..." if len(invalid_quant_modules) > 8 else "")
                        )

                    logging.info(
                        "%s Constructed %d native MixedPrecisionOps.Linear modules "
                        "for the quantized ControlNet layers.",
                        PROJECT_LOG_PREFIX,
                        len(comfy_quant_layers),
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

                meta_parameters = [
                    name
                    for name, parameter in model.named_parameters()
                    if parameter is not None
                    and bool(getattr(parameter, "is_meta", False))
                ]
                meta_buffers = [
                    name
                    for name, buffer in model.named_buffers()
                    if buffer is not None
                    and bool(getattr(buffer, "is_meta", False))
                ]
                if meta_parameters or meta_buffers:
                    raise RuntimeError(
                        "Model loading completed with tensors still on the meta "
                        "device. This would fail during ComfyUI VRAM loading. "
                        f"Meta parameters ({len(meta_parameters)}): "
                        f"{meta_parameters[:12]}; meta buffers "
                        f"({len(meta_buffers)}): {meta_buffers[:12]}"
                    )

                logging.info(
                    "%s Model storage validation passed: no parameters or buffers "
                    "remain on the meta device.",
                    PROJECT_LOG_PREFIX,
                )

                if quant_config:
                    quantized_layers = 0
                    input_scale_layers = 0
                    for module in model.modules():
                        if getattr(module, "quant_format", None) is not None:
                            quantized_layers += 1
                            if getattr(module, "input_scale", None) is not None:
                                input_scale_layers += 1

                    expected_layers = len(comfy_quant_layers)
                    if quantized_layers != expected_layers:
                        raise RuntimeError(
                            "Native quantized-layer binding mismatch: "
                            f"{quantized_layers} modules bound, expected {expected_layers}."
                        )
                    logging.info(
                        "%s Native upstream quant load succeeded: %d quantized layers; "
                        "%d input scales bound.",
                        PROJECT_LOG_PREFIX,
                        quantized_layers,
                        input_scale_layers,
                    )

                del state_dict
                state_dict = None
                gc.collect()

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
