"""
JLC Flux2 ControlNet Loader
---------------------------

- JLC Flux2 ControlNet
  - This node is part of the **JLC Flux2 ControlNet** package developed
    by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/JLC-Flux2-ControlNet

- Node Purpose
  - The **JLC Flux2 ControlNet Loader** is the single user-facing loader for
    supported `FLUX.2-dev-Fun-Controlnet-Union` checkpoints.

  - The node is precision-agnostic. It exposes only checkpoints whose
    safetensors header matches the exact supported FLUX.2 Fun ControlNet
    Union architecture, then passes the selected checkpoint to the shared backend.

- Checkpoint Discovery Filter
  - ComfyUI's `models/controlnet` folder can contain ControlNet checkpoints for
    many unrelated model families. To reduce accidental misuse, this node does
    not display that folder verbatim.

  - For each `.safetensors` candidate, the node reads only the small safetensors
    header and checks for the architectural signature required by the backend:
        • `control_img_in.weight`
        • `control_img_in.bias`
        • `control_transformer_blocks.0.before_proj.weight`
        • `control_transformer_blocks.0.after_proj.weight`
        • `control_transformer_blocks.0.attn.to_q.weight`
        • `control_transformer_blocks.0.attn.norm_q.weight`
        • `control_transformer_blocks.0.ff.linear_in.weight`
        • the exact 6144-hidden / 260-input / 4-block Union architecture

  - Model payload tensors are not loaded for dropdown discovery. Filename text
    is not used as the compatibility contract, so compatible checkpoints may be
    renamed without disappearing from the list.

  - Unrelated Flux.1, Stable Diffusion / SDXL, Wan, and other ControlNet files
    are omitted when they do not match this architectural signature.

  - Header filtering is a user-interface convenience, not the final validation
    boundary. The shared backend repeats authoritative architecture and
    quantization validation when the selected checkpoint is materialized.

- Supported Checkpoint Representations
  - Ordinary dense upstream checkpoints, including the released Alibaba PAI
    BF16 `FLUX.2-dev-Fun-Controlnet-Union-2602` model.

  - Supported ComfyUI-native mixed FP8/BF16 checkpoints, including JLC mixed
    E4M3FN derivatives that carry the required quantization metadata and scale
    tensors.

  - No precision or quantization selector is required. The backend determines
    checkpoint representation from its contents during deferred materialization.

- Loader / Runtime Contract
  - For either supported checkpoint representation, the loader:
        • constructs the same compact FLUX.2 Fun ControlNet side-model
          architecture
        • validates checkpoint structure before publishing the model
        • uses the appropriate dense or native-quantized ComfyUI loading path
        • wraps the model in ComfyUI's native model-patcher lifecycle
        • returns the same reusable `JLC_FLUX2_CONTROLNET` runtime object

  - Because downstream nodes receive the same ControlNet contract, Apply,
    Apply Advanced, Orchestrator, and Inpaint Adapter workflows do not require
    separate BF16 and FP8 node families.

- Runtime Integration
  - Loader-node execution remains lazy: selecting/loading the node registers a
    lightweight handle, while checkpoint tensors are read only when ComfyUI
    requests the side model for sampling.

  - The loaded side model participates in normal ComfyUI loading, offloading,
    device placement, and DynamicVRAM behavior.

  - The loader does not globally patch or replace the native FLUX.2 model.

- Scope
  - This is an inference/runtime node. FP8 calibration and quantization tooling
    remain separate from the public loader interface.

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Built for interoperability with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova
  - Released under the **MIT License**.
"""

from __future__ import annotations

LOADER_NODE_REVISION = "flux2-union-exact-header-filter-v3"

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import folder_paths

from ..jlc_flux2_controlnet_versions import JLC_FLUX2_CONTROLNET_VERSION
from ..jlc_flux2_controlnet.constants import EXPECTED_CONTROL_INPUT_CHANNELS
from ..jlc_flux2_controlnet.loader import load_jlc_flux2_controlnet


MANIFEST = {
    "name": "JLC Flux2 ControlNet Loader",
    "version": JLC_FLUX2_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Single precision-agnostic loader for the supported FLUX.2 Fun Union "
        "ControlNet checkpoints. The node filters ComfyUI's controlnet model "
        "list by safetensors architectural signature without loading model "
        "payloads. The shared backend then automatically distinguishes dense "
        "BF16 checkpoints from supported ComfyUI-native mixed FP8/BF16 "
        "checkpoints and returns the same reusable JLC_FLUX2_CONTROLNET object."
    ),
    "supported_checkpoint_representations": (
        "dense",
        "comfy_native_mixed_fp8_bf16",
    ),
    "checkpoint_discovery": "exact_flux2_union_safetensors_header_filter",
    "precision_selection": "automatic_from_checkpoint",
    "backend_validation": "authoritative",
    "status": "stable",
    "license": "MIT",
}


# Exact released FLUX.2-dev-Fun-Controlnet-Union architecture fingerprint.
# Both the dense BF16 checkpoint and the JLC mixed FP8/BF16 derivative retain
# these payload tensor shapes; quantization adds scale/metadata tensors without
# changing the side-model architecture.
_EXPECTED_FLUX2_UNION_TENSOR_SHAPES = {
    "control_img_in.weight": (6144, EXPECTED_CONTROL_INPUT_CHANNELS),
    "control_img_in.bias": (6144,),
    "control_transformer_blocks.0.before_proj.weight": (6144, 6144),
    "control_transformer_blocks.0.before_proj.bias": (6144,),
    "control_transformer_blocks.0.after_proj.weight": (6144, 6144),
    "control_transformer_blocks.0.attn.to_q.weight": (6144, 6144),
    "control_transformer_blocks.0.attn.norm_q.weight": (128,),
    "control_transformer_blocks.0.ff.linear_in.weight": (36864, 6144),
    "control_transformer_blocks.3.after_proj.weight": (6144, 6144),
}
_EXPECTED_FLUX2_UNION_BLOCK_INDICES = frozenset({0, 1, 2, 3})

# Avoid allocating an unreasonable amount of memory if a malformed file claims
# an absurd safetensors header length. Normal model headers are far smaller.
_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
_NO_COMPATIBLE_CHECKPOINT = "[No compatible FLUX.2 Fun ControlNet checkpoints found]"


def _read_safetensors_header(path: Path) -> dict[str, Any] | None:
    """Read only a safetensors JSON header; never deserialize tensor payloads."""
    if path.suffix.lower() != ".safetensors":
        return None

    try:
        file_size = path.stat().st_size
        if file_size < 10:
            return None

        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                return None

            header_length = int.from_bytes(raw_length, byteorder="little", signed=False)
            remaining_bytes = file_size - 8
            if (
                header_length <= 0
                or header_length > remaining_bytes
                or header_length > _MAX_SAFETENSORS_HEADER_BYTES
            ):
                return None

            raw_header = handle.read(header_length)
            if len(raw_header) != header_length:
                return None

        parsed = json.loads(raw_header.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


def _tensor_shape(header: dict[str, Any], tensor_name: str) -> tuple[int, ...] | None:
    entry = header.get(tensor_name)
    if not isinstance(entry, dict):
        return None

    shape = entry.get("shape")
    if not isinstance(shape, list) or any(
        isinstance(dimension, bool) or not isinstance(dimension, int)
        for dimension in shape
    ):
        return None
    return tuple(shape)


@lru_cache(maxsize=512)
def _matches_flux2_fun_header_cached(
    path_string: str,
    file_size: int,
    modified_ns: int,
) -> bool:
    """Classify one file; size/mtime are cache-key invalidators."""
    del file_size, modified_ns

    header = _read_safetensors_header(Path(path_string))
    if header is None:
        return False

    # Match the exact FLUX.2 Union side-model dimensions, not just the broader
    # Fun-ControlNet naming pattern. Earlier filtering checked only key names
    # plus the 260-channel input contract, which was intentionally too broad and
    # could admit structurally related Flux-family ControlNets.
    for tensor_name, expected_shape in _EXPECTED_FLUX2_UNION_TENSOR_SHAPES.items():
        if _tensor_shape(header, tensor_name) != expected_shape:
            return False

    block_indices: set[int] = set()
    prefix = "control_transformer_blocks."
    for tensor_name in header:
        if not isinstance(tensor_name, str) or not tensor_name.startswith(prefix):
            continue
        remainder = tensor_name[len(prefix):]
        block_token = remainder.split(".", 1)[0]
        if block_token.isdigit():
            block_indices.add(int(block_token))

    if frozenset(block_indices) != _EXPECTED_FLUX2_UNION_BLOCK_INDICES:
        return False

    return True


def _is_compatible_flux2_fun_checkpoint(path: Path) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False

    return _matches_flux2_fun_header_cached(
        str(path.resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _compatible_controlnet_names() -> list[str]:
    compatible: list[str] = []
    for checkpoint_name in folder_paths.get_filename_list("controlnet"):
        checkpoint_path = folder_paths.get_full_path("controlnet", checkpoint_name)
        if checkpoint_path is None:
            continue
        if _is_compatible_flux2_fun_checkpoint(Path(checkpoint_path)):
            compatible.append(checkpoint_name)

    return compatible


print(
    "[JLC Flux2] ControlNet Loader node revision: "
    f"{LOADER_NODE_REVISION}"
)


class JLCFlux2ControlNetLoader:
    @classmethod
    def INPUT_TYPES(cls):
        compatible = _compatible_controlnet_names()
        choices = compatible if compatible else [_NO_COMPATIBLE_CHECKPOINT]
        return {
            "required": {
                "controlnet_name": (
                    choices,
                    {
                        "tooltip": (
                            "Only safetensors checkpoints whose header matches "
                            "the supported FLUX.2 Fun ControlNet architecture are "
                            "shown. Dense BF16 and supported mixed FP8/BF16 "
                            "representations are both accepted automatically."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("JLC_FLUX2_CONTROLNET",)
    RETURN_NAMES = ("controlnet",)
    FUNCTION = "load_controlnet"
    CATEGORY = "Flux2 Controlnet"
    DESCRIPTION = (
        "Loads supported FLUX.2 Fun ControlNet checkpoints through one automatic "
        "dense-or-mixed-precision backend. The dropdown is pre-filtered from "
        "models/controlnet using a lightweight safetensors architecture check."
    )

    def load_controlnet(self, controlnet_name):
        if controlnet_name == _NO_COMPATIBLE_CHECKPOINT:
            raise FileNotFoundError(
                "No compatible FLUX.2 Fun ControlNet checkpoint was found in "
                "ComfyUI's models/controlnet folder."
            )

        checkpoint_path = folder_paths.get_full_path("controlnet", controlnet_name)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Unable to resolve ControlNet checkpoint '{controlnet_name}'."
            )

        path = Path(checkpoint_path)
        if not _is_compatible_flux2_fun_checkpoint(path):
            raise ValueError(
                "The selected checkpoint no longer matches the supported FLUX.2 "
                "Fun ControlNet safetensors header signature. Refresh the node's "
                "model list and select a compatible checkpoint."
            )

        control = load_jlc_flux2_controlnet(
            checkpoint_path,
            checkpoint_name=controlnet_name,
        )
        return (control,)
