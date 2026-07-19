"""
JLC Conditional Save Image
--------------------------

- JLC Flux2 ControlNet
  - This node is part of the **JLC Flux2 ControlNet** package developed
    by **J. L. Córdova**.

  - Repository
    https://github.com/Damkohler/JLC-Flux2-ControlNet

- Node Purpose
  - The **JLC Conditional Save Image** node is an execution-gated image output
    sink for mutually exclusive ComfyUI workflow branches.

  - It is packaged as a practical **companion** to the JLC Flux2 cache-prep
    nodes. A single BOOLEAN control can select between a cache-preparation
    branch and a normal inference-image branch, while this node both:
        • requests only the selected lazy image input
        • saves only in the configured Boolean state(s)

- Workflow Role
  - This node replaces the pattern of:
        • external image Switch
        • separate conditional save sink

  - Typical Flux2 cache workflow:

        switch = TRUE  -> request cache-prep / proof branch
        switch = FALSE -> request final inference image branch

    with:
        • `image_on_true` connected to the TRUE branch
        • `image_on_false` connected to the FALSE branch
        • `save_when="FALSE"` so only the inference image is written

  - When the currently selected branch should not be saved, the node still
    requests that branch so its side effects can occur, but it returns without
    writing a PNG.

- Saving Contract
  - Supports the normal ComfyUI output directory, relative subdirectories inside
    it, or an explicit absolute output directory.

  - PNG workflow metadata is preserved unless ComfyUI metadata saving is
    disabled.

  - PNG compression is lossless:
        • 0 = no compression / fastest / largest
        • 9 = strongest compression / slower / smallest

- Attribution & License
  - Concept and implementation by **J. L. Córdova**
    with development assistance from **ChatGPT (OpenAI)**.

  - Built for interoperability with:
    https://github.com/comfyanonymous/ComfyUI

  - Copyright (c) 2026 J. L. Córdova
  - Released under the **MIT License**.
"""

from __future__ import annotations

from ..jlc_flux2_controlnet_versions import JLC_FLUX2_CONTROLNET_VERSION

import json
import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args


MANIFEST = {
    "name": "JLC Conditional Save Image",
    "version": JLC_FLUX2_CONTROLNET_VERSION,
    "author": "J. L. Córdova",
    "description": (
        "Execution-gated PNG output node with two lazy branch-image inputs and "
        "one shared BOOLEAN selector. Packaged as a companion to the JLC Flux2 "
        "cache-prep nodes so a single control can choose the active branch "
        "while only configured Boolean states write PNG files."
    ),
    "base_package_version": JLC_FLUX2_CONTROLNET_VERSION,
    "role": "cache_prep_companion",
}


_SAVE_WHEN_VALUES = ("FALSE", "TRUE", "BOTH", "NONE")


def _resolve_output_directory(output_folder: str) -> str:
    """Resolve absolute or ComfyUI-output-relative folders safely."""

    raw = os.path.expandvars(os.path.expanduser(str(output_folder or "output").strip()))
    if not raw:
        raw = "output"

    if os.path.isabs(raw):
        target = os.path.abspath(raw)
        os.makedirs(target, exist_ok=True)
        return target

    output_root = os.path.abspath(folder_paths.get_output_directory())
    normalized = raw.replace("\\", "/").strip("/")
    if normalized.lower() in {"", ".", "output"}:
        target = output_root
    else:
        if normalized.lower().startswith("output/"):
            normalized = normalized[len("output/") :]
        target = os.path.abspath(os.path.join(output_root, normalized))

    try:
        inside_output = os.path.commonpath((output_root, target)) == output_root
    except ValueError:
        inside_output = False
    if not inside_output:
        raise ValueError(
            "Relative output_folder must resolve inside the ComfyUI output directory: "
            f"{target}"
        )

    os.makedirs(target, exist_ok=True)
    return target


def _save_is_enabled(switch: bool, save_when: str) -> bool:
    mode = str(save_when or "FALSE").strip().upper()
    if mode == "BOTH":
        return True
    if mode == "NONE":
        return False
    if mode == "TRUE":
        return bool(switch)
    if mode == "FALSE":
        return not bool(switch)
    raise ValueError(f"Unsupported save_when value: {save_when!r}.")


class JLCConditionalSaveImage:
    """Save from one selected branch, with saving optionally disabled per state."""

    CATEGORY = "Flux2 Latents Cache/utils"
    FUNCTION = "save_images"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("selected_image", "image_save_path")
    OUTPUT_NODE = True

    def __init__(self) -> None:
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_on_true": (
                    "IMAGE",
                    {
                        "lazy": True,
                        "tooltip": (
                            "Lazy IMAGE used when switch is TRUE. In Flux2 cache workflows, "
                            "this is commonly the cache-prep or proof branch."
                        ),
                    },
                ),
                "image_on_false": (
                    "IMAGE",
                    {
                        "lazy": True,
                        "tooltip": (
                            "Lazy IMAGE used when switch is FALSE. In Flux2 cache workflows, "
                            "this is commonly the final inference image branch."
                        ),
                    },
                ),
                "switch": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Shared branch selector. Companion to the JLC Flux2 cache-prep "
                            "nodes: TRUE usually selects cache prep; FALSE usually selects "
                            "normal inference."
                        ),
                    },
                ),
                "save_when": (
                    _SAVE_WHEN_VALUES,
                    {
                        "default": "FALSE",
                        "tooltip": (
                            "Which switch states write a PNG. The selected branch is still "
                            "executed either way, which makes this node a cache-prep companion."
                        ),
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                        "tooltip": (
                            "Filename prefix. Standard ComfyUI formatting tokens and "
                            "%batch_num% are supported."
                        ),
                    },
                ),
                "output_folder": (
                    "STRING",
                    {
                        "default": "output",
                        "tooltip": (
                            "Use 'output', a relative subfolder inside the ComfyUI output "
                            "directory, or an absolute destination path."
                        ),
                    },
                ),
                "compress_level": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 9,
                        "step": 1,
                        "tooltip": (
                            "PNG lossless compression: 0 = none/fastest/largest, "
                            "9 = strongest/slower/smallest. Does not affect metadata."
                        ),
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def check_lazy_status(
        self,
        image_on_true: torch.Tensor | None = None,
        image_on_false: torch.Tensor | None = None,
        switch: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        """Request only the currently selected expensive image dependency."""

        if bool(switch):
            return ["image_on_true"] if image_on_true is None else []
        return ["image_on_false"] if image_on_false is None else []

    def save_images(
        self,
        image_on_true: torch.Tensor | None,
        image_on_false: torch.Tensor | None,
        switch: bool,
        save_when: str,
        filename_prefix: str,
        output_folder: str,
        compress_level: int = 4,
        prompt: Any = None,
        extra_pnginfo: dict[str, Any] | None = None,
    ) -> tuple[str]:
        images = image_on_true if bool(switch) else image_on_false

        if images is None:
            selected_name = "image_on_true" if bool(switch) else "image_on_false"
            raise ValueError(
                "JLC Conditional Save Image selected "
                f"{selected_name}, but that lazy IMAGE input was not supplied."
            )
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError(
                "JLC Conditional Save Image expected a rank-4 ComfyUI IMAGE tensor."
            )
        if images.shape[0] < 1:
            raise ValueError("JLC Conditional Save Image received an empty image batch.")

        if not _save_is_enabled(bool(switch), save_when):
            return (images, "")

        target_dir = _resolve_output_directory(output_folder)
        filename_prefix = str(filename_prefix or "ComfyUI") + self.prefix_append
        height = int(images[0].shape[0])
        width = int(images[0].shape[1])
        full_output_folder, filename, counter, _subfolder, _ = (
            folder_paths.get_save_image_path(
                filename_prefix,
                target_dir,
                width,
                height,
            )
        )

        last_saved_path = ""
        compression = max(0, min(9, int(compress_level)))

        for batch_number, image in enumerate(images):
            array = image.detach().to(device="cpu").float().numpy()
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(array)

            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for key, value in extra_pnginfo.items():
                        metadata.add_text(str(key), json.dumps(value))

            filename_with_batch = filename.replace("%batch_num%", str(batch_number))
            base_name = f"{filename_with_batch}_{counter:05}_"
            png_name = base_name + ".png"
            png_path = os.path.join(full_output_folder, png_name)
            pil_image.save(
                png_path,
                pnginfo=metadata,
                compress_level=compression,
            )
            last_saved_path = png_path
            counter += 1

        return (images, last_saved_path)
