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

  - A remote BOOLEAN control selects whether this node is active. The IMAGE
    input is declared lazy and is requested only when the BOOLEAN value matches
    the configured `run_when` value.

- Workflow Role
  - Use one shared BOOLEAN control to select between a cache-preparation branch
    and a normal inference branch. Configure this node to save when that control
    is FALSE or TRUE as appropriate for the workflow.

  - When the save condition does not match, the node returns immediately without
    requesting its IMAGE input. Therefore an inactive sampler, VAE Decode, or
    other expensive upstream image branch is not pulled into the execution graph
    merely because this output node is present.

  - This node is intentionally an output node. Its branch safety comes from the
    lazy IMAGE input and conditional `check_lazy_status` implementation.

- Saving Contract
  - Supports the normal ComfyUI output directory, relative subdirectories inside
    it, or an explicit absolute output directory.

  - PNG workflow metadata is preserved unless ComfyUI metadata saving is disabled.
    An optional caption may be written beside each image using a restricted set
    of plain-text/data extensions.

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
        "Execution-gated PNG output node with a lazy IMAGE input. A remote "
        "BOOLEAN control decides whether the image branch is requested, allowing "
        "inactive sampler and VAE Decode branches to remain outside the active "
        "ComfyUI dependency graph."
    ),
}


_ALLOWED_CAPTION_EXTENSIONS = {
    ".txt",
    ".caption",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".csv",
    ".tsv",
    ".xml",
    ".log",
    ".ini",
    ".toml",
}
_RUN_WHEN_VALUES = ("FALSE", "TRUE")


def _condition_matches(switch: bool, run_when: str) -> bool:
    expected = str(run_when).strip().upper() == "TRUE"
    return bool(switch) == expected


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
        # Treat an optional leading "output/" as referring to the ComfyUI output root.
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


def _sanitize_caption_extension(extension: str) -> str:
    cleaned = os.path.basename(str(extension or ".txt").strip())
    if not cleaned:
        cleaned = ".txt"
    if not cleaned.startswith("."):
        cleaned = "." + cleaned
    cleaned = cleaned.lower()
    if cleaned not in _ALLOWED_CAPTION_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_CAPTION_EXTENSIONS))
        raise ValueError(
            f"Disallowed caption extension {cleaned!r}. Allowed extensions: {allowed}"
        )
    return cleaned


class JLCConditionalSaveImage:
    """Save images only when a remote Boolean matches the configured branch."""

    CATEGORY = "Flux2 ControlNet/Utilities"
    FUNCTION = "save_images"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    OUTPUT_NODE = True

    def __init__(self) -> None:
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Remote branch-control Boolean. The image input is requested only "
                            "when this value matches run_when."
                        ),
                    },
                ),
                "run_when": (
                    _RUN_WHEN_VALUES,
                    {
                        "default": "FALSE",
                        "tooltip": (
                            "Choose which Boolean state activates image saving. For a workflow "
                            "where TRUE prepares caches and FALSE runs inference, use FALSE."
                        ),
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "lazy": True,
                        "tooltip": (
                            "Lazy image branch. It is not requested when switch does not match "
                            "run_when."
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
                        "tooltip": "PNG compression level. Higher values reduce file size but save more slowly.",
                    },
                ),
            },
            "optional": {
                "caption": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Optional text written beside every saved image.",
                    },
                ),
                "caption_file_extension": (
                    "STRING",
                    {
                        "default": ".txt",
                        "tooltip": "Plain-text/data extension for the optional caption sidecar.",
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
        switch: bool,
        run_when: str,
        images: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Request the expensive image dependency only for the active branch."""

        if _condition_matches(switch, run_when) and images is None:
            return ["images"]
        return []

    def save_images(
        self,
        switch: bool,
        run_when: str,
        filename_prefix: str,
        output_folder: str,
        compress_level: int = 4,
        images: torch.Tensor | None = None,
        prompt: Any = None,
        extra_pnginfo: dict[str, Any] | None = None,
        caption: str | None = None,
        caption_file_extension: str = ".txt",
    ) -> tuple[str]:
        if not _condition_matches(switch, run_when):
            return ("",)

        if images is None:
            raise ValueError(
                "JLC Conditional Save Image is active, but its lazy IMAGE input was not supplied."
            )
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError(
                "JLC Conditional Save Image expected a rank-4 ComfyUI IMAGE tensor."
            )
        if images.shape[0] < 1:
            raise ValueError("JLC Conditional Save Image received an empty image batch.")

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

        caption_extension = None
        if caption is not None:
            caption_extension = _sanitize_caption_extension(caption_file_extension)

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

            if caption is not None and caption_extension is not None:
                caption_path = os.path.join(
                    full_output_folder,
                    base_name + caption_extension,
                )
                try:
                    inside_target = (
                        os.path.commonpath(
                            (
                                os.path.abspath(full_output_folder),
                                os.path.abspath(caption_path),
                            )
                        )
                        == os.path.abspath(full_output_folder)
                    )
                except ValueError:
                    inside_target = False
                if not inside_target:
                    raise ValueError(
                        "Refusing to write caption outside the selected output folder: "
                        f"{caption_path}"
                    )
                with open(caption_path, "w", encoding="utf-8") as handle:
                    handle.write(str(caption))

            counter += 1

        return (last_saved_path,)
