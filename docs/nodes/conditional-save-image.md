# Conditional Save Image

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


## JLC Conditional Save Image

**Status:** Stable utility

This output node combines lazy branch selection with optional PNG saving. It is designed as a companion to mutually exclusive cache-preparation and inference branches.

## Inputs

| Input | Type | Default | Purpose |
|---|---|---:|---|
| `image_on_true` | IMAGE, lazy | — | Requested only when `switch` is true. |
| `image_on_false` | IMAGE, lazy | — | Requested only when `switch` is false. |
| `switch` | BOOLEAN | `false` | Selects the active expensive branch. |
| `save_when` | dropdown | `FALSE` | Chooses which switch states write PNG files. |
| `filename_prefix` | STRING | `ComfyUI` | Supports normal ComfyUI formatting and `%batch_num%`. |
| `output_folder` | STRING | `output` | Output root, relative subfolder, or absolute path. |
| `compress_level` | INT | `4` | PNG compression `0–9`. |

Hidden prompt and workflow metadata are embedded unless ComfyUI metadata writing is disabled.

## `save_when` values

| Value | Save behavior |
|---|---|
| `FALSE` | Save only when the switch is false. |
| `TRUE` | Save only when the switch is true. |
| `BOTH` | Save in either state. |
| `NONE` | Execute and return the selected image without saving. |

## Outputs

- `selected_image`
- `image_save_path`

When saving is disabled for the selected state, the image path is an empty string.

## Lazy execution

Only the selected image input is requested. This is the important difference from a node that merely chooses between two already evaluated images.

A common cache workflow uses:

- true branch = cache preparation or proof image;
- false branch = final inference;
- `save_when = FALSE`.

The setup branch executes without writing a final image, and the inference branch is saved.

## Output paths

- `output` writes to the ComfyUI output directory.
- A relative folder resolves inside the output directory.
- An absolute path is accepted.
- A relative path that escapes the ComfyUI output root is rejected.

Inspect example workflow output paths before running on another machine.


---

[Documentation home](../README.md) · [Project README](../../README.md)
