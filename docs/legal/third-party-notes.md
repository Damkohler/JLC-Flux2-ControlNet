# Third-Party Reference Notes

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


The compact model structure and expected mask-aware/control mathematics were informed by public reference work associated with Alibaba VideoX-Fun's FLUX.2 ControlNet implementation and the Flux2Fun ComfyUI experiment.

The referenced VideoX-Fun source identifies its own upstream Diffusers and Black Forest Labs references and licensing information in source headers. Users and distributors should review the current upstream repositories and model-distribution terms directly before redistributing code or model weights.

JLC Flux2 ControlNet:

- implements its own ComfyUI-native integration strategy;
- does not import the reference projects at runtime;
- does not install a global FLUX.2 monkey patch;
- does not bundle ControlNet, FLUX.2, text-encoder, VAE, or LoRA weights;
- releases the source in this repository under the MIT License.

Model files remain governed by the licenses and terms of their original publishers and distributors. The repository MIT License does not relicense third-party model weights.

This note is an attribution and release-review aid, not legal advice.

---

[Documentation home](../README.md) · [Project README](../../README.md)
