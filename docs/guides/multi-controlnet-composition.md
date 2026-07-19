# Multi-ControlNet Composition

> [!NOTE]
> **Documentation status: Draft for Release 1.0.0.**  
> This page follows the current release implementation and supplied workflows, but remains under editorial review. The source code and current ComfyUI node interfaces are authoritative for exact behavior.


## Why the Orchestrator exists

ComfyUI's conventional ControlNet Apply pattern can form a recursive `previous_controlnet` chain. JLC Flux2 ControlNet preserves that conventional behavior in the single Apply nodes, but provides a separate **flat non-recursive Orchestrator** for the validated multi-ControlNet path.

The Orchestrator creates one configured child per active slot, evaluates each child independently against the same native FLUX.2 forward state, and combines the resulting residual requests before one shared injection path.

## Flat ownership model

Each child owns its own:

- control image;
- strength;
- start and end percentages;
- encoded hint-latent cache state;
- diagnostic state;
- shallow configured ControlNet copy.

All children share:

- one lazy checkpoint owner;
- one compact side-model weight set;
- one ComfyUI `CoreModelPatcher`;
- one connected VAE;
- the normal ComfyUI model-management lifecycle.

The side-model weights are not duplicated for every branch.

## Slot contract

The stable Orchestrator supports one to four slots.

- Slot 1 is required.
- Slots 2–4 are optional.
- `slot_count` is authoritative.
- Hidden or stale workflow connections above `slot_count` are ignored.
- Empty optional slots are omitted.
- Later slots are not promoted to replace a missing earlier slot.
- Branch order is retained for diagnostics, cache identity, and experimental inpaint-host selection.

A required slot receiving `None` produces a clear error. A disabled auxiliary-wrapper output should therefore be disconnected or enabled.

## Clean-conditioning requirement

The Orchestrator must receive conditioning that has no existing ControlNet object. Valid upstream sources include:

- text conditioning;
- Flux guidance metadata;
- the JLC Flux2 Reference Image Orchestrator.

Do not place an Apply node before or after the Orchestrator in the same conditioning chain.

## Strength and timing

Each branch has independent:

- `strength` from `0.0` to `2.0`;
- `start_percent` from `0.0` to `1.0`;
- `end_percent` from `0.0` to `1.0`.

`start_percent` must not exceed `end_percent`.

Strength is applied when residuals are injected. A branch at exact zero strength is an exact side-model bypass. This can be useful for A/B testing, but the required control-image input must still be valid.

## Residual injection

The compact side model produces four residual tensors. Release 1.0.0 injects them after native FLUX.2 double blocks:

```text
0, 2, 4, 6
```

For multiple branches, the residual at each target block is the weighted sum of the active child residuals. The composition wrapper owns the shared hook, and children are detached from recursive chaining.

## Apply versus Orchestrator

Use **JLC Flux2 ControlNet Apply** when:

- only one branch is needed;
- conventional chaining behavior is deliberately desired;
- an existing conditioning control should be retained as `previous_controlnet`.

Use **JLC Flux2 ControlNet Orchestrator** when:

- one to four JLC FLUX.2 branches should share one side model;
- branch behavior should be flat and inspectable;
- the experimental In/Out-Paint Adapter may be added downstream.

Use an **Advanced** variant when separate positive and negative conditioning streams must receive the same configured control object.

## Practical control balancing

A useful build order is:

1. Enable the primary structural control alone.
2. Establish a stable strength and full or nearly full timestep range.
3. Add one auxiliary branch at low strength.
4. Limit dense controls to early sampling when possible.
5. Compare changes at a fixed seed.
6. Add a third or fourth branch only after the two-branch result is stable.

OpenPose/DWPose is generally a forgiving primary branch for the experimental inpaint path. Depth, luminance, color, and similar dense controls may compete strongly with prompt, reference, or editable-region guidance.


---

[Documentation home](../README.md) · [Project README](../../README.md)
