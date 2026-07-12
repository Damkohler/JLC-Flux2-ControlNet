"""Explicit non-recursive composition for Flux.2 ControlNet branches."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import comfy.controlnet

from .constants import CONTROL_LAYERS, PROJECT_LOG_PREFIX, REQUESTS_KEY
from .control import JLCFlux2Control
from .hooks import make_injection_hook_group


_EMPTY_CONTROL = {"input": [], "middle": [], "output": []}


class JLCFlux2ComposedControl(comfy.controlnet.ControlBase):
    """Own one or more detached Flux.2 controls without recursive chaining.

    Child controls share their underlying model patchers with the loader outputs,
    but each child owns its own hint, strength, timestep range, and runtime cache.
    The children are evaluated explicitly in a flat loop. Their residuals are
    combined later by the per-forward hook and injected once per native block.
    """

    def __init__(
        self,
        children: Iterable[JLCFlux2Control],
        *,
        diagnostics_enabled: bool = True,
    ):
        super().__init__()
        self.children = tuple(children)
        if len(self.children) < 1:
            raise ValueError("Flux2 composition requires at least one child control.")
        if not all(isinstance(child, JLCFlux2Control) for child in self.children):
            raise TypeError("Flux2 composition accepts only JLCFlux2Control children.")

        for child in self.children:
            # The composite owns evaluation order. No child may recurse into a
            # previous ControlNet, and child hook groups are deliberately hidden.
            child.set_previous_controlnet(None)
            child.extra_hooks = None

        self.extra_hooks = make_injection_hook_group()
        self.diagnostics_enabled = bool(diagnostics_enabled)
        self._wrapper_logged = False
        self._composition_logged = False
        self._reference_skip_logged = False
        self._injected_blocks: set[int] = set()
        self._injection_logged = False

    def copy(self):
        copied_children = []
        for child in self.children:
            child_copy = child.copy()
            child_copy.set_previous_controlnet(None)
            child_copy.extra_hooks = None
            copied_children.append(child_copy)

        copied = JLCFlux2ComposedControl(
            copied_children,
            diagnostics_enabled=self.diagnostics_enabled,
        )
        self.copy_to(copied)
        return copied

    def get_models(self):
        """Materialize and return each active shared side model exactly once."""
        models = []
        seen = set()
        for child in self.children:
            if child.strength == 0.0:
                continue
            for wrapped in child.get_models():
                identity = id(wrapped)
                if identity not in seen:
                    seen.add(identity)
                    models.append(wrapped)
        return models

    def inference_memory_requirements(self, dtype):
        # Weight residency is represented by get_models(). Child activation
        # requirements are currently zero, matching the validated single path.
        return sum(child.inference_memory_requirements(dtype) for child in self.children)

    def pre_run(self, model, percent_to_timestep_function):
        # The composite itself has no timing window; each child owns its range.
        self.timestep_range = None
        self._wrapper_logged = False
        self._composition_logged = False
        self._reference_skip_logged = False
        self._injected_blocks.clear()
        self._injection_logged = False

        for child in self.children:
            child.set_previous_controlnet(None)
            child.pre_run(model, percent_to_timestep_function)

        if self.diagnostics_enabled:
            active = [child for child in self.children if child.strength != 0.0]
            strengths = [float(child.strength) for child in active]
            ranges = [tuple(child.timestep_percent_range) for child in active]
            checkpoints = [child.checkpoint_name or "unnamed checkpoint" for child in active]
            logging.info(
                "%s Explicit non-recursive composition prepared: %d configured branches, %d active, strengths=%s, ranges=%s, checkpoints=%s.",
                PROJECT_LOG_PREFIX,
                len(self.children),
                len(active),
                strengths,
                ranges,
                checkpoints,
            )

    def cleanup(self):
        for child in self.children:
            child.set_previous_controlnet(None)
            child.cleanup()

        self._wrapper_logged = False
        self._composition_logged = False
        self._reference_skip_logged = False
        self._injected_blocks.clear()
        self._injection_logged = False
        self.timestep_range = None
        self.cond_hint = None
        self.extra_concat = None

    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        if self.previous_controlnet is not None:
            raise RuntimeError(
                "JLC Flux2 composed control must be the sole conditioning control. "
                "Do not chain Apply nodes before or after the Composition node."
            )

        requests = transformer_options.setdefault(REQUESTS_KEY, [])
        existing_ids = {id(request.get("control")) for request in requests}

        for slot_index, child in enumerate(self.children, start=1):
            child.set_previous_controlnet(None)
            before = len(requests)
            child.get_control(
                x_noisy,
                t,
                cond,
                batched_number,
                transformer_options,
            )
            for request in requests[before:]:
                control = request.get("control")
                if control is None or id(control) in existing_ids:
                    continue
                request["owner"] = self
                request["composition_slot"] = slot_index
                existing_ids.add(id(control))

        return _EMPTY_CONTROL.copy()

    def note_diagnostic_wrapper(self, x, context, ref_latents):
        if not self.diagnostics_enabled or self._wrapper_logged:
            return
        self._wrapper_logged = True
        ref_count = 0 if ref_latents is None else len(ref_latents)
        logging.info(
            "%s Composite Flux.2 wrapper reached: latent=%s, context=%s, reference_latents=%d.",
            PROJECT_LOG_PREFIX,
            tuple(x.shape),
            None if context is None else tuple(context.shape),
            ref_count,
        )

    def note_composition(self, requests: list[dict]):
        if not self.diagnostics_enabled or self._composition_logged:
            return
        self._composition_logged = True
        slots = [request.get("composition_slot") for request in requests]
        strengths = [float(request.get("strength", 1.0)) for request in requests]
        logging.info(
            "%s Flat side-branch evaluation active for composition slots=%s with strengths=%s; no previous_controlnet recursion is used.",
            PROJECT_LOG_PREFIX,
            slots,
            strengths,
        )

    def note_reference_skip(self):
        if not self.diagnostics_enabled or self._reference_skip_logged:
            return
        self._reference_skip_logged = True
        logging.warning(
            "%s Flux.2 composition does not yet support reference latents; all composed branches are skipped and native output is preserved.",
            PROJECT_LOG_PREFIX,
        )

    def note_composed_injection(
        self,
        *,
        block_index: int,
        strengths: list[float],
        residual,
    ):
        if not self.diagnostics_enabled:
            return
        self._injected_blocks.add(block_index)
        if (
            not self._injection_logged
            and self._injected_blocks == set(CONTROL_LAYERS)
        ):
            logging.info(
                "%s Non-recursive composed residual injection confirmed at double blocks %s from %d branches with strengths=%s; combined dtype=%s, device=%s.",
                PROJECT_LOG_PREFIX,
                CONTROL_LAYERS,
                len(strengths),
                strengths,
                residual.dtype,
                residual.device,
            )
            self._injection_logged = True
