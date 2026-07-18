/*
 * JLC Flux2 Dynamic Slot Visibility Helpers
 * -----------------------------------------
 * Shared frontend companion for FLUX.2 cache-prep and orchestrator nodes.
 * Python predeclares each node's fixed maximum image slots, while this helper
 * exposes only slot_count sockets and hides per-slot widgets above slot_count.
 * Backend nodes must still treat slot_count as authoritative.
 */

const { app } = window.comfyAPI.app;

const SLOT_COUNT_WIDGET = "slot_count";
const UPDATE_BUTTON_LABEL = "Update Visible Slots";

const NODE_CONFIG = {
    JLCFlux2HintLatentCachePrep: {
        maxSlots: 4,
        inputPrefix: "control_image_",
        inputStartIndex: 1,
        slotWidgets: () => [],
        layoutKey: "__jlc_flux2_hint_cache_prep_layout",
        installFlag: "__jlc_flux2_hint_cache_prep_installed",
    },
    JLCFlux2ReferenceLatentCachePrep: {
        maxSlots: 10,
        inputPrefix: "reference_image_",
        inputStartIndex: 1,
        slotWidgets: () => [],
        layoutKey: "__jlc_flux2_reference_cache_prep_layout",
        installFlag: "__jlc_flux2_reference_cache_prep_installed",
    },
    JLCFlux2ReferenceImageOrchestrator: {
        maxSlots: 10,
        inputPrefix: "reference_image_",
        inputStartIndex: 1,
        outputPrefix: "reference_image_",
        outputStartIndex: 1,
        outputTrailingNames: ["diagnostics_json"],
        slotWidgets: (index) => [`enabled_${index}`],
        layoutKey: "__jlc_flux2_reference_orchestrator_layout",
        installFlag: "__jlc_flux2_reference_orchestrator_installed",
    },
    JLCFlux2ControlNetOrchestrator: {
        maxSlots: 4,
        inputPrefix: "control_image_",
        inputStartIndex: 1,
        outputPrefix: "control_image_",
        outputStartIndex: 1,
        outputTrailingNames: [],
        slotWidgets: (index) => [
            `strength_${index}`,
            `start_percent_${index}`,
            `end_percent_${index}`,
        ],
        layoutKey: "__jlc_flux2_controlnet_orchestrator_layout",
        installFlag: "__jlc_flux2_controlnet_orchestrator_installed",
    },
    JLCFlux2ControlNetOrchestratorAdvanced: {
        maxSlots: 4,
        inputPrefix: "control_image_",
        inputStartIndex: 1,
        outputPrefix: "control_image_",
        outputStartIndex: 1,
        outputTrailingNames: [],
        slotWidgets: (index) => [
            `strength_${index}`,
            `start_percent_${index}`,
            `end_percent_${index}`,
        ],
        layoutKey: "__jlc_flux2_controlnet_orchestrator_advanced_layout",
        installFlag: "__jlc_flux2_controlnet_orchestrator_advanced_installed",
    },
};

const JLC_PRIMARY_BUTTON_BLUE = "#0B8CE9";
const JLC_PRIMARY_BUTTON_TEXT = "#FFFFFF";

function slotInputName(index, config) {
    return `${config.inputPrefix}${index}`;
}

function slotOutputName(index, config) {
    return `${config.outputPrefix}${index}`;
}

function getSlotCount(node, config) {
    const widget = node.widgets?.find((w) => w.name === SLOT_COUNT_WIDGET);
    const raw = Number.parseInt(widget?.value ?? 1, 10);
    if (!Number.isFinite(raw)) return 1;
    return Math.max(1, Math.min(config.maxSlots, raw));
}

function findInputIndex(node, name) {
    return node.inputs?.findIndex((input) => input.name === name) ?? -1;
}

function hasInput(node, name) {
    return findInputIndex(node, name) >= 0;
}

function removeInputByName(node, name) {
    const index = findInputIndex(node, name);
    if (index < 0) return false;
    node.removeInput(index);
    return true;
}

function ensureInput(node, name, type, options) {
    if (hasInput(node, name)) return;
    node.addInput(name, type, options);
}

function findOutputIndex(node, name) {
    return node.outputs?.findIndex((output) => output.name === name) ?? -1;
}

function hasOutput(node, name) {
    return findOutputIndex(node, name) >= 0;
}

function removeOutputByName(node, name) {
    const index = findOutputIndex(node, name);
    if (index < 0) return false;
    node.removeOutput(index);
    return true;
}

function syncOutputLinkSlots(node) {
    if (!node.outputs || !node.graph?.links) return;
    for (let slot = 0; slot < node.outputs.length; slot++) {
        for (const linkId of node.outputs[slot].links || []) {
            const link = node.graph.links[linkId];
            if (link) link.origin_slot = slot;
        }
    }
}

function firstTrailingOutputIndex(node, config) {
    for (const name of config.outputTrailingNames || []) {
        const index = findOutputIndex(node, name);
        if (index >= 0) return index;
    }
    return -1;
}

function ensureOutput(node, name, type, options, config) {
    if (hasOutput(node, name)) return;
    node.addOutput(name, type, options);

    const addedIndex = findOutputIndex(node, name);
    const beforeIndex = firstTrailingOutputIndex(node, config);
    if (addedIndex < 0 || beforeIndex < 0 || addedIndex < beforeIndex) return;

    const [output] = node.outputs.splice(addedIndex, 1);
    node.outputs.splice(beforeIndex, 0, output);
    syncOutputLinkSlots(node);
}

function rememberWidgetLayout(widget, config) {
    if (!widget[config.layoutKey]) {
        widget[config.layoutKey] = {
            type: widget.type,
            computeSize: widget.computeSize,
            hidden: widget.hidden,
        };
    }
}

function hideWidget(widget, config) {
    rememberWidgetLayout(widget, config);
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.hidden = true;
}

function showWidget(widget, config) {
    const layout = widget[config.layoutKey];
    if (layout) {
        widget.type = layout.type;
        widget.computeSize = layout.computeSize;
        widget.hidden = layout.hidden ?? false;
    } else {
        widget.hidden = false;
    }
}

function rebuildSlotInputs(node, config, count) {
    if (!node.inputs) node.inputs = [];

    for (let i = config.maxSlots; i > count; i--) {
        removeInputByName(node, slotInputName(i, config));
    }

    for (let i = config.inputStartIndex; i <= count; i++) {
        ensureInput(node, slotInputName(i, config), "IMAGE", { shape: 7 });
    }
}

function rebuildSlotOutputs(node, config, count) {
    if (!config.outputPrefix) return;
    if (!node.outputs) node.outputs = [];

    for (let i = config.maxSlots; i > count; i--) {
        removeOutputByName(node, slotOutputName(i, config));
    }

    for (let i = config.outputStartIndex ?? 1; i <= count; i++) {
        ensureOutput(node, slotOutputName(i, config), "IMAGE", { shape: 7 }, config);
    }

    syncOutputLinkSlots(node);
}

function updateSlotWidgets(node, config, count) {
    if (!node.widgets) return;
    for (let i = 1; i <= config.maxSlots; i++) {
        const visible = i <= count;
        for (const name of config.slotWidgets(i) || []) {
            const widget = node.widgets.find((w) => w.name === name);
            if (!widget) continue;
            if (visible) showWidget(widget, config);
            else hideWidget(widget, config);
        }
    }
}

function resizeNodeToVisibleContent(node) {
    if (!node.computeSize || !node.size) return;
    const currentWidth = node.size[0] ?? 240;
    const computed = node.computeSize();
    if (!computed) return;
    const newWidth = Math.max(currentWidth, computed[0]);
    const newHeight = computed[1];
    if (node.setSize) node.setSize([newWidth, newHeight]);
    else {
        node.size[0] = newWidth;
        node.size[1] = newHeight;
        node.onResize?.(node.size);
    }
}

function roundedRectPath(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + width - r, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + r);
    ctx.lineTo(x + width, y + height - r);
    ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    ctx.lineTo(x + r, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

function stylePrimaryButton(widget) {
    widget.draw = function (ctx, node, widgetWidth, y, widgetHeight) {
        const marginX = 10;
        const marginY = 2;
        const x = marginX;
        const h = Math.max(18, widgetHeight - marginY * 2);
        const w = Math.max(40, widgetWidth - marginX * 2);
        const buttonY = y + marginY;
        ctx.save();
        roundedRectPath(ctx, x, buttonY, w, h, 5);
        ctx.fillStyle = JLC_PRIMARY_BUTTON_BLUE;
        ctx.fill();
        ctx.fillStyle = JLC_PRIMARY_BUTTON_TEXT;
        ctx.font = "bold 12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(widget.name, x + w / 2, buttonY + h / 2);
        ctx.restore();
    };
}

function applyVisibleSlotCount(node, config) {
    const count = getSlotCount(node, config);

    rebuildSlotInputs(node, config, count);
    rebuildSlotOutputs(node, config, count);
    updateSlotWidgets(node, config, count);

    const countWidget = node.widgets?.find((w) => w.name === SLOT_COUNT_WIDGET);
    if (countWidget && countWidget.value !== count) countWidget.value = count;

    resizeNodeToVisibleContent(node);
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
}

function install(node, config) {
    if (node[config.installFlag]) return;
    node[config.installFlag] = true;

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        requestAnimationFrame(() => applyVisibleSlotCount(this, config));
        return result;
    };

    const countWidget = node.widgets?.find((w) => w.name === SLOT_COUNT_WIDGET);
    if (countWidget) {
        const originalCallback = countWidget.callback;
        countWidget.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            if (!arguments[1]) requestAnimationFrame(() => applyVisibleSlotCount(node, config));
            return result;
        };
    }

    const updateButton = node.addWidget("button", UPDATE_BUTTON_LABEL, null, () => {
        applyVisibleSlotCount(node, config);
    });
    stylePrimaryButton(updateButton);

    requestAnimationFrame(() => applyVisibleSlotCount(node, config));
}

app.registerExtension({
    name: "JLC.Flux2.DynamicSlots",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const config = NODE_CONFIG[nodeData?.name];
        if (!config) return;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            install(this, config);
            return result;
        };
    },
});
