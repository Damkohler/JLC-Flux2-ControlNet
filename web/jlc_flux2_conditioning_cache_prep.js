/*
 * JLC Flux2 Conditioning Cache Prep - Dynamic Input Layout
 * ---------------------------------------------------------
 * Dedicated frontend for the additive unified cache-preparation node.
 * It does not modify the released shared dynamic-slot frontend.
 */

const { app } = window.comfyAPI.app;

const NODE_NAME = "JLCFlux2ConditioningCachePrep";
const REFERENCE_WIDGET = "reference_count";
const CONTROL_WIDGET = "control_count";
const INPAINT_WIDGET = "use_inpaint";
const APPLY_BUTTON_LABEL = "Apply Input Layout";
const MAX_REFERENCES = 10;
const MAX_CONTROLS = 4;
const INSTALL_FLAG = "__jlc_flux2_conditioning_cache_layout_installed";
const EMPTY_LATENT_NAME = "empty_flux2_latent";
const CONTRACT_WIDGET = "input_connection_contract";
const CONTRACT_REVISION = "jlc-flux2-conditioning-cache-input-wires-v1";

const BUTTON_BLUE = "#0B8CE9";
const BUTTON_TEXT = "#FFFFFF";

function widget(node, name) {
    return node.widgets?.find((candidate) => candidate.name === name);
}

function boundedInt(value, minimum, maximum) {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed)) return minimum;
    return Math.max(minimum, Math.min(maximum, parsed));
}

function desiredLayout(node) {
    const referenceCount = boundedInt(widget(node, REFERENCE_WIDGET)?.value, 0, MAX_REFERENCES);
    const controlCount = boundedInt(widget(node, CONTROL_WIDGET)?.value, 0, MAX_CONTROLS);
    const rawInpaint = widget(node, INPAINT_WIDGET)?.value;
    const useInpaint = rawInpaint === true || rawInpaint === 1 || rawInpaint === "true";

    const names = [];
    if (controlCount > 0 || useInpaint) names.push(EMPTY_LATENT_NAME);
    for (let index = 1; index <= referenceCount; index++) {
        names.push(`reference_image_${index}`);
    }
    for (let index = 1; index <= controlCount; index++) {
        names.push(`control_image_${index}`);
    }
    if (useInpaint) names.push("inpaint_image", "inpaint_mask");

    return { referenceCount, controlCount, useInpaint, names };
}

function isManagedInput(name) {
    return (
        name === EMPTY_LATENT_NAME ||
        // Legacy v0.1.1 sockets are managed so an unconnected old node can cleanly migrate.
        name === "target_latent" ||
        name === "passthrough_image" ||
        /^reference_image_\d+$/.test(name) ||
        /^control_image_\d+$/.test(name) ||
        name === "inpaint_image" ||
        name === "inpaint_mask"
    );
}

function inputType(name) {
    if (name === EMPTY_LATENT_NAME) return "LATENT";
    if (name === "inpaint_mask") return "MASK";
    return "IMAGE";
}

function inputOptions(name) {
    if (name === EMPTY_LATENT_NAME) {
        return {
            tooltip:
                "Connect the same Empty Flux2 Latent used by the sampler. " +
                "Do not connect the sampler output or any sampled latent.",
        };
    }
    if (inputType(name) === "IMAGE") return { shape: 7 };
    return undefined;
}

function connected(input) {
    return input?.link !== null && input?.link !== undefined;
}


function buildConnectionContract(node) {
    const layout = desiredLayout(node);
    const wired = {};

    // Record every managed socket that physically exists, including any stale
    // connected socket the layout button has refused to remove. This lets Python
    // distinguish muted/pruned paths from genuinely missing or stale wiring.
    for (const input of node.inputs || []) {
        if (!isManagedInput(input.name)) continue;
        wired[input.name] = connected(input);
    }

    // Explicit false entries make selected-but-missing sockets unambiguous.
    for (const name of layout.names) {
        if (!(name in wired)) wired[name] = false;
    }

    return JSON.stringify({
        revision: CONTRACT_REVISION,
        layout: {
            reference_count: layout.referenceCount,
            control_count: layout.controlCount,
            use_inpaint: layout.useInpaint,
        },
        wired,
    });
}

function updateConnectionContract(node) {
    const contractWidget = widget(node, CONTRACT_WIDGET);
    if (!contractWidget) return "{}";
    const value = buildConnectionContract(node);
    contractWidget.value = value;
    return value;
}

function hideConnectionContractWidget(node) {
    const contractWidget = widget(node, CONTRACT_WIDGET);
    if (!contractWidget) return;

    contractWidget.type = "hidden";
    contractWidget.computeSize = () => [0, -4];
    contractWidget.hidden = true;

    // Prompt serialization calls serializeValue when available. Compute the
    // contract at that moment so the queued prompt always receives current wire
    // state even if the last graph edit happened immediately before queueing.
    contractWidget.serializeValue = () => updateConnectionContract(node);
}

function notifyBlocked(names) {
    const message =
        "JLC Flux2 Conditioning Cache Prep cannot remove connected inputs:\n\n" +
        names.join("\n") +
        "\n\nDisconnect them, then press Apply Input Layout again.";
    window.alert(message);
}

function syncInputLinkSlots(node) {
    if (!node.inputs || !node.graph?.links) return;
    for (let slot = 0; slot < node.inputs.length; slot++) {
        const linkId = node.inputs[slot]?.link;
        if (linkId === null || linkId === undefined) continue;
        const link = node.graph.links[linkId];
        if (link) link.target_slot = slot;
    }
}

function reconcileInputs(node) {
    if (!node.inputs) node.inputs = [];
    const layout = desiredLayout(node);
    const desired = new Set(layout.names);

    const blocked = node.inputs
        .filter((input) => isManagedInput(input.name) && !desired.has(input.name) && connected(input))
        .map((input) => input.name);
    if (blocked.length > 0) {
        notifyBlocked(blocked);
        return false;
    }

    // Remove unused managed sockets from bottom to top.
    for (let index = node.inputs.length - 1; index >= 0; index--) {
        const input = node.inputs[index];
        if (isManagedInput(input.name) && !desired.has(input.name)) {
            node.removeInput(index);
        }
    }

    // Add any newly requested sockets.
    for (const name of layout.names) {
        if (!node.inputs.some((input) => input.name === name)) {
            node.addInput(name, inputType(name), inputOptions(name));
        }
    }

    const fixedInputs = node.inputs.filter((input) => !isManagedInput(input.name));
    const managedByName = new Map(
        node.inputs
            .filter((input) => isManagedInput(input.name))
            .map((input) => [input.name, input])
    );

    // When geometry is required, intentionally place Empty Flux2 Latent above VAE.
    const emptyLatent = managedByName.get(EMPTY_LATENT_NAME);
    const remainingManaged = layout.names
        .filter((name) => name !== EMPTY_LATENT_NAME)
        .map((name) => managedByName.get(name))
        .filter(Boolean);
    node.inputs = [
        ...(emptyLatent ? [emptyLatent] : []),
        ...fixedInputs,
        ...remainingManaged,
    ];
    syncInputLinkSlots(node);

    const refWidget = widget(node, REFERENCE_WIDGET);
    const controlWidget = widget(node, CONTROL_WIDGET);
    const inpaintWidget = widget(node, INPAINT_WIDGET);
    if (refWidget) refWidget.value = layout.referenceCount;
    if (controlWidget) controlWidget.value = layout.controlCount;
    if (inpaintWidget) inpaintWidget.value = layout.useInpaint;

    if (node.computeSize) {
        const currentWidth = node.size?.[0] ?? 260;
        const computed = node.computeSize();
        if (computed) node.setSize?.([Math.max(currentWidth, computed[0]), computed[1]]);
    }
    updateConnectionContract(node);
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
    return true;
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

function styleButton(button) {
    button.draw = function (ctx, node, width, y, height) {
        const marginX = 10;
        const marginY = 2;
        const x = marginX;
        const h = Math.max(18, height - marginY * 2);
        const w = Math.max(40, width - marginX * 2);
        const buttonY = y + marginY;
        ctx.save();
        roundedRectPath(ctx, x, buttonY, w, h, 5);
        ctx.fillStyle = BUTTON_BLUE;
        ctx.fill();
        ctx.fillStyle = BUTTON_TEXT;
        ctx.font = "bold 12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(button.name, x + w / 2, buttonY + h / 2);
        ctx.restore();
    };
}

function install(node) {
    if (node[INSTALL_FLAG]) return;
    node[INSTALL_FLAG] = true;

    hideConnectionContractWidget(node);

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        requestAnimationFrame(() => {
            hideConnectionContractWidget(this);
            reconcileInputs(this);
            updateConnectionContract(this);
        });
        return result;
    };

    const originalOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function () {
        const result = originalOnConnectionsChange?.apply(this, arguments);
        requestAnimationFrame(() => updateConnectionContract(this));
        return result;
    };

    const originalOnSerialize = node.onSerialize;
    node.onSerialize = function () {
        updateConnectionContract(this);
        return originalOnSerialize?.apply(this, arguments);
    };

    const button = node.addWidget("button", APPLY_BUTTON_LABEL, null, () => {
        reconcileInputs(node);
        updateConnectionContract(node);
    });
    styleButton(button);

    requestAnimationFrame(() => {
        hideConnectionContractWidget(node);
        reconcileInputs(node);
        updateConnectionContract(node);
    });
}

app.registerExtension({
    name: "JLC.Flux2.ConditioningCachePrep",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            install(this);
            return result;
        };
    },
});
