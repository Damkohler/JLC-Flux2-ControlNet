import { app } from "/scripts/app.js";

const ICON_SIZE = 12;

// These are ComfyUI NODE_CLASS_MAPPINGS keys, not display names.
// Legacy/development aliases are retained so the logo continues to appear
// while the public node export names are finalized.
const FLUX2_CONTROLNET_NODE_NAMES = new Set([
    "JLCFlux2ControlNetLoader",
    "JLCFlux2ControlNetApplyDiagnostic",
    "JLCFlux2ControlNetOrchestrator",
]);

const iconImage = new Image();
iconImage.src = new URL(
    "./assets/icons/jlc-comfyui-nodes_Logo-Dark-0128.png",
    import.meta.url
).href;

function isFlux2ControlNetNode(nodeData) {
    return Boolean(
        nodeData &&
        typeof nodeData.name === "string" &&
        FLUX2_CONTROLNET_NODE_NAMES.has(nodeData.name)
    );
}

app.registerExtension({
    name: "JLC.Flux2ControlNet.Icons",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isFlux2ControlNetNode(nodeData)) {
            return;
        }

        if (nodeType.prototype.__jlcFlux2ControlNetIconApplied) {
            return;
        }

        nodeType.prototype.__jlcFlux2ControlNetIconApplied = true;

        const originalOnDrawForeground = nodeType.prototype.onDrawForeground;

        nodeType.prototype.onDrawForeground = function (ctx) {
            if (originalOnDrawForeground) {
                originalOnDrawForeground.apply(this, arguments);
            }

            try {
                if (!iconImage.complete || iconImage.naturalWidth <= 0) {
                    return;
                }

                ctx.save();
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = "high";

                const x = ICON_SIZE + 18;
                const y = -(ICON_SIZE + 9);

                ctx.drawImage(iconImage, x, y, ICON_SIZE, ICON_SIZE);
                ctx.restore();
            } catch (err) {
                try {
                    ctx.restore();
                } catch (_) {
                    // no-op
                }
                console.warn("[JLC Flux2 ControlNet Icons] draw skipped:", err);
            }
        };
    },
});
