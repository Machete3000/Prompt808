import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "Prompt808.Bridge",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Prompt808 Generate") return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      origOnNodeCreated?.apply(this, arguments);
      refreshDropdowns(this);
      hookLibraryWidget(this);
    };
  },

  async nodeCreated(node) {
    if (node.comfyClass !== "Prompt808 Generate") return;

    const refreshBtn = node.addWidget("button", "Refresh Options", null, () => {
      refreshDropdowns(node);
    });
    refreshBtn.serialize = false;
  },
});

/**
 * Watch the library widget for changes and refresh archetypes accordingly.
 */
function hookLibraryWidget(node) {
  const libWidget = node.widgets?.find((w) => w.name === "library");
  if (!libWidget) return;

  const origCallback = libWidget.callback;
  libWidget.callback = function (value) {
    origCallback?.call(this, value);
    refreshArchetypes(node, value);
  };
}

/**
 * Fetch archetypes scoped to a specific library and update the widget.
 */
async function refreshArchetypes(node, libraryName) {
  try {
    const headers = {};
    if (libraryName) {
      headers["X-Library"] = libraryName;
    }
    const resp = await fetch("/prompt808/api/generate/options", {
      headers,
      signal: AbortSignal.timeout(3000),
    });
    if (!resp.ok) return;
    const data = await resp.json();

    const archWidget = node.widgets?.find((w) => w.name === "archetype");
    if (archWidget && data.archetypes?.length) {
      const prev = archWidget.value;
      archWidget.options.values = data.archetypes;
      archWidget.value = data.archetypes.includes(prev) ? prev : data.archetypes[0];
    }

    node.setDirtyCanvas(true, true);
  } catch {
    // Server not reachable — silently ignore
  }
}

async function refreshDropdowns(node) {
  try {
    // Use /prompt808/api/ routes (registered on PromptServer)
    const optResp = await fetch("/prompt808/api/generate/options", {
      signal: AbortSignal.timeout(3000),
    });
    if (optResp.ok) {
      const data = await optResp.json();

      const modelWidget = node.widgets?.find((w) => w.name === "llm_model");
      if (modelWidget && data.models?.length) {
        const prev = modelWidget.value;
        modelWidget.options.values = data.models;
        modelWidget.value = data.models.includes(prev) ? prev : data.models[0];
      }

      const archWidget = node.widgets?.find((w) => w.name === "archetype");
      if (archWidget && data.archetypes?.length) {
        const prev = archWidget.value;
        archWidget.options.values = data.archetypes;
        archWidget.value = data.archetypes.includes(prev) ? prev : data.archetypes[0];
      }
    }

    const libResp = await fetch("/prompt808/api/libraries", {
      signal: AbortSignal.timeout(3000),
    });
    if (libResp.ok) {
      const libData = await libResp.json();
      const libs = libData.libraries || [];
      if (libs.length) {
        const libWidget = node.widgets?.find((w) => w.name === "library");
        if (libWidget) {
          const prev = libWidget.value;
          const names = libs.map((l) => l.name);
          const activeName = libs.find((l) => l.active)?.name || names[0];
          libWidget.options.values = names;
          libWidget.value = names.includes(prev) ? prev : activeName;
        }
      }
    }

    node.setDirtyCanvas(true, true);
  } catch {
    // Server not reachable — silently ignore
  }
}
