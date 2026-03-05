import { app } from "../../scripts/app.js";

/** All live Prompt808 Generate nodes — refreshed on sidebar events. */
const _liveNodes = new Set();

/** Per-node AbortController to cancel stale in-flight refreshes. */
const _refreshControllers = new WeakMap();

app.registerExtension({
  name: "Prompt808.Bridge",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "Prompt808 Generate") return;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      origOnNodeCreated?.apply(this, arguments);
      _liveNodes.add(this);
      refreshDropdowns(this);
      hookLibraryWidget(this);
    };

    const origOnRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      _liveNodes.delete(this);
      _refreshControllers.get(this)?.abort();
      _refreshControllers.delete(this);
      origOnRemoved?.apply(this, arguments);
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

// Listen for sidebar events (library CRUD, NSFW toggle) and refresh all nodes.
document.addEventListener("prompt808:options-changed", () => {
  for (const node of _liveNodes) {
    refreshDropdowns(node);
  }
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
  // Cancel any previous in-flight refresh for this node
  _refreshControllers.get(node)?.abort();
  const ac = new AbortController();
  _refreshControllers.set(node, ac);

  try {
    const headers = {};
    if (libraryName) {
      headers["X-Library"] = libraryName;
    }
    const resp = await fetch("/prompt808/api/generate/options", {
      headers,
      signal: ac.signal,
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
    // Server not reachable or request aborted — silently ignore
  }
}

async function refreshDropdowns(node) {
  // Cancel any previous in-flight refresh for this node
  _refreshControllers.get(node)?.abort();
  const ac = new AbortController();
  _refreshControllers.set(node, ac);

  try {
    // Use /prompt808/api/ routes (registered on PromptServer)
    const libWidget = node.widgets?.find((w) => w.name === "library");
    const headers = {};
    if (libWidget?.value) headers["X-Library"] = libWidget.value;

    // Fetch options and libraries in parallel
    const [optResp, libResp] = await Promise.all([
      fetch("/prompt808/api/generate/options", {
        headers,
        signal: ac.signal,
      }),
      fetch("/prompt808/api/libraries", {
        signal: ac.signal,
      }),
    ]);

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

      const ptypeWidget = node.widgets?.find((w) => w.name === "prompt_type");
      if (ptypeWidget && data.prompt_types?.length) {
        const prev = ptypeWidget.value;
        ptypeWidget.options.values = data.prompt_types;
        ptypeWidget.value = data.prompt_types.includes(prev) ? prev : data.prompt_types[0];
      }

      const moodWidget = node.widgets?.find((w) => w.name === "mood");
      if (moodWidget && data.moods?.length) {
        const prev = moodWidget.value;
        moodWidget.options.values = data.moods;
        moodWidget.value = data.moods.includes(prev) ? prev : data.moods[0];
      }
    }

    if (libResp.ok) {
      const libData = await libResp.json();
      const libs = libData.libraries || [];
      if (libs.length) {
        const libWidget2 = node.widgets?.find((w) => w.name === "library");
        if (libWidget2) {
          const prev = libWidget2.value;
          const names = libs.map((l) => l.name);
          const activeName = libs.find((l) => l.active)?.name || names[0];
          libWidget2.options.values = names;
          libWidget2.value = names.includes(prev) ? prev : activeName;
        }
      }
    }

    node.setDirtyCanvas(true, true);
  } catch {
    // Server not reachable or request aborted — silently ignore
  }
}
