/**
 * Prompt808 — ComfyUI native extension.
 *
 * Registers the Prompt808 panel as a sidebar tab (or fallback menu button).
 * Tab-based internal navigation: Analyze | Library | Photos | Archetypes | Style
 */

import { app } from "../../scripts/app.js";
import * as api from "./api.js";
import { $el, Prompt808Dialog, confirmDialog, toast } from "./utils.js";

// Load stylesheet — ComfyUI only auto-loads .js, not .css
const _link = document.createElement("link");
_link.rel = "stylesheet";
_link.href = new URL("./prompt808.css", import.meta.url).href;
document.head.appendChild(_link);

// Page modules — loaded on demand
const pageLoaders = {
  analyze:     () => import("./analyze.js"),
  library:     () => import("./library.js"),
  photos:      () => import("./photos.js"),
  archetypes:  () => import("./archetypes.js"),
  style:       () => import("./style.js"),
};

const TABS = [
  { id: "analyze",    label: "Analyze" },
  { id: "library",    label: "Library" },
  { id: "photos",     label: "Photos" },
  { id: "archetypes", label: "Archetypes" },
  { id: "style",      label: "Style" },
];

// Global state
let _activeTab = "analyze";
let _libraryState = { libraries: [], active: null, dataVersion: 0 };
let _pageCache = {};      // { tabId: { module, container } }
let _rootContainer = null;
let _tabBar = null;
let _contentArea = null;
let _libSwitcher = null;

// ---------------------------------------------------------------------------
// Library state management (replaces React LibraryContext)
// ---------------------------------------------------------------------------

async function refreshLibraries() {
  try {
    const data = await api.getLibraries();
    _libraryState.libraries = data.libraries || [];
    const current = _libraryState.libraries.find(l => l.active);
    _libraryState.active = current ? current.name : null;
    if (_libraryState.active) api.setActiveLibrary(_libraryState.active);
    _renderLibSwitcher();
  } catch (e) {
    console.error("Failed to fetch libraries:", e);
  }
}

function invalidateData() {
  _libraryState.dataVersion++;
  // Notify active page
  const cached = _pageCache[_activeTab];
  if (cached?.module?.onDataVersionChanged) {
    cached.module.onDataVersionChanged(_libraryState.dataVersion);
  }
}

/** Public library state for pages */
export function getLibraryState() {
  return _libraryState;
}

/** Notify bridge nodes that dropdown options may have changed. */
function notifyNodeRefresh() {
  document.dispatchEvent(new CustomEvent("prompt808:options-changed"));
}

export { invalidateData, refreshLibraries };

// ---------------------------------------------------------------------------
// Library switcher UI
// ---------------------------------------------------------------------------

function _renderLibSwitcher() {
  if (!_libSwitcher) return;
  _libSwitcher.innerHTML = "";

  // No libraries — show prominent "Create" CTA
  if (_libraryState.libraries.length === 0) {
    _libSwitcher.appendChild($el("button.p8-btn.p8-btn--primary.p8-lib-create-cta", {
      textContent: "+ Create Your First Library",
      onClick: async () => {
        const name = prompt("Library name:");
        if (!name?.trim()) return;
        try {
          await api.createLibrary(name.trim());
          await api.switchLibrary(name.trim());
          api.setActiveLibrary(name.trim());
          toast(`Library "${name.trim()}" created`, "success");
          await refreshLibraries();
          invalidateData();
          notifyNodeRefresh();
          _switchTab("analyze");
        } catch (err) {
          toast("Create failed: " + err.message, "error");
        }
      },
    }));
    return;
  }

  // Sort libraries according to user preference
  const sortOrder = app.ui.settings.getSettingValue("Prompt808.General.LibrarySort") ?? "Newest First";
  const libs = [..._libraryState.libraries];
  if (sortOrder === "A-Z") {
    libs.sort((a, b) => a.name.localeCompare(b.name));
  } else if (sortOrder === "Newest First") {
    libs.sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
  } else if (sortOrder === "Oldest First") {
    libs.sort((a, b) => (a.created_at || "").localeCompare(b.created_at || ""));
  }

  const select = $el("select.p8-lib-select", {
    onChange: async (e) => {
      const val = e.target.value;
      if (val === _libraryState.active) return;
      try {
        await api.switchLibrary(val);
        api.setActiveLibrary(val);
        toast(`Switched to "${val}"`, "success");
        await refreshLibraries();
        invalidateData();
        // Re-render active page
        _switchTab(_activeTab);
      } catch (err) {
        toast("Switch failed: " + err.message, "error");
      }
    },
  }, libs.map(lib =>
    $el("option", {
      value: lib.name,
      textContent: `${lib.name} (${lib.element_count || 0})`,
      selected: lib.active,
    })
  ));

  _libSwitcher.appendChild(select);

  // New library button
  _libSwitcher.appendChild($el("button.p8-lib-action", {
    textContent: "+",
    title: "New library",
    onClick: async () => {
      const name = prompt("New library name:");
      if (!name?.trim()) return;
      try {
        await api.createLibrary(name.trim());
        await api.switchLibrary(name.trim());
        api.setActiveLibrary(name.trim());
        toast(`Library "${name.trim()}" created`, "success");
        await refreshLibraries();
        invalidateData();
        notifyNodeRefresh();
        _switchTab(_activeTab);
      } catch (err) {
        toast("Create failed: " + err.message, "error");
      }
    },
  }));

  // Rename button
  _libSwitcher.appendChild($el("button.p8-lib-action", {
    textContent: "\u270E",
    title: "Rename library",
    onClick: async () => {
      const active = _libraryState.active;
      if (!active) return;
      const newName = prompt("Rename library:", active);
      if (!newName?.trim() || newName.trim() === active) return;
      try {
        await api.renameLibrary(active, newName.trim());
        api.setActiveLibrary(newName.trim());
        toast(`Renamed to "${newName.trim()}"`, "success");
        await refreshLibraries();
        invalidateData();
        notifyNodeRefresh();
        _switchTab(_activeTab);
      } catch (err) {
        toast("Rename failed: " + err.message, "error");
      }
    },
  }));

  // Delete button (hidden when only 1 library)
  if (_libraryState.libraries.length > 1) {
    _libSwitcher.appendChild($el("button.p8-lib-action.p8-lib-action--danger", {
      textContent: "\u00d7",
      title: "Delete library",
      onClick: async () => {
        const active = _libraryState.active;
        if (!active) return;
        const ok = await confirmDialog(
          "Delete Library",
          `Permanently delete "${active}" and all its elements, photos, archetypes, and style profiles? This cannot be undone.`,
        );
        if (!ok) return;
        try {
          await api.deleteLibrary(active);
          toast(`Library "${active}" deleted`, "success");
          await refreshLibraries();
          invalidateData();
          notifyNodeRefresh();
          _switchTab(_activeTab);
        } catch (err) {
          toast("Delete failed: " + err.message, "error");
        }
      },
    }));
  }

  // Import button
  const importInput = $el("input", {
    type: "file",
    accept: ".p808",
    style: { display: "none" },
    onChange: async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      try {
        const result = await api.importLibrary(file);
        if (result.status === "error") {
          toast(result.message, "error");
        } else {
          const counts = result.imported || {};
          const total = Object.values(counts).reduce((sum, v) => {
            if (typeof v === "number") return sum + v;
            return sum + (v.inserted || 0) + (v.replaced || 0);
          }, 0);
          let msg = `Imported "${result.library_name}" (${total} records, ${result.thumbnails || 0} thumbnails)`;
          if (result.warnings?.length) {
            msg += ` — ${result.warnings.length} warning(s)`;
            console.warn("Import warnings:", result.warnings);
          }
          toast(msg, "success");
          await refreshLibraries();
          invalidateData();
          notifyNodeRefresh();
          _switchTab(_activeTab);
        }
      } catch (err) {
        toast("Import failed: " + err.message, "error");
      }
      importInput.value = "";
    },
  });
  _libSwitcher.appendChild(importInput);
  _libSwitcher.appendChild($el("button.p8-lib-action", {
    textContent: "\u2b07",
    title: "Import library",
    onClick: () => importInput.click(),
  }));

  // Export button
  _libSwitcher.appendChild($el("button.p8-lib-action", {
    textContent: "\u2b06",
    title: "Export library",
    onClick: () => {
      const body = $el("div", {}, [
        $el("p.p8-dialog__message", { textContent: "Export with or without thumbnail images?" }),
        $el("div.p8-dialog__actions", {}, [
          $el("button.p8-btn.p8-btn--secondary", {
            textContent: "Without Thumbnails",
            onClick: async () => {
              dlg.close();
              try {
                await api.exportLibrary(false);
                toast("Library exported", "success");
              } catch (err) {
                toast("Export failed: " + err.message, "error");
              }
            },
          }),
          $el("button.p8-btn", {
            textContent: "With Thumbnails",
            onClick: async () => {
              dlg.close();
              try {
                await api.exportLibrary(true);
                toast("Library exported", "success");
              } catch (err) {
                toast("Export failed: " + err.message, "error");
              }
            },
          }),
        ]),
      ]);
      const dlg = new Prompt808Dialog("Export Library", body);
      dlg.show();
    },
  }));
}

// ---------------------------------------------------------------------------
// Tab navigation
// ---------------------------------------------------------------------------

function _renderTabs() {
  _tabBar.innerHTML = "";
  for (const tab of TABS) {
    const btn = $el("button", {
      classList: ["p8-tab", tab.id === _activeTab ? "p8-tab--active" : ""],
      textContent: tab.label,
      onClick: () => _switchTab(tab.id),
    });
    _tabBar.appendChild(btn);
  }
}

async function _switchTab(tabId) {
  _activeTab = tabId;
  _renderTabs();

  // Hide all pages
  for (const [id, cached] of Object.entries(_pageCache)) {
    cached.container.style.display = id === tabId ? "" : "none";
  }

  // Load & render page if not cached
  if (!_pageCache[tabId]) {
    const loader = pageLoaders[tabId];
    if (!loader) return;

    const container = $el("div.p8-page");
    _contentArea.appendChild(container);

    try {
      const mod = await loader();
      _pageCache[tabId] = { module: mod, container };
      mod.render(container);
    } catch (e) {
      container.textContent = `Failed to load ${tabId}: ${e.message}`;
      console.error(e);
    }
  } else {
    // Page already loaded — show and optionally refresh
    _pageCache[tabId].container.style.display = "";
    const mod = _pageCache[tabId].module;
    if (mod.onActivated) mod.onActivated();
  }
}

// ---------------------------------------------------------------------------
// Build root container
// ---------------------------------------------------------------------------

function renderPrompt808App(container) {
  container.innerHTML = "";

  // Re-attach existing DOM tree if we've rendered before (sidebar close/reopen)
  if (_rootContainer) {
    container.appendChild(_rootContainer);
    return;
  }

  // First render — build from scratch
  _rootContainer = $el("div.prompt808", {}, [
    // Top bar: library switcher
    _libSwitcher = $el("div.p8-lib-switcher"),
    // Tab bar
    _tabBar = $el("div.p8-tab-bar"),
    // Content
    _contentArea = $el("div.p8-content"),
  ]);

  container.appendChild(_rootContainer);

  // Init — await libraries so pages see the active library on first render
  _renderTabs();
  refreshLibraries().then(() => _switchTab(_activeTab));
}

// ---------------------------------------------------------------------------
// ComfyUI extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
  name: "Prompt808",

  settings: [
    {
      id: "Prompt808. Prompt808",
      name: "Version 1.0.0",
      type: () => {
        const a = document.createElement("a");
        a.textContent = "www.prompt808.com";
        a.href = "https://www.prompt808.com";
        a.target = "_blank";
        return a;
      },
    },
    {
      id: "Prompt808.General.NSFW",
      name: "NSFW Content",
      type: "boolean",
      defaultValue: false,
      tooltip: "Show adult content styles (Boudoir, Erotica) and moods (Sensual, Provocative) in the Generate panel and node dropdowns.",
      onChange: (value) => {
        api.saveAppSettings({ nsfw: value }).then(() => notifyNodeRefresh()).catch(() => {
          toast("Failed to save NSFW setting — change may not persist after restart", "error");
          notifyNodeRefresh();
        });
      },
    },
    {
      id: "Prompt808.General.LibrarySort",
      name: "Library Sort Order",
      type: "combo",
      defaultValue: "Newest First",
      options: ["A-Z", "Newest First", "Oldest First"],
      tooltip: "How libraries are sorted in the sidebar dropdown.",
      onChange: (value) => {
        api.saveAppSettings({ library_sort: value }).catch(() => {});
        _renderLibSwitcher();
      },
    },
    {
      id: "Prompt808.Troubleshooting.Debug",
      name: "Debug Mode",
      type: "boolean",
      defaultValue: false,
      tooltip: "Log full LLM prompts and responses to the ComfyUI console for troubleshooting prompt composition. Only applies when an LLM model is selected.",
      onChange: (value) => {
        api.saveAppSettings({ debug: value }).catch(() => {});
      },
    },
    {
      id: "Prompt808.Support.BuyMeACoffee",
      name: "Support Prompt808",
      type: () => {
        const wrap = document.createElement("div");
        wrap.style.cssText = "text-align:center;padding:8px 0";
        const a = document.createElement("a");
        a.href = "https://buymeacoffee.com/machete3000";
        a.target = "_blank";
        a.title = "Buy Me a Coffee";
        const img = document.createElement("img");
        img.src = "/prompt808/img/qr-code.png";
        img.alt = "Buy Me a Coffee QR Code";
        img.style.cssText = "max-width:180px;border-radius:8px";
        a.appendChild(img);
        wrap.appendChild(a);
        return wrap;
      },
    },
  ],

  async setup() {
    // Hydrate settings from server (sync across browsers)
    api.getAppSettings().then(saved => {
      if (saved.nsfw !== undefined) app.ui.settings.setSettingValue("Prompt808.General.NSFW", saved.nsfw);
      if (saved.debug !== undefined) app.ui.settings.setSettingValue("Prompt808.Troubleshooting.Debug", saved.debug);
      if (saved.library_sort !== undefined) app.ui.settings.setSettingValue("Prompt808.General.LibrarySort", saved.library_sort);
    }).catch(() => {});

    // Try sidebar tab first (newer ComfyUI builds)
    if (app.extensionManager?.registerSidebarTab) {
      app.extensionManager.registerSidebarTab({
        id: "prompt808",
        icon: "pi pi-camera",
        title: "Prompt808",
        type: "custom",
        render: (container) => renderPrompt808App(container),
      });
      return;
    }

    // Fallback: floating dialog triggered by menu button
    const menuBtn = $el("button.p8-menu-btn", {
      textContent: "Prompt808",
      onClick: () => {
        // Toggle panel visibility
        if (_rootContainer && document.body.contains(_rootContainer)) {
          _rootContainer.parentElement.remove();
          _rootContainer = null;
          return;
        }
        const panel = $el("div.p8-floating-panel");
        renderPrompt808App(panel);
        document.body.appendChild(panel);
      },
    });

    // Add to ComfyUI menu area
    const menuArea = document.querySelector(".comfy-menu") ||
                     document.querySelector("#comfy-menu") ||
                     document.querySelector("header");
    if (menuArea) {
      menuArea.appendChild(menuBtn);
    } else {
      // Last resort: fixed button
      menuBtn.style.cssText = "position:fixed;top:8px;right:200px;z-index:999;";
      document.body.appendChild(menuBtn);
    }
  },
});
