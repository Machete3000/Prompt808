/**
 * Prompt808 — Style Profiles page.
 */

import * as api from "./api.js";
import { $el, helpButton, spinner, toast } from "./utils.js";

let _container = null;
let _genres = [];
let _summary = {};
let _loading = true;
let _expanded = {};
let _details = {};

let _listEl, _headerCount;

export function render(container) {
  _container = container;
  container.innerHTML = "";

  const page = $el("div", {}, [
    $el("div.p8-page-header", {}, [
      $el("h2.p8-page-title", { textContent: "Style Profiles" }),
      _headerCount = $el("span.p8-page-count"),
      helpButton("Style Profiles", [
        "Style profiles capture the recurring aesthetic patterns in your element library, organized by genre (e.g. portrait, landscape, street). They track your preferred lighting setups, color palettes, composition tendencies, and other stylistic dimensions.",
        "The generator uses style profiles as context during LLM composition — when building a prompt for a particular genre, the profile guides the LLM toward your established aesthetic preferences.",
        "Expand any genre card to see its full style context — a natural-language summary of the detected patterns. The top patterns row shows the dominant value for each dimension at a glance.",
        "Profiles update automatically as photos are added, deleted, or edited. \"Recalculate\" recomputes all profiles from the current library elements if you ever want a clean refresh.",
      ]),
      $el("button.p8-btn.p8-btn--danger-outline", {
        textContent: "Recalculate",
        style: { marginLeft: "auto" },
        dataset: { sf: "resetAllBtn" },
        onClick: _handleResetAll,
      }),
    ]),
    _listEl = $el("div"),
  ]);

  container.appendChild(page);
  _fetchProfiles();
}

export function onActivated() { _fetchProfiles(); }
export function onDataVersionChanged() { _fetchProfiles(); }

async function _fetchProfiles() {
  _loading = true;
  _expanded = {};
  _details = {};
  _renderList();
  try {
    const data = await api.getStyleProfiles();
    _genres = data.genres || [];
    _summary = data.summary || {};
  } catch (e) {
    toast("Failed to load profiles: " + e.message, "error");
  }
  _loading = false;
  _headerCount.textContent = `${_genres.length} genre(s)`;
  // Show/hide reset all button
  const btn = _container.querySelector("[data-sf='resetAllBtn']");
  if (btn) btn.style.display = _genres.length > 0 ? "" : "none";
  _renderList();
}

async function _handleResetAll() {
  if (!confirm("Recalculate all style profiles from current library data?")) return;
  try {
    await api.resetAllProfiles();
    toast("Profiles recalculated", "success");
    _details = {};
    _fetchProfiles();
  } catch (e) {
    toast("Recalculate failed: " + e.message, "error");
  }
}

async function _toggleGenre(genre) {
  const isOpen = _expanded[genre];
  _expanded[genre] = !isOpen;

  if (!isOpen && !_details[genre]) {
    try {
      const data = await api.getStyleProfile(genre);
      _details[genre] = data;
    } catch (e) {
      toast("Failed to load profile detail: " + e.message, "error");
    }
  }

  _renderList();
}

async function _handleResetGenre(genre) {
  if (!confirm(`Recalculate style profile for "${genre}"?`)) return;
  try {
    await api.resetStyleProfile(genre);
    toast(`Profile "${genre}" recalculated`, "success");
    delete _details[genre];
    _fetchProfiles();
  } catch (e) {
    toast("Recalculate failed: " + e.message, "error");
  }
}

function _renderList() {
  _listEl.innerHTML = "";
  if (_loading) {
    _listEl.appendChild($el("div.p8-center", {}, [spinner(24)]));
    return;
  }
  if (_genres.length === 0) {
    _listEl.appendChild($el("p.p8-empty", { textContent: "No style profiles yet. Analyze photos to start building your stylistic DNA." }));
    return;
  }

  const list = $el("div.p8-style-list");
  for (const genre of _genres) {
    const info = _summary[genre] || {};
    const detail = _details[genre];
    const isOpen = _expanded[genre];

    const card = $el("div.p8-style-card", {}, [
      $el("div.p8-style-header", {
        onClick: () => _toggleGenre(genre),
      }, [
        $el("div.p8-style-title", {}, [
          $el("span.p8-arch-arrow", { textContent: isOpen ? "\u25BC" : "\u25B6" }),
          $el("h3", { textContent: genre }),
          $el("span.p8-style-obs", { textContent: `${info.observations ?? 0} observations` }),
        ]),
        $el("button.p8-btn.p8-btn--small.p8-btn--danger-outline", {
          textContent: "Recalculate",
          onClick: (e) => { e.stopPropagation(); _handleResetGenre(genre); },
        }),
      ]),
    ]);

    // Top patterns (always visible)
    if (info.top_patterns && Object.keys(info.top_patterns).length > 0) {
      const patterns = $el("div.p8-style-patterns");
      for (const [dim, tag] of Object.entries(info.top_patterns)) {
        patterns.appendChild($el("div.p8-style-pattern-row", {}, [
          $el("span.p8-style-dim", { textContent: dim }),
          $el("span.p8-style-tag", { textContent: tag }),
        ]));
      }
      card.appendChild(patterns);
    }

    // Expanded detail
    if (isOpen && detail) {
      const detailEl = $el("div.p8-style-detail");
      if (detail.context_text) {
        detailEl.appendChild($el("div.p8-style-context-box", {}, [
          $el("h4.p8-style-context-title", { textContent: "Style Context" }),
          $el("p.p8-style-context-text", { textContent: detail.context_text }),
        ]));
      }
      if (info.last_updated) {
        detailEl.appendChild($el("p.p8-style-updated", { textContent: `Last updated: ${info.last_updated}` }));
      }
      card.appendChild(detailEl);
    }

    list.appendChild(card);
  }
  _listEl.appendChild(list);
}
