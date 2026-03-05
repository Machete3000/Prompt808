/**
 * Prompt808 API client — plain ES module for ComfyUI native integration.
 * All endpoints use /prompt808/api/* prefix (registered on PromptServer).
 */

const BASE = "/prompt808/api";
const THUMB_BASE = "/prompt808/thumbnails";

let _activeLibrary = null;

export function setActiveLibrary(name) {
  _activeLibrary = name;
}

export function getActiveLibraryName() {
  return _activeLibrary;
}

export function getThumbnailUrl(filename) {
  return `${THUMB_BASE}/${encodeURIComponent(filename)}`;
}

async function request(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...options.headers };
  if (_activeLibrary) {
    headers["X-Library"] = _activeLibrary;
  }
  const res = await fetch(`${BASE}${path}`, { headers, ...options });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

// Health
export function checkHealth() {
  return request("/health");
}

// ---- Analysis ----

export async function analyzePhoto(formData, onProgress) {
  const headers = {};
  if (_activeLibrary) headers["X-Library"] = _activeLibrary;
  const res = await fetch(`${BASE}/analyze`, {
    method: "POST",
    body: formData,
    headers,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop();
    for (const part of parts) {
      let type, data;
      for (const line of part.split("\n")) {
        if (line.startsWith("event:")) type = line.slice(6);
        if (line.startsWith("data:")) data = JSON.parse(line.slice(5));
      }
      if (type === "progress" && onProgress) onProgress(data.message);
      else if (type === "result") result = data;
      else if (type === "error") throw new Error(data.message);
    }
  }

  if (!result) throw new Error("No result received from analysis");
  return result;
}

export function getAnalyzeOptions() {
  return request("/analyze/options");
}

export function analysisCleanup() {
  return request("/analyze/cleanup", { method: "POST" });
}

// ---- Elements ----

export function getElements({ category, offset = 0, limit = 50 } = {}) {
  const params = new URLSearchParams({ offset, limit });
  if (category) params.set("category", category);
  return request(`/library/elements?${params}`);
}

export function getElement(id) {
  return request(`/library/elements/${encodeURIComponent(id)}`);
}

export function deleteElement(id) {
  return request(`/library/elements/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export function updateElement(id, updates) {
  return request(`/library/elements/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });
}

export function getCategories() {
  return request("/library/categories");
}

export function getStats() {
  return request("/library/stats");
}

export function resetAllData() {
  return request("/library/reset", { method: "DELETE" });
}

// ---- Photos ----

export function getPhotos() {
  return request("/library/photos");
}

export function getPhotoElements(thumbnail) {
  return request(`/library/photos/${encodeURIComponent(thumbnail)}/elements`);
}

export function deletePhoto(thumbnail) {
  return request(`/library/photos/${encodeURIComponent(thumbnail)}`, { method: "DELETE" });
}

// ---- Archetypes ----

export function getArchetypes() {
  return request("/library/archetypes");
}

export function deleteArchetype(id) {
  return request(`/library/archetypes/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export function regenerateArchetypes() {
  return request("/library/archetypes/regenerate", { method: "POST" });
}

// ---- Generation (node bridge) ----

export function getGenerateOptions() {
  return request("/generate/options");
}

// ---- Style Profiles ----

export function getStyleProfiles() {
  return request("/style/profiles");
}

export function getStyleProfile(genre) {
  return request(`/style/profiles/${encodeURIComponent(genre)}`);
}

export function resetStyleProfile(genre) {
  return request(`/style/profiles/${encodeURIComponent(genre)}/reset`, { method: "POST" });
}

export function resetAllProfiles() {
  return request("/style/profiles/reset", { method: "POST" });
}

// ---- Libraries ----

export function getLibraries() {
  return request("/libraries");
}

export function createLibrary(name) {
  return request("/libraries", { method: "POST", body: JSON.stringify({ name }) });
}

export function switchLibrary(name) {
  return request("/libraries/active", { method: "PUT", body: JSON.stringify({ name }) });
}

export function renameLibrary(oldName, newName) {
  return request(`/libraries/${encodeURIComponent(oldName)}`, {
    method: "PATCH",
    body: JSON.stringify({ name: newName }),
  });
}

export function deleteLibrary(name) {
  return request(`/libraries/${encodeURIComponent(name)}`, { method: "DELETE" });
}

// ---- App Settings ----

export function getAppSettings() {
  return request("/settings");
}

export function saveAppSettings(settings) {
  return request("/settings", {
    method: "PUT",
    body: JSON.stringify(settings),
  });
}

// ---- Export / Import ----

export async function exportLibrary(includeThumbnails = true) {
  const headers = { "Content-Type": "application/json" };
  if (_activeLibrary) headers["X-Library"] = _activeLibrary;
  const res = await fetch(`${BASE}/library/export`, {
    method: "POST",
    headers,
    body: JSON.stringify({ include_thumbnails: includeThumbnails }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  const ct = res.headers.get("Content-Type") || "";
  if (ct.includes("application/json")) {
    // Pro-gated upsell or error response
    return res.json();
  }
  // Binary download
  const blob = await res.blob();
  const name = _activeLibrary || "library";
  const a = document.createElement("a");
  const blobUrl = URL.createObjectURL(blob);
  a.href = blobUrl;
  a.download = `${name}.p808`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Defer revocation so the browser has time to start the download
  setTimeout(() => URL.revokeObjectURL(blobUrl), 10000);
  return { status: "downloaded", filename: `${name}.p808` };
}

export async function importLibrary(file, targetName) {
  const formData = new FormData();
  formData.append("file", file);
  if (targetName) formData.append("name", targetName);
  const headers = {};
  if (_activeLibrary) headers["X-Library"] = _activeLibrary;
  const res = await fetch(`${BASE}/library/import`, {
    method: "POST",
    body: formData,
    headers,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}
