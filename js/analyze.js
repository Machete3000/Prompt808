/**
 * Prompt808 — Analyze page (photo upload + element extraction).
 */

import * as api from "./api.js";
import { getLibraryState, invalidateData } from "./prompt808.js";
import { $el, confirmDialog, helpButton, spinner, toast } from "./utils.js";

let _container = null;
let _files = [];
let _previews = [];  // { name, url }
let _analyzing = false;
let _dragging = false;
let _history = [];
let _cancelled = false;
let _stopping = false;
let _batchIndex = 0;
let _batchTotal = 0;
let _currentPhase = "";

// Settings (persisted to localStorage)
let _showSettings = false;
let _visionModel = localStorage.getItem("sf_visionModel") || "Qwen3-VL-8B-Instruct-FP8";
let _quantization = localStorage.getItem("sf_quantization") || "FP8";
let _maxTokens = Number(localStorage.getItem("sf_maxTokens")) || 2048;
let _apiUrl = localStorage.getItem("sf_visionApiUrl") || "http://127.0.0.1:1234";
let _force = false;
let _options = null;

let _dropzoneEl, _fileActionsEl, _settingsEl, _btnRowEl, _hintEl, _progressEl, _phaseEl, _historyEl, _fileInput;

function _hasLibrary() {
  return !!getLibraryState().active;
}

export function render(container) {
  _container = container;
  container.innerHTML = "";

  _fileInput = $el("input", {
    type: "file", accept: "image/*", multiple: true,
    style: { display: "none" },
    onChange: (e) => { if (e.target.files?.length) _addFiles(e.target.files); e.target.value = ""; },
  });

  const page = $el("div", {}, [
    $el("div.p8-page-header", {}, [
      $el("h2.p8-page-title", { textContent: "Analyze Photos" }),
      helpButton("Analyze", [
        "Drop, paste, or browse for images to extract visual elements. Each photo is sent to a local vision model that identifies subjects, lighting, color palette, composition, mood, textures, and more.",
        "Extracted elements are automatically categorized and added to your element library. Duplicate detection prevents the same element from being added twice — use the \"Skip duplicate check\" option to force re-analysis.",
        "Vision Model Settings lets you choose how photos are analyzed. \"API\" (recommended) routes inference through an OpenAI-compatible server like LM Studio, Ollama, llama.cpp, or vLLM — typically 10-20x faster than the local path on the same GPU. The other entries load a Qwen-VL model locally via HuggingFace transformers; useful for self-contained installs but slower. Quantization (FP8, 8-bit, 4-bit) only applies to the local path and reduces VRAM at some quality cost.",
        "You can queue multiple images at once — they are processed sequentially with a progress bar and live status updates showing the current analysis phase. The image being processed is highlighted with a spinner overlay. Results appear in the Analysis Results section below as each image completes.",
      ], { marginLeft: "auto" }),
    ]),
    _dropzoneEl = _hasLibrary()
      ? $el("div.p8-dropzone", {
          onClick: () => _fileInput.click(),
          onDrop: _handleDrop,
          onDragover: (e) => { e.preventDefault(); e.stopPropagation(); _setDragging(true); },
          onDragleave: (e) => { e.stopPropagation(); _setDragging(false); },
        })
      : $el("div.p8-dropzone.p8-dropzone--disabled"),
    _fileInput,
    _fileActionsEl = $el("div"),
    _settingsEl = $el("div"),
    _btnRowEl = $el("div"),
    _hintEl = $el("div"),
    _progressEl = $el("div"),
    _phaseEl = $el("div.p8-phase-text"),
    _historyEl = $el("div"),
  ]);

  container.appendChild(page);
  _renderDropzone();
  _renderSettings();
  _renderBtnRow();
  _renderProgress();
  _renderPhase(_currentPhase);
  _renderHistory();
  _renderHint();

  // Load options
  api.getAnalyzeOptions().then(opts => { _options = opts; _renderSettings(); }).catch(() => {});

  // Paste listener — remove old one first to avoid duplicates on re-render
  if (_pasteHandler) document.removeEventListener("paste", _pasteHandler, true);
  _pasteHandler = (e) => {
    // Only intercept when the Analyze tab is visible and a library is active
    if (!_container || !_container.offsetParent) return;
    if (!_hasLibrary()) return;
    const items = e.clipboardData?.items;
    if (!items) return;
    const imageFiles = [];
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) imageFiles.push(file);
      }
    }
    if (imageFiles.length > 0) {
      e.preventDefault();
      e.stopPropagation();
      _addFiles(imageFiles);
    }
  };
  document.addEventListener("paste", _pasteHandler, true);
}

let _pasteHandler = null;

export function onDataVersionChanged() {
  if (_analyzing) return;
  _history = [];
  if (_historyEl) _renderHistory();
}

export function onActivated() {
  if (_container) render(_container);
}

function _addFiles(newFiles) {
  const imageFiles = Array.from(newFiles).filter(f => f.type.startsWith("image/"));
  if (imageFiles.length === 0) return;
  _files.push(...imageFiles);
  for (const f of imageFiles) {
    _previews.push({ name: f.name, url: URL.createObjectURL(f) });
  }
  _renderDropzone();
  _renderFileActions();
  _renderBtnRow();
}

function _removeFile(index) {
  URL.revokeObjectURL(_previews[index]?.url);
  _files.splice(index, 1);
  _previews.splice(index, 1);
  _renderDropzone();
  _renderFileActions();
  _renderBtnRow();
}

function _clearFiles() {
  _previews.forEach(p => URL.revokeObjectURL(p.url));
  _files = [];
  _previews = [];
  _renderDropzone();
  _renderFileActions();
  _renderBtnRow();
}

function _setDragging(val) {
  _dragging = val;
  _dropzoneEl.classList.toggle("p8-dropzone--dragging", val);
}

async function _handleDrop(e) {
  e.preventDefault();
  e.stopPropagation();
  _setDragging(false);

  if (e.dataTransfer.files?.length) {
    _addFiles(e.dataTransfer.files);
    return;
  }

  // Browser image drops
  const html = e.dataTransfer.getData("text/html");
  const uri = e.dataTransfer.getData("text/uri-list") || e.dataTransfer.getData("text/plain");
  let imageUrl = null;

  if (html) {
    const match = html.match(/<img[^>]+src=["']([^"']+)["']/i);
    if (match) imageUrl = match[1];
  }
  if (!imageUrl && uri && /^https?:\/\/.+/i.test(uri)) {
    imageUrl = uri.trim();
  }
  if (!imageUrl) return;

  try {
    toast("Downloading image...", "info");
    const resp = await fetch(imageUrl);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const blob = await resp.blob();
    if (!blob.type.startsWith("image/")) { toast("Dropped item is not an image", "error"); return; }
    const urlPath = new URL(imageUrl).pathname;
    const ext = blob.type.split("/")[1]?.replace("jpeg", "jpg") || "jpg";
    let name = urlPath.split("/").pop() || "image";
    if (!name.includes(".")) name = `${name}.${ext}`;
    const file = new File([blob], name, { type: blob.type });
    _addFiles([file]);
  } catch (err) {
    toast("Failed to download image: " + err.message, "error");
  }
}

function _renderDropzone() {
  _dropzoneEl.innerHTML = "";
  _dropzoneEl.classList.toggle("p8-dropzone--has-files", _files.length > 0);

  if (!_hasLibrary()) {
    _dropzoneEl.appendChild($el("div", { style: { display: "flex", flexDirection: "column", alignItems: "center", gap: "10px", padding: "10px 20px" } }, [
      $el("span", { textContent: "\u25B2", style: { fontSize: "28px", color: "var(--p8-accent)", lineHeight: "1" } }),
      $el("strong", { textContent: "Create a library to get started", style: { fontSize: "14px", color: "var(--p8-text)" } }),
      $el("p", {
        style: { fontSize: "12px", color: "var(--p8-text-muted)", margin: "0", textAlign: "center", lineHeight: "1.6" },
        innerHTML: 'Click the <strong>+ Create Your First Library</strong> button at the top of this panel.<br>Name your library, then come back here to analyze photos.',
      }),
    ]));
    return;
  }

  if (_files.length > 0) {
    const grid = $el("div.p8-preview-grid", { onClick: (e) => e.stopPropagation() });
    _previews.forEach((p, i) => {
      grid.appendChild($el("div.p8-preview-item", {}, [
        $el("img.p8-preview-thumb", { src: p.url, alt: p.name }),
        $el("span.p8-preview-name", { textContent: p.name }),
        !_analyzing ? $el("button.p8-preview-remove", {
          textContent: "x",
          onClick: () => _removeFile(i),
        }) : null,
      ].filter(Boolean)));
    });
    if (!_analyzing) {
      grid.appendChild($el("div.p8-add-more", {
        textContent: "+ Add more",
        onClick: () => _fileInput.click(),
      }));
    }
    _dropzoneEl.appendChild(grid);
  } else {
    _dropzoneEl.appendChild($el("div", { style: { display: "flex", flexDirection: "column", alignItems: "center", gap: "6px" } }, [
      $el("p.p8-drop-label", { textContent: "Drop images here or click to browse" }),
      $el("p.p8-drop-hint", { textContent: "JPG, PNG, WebP, AVIF, BMP, TIFF, HEIC \u2014 multiple files supported" }),
    ]));
  }
}

function _renderFileActions() {
  _fileActionsEl.innerHTML = "";
  if (_files.length > 0 && !_analyzing) {
    _fileActionsEl.appendChild($el("div.p8-file-actions", {}, [
      $el("span", { textContent: `${_files.length} image(s) selected` }),
      $el("button.p8-btn.p8-btn--small.p8-btn--danger-outline", {
        textContent: "Clear all",
        onClick: _clearFiles,
      }),
    ]));
  }
}

function _renderSettings() {
  _settingsEl.innerHTML = "";

  _settingsEl.appendChild($el("button.p8-settings-toggle", {
    textContent: (_showSettings ? "\u25BC" : "\u25B6") + " Vision Model Settings",
    onClick: () => { _showSettings = !_showSettings; _renderSettings(); },
  }));

  if (_showSettings) {
    const models = _options?.vision_models || ["Qwen3-VL-8B-Instruct-FP8"];
    const quants = _options?.quantizations || ["FP16", "FP8", "8-bit", "4-bit"];

    const isApi = _visionModel === "API";

    _settingsEl.appendChild($el("div.p8-settings-panel", {}, [
      $el("div.p8-field", {}, [
        $el("label.p8-label", { textContent: "Vision Model" }),
        $el("select.p8-select", {
          onChange: (e) => {
            _visionModel = e.target.value;
            localStorage.setItem("sf_visionModel", _visionModel);
            _renderSettings();
          },
        }, models.map(m => $el("option", { value: m, textContent: m, selected: m === _visionModel }))),
      ]),
      isApi
        ? $el("div.p8-field", { style: { gridColumn: "1 / -1" } }, [
            $el("label.p8-label", { textContent: "API URL" }),
            $el("input.p8-input", {
              type: "text", value: _apiUrl,
              placeholder: "http://127.0.0.1:1234",
              onInput: (e) => {
                _apiUrl = e.target.value;
                localStorage.setItem("sf_visionApiUrl", _apiUrl);
              },
              title: "OpenAI-compatible vision endpoint (LM Studio, Ollama, llama.cpp, vLLM, etc.)",
            }),
          ])
        : $el("div.p8-field", {}, [
            $el("label.p8-label", { textContent: "Quantization" }),
            $el("select.p8-select", {
              onChange: (e) => { _quantization = e.target.value; localStorage.setItem("sf_quantization", _quantization); },
            }, quants.map(q => $el("option", { value: q, textContent: q, selected: q === _quantization }))),
          ]),
      $el("div.p8-field", {}, [
        $el("label.p8-label", { textContent: "Max Tokens" }),
        $el("input.p8-input", {
          type: "number", value: _maxTokens, min: 256, max: 4096,
          onInput: (e) => { _maxTokens = Number(e.target.value); localStorage.setItem("sf_maxTokens", String(_maxTokens)); },
        }),
      ]),
      $el("label.p8-check", { style: { gridColumn: "1 / -1" } }, [
        $el("input", { type: "checkbox", checked: _force, onChange: (e) => _force = e.target.checked }),
        "Skip duplicate check (force re-analyze)",
      ]),
    ]));
  }
}

function _renderBtnRow() {
  _btnRowEl.innerHTML = "";
  const row = $el("div", { style: { display: "flex", alignItems: "center", gap: "10px", marginTop: "12px" } });

  const analyzeBtn = $el("button.p8-btn.p8-btn--primary.p8-btn--large", {
    disabled: _files.length === 0 || _analyzing || !_hasLibrary(),
    dataset: { sf: "analyzeBtn" },
    onClick: _handleAnalyze,
  });

  if (_analyzing) {
    analyzeBtn.appendChild(spinner(14));
    analyzeBtn.appendChild(document.createTextNode(` Analyzing ${_batchIndex} of ${_batchTotal}...`));
  } else {
    analyzeBtn.textContent = `Analyze ${_files.length || ""} Photo${_files.length !== 1 ? "s" : ""}`;
  }
  row.appendChild(analyzeBtn);

  if (_analyzing) {
    const stopBtn = $el("button.p8-btn.p8-btn--danger-outline", {
      disabled: _stopping,
      onClick: () => { _cancelled = true; _stopping = true; _renderBtnRow(); },
    });
    if (_stopping) {
      stopBtn.appendChild(spinner(14));
      stopBtn.appendChild(document.createTextNode(" Stopping..."));
    } else {
      stopBtn.textContent = "Stop";
    }
    row.appendChild(stopBtn);
  }

  _btnRowEl.appendChild(row);
}

function _renderHint() {
  _hintEl.innerHTML = "";
  if (_analyzing) return;
  const libState = getLibraryState();
  const activeLib = libState.libraries.find(l => l.active);
  if (!activeLib || activeLib.element_count === 0) return;

  _hintEl.appendChild($el("p.p8-hint", {
    innerHTML: 'Add the <strong>Prompt808 Generate</strong> node to your ComfyUI workflow to generate prompts from this library.',
  }));
}

function _renderProgress() {
  _progressEl.innerHTML = "";
  if (!_analyzing || _batchTotal <= 1) return;
  const pct = (_batchIndex / _batchTotal) * 100;
  _progressEl.appendChild($el("div.p8-batch-progress", {}, [
    $el("div.p8-batch-fill", { style: { width: pct + "%" } }),
    $el("span.p8-batch-text", { textContent: `${_batchIndex} / ${_batchTotal}` }),
  ]));
}

function _renderPhase(msg) {
  _currentPhase = msg || "";
  _phaseEl.textContent = _currentPhase;
}

function _renderHistory() {
  _historyEl.innerHTML = "";
  if (_history.length === 0) return;

  _historyEl.appendChild($el("div.p8-history", {}, [
    $el("h3.p8-history-title", { textContent: "Analysis Results" }),
    ..._history.map(item =>
      $el("div", {
        classList: ["p8-history-card", item.status === "error" ? "p8-history-card--error" : ""],
      }, [
        $el("div.p8-history-header", {}, [
          $el("strong", { textContent: item.filename }),
          $el("span.p8-history-timestamp", { textContent: item.timestamp }),
        ]),
        item.status === "error"
          ? $el("div.p8-history-meta", {}, [$el("span.p8-history-error", { textContent: "Failed: " + item.error })])
          : $el("div.p8-history-meta", {}, [
              item.subject_type ? $el("span.p8-badge", { textContent: item.subject_type }) : null,
              $el("span", { textContent: `+${item.elements_added} elements` }),
              item.duplicates_rejected > 0 ? $el("span.p8-history-dupes", { textContent: `-${item.duplicates_rejected} duplicates` }) : null,
            ].filter(Boolean)),
      ])
    ),
  ]));
}

async function _handleAnalyze() {
  if (_files.length === 0) return;

  // If the library already has elements, confirm the user wants to add to it
  const libState = getLibraryState();
  const activeLib = libState.libraries.find(l => l.active);
  if (activeLib && activeLib.element_count > 0) {
    const ok = await confirmDialog(
      "Add to existing library?",
      `"${activeLib.name}" already has ${activeLib.element_count} elements. New photos will be added to this library. To start a new library, cancel and use the + button above.`,
    );
    if (!ok) return;
  }

  _cancelled = false;
  _stopping = false;
  _analyzing = true;
  _history = [];
  _renderHistory();

  const toProcess = [..._files];
  const totalCount = toProcess.length;
  _batchIndex = 0;
  _batchTotal = totalCount;

  _renderBtnRow();
  _renderProgress();

  let totalAdded = 0;
  let totalDupes = 0;
  let succeeded = 0;

  for (let i = 0; i < toProcess.length; i++) {
    if (_cancelled) break;

    _batchIndex = i + 1;
    _renderBtnRow();
    _renderProgress();
    const file = toProcess[i];

    // Mark the first thumbnail as currently being analyzed
    const firstItem = _dropzoneEl.querySelector(".p8-preview-item");
    if (firstItem) firstItem.classList.add("p8-preview-analyzing");

    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("vision_model", _visionModel);
      formData.append("quantization", _quantization);
      formData.append("max_tokens", String(_maxTokens));
      if (_visionModel === "API") formData.append("api_url", _apiUrl);
      if (_force) formData.append("force", "true");

      const result = await api.analyzePhoto(formData, (msg) => _renderPhase(msg));
      totalAdded += result.elements_added;
      totalDupes += result.duplicates_rejected;
      succeeded++;

      _history.unshift({
        filename: file.name,
        timestamp: new Date().toLocaleTimeString(),
        ...result,
      });
    } catch (e) {
      _history.unshift({
        filename: file.name,
        timestamp: new Date().toLocaleTimeString(),
        status: "error",
        error: e.message,
        elements_added: 0,
        duplicates_rejected: 0,
      });
    }

    _renderPhase("");

    // Remove processed file
    if (_previews.length > 0) URL.revokeObjectURL(_previews[0]?.url);
    _files.shift();
    _previews.shift();
    _renderDropzone();
    _renderFileActions();
    _renderHistory();
    invalidateData();
  }

  const wasCancelled = _cancelled;
  _analyzing = false;
  _stopping = false;
  _batchIndex = 0;
  _batchTotal = 0;
  _renderBtnRow();
  _renderProgress();
  _renderPhase("");

  if (succeeded > 0) _renderHint();

  if (!wasCancelled) {
    api.analysisCleanup().catch(() => {});
  }

  if (wasCancelled) {
    toast(`Stopped after ${succeeded}/${totalCount} photos, +${totalAdded} elements`, "warning");
  } else if (succeeded === totalCount) {
    toast(`Batch complete: ${succeeded} photos, +${totalAdded} elements, -${totalDupes} duplicates`, "success");
  } else {
    toast(`Batch done: ${succeeded}/${totalCount} succeeded, +${totalAdded} elements`, "warning");
  }
}
