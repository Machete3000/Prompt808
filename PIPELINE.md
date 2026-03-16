# Prompt808 — Analysis & Generation Pipeline

This document explains in detail how Prompt808 analyzes images and generates prompts.

## Table of Contents

- [Overview](#overview)
- [Multi-Library Architecture](#multi-library-architecture)
- [Analysis Pipeline](#analysis-pipeline)
  - [1. Image Upload & Validation](#1-image-upload--validation)
  - [2. Photo-Level Deduplication (CLIP)](#2-photo-level-deduplication-clip)
  - [3. Medium Detection](#3-medium-detection)
  - [4. QwenVL Element Extraction (Dual Pipeline)](#4-qwenvl-element-extraction-dual-pipeline)
  - [5. JSON Parsing & Retry Strategy](#5-json-parsing--retry-strategy)
  - [6. Element Normalization](#6-element-normalization)
  - [7. Description Deduplication](#7-description-deduplication)
  - [8. Tag Normalization](#8-tag-normalization)
  - [9. Store Commitment](#9-store-commitment)
  - [10. Archetype Regeneration](#10-archetype-regeneration)
  - [11. Style Profile Update](#11-style-profile-update)
- [Archetype Generation](#archetype-generation)
  - [Photo Grouping](#photo-grouping)
  - [Feature Embedding](#feature-embedding)
  - [Agglomerative Clustering](#agglomerative-clustering)
  - [Cluster Merging](#cluster-merging)
  - [Archetype Building](#archetype-building)
  - [Naming](#naming)
- [Prompt Generation Pipeline](#prompt-generation-pipeline)
  - [1. Cache Lookup](#1-cache-lookup)
  - [2. Archetype Resolution & Element Filtering](#2-archetype-resolution--element-filtering)
  - [3. Frequency-Weighted Element Selection](#3-frequency-weighted-element-selection)
  - [4. Prompt Composition](#4-prompt-composition)
  - [5. Negative Prompt Assembly](#5-negative-prompt-assembly)
  - [6. Cache Storage & WebSocket Broadcast](#6-cache-storage--websocket-broadcast)
- [Multi-Library Generation](#multi-library-generation)
- [Style Profiles](#style-profiles)
- [Library Export & Import](#library-export--import)
- [Models & Embedding Layers](#models--embedding-layers)
- [Data Flow Diagram](#data-flow-diagram)

---

## Overview

Prompt808 operates as two distinct pipelines that share a common element library:

1. **Analysis pipeline** — Takes an image, detects its artistic medium, extracts structured elements via the appropriate pipeline (photography or native), deduplicates, normalizes, and stores them.
2. **Generation pipeline** — Takes a seed + parameters, selects elements from the library, and composes a natural-language prompt via LLM rewriting or style-differentiated concatenation. Supports multi-library generation — combining elements from multiple libraries into a single merged pool.

The library grows with each analyzed image. Elements are grouped into **archetypes** (scene-type clusters like "Studio Side Light" or "Beach Golden Hour") that act as coherent element pools for generation.

Prompt808 runs as a ComfyUI custom node. All API routes are registered on ComfyUI's PromptServer under `/prompt808/api/*`. The sidebar UI is vanilla JavaScript loaded by ComfyUI's extension system.

---

## Multi-Library Architecture

All data is organized into isolated **libraries**. Each library has its own elements, archetypes, vocabulary, embeddings cache, style profiles, and prompt cache — all stored in a single SQLite database.

**Storage layout:**

```
user_data/
├── prompt808.db                # SQLite database (all libraries, all tables)
└── libraries/
    ├── default/
    │   └── thumbnails/         # Photo thumbnails (images stay on disk)
    └── Portraits/
        └── thumbnails/
```

**Database schema:** The `prompt808.db` file contains 9 tables, all scoped by a `library_id` foreign key:

| Table                    | Contents                                                                |
| ------------------------ | ----------------------------------------------------------------------- |
| `libraries`              | Library metadata (name, active flag)                                    |
| `elements`               | Extracted scene elements (tags/attributes stored as JSON columns)       |
| `archetypes`             | Photo-level scene clusters (compatible tags, element IDs)               |
| `vocabulary`             | Canonical tag forms with aliases                                        |
| `style_profiles`         | Per-genre style patterns (EMA-weighted)                                 |
| `embeddings_cache`       | Sentence-transformer embeddings (BLOB, 384 floats)                      |
| `image_embeddings_cache` | CLIP image embeddings (BLOB, 512 floats)                                |
| `prompt_cache`           | Deterministic prompt cache keyed by input hash                          |
| `generate_settings`      | App-wide settings (NSFW, debug, balance_libraries) |

The database uses WAL journal mode for concurrent reads with serialized writes via a single `threading.Lock`. All embeddings are stored as raw binary BLOBs (`np.float32.tobytes()`), which is ~6x smaller than JSON float arrays.

**Event loop safety:** All route handlers offload synchronous SQLite operations to the thread pool via `asyncio.to_thread()`. This prevents database access from blocking ComfyUI's aiohttp event loop, keeping the UI responsive during all API calls.

**VRAM management exception:** ComfyUI VRAM reclamation must run synchronously on the event-loop thread, **not** in a thread pool. ComfyUI's model patching internals (tensor operations, parameter re-wrapping) are not thread-safe — dispatching to `asyncio.to_thread()` corrupts tensor state and causes `RuntimeError: Cannot set version_counter for inference tensor` on the next workflow execution. The pre-analysis VRAM reclamation in `routes.py` calls `_free_vram_for_analysis()` directly (synchronously) using ComfyUI's cooperative `free_memory()` API. This asks ComfyUI to free only the VRAM needed for the analysis models, avoiding interference with concurrently running workflows on high-VRAM systems.

**Library scoping:** All store and core modules call `library_manager.get_library_id()` to get the integer primary key for the current library context. Every query is scoped by `WHERE library_id=?`. Deleting a library cascades to all related rows via `ON DELETE CASCADE`.

**Per-request scoping:** The frontend sends an `X-Library` header with every API call. The route handler reads this header and scopes the request to that library via `library_manager.set_request_library()`. This means:

- Multiple browser tabs can operate on different libraries concurrently
- Requests without the header fall back to the persisted active library

**Settings sharing:** App-wide settings (NSFW content toggle, debug mode) are stored in the `generate_settings` table with key `'app'`, synced from ComfyUI's settings dialog via the `/prompt808/api/settings` endpoint. The node's `INPUT_TYPES` reads the NSFW setting to filter adult content styles and moods from its dropdowns. All generation settings (style, archetype, model, temperature, etc.) are exposed directly as node inputs.

**Initialization:** On startup, `library_manager.migrate_if_needed()` creates the database schema and loads the active library from the `libraries` table. No libraries exist on fresh install — the user creates their first library via the sidebar, which auto-activates it.

---

## Analysis Pipeline

Entry point: `POST /prompt808/api/analyze` (`server/routes.py` → `server/core/analyzer.py`)

### 1. Image Upload & Validation

The user uploads an image via multipart form data. The server:

- Validates the file extension against supported formats (JPG, PNG, WebP, BMP, TIFF, HEIC)
- Saves the upload to a temporary file in the active library's `thumbnails/` directory
- Computes an MD5 content hash of the raw bytes (used for dedup and thumbnail naming)
- **Downscales large images** — Before vision model inference, images larger than 1536px on their longest side are resized down (Lanczos). The Qwen VL processor slices images into 28×28 patches, so 4K+ resolution images generate massive token counts and peg the CPU for minutes. 1536px is sufficient for element extraction while keeping preprocessing under a few seconds.

### 2. Photo-Level Deduplication (CLIP)

**Module:** `server/core/image_embeddings.py`
**Model:** `openai/clip-vit-base-patch32` (~350 MB, 512-dim embeddings)

Before running the expensive QwenVL inference, the system checks if the image is a near-duplicate of a previously analyzed image in the same library:

1. **Exact hash match** — If the MD5 content hash exists in the image embeddings cache, it's an identical file. Rejected immediately.
2. **Semantic similarity** — The image is embedded via CLIP's image encoder into a normalized 512-dimensional vector. This vector is compared (cosine similarity) against all cached photo embeddings.
3. **Threshold check** — If similarity >= 0.95, the image is rejected as a duplicate.

The cache is stored in the `image_embeddings_cache` table as BLOB data (512 floats), keyed by content hash and scoped by library. Photos are registered in the cache only after successful analysis.

Deduplication is per-library. If the user passes `force=true`, the dedup check is skipped.

### 3. Medium Detection

**Module:** `server/core/analyzer.py` (`detect_medium()`)

A dedicated QwenVL prompt runs first on every image to identify its artistic medium:

```json
{
    "is_photograph": true/false,
    "medium": "pen and ink line art",
    "medium_tags": ["ink", "line_art", "crosshatching"],
    "technique_notes": "fine ink lines with detailed cross-hatching..."
}
```

The model distinguishes between:

- **Traditional media:** watercolor, oil paint, ink, charcoal, pencil, pastel, gouache
- **Digital:** digital painting, vector art, pixel art, 3D render
- **Photography:** film, digital photograph, infrared, long exposure
- **Mixed:** collage, mixed media

This detection determines which extraction pipeline runs next.

### 4. QwenVL Element Extraction (Dual Pipeline)

**Module:** `server/core/analyzer.py`
**Model:** Any QwenVL variant from `models.json`

Based on the medium detection result, one of two extraction prompts is used:

#### Photography pipeline (`is_photograph=true`)

Uses the standard extraction prompt with photography-specific categories:

**Tier 1 — Mandatory (exactly 6 elements, one per category):**

| Category      | What it captures                   |
| ------------- | ---------------------------------- |
| `environment` | Physical setting/location          |
| `lighting`    | Light quality, direction, source   |
| `camera`      | Lens choice, angle, DOF, technique |
| `palette`     | Dominant colors and tonal quality  |
| `composition` | Framing, arrangement, visual flow  |
| `mood`        | Emotional tone, atmosphere         |

**Tier 2 — Subject-specific (3-8 additional elements):**

Categories relevant to the specific image:

- Portrait: `pose`, `expression`, `hair`, `clothing`, `skin`, `accessories`
- Nude/Erotic: `pose`, `body`, `nudity_level`, `clothing_state`, `intimate_detail`, `skin`, `expression`, `body_features`
- Landscape: `terrain`, `sky`, `weather`, `texture`
- Other: any categories the model finds relevant

When the subject is nude or erotic, the extraction prompt instructs the vision model to describe body details, nudity level, positioning, and anatomical features with the same precision used for other photographic elements.

#### Native pipeline (`is_photograph=false`)

Uses a medium-aware extraction prompt where **`technique`** replaces **`camera`**:

**Tier 1 — Mandatory (exactly 6 elements):**

| Category      | What it captures                           |
| ------------- | ------------------------------------------ |
| `environment` | Setting/background/scene                   |
| `lighting`    | Light quality, direction, shadows          |
| `technique`   | Artistic technique, tools, rendering style |
| `palette`     | Dominant colors, color relationships       |
| `composition` | Arrangement, framing, visual flow          |
| `mood`        | Emotional tone, atmosphere                 |

The technique category captures medium-specific details: "fine ink lines with detailed cross-hatching and stippling for texture", "wet-on-wet watercolor washes with controlled color bleeding", "cel-shaded digital art with clean outlines".

Both pipelines produce the same element structure:

```json
{
  "category": "technique",
  "desc": "fine ink lines with detailed cross-hatching and stippling for texture",
  "tags": ["ink", "crosshatching", "stippling", "line_art"],
  "attributes": {}
}
```

The detected medium is attached to the analysis result and stored on each element as metadata.

**FP8 fallback:** Vision models with non-standard tensor dimensions may fail FP8 block quantization. The loader catches this and automatically falls back to FP16.

### 5. JSON Parsing & Retry Strategy

**Module:** `server/core/json_parser.py`

LLM output is parsed through layered strategies:

1. **Strip thinking tokens** — Qwen3 Thinking models wrap reasoning in `<think>...</think>` tags. These are removed.
2. **Strip markdown fences** — Remove ` ```json ... ``` ` wrappers.
3. **JSON object extraction** — Find the outermost `{...}` and parse it.
4. **JSON array extraction** — Find the outermost `[...]` and parse it.
5. **Regex fallback** — Extract `"type": "...", "desc": "..."` pairs from malformed output.

If extraction returns no elements, the system retries up to 3 times with progressively stricter prompts.

### 6. Element Normalization

**Function:** `analyzer._normalize_elements()`

Raw extracted elements are validated and normalized:

1. **Category normalization** — Lowercased, spaces replaced with underscores. Aliases are applied:
   
   | Raw category                                           | Canonical      |
   | ------------------------------------------------------ | -------------- |
   | `background`, `setting`, `location`                    | `environment`  |
   | `light`                                                | `lighting`     |
   | `color`, `colors`, `colour`, `color_palette`, `tone`   | `palette`      |
   | `framing`                                              | `composition`  |
   | `emotion`, `atmosphere`                                | `mood`         |
   | `attire`, `outfit`, `garment`                          | `clothing`     |
   | `hairstyle`, `hair_style`                              | `hair`         |
   | `facial_expression`, `face`                            | `expression`   |
   | `tattoo`                                               | `tattoos`      |
   | `body_features`, `body_detail`, `anatomy`, `genitalia` | `body`         |
   | `jewelry`, `jewellery`                                 | `accessories`  |
   | `nudity`, `nude_level`     | `nudity_level` |

2. **Subject-type collision skip** — If a category matches the subject type (e.g., `portrait` as both), the element is skipped.

3. **Tag format normalization** — Each tag is lowercased, non-alphanumeric characters replaced with underscores, multiple underscores collapsed. Example: `"Golden Hour"` → `"golden_hour"`.

4. **Metadata attachment** — Each element receives `source_photo`, `subject_type`, `added` (today's date), and `medium` (if non-photographic).

### 7. Description Deduplication

**Module:** `server/core/embeddings.py`
**Model:** `sentence-transformers/all-MiniLM-L6-v2` (~80 MB, 384-dim embeddings)

Before committing each element, its description is checked against all existing descriptions in the library:

1. The new description is embedded via sentence-transformer into a 384-dimensional vector.
2. Cosine similarity is computed against all existing description embeddings.
3. If similarity >= 0.90, the element is rejected as a duplicate.

Existing description embeddings are pre-computed once before the loop and incrementally updated as new elements are added, avoiding O(n^2) cost.

### 8. Tag Normalization

**Module:** `server/core/embeddings.py`

Each element's tags are checked against the existing vocabulary:

1. All new tags and all existing canonical tags are batch-embedded.
2. For each new tag, cosine similarity is computed against all existing tags.
3. If similarity >= 0.85, the new tag is mapped to the existing canonical form.
4. The mapping is recorded in the vocabulary store.

This prevents vocabulary fragmentation — semantically identical tags are consolidated.

### 9. Store Commitment

After dedup and normalization:

1. Each surviving element receives a unique ID: `{category}_{first_3_desc_words}` with numeric suffix if needed.
2. Elements are batch-inserted into the `elements` table (scoped by `library_id`).
3. New tags are registered in the `vocabulary` table.
4. The prompt cache is invalidated.

### 10. Archetype Regeneration

After committing elements, archetypes are regenerated from the full library. See [Archetype Generation](#archetype-generation) below.

### 11. Style Profile Update

**Module:** `server/core/style_profile.py`

The analysis result updates the per-genre style profile. See [Style Profiles](#style-profiles) below.

### Analysis Flow Summary

```
Image Upload (drag-and-drop, clipboard paste, browser image drop, file picker)
    |
    v
Validate format --> reject if unsupported
    |
    v
MD5 hash --> exact match in library cache? --> reject as duplicate
    |
    v
CLIP embed --> similarity > 0.95 in library? --> reject as duplicate
    |
    v
Medium Detection (QwenVL) --> is_photograph? / medium type
    |
    ├── photograph=true ──> Photography Extraction (camera, lens, DOF)
    │                        Tier 1: environment, lighting, camera, palette, composition, mood
    │
    └── photograph=false ──> Native Extraction (technique, rendering, style)
                              Tier 1: environment, lighting, technique, palette, composition, mood
    |
    v
JSON parsing (layered strategies, up to 3 retries)
    |
    v
Element normalization (categories, tags, aliases)
    |
    v
For each element:
    |-- Embed description
    |-- Compare against existing (cosine sim >= 0.90?) --> reject duplicate
    |-- Normalize tags against vocabulary (cosine sim >= 0.85?)
    |-- Generate unique ID
    |-- Commit to store
    |
    v
Register photo in CLIP cache
    |
    v
Regenerate archetypes (photo-level clustering)
    |
    v
Update style profile (per-genre accumulation)
    |
    v
Invalidate prompt cache
```

---

## Archetype Generation

**Module:** `server/core/archetypes.py`

Archetypes are auto-generated clusters of photos that share similar scene characteristics. They serve as coherent element pools for prompt generation — selecting "Studio Side Light" ensures you get studio-appropriate lighting, environment, and mood elements together.

### Photo Grouping

Elements are grouped by their source photo (using the `thumbnail` field, falling back to `source_photo`). This produces a mapping of `{photo_key: [elements]}`.

If there are fewer than 4 distinct photos, a single flat "All Elements" archetype is created instead of clustering.

### Feature Embedding

For each photo group, a focused text profile is created from only the **environment** and **lighting** descriptions — the features that best distinguish scene types. Subject-specific categories like pose, hair, and clothing are excluded because they would prevent meaningful scene-type separation.

These texts are embedded via sentence-transformer into 384-dim vectors.

### Agglomerative Clustering

A cosine distance matrix is computed from the photo embeddings, then **Agglomerative Clustering** with average linkage finds the optimal number of clusters:

1. The algorithm tries cluster counts from 3 to sqrt(n_photos), capped at 12.
2. For each count, it computes the **silhouette score**.
3. The cluster count with the highest silhouette score is selected.

### Cluster Merging

Clusters with fewer than 2 photos are merged into their nearest neighbor. After merging, cluster labels are renumbered to be contiguous.

### Archetype Building

For each cluster, an archetype dict is constructed:

```json
{
  "id": "studio_side_light_0",
  "name": "Studio Side Light",
  "compatible": {
    "environment_tags": ["studio", "minimalist"],
    "lighting_tags": ["side_light", "dramatic"],
    ...
  },
  "element_ids": ["environment_dark_studio", "lighting_hard_flash", ...],
  "category_weights": {
    "environment": 0.98,
    "lighting": 0.63,
    "tattoos": 0.034,
    ...
  },
  "photo_count": 44,
  "negative_hints": ["landscape", "wildlife"],
  "generated": "2026-02-26"
}
```

- **`compatible`** — Aggregated tag sets per category, used for element filtering during generation
- **`element_ids`** — Direct membership list of all elements in the cluster
- **`category_weights`** — Fraction of photos with each category
- **`negative_hints`** — Terms to avoid, derived from absent categories

### Naming

Archetypes are named using **TF-IDF distinctive tag scoring** across all sibling archetypes in the same library. For each archetype, every tag from naming-eligible categories (mood, environment, lighting, palette) is scored:

```
score = TF × IDF × category_priority
```

- **TF** — Tag frequency within this cluster (how representative is the tag)
- **IDF** — Inverse document frequency across sibling clusters (how unique is the tag to this cluster)
- **Category priority** — Mood (1.5) > environment (1.3) > lighting (1.0) > palette (0.8)

Up to 3 highest-scoring tags from **different** categories are selected, with a word-overlap check to prevent stuttering (e.g., "Studio Studio Light"). Tags are sorted by natural word order (environment → mood → palette → lighting) and joined into the name.

A disambiguation pass resolves any remaining duplicate names within a library by appending a distinguishing tag unique to each cluster. If no unique tag exists, a numeric suffix (I, II, III) is used as a last resort.

Generic tags (e.g., `soft_light`, `minimalist`, `indoor`, `skin_tone`) are excluded from naming via a skip list.

Optional LLM naming can override this — when enabled, the distinctive tags are passed as context to the LLM, which composes a 3-5 word evocative name.

---

## Prompt Generation Pipeline

Entry point: `bridge_node.py` (Generate Prompt node) or `library_select_node.py` (Select Libraries node)
Core logic: `server/core/generator.py`

### 1. Cache Lookup

**Module:** `server/core/prompt_cache.py`

A deterministic cache key is computed from all generation inputs: `SHA256(seed, archetype_id, style, mood, model_name, quantization, library_version)`. If a cache hit is found, the prompt is returned instantly.

The cache is invalidated whenever the element library changes.

### 2. Archetype Resolution & Element Filtering

If a specific archetype is selected (not "Any"), a **favored set** of element object identities is computed by `_get_archetype_element_ids()`:

1. **Direct ID matching** — Elements whose IDs are in the archetype's `element_ids` list are included, but only if they belong to the **same library** as the archetype. This prevents cross-library ID collisions when multiple libraries are merged into a single pool.
2. **Tag matching** — Elements matching the archetype's `compatible` tags are included cross-library. A minimum overlap threshold prevents overly broad matches: `max(2, ⌊len(arch_tags) × 0.2⌋)` matching tags required for tag lists of 5+ tags, or 1 tag for smaller lists.
3. **Negative hint exclusion** — Elements with tags matching `negative_hints` are excluded from both matching methods.

The favored set is used for **weighting**, not filtering — an empty favored set means uniform random selection across all elements.

If "Any" is selected, all library elements are used with no favored set.

**Native style filtering:** Native means "truthful to source medium." The Native pool includes elements matching any of three rules:

1. `extraction_type="native"` — non-photo art's native description
2. `is_photograph=true` — for photographs, the photo-pass extraction IS the native description
3. `extraction_type` is NULL — curated/universal elements (e.g. poses from `.p808` imports) that are medium-agnostic

For non-photo art (3D renders, paintings), only their `extraction_type="native"` elements are included (their photo-pass descriptions are excluded since those misrepresent the medium). Camera-category elements are always excluded from the Native pool since lens/f-stop specs don't apply when the pool may contain non-photography art. If no native elements exist, all non-camera elements are used as fallback. When a photography style is selected, only non-native elements (`extraction_type != "native"`) are used.

### 3. Frequency-Weighted Element Selection

**Function:** `generator._select_elements()`

Elements are grouped by category. Selection uses a seeded RNG for deterministic output:

**Tier 1 categories** (environment, lighting, camera, technique, palette, composition, mood) are **always** included — one element per category.

**Tier 2 categories** (pose, hair, expression, clothing, tattoos, accessories, etc.) are included **probabilistically**, based on their photo fraction:

```
inclusion_probability = (unique photos with this category) / (total unique photos)
```

This means:

- `pose` present in 93% of photos → included in ~93% of prompts
- `tattoos` present in 3% of photos → included in ~3% of prompts

Within each included category, element selection is controlled by `_pick_element()`, which handles archetype weighting and library balancing.

#### Archetype Influence

**Node input:** Archetype Influence (0–100%, default 70%) on the Generate Prompt node.

When an archetype is selected, the favored set biases element selection via **adaptive per-category weighting**. The formula auto-scales the weight based on the ratio of favored to non-favored elements in each category:

```
weight = target_p × n_other / (n_favored × (1 - target_p))
weight = max(1.0, weight)   # never downweight favored elements
```

- **Sparse categories** (few favored elements in a large pool) get high weights to reach the target probability
- **Broad categories** (many favored elements) get weight clamped to 1.0 (no boost needed)
- At **100% influence**, only favored elements are selected; categories with no favored elements are skipped
- At **0% influence**, selection is uniform across all elements

**Tier 2 floor behavior:** Archetype influence also acts as a minimum inclusion probability for Tier 2 categories that the archetype covers. For example, a pose-only archetype at 80% guarantees the `poses` category is included in at least 80% of prompts, even if poses appear in only 3% of analyzed photos. The formula: `weight = max(photo_fraction, archetype_influence)` for favored categories.

#### Balance Libraries

**Setting:** Balance Libraries toggle in Settings > Prompt808 > General (default: on).

When enabled with multiple libraries in play, element selection uses a two-stage process to give each library equal representation regardless of size:

1. Pick a library uniformly at random
2. Pick an element from that library

Without this toggle, a library with 34 elements would be nearly invisible alongside one with 5,000.

**Interaction with Archetype Influence:** When both features are active, the archetype decides first — with probability equal to the archetype influence, a favored element is chosen directly. Only the remaining non-favored picks (the `1 - influence` portion) use library balancing. This ensures archetypes maintain coherence while balancing applies to the creative-freedom portion of the selection.

### 4. Prompt Composition

The selected elements are composed via one of two paths:

#### LLM Composition (when a text model is selected)

The chosen elements are formatted as a bullet list and sent to the text model with:

- A **style instruction** (Photo-Architectural, Photo-Boudoir, Photo-Cinematic, Photo-Documentary, Photo-Erotica, Photo-Fashion, Photo-Fine Art, Photo-Portrait, Photo-Street, or Native) that sets the linguistic register
- A **mood modifier** (Dramatic, Elegant, Ethereal, Gritty, Melancholic, Mysterious, Provocative, Romantic, Sensual, Serene, or None) that biases the atmosphere. An "Any" option picks a random mood per generation
- An **enrichment level** that controls creative freedom
- A **style profile context** from the user's per-genre style data (if enough observations exist)
- A **fidelity rule** instructing the LLM to reproduce all element descriptions faithfully without euphemizing, censoring, or omitting anatomical, sexual, or explicit content

The LLM outputs a JSON response with `prompt` and `negative_prompt` fields.

**Enrichment levels control the LLM's behavior:**

| Level      | Temperature offset | Effective (base 0.7) | Behavior                                                  |
| ---------- | ------------------ | -------------------- | --------------------------------------------------------- |
| Baseline   | +0.0               | 0.7                  | Rewrites descriptions as vivid phrases, preserves meaning |
| Vivid      | +0.1               | 0.8                  | Adds sensory/textural detail, stays close to original     |
| Expressive | +0.15              | 0.85                 | Reinterprets with mood imagery, allows artistic license   |
| Poetic     | +0.2               | 0.9                  | Uses metaphor, art references, heightened language        |
| Lyrical    | +0.3               | 1.0                  | Generates original phrases inspired by tags only          |
| Freeform   | +0.15              | 0.85                 | Director's instructions grounded in scene context         |

#### Style-Differentiated Simple Composition (no text model / fallback)

Without an LLM, prompts are composed using four style-specific configuration layers that produce distinctly different output per style:

**1. Style prefix** — Sets the opening tone:

- Boudoir: `"intimate boudoir photography,"`
- Cinematic: `"cinematic film still,"`
- Erotica: `"explicit erotic photography,"`
- Fine Art: `"fine art photography, gallery-quality,"`
- Fashion: `"high-fashion editorial photography,"`
- Documentary: `"documentary photography, candid,"`
- Native: Uses the detected medium (e.g., `"watercolor painting,"`)

**2. Category ordering** — Elements are sorted by style priority. Each style leads with its most characteristic categories:

- Boudoir: subject → body → pose → lighting → skin → nudity_level → ...
- Cinematic: camera → lighting → composition → environment → ...
- Erotica: subject → body → intimate_detail → pose → nudity_level → ...
- Fine Art: lighting → palette → mood → composition → ...
- Fashion: clothing → subject → lighting → palette → ...
- Documentary: environment → subject → mood → lighting → ...

**3. Category boosters** — Per-category prefix/suffix tuples add style flavor:

- Boudoir skin elements get `" tactile detail"` suffix
- Cinematic camera elements get `"shot on "` prefix
- Erotica body elements get `" explicit detail"` suffix
- Fine Art lighting gets `", sculpting form"` suffix
- Fashion clothing gets `", editorial styling"` suffix
- Documentary environment gets `", found location"` suffix

**4. Style connectors** — Bridging phrases between elements (cycled):

- Boudoir: `", draped in "`, `", revealing "`, `", bathed in "`
- Cinematic: `", captured with "`, `", revealing "`, `", framed by "`
- Erotica: `", exposing "`, `", positioned in "`, `", against "`
- Fine Art: `", dissolving into "`, `", rendered in "`, `", bathed in "`
- Fashion: `", styled with "`, `", against "`, `", accentuated by "`
- Documentary: `", amid "`, `", witnessing "`, `", set against "`

**5. Quality suffix** — Closing phrases per style:

- Boudoir: `"skin detail, intimate atmosphere, sensual lighting, shallow depth of field"`
- Cinematic: `"cinematic depth of field, anamorphic lens quality, film grain"`
- Erotica: `"anatomical precision, explicit detail, professional lighting, sharp focus on subject"`
- Fine Art: `"gallery-quality print, contemplative mood, fine detail"`
- Fashion: `"editorial finish, magazine-quality"`
- Documentary: `"raw authenticity, decisive moment, natural imperfection"`

### 5. Negative Prompt Assembly

Negative prompts combine three sources:

1. **LLM negatives** — If the LLM generated a `negative_prompt` field.
2. **Style base negatives** — Each style has a safety-net set of terms. The Native style uses minimal negatives to avoid conflicting with the detected medium. Boudoir and Erotica styles include anatomical quality terms (e.g., "bad anatomy", "deformed genitalia").
3. **Archetype anti-affinity hints** — Terms derived from categories NOT present in the archetype.

All terms are deduplicated and sorted alphabetically.

### 6. Cache Storage

The generated prompt + negative prompt are stored in the cache keyed by the deterministic hash.

### Generation Flow Summary

```
seed + archetype + style + mood + model + enrichment
    |
    v
Resolve library (X-Library header or persisted active)
    |
    v
Cache lookup (SHA256 of all inputs) --> hit? return cached
    |
    v
Load all elements from library store
    |
    v
Build favored set (library-scoped ID match + tag match - negative hints)
    |
    v
Filter by style (Native → non-photo elements; Photo styles → photo elements)
    |
    v
Group by category, compute photo fractions
    |
    v
Tier 1 categories: always select one element
Tier 2 categories: include with P = max(photo_fraction, archetype_influence) for favored categories
    |
    v
Per-category element selection (_pick_element):
    |-- Archetype influence roll → favored element (if archetype active)
    |-- Otherwise: balanced library pick (if Balance Libraries on)
    |              or uniform pool pick (default)
    |
    v
LLM available?
    |-- Yes: compose via LLM with style + mood + enrichment + style profile
    |-- No:  style-differentiated composition (ordering + boosters + connectors + quality suffix)
    |
    v
Build negative prompt (LLM + style base + archetype hints)
    |
    v
Cache result
    |
    v
Broadcast to sidebar via WebSocket
    |
    v
Return prompt + negative + metadata
```

---

## Multi-Library Generation

**Modules:** `bridge_node.py`, `library_select_node.py`, `js/prompt808_library_select.js`

Generation can draw elements from multiple libraries simultaneously, creating a virtual merged pool that multiplies the combinatorial space.

### Activation

There are two ways to enable multi-library generation:

1. **"All" option** — Select "All" in the Generate Prompt node's library dropdown. Resolves to all libraries via `library_manager.list_libraries()`.
2. **Select Libraries node** — A dedicated node (`library_select_node.py`) with dynamic library slots. Each slot has an on/off toggle and a library dropdown. Outputs a `P808_LIBRARIES` list that connects to the Generate Prompt node's `libraries` input. When connected, the library dropdown on the Generate node is automatically hidden.

### Library Resolution

`bridge_node._resolve_libraries()` determines which libraries to use, with this priority:

1. **Library Select node connected** — uses the list from the `libraries` input, validates each name exists
2. **"All" selected** — resolves to all library names
3. **Single library** — uses the dropdown value (default behavior)

Invalid library names are logged and skipped. If all names are invalid, the node returns an error status.

### Data Merging

`bridge_node._gather_multi_library_data()` iterates the resolved library names. For each library, it sets the `_request_library` context variable, then collects:

- **Elements** via `elements.get_all()`
- **Archetypes** via `archetypes.get_all()`
- **Style contexts** via `style_profile.get_style_context(genre)`

Each element is tagged with `_library` (its source library name) and each archetype with `_library`. All elements and archetypes are concatenated into flat lists. Style contexts are collected per genre.

### Wrapper Stores

Three duck-typed wrapper classes present the merged data with the same interface as the real store modules, so `generator.generate_prompt()` works unchanged:

| Wrapper | Interface | Purpose |
|---------|-----------|---------|
| `_MergedElementStore` | `get_all()`, `count()`, `get_library_version()` | Merged element pool |
| `_MergedArchetypeStore` | `get_all()`, `get_by_id()`, `get_by_name()`, `get_names()` | Merged archetype pool |
| `_MergedStyleProfile` | `get_style_context(genre)` | Picks one style context at random from all libraries using the seed |

### Cache Key

A composite version string `"multi:LibA=5_10|LibB=3_8"` (encoding element count and archetype count per library) is used as the library version for cache key computation, preventing false cache hits between single-library and multi-library generations.

---

## Style Profiles

**Module:** `server/core/style_profile.py`

Style profiles track the user's aesthetic patterns per genre (e.g., portrait, landscape, street). They are updated after each analysis and used as context during LLM prompt composition.

### How Profiles Work

1. Elements are grouped by style-relevant categories: lighting, camera, technique, palette, composition, mood, environment.
2. Tag frequencies within each category are normalized to weights.
3. Weights are merged into the genre profile using a running average — all photos contribute equally.
4. After enough observations (>= 3), the profile generates natural language style context (e.g., "Style profile: Lighting preference: side_light, golden_hour") that is included in future prompt generation.

### Rebuild

Profiles can be rebuilt from current library elements at any time via the "Recalculate" button in the Style tab or the `POST /prompt808/api/style/profiles/reset` endpoint. This recalculates all profiles from scratch using the current element library, ensuring they stay accurate after photo or element deletions.

Rebuild is also triggered automatically when:

- A photo is deleted
- An element is deleted
- An element's tags are edited

---

## Library Export & Import

**Modules:** `server/plugins/export/export.py`, `server/plugins/export/_import.py`

Libraries can be exported as `.p808` files (zip archives) for backup or sharing, and imported to create new libraries.

### Export

`POST /prompt808/api/library/export` creates a `.p808` zip file containing:

| File | Contents |
|------|----------|
| `metadata.json` | Library name, export timestamp, format version, element/archetype/vocabulary counts |
| `elements.json` | All elements with JSON columns parsed to native objects |
| `archetypes.json` | All archetypes |
| `vocabulary.json` | Tag vocabulary with canonical forms and aliases |
| `style_profiles.json` | Per-genre style profiles |
| `thumbnails/` | Photo thumbnail images (optional) |

### Import

`POST /prompt808/api/library/import` accepts a multipart upload of one or more `.p808` files. For each file:

1. **Name deduplication** — If a library with the same name exists, appends `(2)`, `(3)`, etc. (up to 50-char limit)
2. **Format validation** — Checks format version (currently `v1`) and required metadata fields
3. **Column whitelist** — Only accepts known columns per table, rejecting unexpected data
4. **Row validation** — Ensures required fields are present and JSON columns are well-formed
5. **Referential integrity** — Removes dangling `element_ids` references in archetypes
6. **Security** — Rejects thumbnail paths with directory traversal attempts
7. **Conflict resolution** — Uses `INSERT OR REPLACE` for idempotent re-imports

The sidebar import picker supports selecting multiple `.p808` files at once for batch import.

---

## Models & Embedding Layers

Prompt808 uses three distinct model types, each loaded lazily as singletons:

| Model                           | Purpose                               | Size      | When loaded                     |
| ------------------------------- | ------------------------------------- | --------- | ------------------------------- |
| **QwenVL** (vision)             | Medium detection + element extraction | 7.5-28 GB | During analysis                 |
| **CLIP** (image)                | Photo-level dedup                     | ~350 MB   | During analysis (before QwenVL) |
| **Sentence-Transformer** (text) | Tag normalization + description dedup | ~80 MB    | During analysis (commit phase)  |
| **Qwen3/2.5** (text LLM)        | Prompt composition + enrichment       | 0.7-28 GB | During generation (if selected) |

Analysis models (vision, CLIP, sentence-transformer) can be unloaded via `POST /prompt808/api/analyze/cleanup`. The text LLM is managed by the node's `keep_model_loaded` input — when enabled, the model is offloaded to CPU RAM after generation; when disabled, it is fully unloaded.

### Memory Management

- Vision and CLIP models are unloaded together after analysis cleanup
- Text models support quantization (FP16, FP8, 8-bit, 4-bit)
- Models that use BitsAndBytes quantization (4-bit, 8-bit) cannot be offloaded to CPU RAM — they are fully unloaded
- FP16/FP8 models can be offloaded to CPU RAM when GPU VRAM is needed
- Vision models with incompatible tensor dimensions automatically fall back from FP8 to FP16
- **Pre-analysis VRAM reclamation:** Before loading analysis models, `routes.py` calls `comfy.model_management.unload_all_models()` synchronously on the event-loop thread to free VRAM occupied by ComfyUI workflow models. This call must not be dispatched to a thread pool — ComfyUI's model patching internals are not thread-safe (see Event loop safety above)
- **Deep cleanup:** Every model unload (`embeddings.unload_model()`, `image_embeddings.unload_model()`, `model_manager.unload_model()`) calls `gc.collect()` followed by `comfy.model_management.soft_empty_cache()` (or `torch.cuda.empty_cache()` as fallback) to release VRAM pages held by PyTorch's caching allocator. The analysis cleanup endpoint performs a final combined cleanup pass after all models are freed

---

## Data Flow Diagram

```
                            ANALYSIS
                            ========

     Image ──> Validate ──> CLIP Dedup ──> Resize (≤1536px) ──> Medium Detection ──> is_photograph?
                                |               |                      |
                           image_embeddings QwenVL prompt          ┌────┴────┐
                           _cache table                            v         v
                           (SQLite BLOB)                    Photography   Native
                                                            Pipeline     Pipeline
                                                            (camera)    (technique)
                                                                 |         |
                                                                 └────┬────┘
                                                                      v
                                                            JSON Parsing
                                                            (layered strategies)
                                                                      |
                                                                      v
                                                           Normalize Elements
                                                           (categories, tags,
                                                            aliases, medium)
                                                                      |
                                                                      v
                                                     ┌── Dedup Descriptions ──┐
                                                     │   (cosine sim >= 0.90) │
                                                     v                        v
                                                  REJECT              Normalize Tags
                                                                 (cosine sim >= 0.85)
                                                                         |
                                                                         v
                                                                 Commit Elements
                                                              ┌──────┴──────┐
                                                              v              v
                                                       elements        vocabulary
                                                       table           table
                                                              |
                                                              v
                                                     Regenerate Archetypes
                                                     (photo-level clustering)
                                                              |
                                                              v
                                                       archetypes table
                                                              |
                                                              v
                                                     Update Style Profile
                                                              |
                                                              v
                                                       style_profiles table


                           GENERATION
                           ==========

     seed + params ──> Resolve Library ──> Cache Hit? ──> YES ──> Return cached
                                               |
                                               NO
                                               |
                                               v
                                        Load all elements
                                               |
                                               v
                                        Filter by archetype
                                        + Filter by style (photo vs native)
                                               |
                                               v
                                        Frequency-weighted selection
                                        ┌─────────────────────────────┐
                                        │ Tier 1: always pick 1/cat   │
                                        │ Tier 2: P = photo_fraction  │
                                        └─────────────────────────────┘
                                               |
                                               v
                                        ┌──────┴──────┐
                                        v              v
                                  LLM Compose    Style-Differentiated
                                  (enrichment)   Simple Compose
                                  (style +       (ordering + boosters
                                   profile)       + connectors + suffix)
                                        |              |
                                        └──────┬──────┘
                                               v
                                        Build negative prompt
                                        (LLM + style base + anti-affinity)
                                               |
                                               v
                                        Cache result
                                               |
                                               v
                                        WebSocket broadcast ──> Sidebar UI
                                               |
                                               v
                                        Return prompt + negative + metadata
```

---

## Architecture

```
Prompt808/                        # ComfyUI custom node
├── __init__.py                   # Node registration, route registration, library initialization
├── bridge_node.py                # Generate Prompt node (all settings as inputs)
├── library_select_node.py        # Select Libraries node (multi-library selection)
├── pyproject.toml                # ComfyUI node metadata
├── models.json                   # Model registry (text + vision models)
├── requirements.txt              # Python dependencies
├── js/                           # Frontend (vanilla JS, loaded by ComfyUI)
│   ├── prompt808.js              # Sidebar panel registration, tab navigation, library switcher
│   ├── prompt808.css             # All styles
│   ├── prompt808_bridge.js       # Generate node UI (dropdown refresh, library widget auto-hide)
│   ├── prompt808_library_select.js  # Select Libraries node UI (dynamic slots, toggle widgets)
│   ├── api.js                    # API client (X-Library header, all endpoints)
│   ├── utils.js                  # DOM helpers ($el, toast, spinner, helpButton)
│   ├── analyze.js                # Analyze tab (image upload, vision model selection)
│   ├── library.js                # Library tab (element browser, edit, delete)
│   ├── photos.js                 # Photos tab (thumbnail grid, photo management)
│   ├── archetypes.js             # Archetypes tab (cluster viewer, regenerate)
│   └── style.js                  # Style tab (per-genre style profiles, recalculate)
├── server/                       # Backend (aiohttp routes on ComfyUI's PromptServer)
│   ├── routes.py                 # Route registration, delegates to api/ handlers
│   ├── app.py                    # FastAPI app for testing
│   ├── api/                      # Request handlers (split by domain)
│   │   ├── analysis.py           # Analysis endpoints (upload, options, cleanup)
│   │   ├── libraries.py          # Library CRUD endpoints
│   │   ├── library.py            # Library data endpoints (elements, photos, archetypes)
│   │   └── style.py              # Style profile endpoints
│   ├── core/                     # Business logic
│   │   ├── database.py           # SQLite singleton, schema, WAL mode, write lock
│   │   ├── analyzer.py           # Medium detection + QwenVL extraction (dual pipeline)
│   │   ├── generator.py          # Prompt composition (LLM + style-differentiated simple)
│   │   ├── model_manager.py      # Text model loading, inference, lifecycle
│   │   ├── library_manager.py    # Multi-library CRUD, path resolution, initialization
│   │   ├── archetypes.py         # Photo-level agglomerative clustering
│   │   ├── embeddings.py         # Tag normalization + description dedup (sentence-transformers)
│   │   ├── image_embeddings.py   # Photo dedup via CLIP embeddings
│   │   ├── coherence.py          # Color harmony, tag filtering, seeded selection
│   │   ├── style_profile.py      # Per-genre style learning + rebuild from library
│   │   ├── prompt_cache.py       # Deterministic prompt caching
│   │   └── json_parser.py        # Robust LLM JSON extraction
│   ├── store/                    # SQLite-backed persistence
│   │   ├── elements.py           # Element CRUD
│   │   ├── archetypes.py         # Archetype CRUD
│   │   └── vocabulary.py         # Tag vocabulary management
│   └── plugins/                  # Pluggable extensions
│       └── export/               # Library export/import
│           ├── export.py         # Export library to .p808 zip file
│           └── _import.py        # Import .p808 zip into new library
└── user_data/                    # Per-user persistent data (auto-created)
    ├── prompt808.db              # SQLite database (all libraries, elements, caches)
    └── libraries/                # Per-library disk data
        └── <library_name>/
            └── thumbnails/       # Photo thumbnails (images stay on disk)
```

---

## API Reference

All endpoints are registered on ComfyUI's PromptServer under `/prompt808/api/*`. All endpoints accept an `X-Library` header to scope the request to a specific library.

### Analysis

| Method | Endpoint                         | Description                                                                                                                              |
| ------ | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| POST   | `/prompt808/api/analyze`         | Upload an image for analysis (SSE streaming). Multipart form: `image`, `vision_model`, `quantization`, `device`, `attention_mode`, `max_tokens`, `force`. Returns SSE `progress` events with phase updates and a final `result` event |
| GET    | `/prompt808/api/analyze/options` | Available vision models and quantization options                                                                                         |
| POST   | `/prompt808/api/analyze/cleanup` | Unload analysis models (vision, CLIP, sentence-transformer) to free VRAM                                                                 |

### Generation

| Method | Endpoint                           | Description                                             |
| ------ | ---------------------------------- | ------------------------------------------------------- |
| GET    | `/prompt808/api/generate/options`  | Available styles, moods, archetypes, and text models    |

Generation is handled directly by the Generate Prompt node via `bridge_node.py`, not through HTTP endpoints.

### Libraries

| Method | Endpoint                          | Description                                            |
| ------ | --------------------------------- | ------------------------------------------------------ |
| GET    | `/prompt808/api/libraries`        | List all libraries with active flag and element counts |
| POST   | `/prompt808/api/libraries`        | Create new library. JSON body: `{"name": "..."}`       |
| PUT    | `/prompt808/api/libraries/active` | Switch active library. JSON body: `{"name": "..."}`    |
| PATCH  | `/prompt808/api/libraries/{name}` | Rename library. JSON body: `{"name": "new_name"}`      |
| DELETE | `/prompt808/api/libraries/{name}` | Delete library (refuses if last one)                   |

### Library Data

| Method | Endpoint                                             | Description                                                              |
| ------ | ---------------------------------------------------- | ------------------------------------------------------------------------ |
| GET    | `/prompt808/api/library/elements`                    | List elements. Query: `category`, `offset`, `limit`                      |
| GET    | `/prompt808/api/library/elements/{id}`               | Get single element                                                       |
| PATCH  | `/prompt808/api/library/elements/{id}`               | Update element (desc, tags, attributes)                                  |
| DELETE | `/prompt808/api/library/elements/{id}`               | Delete element (auto-regenerates archetypes and rebuilds style profiles) |
| GET    | `/prompt808/api/library/categories`                  | List categories with counts                                              |
| GET    | `/prompt808/api/library/archetypes`                  | List all archetypes                                                      |
| GET    | `/prompt808/api/library/archetypes/{id}`             | Get single archetype                                                     |
| DELETE | `/prompt808/api/library/archetypes/{id}`             | Delete archetype                                                         |
| POST   | `/prompt808/api/library/archetypes/regenerate`       | Force archetype regeneration (with LLM naming)                           |
| GET    | `/prompt808/api/library/stats`                       | Library statistics                                                       |
| GET    | `/prompt808/api/library/photos`                      | List analyzed photos with element counts                                 |
| GET    | `/prompt808/api/library/photos/{thumbnail}/elements` | Get elements for a specific photo                                        |
| DELETE | `/prompt808/api/library/photos/{thumbnail}`          | Delete photo + all associated elements, archetypes, and style profiles   |
| DELETE | `/prompt808/api/library/reset`                       | Reset all data in the active library                                     |

### Export & Import

| Method | Endpoint                          | Description                                                    |
| ------ | --------------------------------- | -------------------------------------------------------------- |
| POST   | `/prompt808/api/library/export`   | Export active library as `.p808` zip file (binary download)    |
| POST   | `/prompt808/api/library/import`   | Import `.p808` file (multipart upload, creates a new library)  |

### Style Profiles

| Method | Endpoint                                      | Description                                    |
| ------ | --------------------------------------------- | ---------------------------------------------- |
| GET    | `/prompt808/api/style/profiles`               | List all genre profiles with summary           |
| GET    | `/prompt808/api/style/profiles/{genre}`       | Get genre profile with context text            |
| POST   | `/prompt808/api/style/profiles/{genre}/reset` | Recalculate genre profile from current library |
| POST   | `/prompt808/api/style/profiles/reset`         | Recalculate all profiles from current library  |

### App Settings

| Method | Endpoint                  | Description                            |
| ------ | ------------------------- | -------------------------------------- |
| GET    | `/prompt808/api/settings` | Read app-wide settings (NSFW, debug, balance_libraries) |
| PUT    | `/prompt808/api/settings` | Save app-wide settings (partial merge)                                       |

### Health

| Method | Endpoint                | Description                                                          |
| ------ | ----------------------- | -------------------------------------------------------------------- |
| GET    | `/prompt808/api/health` | Server status + element/archetype/vocabulary counts + active library |
