"""QwenVL-powered photo analysis pipeline.

Analyzes reference photos to extract photographic elements with dynamic
categories. Uses a two-tier extraction:
  Tier 1: Universal attributes (environment, lighting, camera, palette, composition, mood)
  Tier 2: Subject-specific attributes (whatever is relevant to the photo)

For non-photographic images, a second "native" extraction runs with
medium-aware categories (technique instead of camera).

Elements are auto-committed to the store, tags are normalized against the
existing vocabulary, and duplicate descriptions are rejected.
"""

import json
import logging
import re
from datetime import date

import numpy as np

from . import embeddings as emb_module
from .json_parser import build_retry_prompt, parse_element_extraction, parse_llm_json

log = logging.getLogger("prompt808.analyzer")

# Maximum retries for extraction
MAX_RETRIES = 3

# Tag format guide included in extraction prompts
TAG_FORMAT_GUIDE = (
    "Tag format rules:\n"
    "- Lowercase, underscore-separated (e.g., golden_hour, shallow_dof)\n"
    "- No adjectives as standalone tags — combine with noun (e.g., dramatic_light not dramatic)\n"
    "- 2-5 tags per element, most specific first\n"
)

# Medium detection prompt — runs first on every image
MEDIUM_DETECTION_PROMPT = """Look at this image and identify its artistic medium and technique.

Respond with ONLY valid JSON:
{
    "is_photograph": true/false,
    "medium": "short description of the artistic medium (e.g., 'photograph', 'pen and ink line art', 'watercolor painting', 'digital illustration', '3D render', 'oil painting', 'anime illustration', 'pixel art')",
    "medium_tags": ["tag1", "tag2", "tag3"],
    "technique_notes": "brief description of the specific techniques visible (e.g., cross-hatching, wet-on-wet, cel shading, bokeh, etc.)"
}

Be specific about the medium. Do NOT default to 'photograph' or 'digital art' unless it truly is one. Distinguish between:
- Traditional media: watercolor, oil paint, ink, charcoal, pencil, pastel, gouache
- Digital: digital painting, vector art, pixel art, 3D render
- Photography: film, digital photograph, infrared, long exposure
- Mixed: collage, mixed media

Focus on WHAT the image IS, not what it depicts."""

# Extraction prompt for QwenVL (photography-focused, unchanged)
EXTRACTION_PROMPT = """Analyze this photograph and extract its photographic elements as structured JSON.

First, identify the subject type (e.g., landscape, portrait, automotive, street, food, macro, architecture, wildlife, fashion, nude, erotic, boudoir, still_life, or other).

Then extract elements in two tiers:

**Tier 1 — MANDATORY for every photo** (extract exactly one element for EACH of these 6 categories):
- environment: The physical setting/location (e.g., "sunlit forest clearing with dappled shade")
- lighting: Light quality, direction, source (e.g., "golden hour side light raking across surfaces")
- camera: Lens choice, angle, DOF, technique (e.g., "85mm portrait lens, shallow depth of field")
- palette: Dominant colors and tonal quality (e.g., "warm golden tones against cool blue shadows")
- composition: Framing, arrangement, visual flow (e.g., "centered subject with leading lines from foreground")
- mood: Emotional tone, atmosphere (e.g., "solitude and quiet contemplation")

**Tier 2 — Subject-specific** (add 3-8 elements with categories relevant to THIS photo):
- Portrait: pose, expression, hair, clothing, skin, accessories
- Nude/Erotic: pose, body, nudity_level, clothing_state, intimate_detail, skin, expression, body_features
- Landscape: terrain, sky, weather, season, vegetation, water_features
- Automotive: body_style, finish, angle, detail, road_surface
- Street: pedestrians, signage, atmosphere, time_of_day
- Food: presentation, texture, garnish, vessel, ingredient

When the subject is nude or erotic, describe body details, nudity level, positioning, and anatomical features with the same precision used for other photographic elements. Do not euphemize, omit, or soften explicit content.

{tag_format_guide}
Respond with ONLY valid JSON in this format:
{{
    "subject_type": "portrait",
    "elements": [
        {{"category": "environment", "desc": "sunlit studio with large windows and white walls", "tags": ["studio", "window_light", "white_walls", "bright"], "attributes": {{}}}},
        {{"category": "lighting", "desc": "soft directional window light creating gentle shadows on skin", "tags": ["window_light", "soft_light", "directional", "skin_highlight"], "attributes": {{}}}},
        {{"category": "camera", "desc": "85mm lens at f/2.8, shallow depth of field isolating subject", "tags": ["85mm", "shallow_dof", "portrait_lens", "selective_focus"], "attributes": {{}}}},
        {{"category": "palette", "desc": "warm skin tones against neutral grey background", "tags": ["warm_tones", "neutral_background", "skin_tone"], "attributes": {{}}}},
        {{"category": "composition", "desc": "centered subject with negative space on left side", "tags": ["centered", "negative_space", "balanced_frame"], "attributes": {{}}}},
        {{"category": "mood", "desc": "serene confidence with quiet introspection", "tags": ["serene", "confident", "introspective", "calm"], "attributes": {{}}}},
        {{"category": "pose", "desc": "seated with one hand resting on knee, head tilted slightly", "tags": ["seated", "hand_on_knee", "head_tilt", "relaxed_pose"], "attributes": {{}}}},
        {{"category": "expression", "desc": "soft gaze directed slightly off-camera with subtle smile", "tags": ["soft_gaze", "off_camera", "subtle_smile"], "attributes": {{}}}}
    ]
}}

You MUST include all 6 tier 1 categories (environment, lighting, camera, palette, composition, mood). Add 3-8 tier 2 elements on top. Total: 9-14 elements.
"""

# Native extraction prompt — for non-photographic images (technique instead of camera)
NATIVE_EXTRACTION_PROMPT = """Analyze this {medium} and extract its visual elements as structured JSON.

First, identify the subject type (e.g., landscape, portrait, automotive, street, food, macro, architecture, wildlife, fashion, nude, erotic, boudoir, still_life, or other).

Then extract elements in two tiers:

**Tier 1 — MANDATORY** (extract exactly one element for EACH of these 6 categories):
- environment: The setting or scene depicted (e.g., "moonlit forest clearing with tall pines")
- lighting: Light quality, direction, how it creates atmosphere (e.g., "warm golden light from upper left creating long shadows")
- technique: The artistic technique, tools, and rendering style (e.g., "fine ink lines with detailed cross-hatching and stippling for texture", "wet-on-wet watercolor washes with controlled color bleeding", "cel-shaded digital art with clean outlines")
- palette: Dominant colors, color relationships, and tonal quality (e.g., "muted earth tones with accents of deep crimson")
- composition: Framing, arrangement, visual flow (e.g., "centered subject with diagonal leading lines")
- mood: Emotional tone, atmosphere (e.g., "contemplative solitude with underlying warmth")

**Tier 2 — Subject-specific** (add 3-8 elements with categories relevant to THIS image):
- Portrait: pose, expression, hair, clothing, skin, accessories
- Nude/Erotic: pose, body, nudity_level, clothing_state, intimate_detail, skin, expression, body_features
- Landscape: terrain, sky, weather, season, vegetation, water_features
- Other: add whatever categories best describe the unique aspects of this image

When the subject is nude or erotic, describe body details, nudity level, positioning, and anatomical features with the same precision used for other visual elements. Do not euphemize, omit, or soften explicit content.

{tag_format_guide}
Respond with ONLY valid JSON in this format:
{{
    "subject_type": "portrait",
    "elements": [
        {{"category": "environment", "desc": "...", "tags": [...], "attributes": {{}}}},
        {{"category": "lighting", "desc": "...", "tags": [...], "attributes": {{}}}},
        {{"category": "technique", "desc": "...", "tags": [...], "attributes": {{}}}},
        {{"category": "palette", "desc": "...", "tags": [...], "attributes": {{}}}},
        {{"category": "composition", "desc": "...", "tags": [...], "attributes": {{}}}},
        {{"category": "mood", "desc": "...", "tags": [...], "attributes": {{}}}}
    ]
}}

You MUST include all 6 tier 1 categories (environment, lighting, technique, palette, composition, mood). Add 3-8 tier 2 elements on top. Total: 9-14 elements.
"""


def detect_medium(image_path, vision_model_manager):
    """Detect the artistic medium of an image via VLLM.

    Returns:
        dict with keys: is_photograph, medium, medium_tags, technique_notes
        Falls back to photography assumption on failure.
    """
    fallback = {
        "is_photograph": True,
        "medium": "photograph",
        "medium_tags": ["photograph"],
        "technique_notes": "",
    }

    try:
        raw = vision_model_manager.generate_with_image(
            image_path, MEDIUM_DETECTION_PROMPT,
            max_tokens=512,
            temperature=0.1,
            seed=42,
        )
    except Exception as e:
        log.error("Medium detection failed: %s", e)
        return fallback

    parsed = parse_llm_json(raw)
    if isinstance(parsed, dict) and "medium" in parsed:
        log.info("Medium detected: %s (is_photograph=%s)",
                 parsed.get("medium"), parsed.get("is_photograph"))
        return {
            "is_photograph": bool(parsed.get("is_photograph", True)),
            "medium": parsed.get("medium", "photograph"),
            "medium_tags": parsed.get("medium_tags", []),
            "technique_notes": parsed.get("technique_notes", ""),
        }

    log.warning("Medium detection returned unparseable result, assuming photograph")
    return fallback


def _extract_elements(image_path, vision_model_manager, prompt, max_tokens=2048):
    """Run element extraction with retry logic. Returns (subject_type, raw_elements)."""
    subject_type = None
    raw_elements = []

    for attempt in range(MAX_RETRIES):
        retry_prompt = build_retry_prompt(prompt, attempt)

        try:
            raw_response = vision_model_manager.generate_with_image(
                image_path, retry_prompt,
                max_tokens=max_tokens,
                temperature=0.3,
                seed=42,
            )
        except Exception as e:
            log.error("Vision model inference failed (attempt %d): %s", attempt + 1, e)
            continue

        subject_type, raw_elements = parse_element_extraction(raw_response)

        if raw_elements:
            log.info("Extracted %d elements (attempt %d), subject_type=%s",
                     len(raw_elements), attempt + 1, subject_type)
            break

        log.warning("Extraction attempt %d returned no elements", attempt + 1)

    return subject_type, raw_elements


def analyze_photo(image_path, vision_model_manager, quantization="FP8",
                  device="auto", attention_mode="auto", max_tokens=2048,
                  progress_cb=None):
    """Analyze an image and extract elements.

    Runs medium detection first, then photo-style extraction (always), then
    native-style extraction (if the image is not a photograph).

    Args:
        image_path: Path to the image file.
        vision_model_manager: Module or object with generate_with_image().
        quantization: Quantization level for the vision model.
        device: Device to run inference on.
        attention_mode: Attention implementation.
        max_tokens: Maximum tokens for generation.
        progress_cb: Optional callable(str) for reporting progress phases.

    Returns:
        dict with keys:
            subject_type: str or None
            elements: list of element dicts (normalized)
            raw_elements: list of raw element dicts before normalization
            medium: str — detected medium name
            medium_tags: list of medium tag strings
            is_photograph: bool
            status: str describing the outcome
    """
    _progress = progress_cb or (lambda msg: None)

    # Step 1: Detect medium
    _progress("Detecting medium...")
    medium_info = detect_medium(image_path, vision_model_manager)

    # Step 2: Photo extraction (always runs — unchanged)
    _progress("Extracting elements...")
    photo_prompt = EXTRACTION_PROMPT.format(tag_format_guide=TAG_FORMAT_GUIDE)
    subject_type, raw_elements = _extract_elements(
        image_path, vision_model_manager, photo_prompt, max_tokens
    )

    if not raw_elements:
        return {
            "subject_type": None,
            "elements": [],
            "raw_elements": [],
            "medium": medium_info.get("medium", "photograph"),
            "medium_tags": medium_info.get("medium_tags", []),
            "is_photograph": medium_info.get("is_photograph", True),
            "status": f"extraction_failed after {MAX_RETRIES} attempts",
        }

    # Normalize photo elements
    photo_normalized = _normalize_elements(
        raw_elements, image_path, subject_type,
        extraction_type="photo", medium_info=medium_info,
    )

    # Step 3: Native extraction (only for non-photographs)
    native_normalized = []
    native_raw = []
    if not medium_info.get("is_photograph", True):
        _progress("Extracting native elements...")
        medium_name = medium_info.get("medium", "image")
        native_prompt = NATIVE_EXTRACTION_PROMPT.format(
            medium=medium_name, tag_format_guide=TAG_FORMAT_GUIDE,
        )
        native_subject_type, native_raw = _extract_elements(
            image_path, vision_model_manager, native_prompt, max_tokens
        )
        if native_raw:
            native_normalized = _normalize_elements(
                native_raw, image_path,
                native_subject_type or subject_type,
                extraction_type="native", medium_info=medium_info,
            )
            log.info("Native extraction: %d elements for medium '%s'",
                     len(native_normalized), medium_name)

    all_elements = photo_normalized + native_normalized
    n_photo = len(photo_normalized)
    n_native = len(native_normalized)

    return {
        "subject_type": subject_type,
        "elements": all_elements,
        "raw_elements": raw_elements + native_raw,
        "medium": medium_info.get("medium", "photograph"),
        "medium_tags": medium_info.get("medium_tags", []),
        "is_photograph": medium_info.get("is_photograph", True),
        "status": f"ok: {n_photo} photo + {n_native} native elements extracted",
    }


def process_and_commit(analysis_result, element_store, vocabulary_store,
                       archetype_generator=None, model_manager=None,
                       regenerate_archetypes=True):
    """Process analyzed elements: normalize tags, dedup, commit to store.

    Args:
        analysis_result: Output from analyze_photo().
        element_store: The store.elements module.
        vocabulary_store: The store.vocabulary module.
        archetype_generator: Optional core.archetypes module for regeneration.
        model_manager: Optional model_manager module for LLM naming.
        regenerate_archetypes: Whether to regenerate archetypes after commit.
            Set False during batch analysis to avoid expensive O(n²) regen
            after every single photo.

    Returns:
        dict with keys:
            added: list of committed element dicts
            duplicates: list of rejected duplicate elements
            status: str describing the outcome
    """
    elements = analysis_result.get("elements", [])
    thumbnail = analysis_result.get("thumbnail")
    if not elements:
        return {"added": [], "duplicates": [], "status": "no elements to commit"}

    # Attach thumbnail to all elements from this analysis
    if thumbnail:
        for elem in elements:
            elem["thumbnail"] = thumbnail

    # Get existing data for dedup and normalization
    existing_elements = element_store.get_all()
    existing_tags = vocabulary_store.get_all_canonical_tags()

    existing_tag_embeddings = None
    if existing_tags:
        existing_tag_embeddings = emb_module.embed_texts(existing_tags)

    added = []
    duplicates = []

    for elem in elements:
        # Check for duplicate descriptions (only within same extraction type)
        desc = elem.get("desc", "")
        elem_ext_type = elem.get("extraction_type", "photo")
        if desc and (existing_elements or added):
            # Build list of descriptions from same extraction type for dedup
            same_type_descs = []
            same_type_indices = []
            for idx, ex_elem in enumerate(existing_elements):
                ex_desc = ex_elem.get("desc", "")
                if ex_desc and ex_elem.get("extraction_type", "photo") == elem_ext_type:
                    same_type_descs.append(ex_desc)
                    same_type_indices.append(idx)
            # Also check against already-added elements in this batch
            for a_elem in added:
                if a_elem.get("extraction_type", "photo") == elem_ext_type:
                    a_desc = a_elem.get("desc", "")
                    if a_desc:
                        same_type_descs.append(a_desc)

            if same_type_descs:
                same_type_embeddings = emb_module.embed_texts(same_type_descs)
                is_dup, similar_to, sim = emb_module.is_duplicate(
                    desc, same_type_descs, existing_embeddings=same_type_embeddings
                )
                if is_dup:
                    log.info("Duplicate element rejected: '%.50s' similar to '%.50s' (%.3f)",
                             desc, similar_to, sim)
                    duplicates.append({**elem, "similar_to": similar_to, "similarity": sim})
                    continue

        # Normalize tags against vocabulary
        raw_tags = elem.get("tags", [])
        if raw_tags and existing_tags:
            normalized_tags = emb_module.normalize_tags(
                raw_tags, existing_tags, existing_embeddings=existing_tag_embeddings
            )
            elem["tags"] = normalized_tags
        else:
            normalized_tags = raw_tags

        # Register tags in vocabulary
        if normalized_tags:
            for i, (raw, norm) in enumerate(zip(raw_tags, normalized_tags)):
                if raw != norm:
                    vocabulary_store.add_tag(raw, canonical=norm)
                else:
                    vocabulary_store.add_tag(raw)

        # Generate unique ID
        elem["id"] = _generate_element_id(elem, existing_elements + added)

        # Commit element
        added.append(elem)

    # Batch-add to store
    if added:
        element_store.add_many(added)
        log.info("Committed %d elements to library", len(added))

    # Regenerate archetypes if available and requested
    if archetype_generator and added and regenerate_archetypes:
        try:
            from ..store import archetypes as archetype_store
            all_elements = element_store.get_all()
            new_archetypes = archetype_generator.generate_archetypes(
                all_elements,
                use_llm_naming=False,
                model_manager=None,
            )
            archetype_store.replace_all(new_archetypes)
            log.info("Archetypes regenerated: %d total", len(new_archetypes))
        except Exception as e:
            log.error("Archetype regeneration failed: %s", e)

    # Invalidate prompt cache since library changed
    if added:
        try:
            from . import prompt_cache
            prompt_cache.invalidate()
        except Exception as e:
            log.warning("Prompt cache invalidation failed: %s", e)

    return {
        "added": added,
        "duplicates": duplicates,
        "status": f"ok: {len(added)} added, {len(duplicates)} duplicates rejected",
    }


# Category synonyms that QwenVL sometimes produces — map to canonical forms
_CATEGORY_ALIASES = {
    "tattoo": "tattoos",
    "body_features": "body",
    "background": "environment",
    "setting": "environment",
    "location": "environment",
    "light": "lighting",
    "colour": "palette",
    "color": "palette",
    "colors": "palette",
    "colours": "palette",
    "color_palette": "palette",
    "tone": "palette",
    "framing": "composition",
    "emotion": "mood",
    "atmosphere": "mood",
    "attire": "clothing",
    "outfit": "clothing",
    "garment": "clothing",
    "hairstyle": "hair",
    "hair_style": "hair",
    "facial_expression": "expression",
    "face": "expression",
    "jewelry": "accessories",
    "jewellery": "accessories",
    "nudity": "nudity_level",
    "nude_level": "nudity_level",
    "sexual_position": "intimate_detail",
    "body_detail": "body",
    "anatomy": "body",
    "genitalia": "body",
}


def _normalize_elements(raw_elements, image_path, subject_type,
                        extraction_type="photo", medium_info=None):
    """Validate and normalize raw extracted elements."""
    today = date.today().isoformat()
    normalized = []

    medium = (medium_info or {}).get("medium", "photograph")
    is_photograph = (medium_info or {}).get("is_photograph", True)

    for elem in raw_elements:
        # Must have at least category and desc
        if not elem.get("category") or not elem.get("desc"):
            log.warning("Skipping element missing category or desc: %s", elem)
            continue

        # Normalize category to lowercase with underscores
        category = elem["category"].lower().strip().replace(" ", "_")

        # Apply category aliases
        category = _CATEGORY_ALIASES.get(category, category)

        # Skip if category is the same as subject_type (e.g., "portrait" as category)
        if category == subject_type:
            log.debug("Skipping element with category matching subject_type: %s", category)
            continue

        # Normalize tags
        tags = elem.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        tags = [t for t in (_normalize_tag_format(t) for t in tags if t) if t]

        # Build normalized element
        normalized.append({
            "category": category,
            "desc": elem["desc"].strip(),
            "tags": tags,
            "attributes": elem.get("attributes", {}),
            "source_photo": str(image_path) if image_path else None,
            "subject_type": subject_type,
            "added": today,
            "extraction_type": extraction_type,
            "medium": medium,
            "is_photograph": is_photograph,
        })

    return normalized


def _normalize_tag_format(tag):
    """Normalize a tag to lowercase_underscore format."""
    tag = tag.strip().lower()
    tag = re.sub(r'[^a-z0-9_]', '_', tag)
    tag = re.sub(r'_+', '_', tag)
    return tag.strip('_')


def _generate_element_id(element, existing_elements):
    """Generate a unique slug ID for an element."""
    base = element.get("category", "unknown")
    desc_words = element.get("desc", "").lower().split()[:3]
    if desc_words:
        slug = "_".join(re.sub(r'[^a-z0-9]', '', w) for w in desc_words if w)
        base = f"{base}_{slug}" if slug else base

    # Ensure uniqueness
    existing_ids = {e.get("id") for e in existing_elements}
    candidate = base
    counter = 1
    while candidate in existing_ids:
        candidate = f"{base}_{counter}"
        counter += 1

    return candidate
