"""LLM-driven prompt assembly for image generation.

Replaces v1's 4 hardcoded template functions with a single LLM composition
call. Elements are gathered from the library via archetype matching, then
fed to the LLM with a style instruction. The LLM composes natural-language
prompts for any genre — landscape, portrait, automotive, etc.

Supports a "Native" style that generates prompts matching the source
material's detected artistic medium (e.g., watercolor, line art, 3D render)
instead of defaulting to photography language.

Negative prompts use a hybrid approach:
  - LLM generates style-appropriate negatives
  - Template base provides safety-net terms
  - Dynamic anti-affinity from archetype element types
"""

import logging
import random
import re
from collections import Counter

from . import prompt_cache
from .coherence import check_color_harmony, filter_by_ids, filter_by_tags, pick
from .model_manager import ENRICHMENT_LEVELS

log = logging.getLogger("prompt808.generator")

# Photo styles — always available regardless of library content
PHOTO_STYLES = ["Architectural", "Boudoir", "Cinematic", "Documentary", "Erotica", "Fashion", "Fine Art", "Portrait", "Street"]

# Style instructions that replace v1's 4 template functions
STYLE_INSTRUCTIONS = {
    "Architectural": (
        "Compose an architectural photography prompt. Emphasize geometry, symmetry, "
        "leading lines, and structural detail. Use precise spatial language and "
        "reference perspective control techniques. The result should convey the "
        "scale, materiality, and design intent of the built environment."
    ),
    "Boudoir": (
        "Compose an intimate boudoir photography prompt. Focus on the subject's body, "
        "skin texture, and physical presence. Use language that captures sensuality, "
        "vulnerability, and corporeal detail. Describe lighting on skin, body positioning, "
        "and the interplay between concealment and exposure. "
        "The result should feel intimate, body-conscious, and seductive."
    ),
    "Cinematic": (
        "Compose a cinematic photography prompt. Use precise technical language, "
        "camera specifications, and directorial framing. Reference film-quality "
        "lighting setups and lens choices. The result should read like a Director "
        "of Photography's shot description."
    ),
    "Documentary": (
        "Compose a documentary photography prompt. Use observational, raw, "
        "found-moment language. Emphasize authenticity, environmental context, "
        "and candid human presence. The result should feel unstaged and real."
    ),
    "Erotica": (
        "Compose an explicit erotic photography prompt. Describe bodies, sexual positioning, "
        "anatomical detail, and physical interaction with direct, unambiguous language. "
        "Do not euphemize — use precise terms for body parts, states of arousal, and sexual acts. "
        "Treat explicit content with the same technical precision as any other photographic genre. "
        "The result should be visually specific and sexually explicit."
    ),
    "Fashion": (
        "Compose an editorial fashion photography prompt. Use sharp, stylistic "
        "language with attention to fabric behavior, pose dynamics, and editorial "
        "attitude. The result should feel like a creative brief for a high-fashion shoot."
    ),
    "Fine Art": (
        "Compose a fine art photography prompt. Use painterly, poetic language "
        "with art-historical references. Describe light as a sculptor of form. "
        "The result should evoke gallery-quality contemplative imagery."
    ),
    "Native": (
        "Compose a prompt that matches the original artistic medium of the source "
        "material. Use language appropriate to the detected medium — do NOT use "
        "photography terminology unless the source is a photograph. Describe the "
        "artistic qualities, technique, and visual characteristics specific to "
        "this medium."
    ),
    "Portrait": (
        "Compose a portrait photography prompt. Focus on the subject first — "
        "expression, skin quality, and presence. Use language that describes "
        "lighting on the face and body, shallow depth of field, and emotional "
        "connection. The result should feel intimate and subject-driven."
    ),
    "Street": (
        "Compose a street photography prompt. Emphasize candid timing, urban "
        "geometry, and found compositions. Use language that captures decisive "
        "moments, layered scenes, and the energy of public spaces. The result "
        "should feel spontaneous, observational, and graphically composed."
    ),
}

# Base negative prompts per style (safety-net terms)
BASE_NEGATIVES = {
    "Architectural": "blurry, distorted perspective, tilted horizon, cluttered, cartoon, illustration, watermark, text, deformed, low quality, fisheye distortion",
    "Boudoir": "blurry, bad anatomy, disfigured, cluttered, cartoon, illustration, watermark, text, low quality, unflattering angle, harsh flash",
    "Cinematic": "blurry, out of focus, amateur, cartoon, painting, illustration, watermark, text, deformed, disfigured, low quality",
    "Documentary": "posed, artificial, over-processed, oversaturated, cartoon, illustration, watermark, text, deformed, studio lighting, fashion editorial",
    "Erotica": "blurry, bad anatomy, disfigured, cartoon, illustration, watermark, text, low quality, deformed genitalia, extra limbs, unflattering angle",
    "Fashion": "amateur, unflattering, blurry, cluttered, text, watermark, deformed, cartoon, illustration, low quality, bad anatomy",
    "Fine Art": "snapshot, amateur, cluttered, busy, text, watermark, deformed, cartoon, illustration, low quality, oversaturated",
    "Native": "blurry, low quality, deformed, text, watermark",
    "Portrait": "blurry, bad anatomy, disfigured, unflattering, cluttered background, cartoon, illustration, watermark, text, low quality, amateur lighting",
    "Street": "posed, staged, studio lighting, artificial, over-processed, cartoon, illustration, watermark, text, deformed, low quality",
}

# Universal safety negatives always included
SAFETY_NEGATIVES = "watermark, text, deformed, disfigured, low quality, blurry"

# Mood descriptors for prompt flavoring
MOOD_MODIFIERS = {
    "Dramatic": "with intense contrast, deep shadows, and emotional weight",
    "Serene": "with calm stillness, soft light, and peaceful atmosphere",
    "Mysterious": "with enigmatic shadows, fog or haze, and hidden narratives",
    "Romantic": "with warm tones, intimate framing, and tender atmosphere",
    "Melancholic": "with muted tones, solitary subjects, and wistful nostalgia",
    "Ethereal": "with dreamlike quality, soft diffusion, and otherworldly glow",
    "Gritty": "with raw texture, harsh reality, and unflinching directness",
    "Elegant": "with refined composition, luxurious detail, and graceful form",
    "Sensual": "with warm skin tones, languid energy, tactile textures, and slow intimate atmosphere",
    "Provocative": "with bold confrontational gaze, deliberate exposure, high contrast, and unapologetic presence",
}

# ---- Simple-compose style configuration (no-LLM path) ----

# Element category ordering by style priority — earlier categories appear first
STYLE_CATEGORY_ORDER = {
    "Architectural": [
        "environment", "composition", "lighting", "camera", "palette",
        "mood", "subject", "prop",
    ],
    "Boudoir": [
        "subject", "body", "pose", "lighting", "skin", "nudity_level",
        "intimate_detail", "palette", "mood", "composition", "environment",
        "clothing", "camera",
    ],
    "Cinematic": [
        "camera", "lighting", "composition", "environment", "palette",
        "mood", "subject", "clothing", "prop",
    ],
    "Documentary": [
        "environment", "subject", "mood", "lighting", "composition",
        "camera", "palette", "clothing", "prop",
    ],
    "Erotica": [
        "subject", "body", "intimate_detail", "pose", "nudity_level",
        "lighting", "skin", "body_features", "clothing_state",
        "mood", "composition", "environment", "camera",
    ],
    "Fashion": [
        "clothing", "subject", "lighting", "palette",
        "composition", "environment", "mood", "camera",
    ],
    "Fine Art": [
        "lighting", "palette", "mood", "composition", "environment",
        "camera", "subject", "clothing", "prop",
    ],
    # Native has no fixed ordering — elements are shuffled via RNG for variety
    "Portrait": [
        "subject", "lighting", "camera", "palette", "mood",
        "composition", "environment", "clothing", "prop",
    ],
    "Street": [
        "environment", "subject", "composition", "lighting", "mood",
        "camera", "palette", "clothing", "prop",
    ],
}

# Short bridging phrases between element descriptions (cycled)
STYLE_CONNECTORS = {
    "Architectural": [", intersected by ", ", anchored in ", ", defined by "],
    "Boudoir": [", draped in ", ", revealing ", ", bathed in "],
    "Cinematic": [", captured with ", ", revealing ", ", framed by "],
    "Documentary": [", amid ", ", witnessing ", ", set against "],
    "Erotica": [", exposing ", ", positioned in ", ", against "],
    "Fashion": [", styled with ", ", against ", ", accentuated by "],
    "Fine Art": [", dissolving into ", ", rendered in ", ", bathed in "],
    "Native": [", with ", ", featuring ", ", rendered with "],
    "Portrait": [", illuminated by ", ", revealing ", ", softened by "],
    "Street": [", caught amid ", ", layered against ", ", framed within "],
}

# (prefix, suffix) tuples applied to element descriptions by category
STYLE_CATEGORY_BOOSTERS = {
    "Architectural": {
        "environment": ("", " structural form"),
        "composition": ("", " geometric precision"),
    },
    "Boudoir": {
        "subject": ("", " intimate presence"),
        "skin": ("", " tactile detail"),
        "lighting": ("", " sculpting skin"),
    },
    "Cinematic": {
        "camera": ("shot on ", ""),
        "lighting": ("", " film-quality lighting"),
    },
    "Documentary": {
        "environment": ("", " found location"),
        "mood": ("", " unstaged moment"),
    },
    "Erotica": {
        "body": ("", " explicit detail"),
        "intimate_detail": ("", " unambiguous clarity"),
        "lighting": ("", " revealing every surface"),
    },
    "Fashion": {
        "clothing": ("", " editorial styling"),
        "subject": ("", " haute couture presence"),
    },
    "Fine Art": {
        "lighting": ("", " sculpting form"),
        "palette": ("", " painterly tones"),
    },
    "Native": {},  # Medium prefix provides sufficient context
    "Portrait": {
        "subject": ("", " intimate presence"),
        "lighting": ("", " portrait lighting"),
    },
    "Street": {
        "environment": ("", " urban setting"),
        "composition": ("", " decisive geometry"),
    },
}

# Closing quality descriptors per style
STYLE_QUALITY_SUFFIXES = {
    "Architectural": "tilt-shift precision, clean lines, structural clarity",
    "Boudoir": "skin detail, intimate atmosphere, sensual lighting, shallow depth of field",
    "Cinematic": "cinematic depth of field, anamorphic lens quality, film grain",
    "Documentary": "raw authenticity, decisive moment, natural imperfection",
    "Erotica": "anatomical precision, explicit detail, professional lighting, sharp focus on subject",
    "Fashion": "editorial finish, magazine-quality",
    "Fine Art": "gallery-quality print, contemplative mood, fine detail",
    "Native": "highly detailed, professional quality",
    "Portrait": "shallow depth of field, skin detail, catchlight in eyes",
    "Street": "decisive moment, urban texture, candid authenticity",
}


def generate_prompt(seed, archetype_id="Any", style="Any", mood="Any",
                    model_name=None, quantization="FP16", enrichment="Vivid",
                    temperature=0.7, max_tokens=1024,
                    model_manager=None, element_store=None, archetype_store=None,
                    style_profile_module=None, debug=False, nsfw=False):
    """Generate a photography prompt from the element library.

    Args:
        seed: Random seed for deterministic element selection.
        archetype_id: Archetype to filter elements by, or "Any".
        style: Style instruction key (Cinematic, Fine Art, Fashion, Documentary).
        mood: Mood modifier key, or "Any".
        model_name: LLM model name for composition.
        quantization: LLM quantization level.
        enrichment: Enrichment level for descriptions.
        model_manager: The model_manager module (for LLM calls).
        element_store: The store.elements module.
        archetype_store: The store.archetypes module.
        style_profile_module: The core.style_profile module (optional).
        debug: Enable debug logging.

    Returns:
        dict with keys:
            prompt: Generated positive prompt string.
            negative_prompt: Generated negative prompt string.
            archetype_used: Name of archetype used.
            elements_used: List of element IDs used.
            status: Description of outcome.
    """
    # Strip display prefix (e.g. "Photo-Cinematic" → "Cinematic")
    style = _strip_style_prefix(style)

    # Resolve "Any" style to a concrete one using the seed
    all_styles = PHOTO_STYLES + ["Native"]
    if not nsfw:
        all_styles = [s for s in all_styles if s not in ("Boudoir", "Erotica")]
    if style == "Any" or style not in STYLE_INSTRUCTIONS:
        style_rng = random.Random(seed)
        style = style_rng.choice(all_styles)

    # Resolve "Any" enrichment to a concrete level using the seed
    if enrichment == "Any":
        enrichment_rng = random.Random(seed + 1)
        enrichment = enrichment_rng.choice(list(ENRICHMENT_LEVELS.keys()))

    # Resolve archetype name early (needed for both cache hit and miss paths)
    archetype = None
    archetype_name = "Any"

    if archetype_id != "Any" and archetype_store:
        archetype = archetype_store.get_by_id(archetype_id)
        if not archetype:
            archetype = archetype_store.get_by_name(archetype_id)
        if archetype:
            archetype_name = archetype.get("name", archetype_id)

    # Check prompt cache first
    if element_store:
        library_version = element_store.get_library_version()
    else:
        library_version = "unknown"

    cached = prompt_cache.get(
        seed, archetype_id, style, mood, model_name, quantization, library_version,
        enrichment, temperature,
    )
    if cached:
        prompt_text, negative_text = cached
        return {
            "prompt": prompt_text,
            "negative_prompt": negative_text,
            "archetype_used": archetype_name,
            "elements_used": [],
            "status": "cache_hit",
        }

    # Step 1: Gather elements
    all_elements = element_store.get_all() if element_store else []
    if not all_elements:
        return {
            "prompt": "",
            "negative_prompt": SAFETY_NEGATIVES,
            "archetype_used": "none",
            "elements_used": [],
            "status": "empty_library",
        }

    # Step 2: Filter elements by extraction type based on style
    if style == "Native":
        # Prefer native elements; supplement with photo elements when
        # the native pool is too thin (e.g., only 1 source photo analyzed
        # as non-photography). Camera-category elements are always excluded
        # since lens/f-stop specs don't apply to non-photography art.
        native = [e for e in all_elements if e.get("extraction_type") == "native"]
        if native:
            native_sources = {
                e.get("source_photo") or e.get("thumbnail") or "unknown"
                for e in native
            }
            if len(native_sources) >= 2:
                all_elements = native
            else:
                # Single source — supplement for variety
                photo_supplement = [
                    e for e in all_elements
                    if e.get("extraction_type") != "native"
                    and e.get("category") != "camera"
                ]
                all_elements = native + photo_supplement
                log.info(
                    "Native pool from %d source(s), supplementing with "
                    "%d photo elements for variety",
                    len(native_sources), len(photo_supplement),
                )
        else:
            # No native elements — use all but exclude camera category
            all_elements = [
                e for e in all_elements if e.get("category") != "camera"
            ]
            log.info("No native elements in library, Native style using "
                     "non-camera photo elements")
    else:
        # Photo styles: exclude native-only elements
        photo = [e for e in all_elements if e.get("extraction_type") != "native"]
        if photo:
            all_elements = photo

    # Step 3: Filter elements by archetype
    selected_elements = all_elements

    if archetype:
        selected_elements = _filter_elements_by_archetype(all_elements, archetype)
        # Fall back to all elements if archetype filter is too restrictive
        if len(selected_elements) < 3:
            log.warning("Archetype '%s' matched only %d elements, using all",
                        archetype_name, len(selected_elements))
            selected_elements = all_elements
            archetype_name = "Any (fallback)"

    # Step 4: Seeded random selection of elements per category
    rng = random.Random(seed)
    chosen_elements = _select_elements(rng, selected_elements)

    if not chosen_elements:
        return {
            "prompt": "",
            "negative_prompt": SAFETY_NEGATIVES,
            "archetype_used": archetype_name,
            "elements_used": [],
            "status": "no_elements_selected",
        }

    # Resolve medium for Native style from selected elements
    resolved_medium = None
    if style == "Native":
        mediums = [e.get("medium") for e in chosen_elements if e.get("medium")]
        if mediums:
            resolved_medium = Counter(mediums).most_common(1)[0][0]
        else:
            resolved_medium = "illustration"

    # Step 5: Build the prompt
    if model_manager and model_name and model_name != "None":
        # LLM-driven composition
        prompt_text, negative_text = _llm_compose(
            chosen_elements, style, mood, archetype, seed,
            model_manager, model_name, quantization, enrichment,
            style_profile_module, debug, temperature, max_tokens,
            resolved_medium=resolved_medium,
        )
    else:
        # Fallback: simple concatenation (no LLM available)
        prompt_text = _simple_compose(chosen_elements, style, mood,
                                      resolved_medium=resolved_medium, rng=rng)
        negative_text = _build_negative(style, archetype, llm_negatives=None,
                                        resolved_medium=resolved_medium)

    element_ids = [e.get("id", "") for e in chosen_elements]

    # Cache the result
    prompt_cache.put(
        seed, archetype_id, style, mood, model_name, quantization,
        library_version, prompt_text, negative_text, enrichment, temperature,
    )

    return {
        "prompt": prompt_text,
        "negative_prompt": negative_text,
        "archetype_used": archetype_name,
        "elements_used": element_ids,
        "status": "ok",
    }


def _filter_elements_by_archetype(elements, archetype):
    """Filter elements using archetype's compatible tags and element_ids."""
    compatible = archetype.get("compatible", {})
    element_ids = archetype.get("element_ids", [])
    negative_hints = archetype.get("negative_hints", [])

    # First: include elements by ID (direct membership)
    id_matches = filter_by_ids(elements, element_ids)

    # Second: include elements by tag matching
    tag_matches = []
    for elem in elements:
        if elem in id_matches:
            continue
        cat = elem.get("category", "")
        tag_key = f"{cat}_tags"
        if tag_key in compatible:
            required = compatible[tag_key]
            if filter_by_tags([elem], "tags", required):
                tag_matches.append(elem)

    combined = id_matches + tag_matches

    # Exclude elements matching negative hints
    if negative_hints:
        combined = [e for e in combined
                    if not set(e.get("tags", [])) & set(negative_hints)]

    return combined


def _select_elements(rng, elements, max_retries=5):
    """Select elements per category using seeded RNG with frequency weighting
    and color harmony validation.

    Tier 1 categories (environment, lighting, camera, palette, composition,
    mood) are always included. Tier 2 categories are included with probability
    proportional to their photo fraction — if only 3 out of 44 photos have
    tattoos, tattoo elements appear in ~7% of prompts, not 100%.

    After selection, check_color_harmony validates that palette, lighting, and
    environment tags don't clash (e.g., neon cyberpunk + pastoral meadow).
    Retries up to max_retries times on clash before accepting the result.
    """
    TIER1 = {"environment", "lighting", "camera", "technique", "palette", "composition", "mood"}

    by_category = {}
    cat_photos = {}
    all_photos = set()

    for elem in elements:
        cat = elem.get("category", "unknown")
        by_category.setdefault(cat, []).append(elem)
        photo = elem.get("thumbnail") or elem.get("source_photo") or "unknown"
        all_photos.add(photo)
        cat_photos.setdefault(cat, set()).add(photo)

    n_photos = max(len(all_photos), 1)

    for attempt in range(max_retries):
        chosen_dict = {}

        for cat in sorted(by_category.keys()):
            if cat in TIER1:
                picked = pick(rng, by_category[cat])
                if picked:
                    chosen_dict[cat] = picked
            else:
                weight = len(cat_photos.get(cat, set())) / n_photos
                if rng.random() < weight:
                    picked = pick(rng, by_category[cat])
                    if picked:
                        chosen_dict[cat] = picked

        if "palette" in chosen_dict:
            if check_color_harmony(chosen_dict):
                return list(chosen_dict.values())
            else:
                log.debug("Color harmony clash on attempt %d, retrying...", attempt + 1)
        else:
            return list(chosen_dict.values())

    log.warning("Could not resolve color harmony after %d attempts, proceeding anyway", max_retries)
    return list(chosen_dict.values())


def _llm_compose(chosen_elements, style, mood, archetype, seed,
                 model_manager, model_name, quantization, enrichment,
                 style_profile_module, debug, temperature=0.7, max_tokens=1024,
                 resolved_medium=None):
    """Use LLM to compose a natural-language prompt from elements.

    The composition prompt adapts based on the enrichment level — higher
    levels loosen fidelity constraints and inject creative rules.
    """
    # Look up enrichment config
    enrich_cfg = ENRICHMENT_LEVELS.get(enrichment, ENRICHMENT_LEVELS["Vivid"])

    # Build element descriptions for the LLM
    element_lines = []
    for elem in chosen_elements:
        cat = elem.get("category", "unknown").replace("_", " ").title()
        if enrich_cfg.get("omit_original_desc"):
            # Higher creativity levels: send only tags, not descriptions
            tags = ", ".join(elem.get("tags", []))
            element_lines.append(f"- {cat}: [{tags}]")
        else:
            desc = elem.get("desc", "")
            element_lines.append(f"- {cat}: {desc}")

    elements_block = "\n".join(element_lines)

    # Style instruction — dynamic for Native, static for photo styles
    if style == "Native" and resolved_medium:
        style_instruction = (
            f"Compose a {resolved_medium} prompt. Use language appropriate to this "
            f"artistic medium. Describe the visual qualities, technique, and "
            f"characteristics specific to {resolved_medium}. Do NOT use photography "
            f"terminology such as lens, f-stop, depth of field, or camera angle."
        )
    else:
        style_instruction = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["Cinematic"])

    # Task instruction from enrichment level
    task_instruction = enrich_cfg["task_instruction"]

    # Mood modifier
    mood_clause = ""
    if mood and mood != "Any" and mood in MOOD_MODIFIERS:
        mood_clause = f"\nMood direction: {MOOD_MODIFIERS[mood]}"

    # Scene context for Freeform mode (needs_scene_context)
    scene_context = ""
    if enrich_cfg.get("needs_scene_context") and archetype:
        scene_context = f"\nScene context: {archetype.get('name', 'unknown scene')}"

    # Style profile context (if available)
    style_context = ""
    if style_profile_module:
        subject_types = set()
        for elem in chosen_elements:
            st = elem.get("subject_type")
            if st:
                subject_types.add(st)
        for genre in subject_types:
            ctx = style_profile_module.get_style_context(genre)
            if ctx:
                style_context = f"\n\n{ctx}"
                break

    # Build rules block
    word_range = enrich_cfg.get("word_range", "5-20 words")
    fidelity_rule = enrich_cfg.get("fidelity_rule", "")
    creative_rules = enrich_cfg.get("creative_rules", [])

    rules_lines = [
        "- Output ONE continuous prompt paragraph (no line breaks)",
        "- Weave all elements into a coherent scene description",
        f"- Fidelity: {fidelity_rule}",
        f"- {word_range} per element, 40-120 words total",
        "- Reproduce all element descriptions faithfully — do not euphemize, censor, or omit anatomical, sexual, or explicit content",
    ]
    for rule in creative_rules:
        rules_lines.append(f"- {rule}")

    rules_block = "\n".join(rules_lines)

    # Build the composition prompt
    composition_prompt = (
        f"{style_instruction}\n\n"
        f"{task_instruction}\n\n"
        f"Elements:\n"
        f"{elements_block}"
        f"{mood_clause}"
        f"{scene_context}"
        f"{style_context}\n\n"
        f"Rules:\n"
        f"{rules_block}\n\n"
        f"Also provide a negative prompt (comma-separated terms to avoid).\n\n"
        f"Respond with ONLY JSON:\n"
        f'{{"prompt": "your composed prompt here", "negative_prompt": "terms to avoid"}}'
    )

    # Apply temperature nudge from enrichment level
    effective_temp = temperature + enrich_cfg.get("temperature_nudge", 0.0)
    effective_temp = min(max(effective_temp, 0.1), 1.5)

    try:
        # Load model if needed
        model_manager.load_model(model_name, quantization=quantization)

        raw = model_manager.generate_text(
            composition_prompt,
            max_tokens=max_tokens,
            temperature=effective_temp,
            seed=seed,
            debug=debug,
        )

        # Parse the LLM response
        from .json_parser import parse_llm_json
        parsed = parse_llm_json(raw)

        if isinstance(parsed, dict) and parsed.get("prompt"):
            prompt_text = parsed["prompt"].strip()
            llm_negatives = parsed.get("negative_prompt", "")
            negative_text = _build_negative(style, archetype, llm_negatives,
                                            resolved_medium=resolved_medium)
            return prompt_text, negative_text

    except Exception as e:
        log.error("LLM composition failed: %s", e)

    # Fallback to simple composition
    fallback_rng = random.Random(seed)
    prompt_text = _simple_compose(chosen_elements, style, mood,
                                  resolved_medium=resolved_medium,
                                  rng=fallback_rng)
    negative_text = _build_negative(style, archetype, llm_negatives=None,
                                    resolved_medium=resolved_medium)
    return prompt_text, negative_text


def _order_by_style(elements, category_order, rng=None):
    """Sort elements by a style's category priority list.

    When rng is provided, randomizes order within equal-priority groups,
    ensuring different seeds produce different element arrangements.
    """
    order_map = {cat: i for i, cat in enumerate(category_order)}
    default_rank = len(category_order)

    if rng:
        # Stable sort by priority with random tie-breaking
        decorated = [(order_map.get(e.get("category", ""), default_rank), rng.random(), e)
                     for e in elements]
        decorated.sort(key=lambda t: (t[0], t[1]))
        return [e for _, _, e in decorated]

    return sorted(elements,
                  key=lambda e: order_map.get(e.get("category", ""), default_rank))


def _simple_compose(chosen_elements, style, mood, resolved_medium=None, rng=None):
    """Fallback: compose prompt with style-aware ordering and connectors.

    When rng is provided, introduces seed-dependent variation by shuffling
    element order and cycling connector start positions, ensuring different
    seeds produce different prompts even with identical element sets.
    """
    parts = []

    # 1. Style prefix
    if style == "Native" and resolved_medium:
        parts.append(f"{resolved_medium},")
    else:
        style_prefix = {
            "Boudoir": "intimate boudoir photography,",
            "Cinematic": "cinematic film still,",
            "Erotica": "explicit erotic photography,",
            "Fine Art": "fine art photography, gallery-quality,",
            "Fashion": "high-fashion editorial photography,",
            "Documentary": "documentary photography, candid,",
        }
        parts.append(style_prefix.get(style, "photography,"))

    # 2. Mood
    if mood and mood != "Any" and mood in MOOD_MODIFIERS:
        parts.append(mood.lower() + " atmosphere,")

    # 3. Sort elements by style priority, with seed-based variation
    order = STYLE_CATEGORY_ORDER.get(style, [])
    if order:
        ordered = _order_by_style(chosen_elements, order, rng=rng)
    elif rng:
        # No style ordering (e.g., Native) — shuffle for variety
        ordered = list(chosen_elements)
        rng.shuffle(ordered)
    else:
        ordered = list(chosen_elements)

    # 4. Build descriptions with boosters
    boosters = STYLE_CATEGORY_BOOSTERS.get(style, {})
    desc_parts = []
    for elem in ordered:
        desc = elem.get("desc", "")
        if not desc:
            continue
        cat = elem.get("category", "")
        pfx, sfx = boosters.get(cat, ("", ""))
        desc_parts.append(f"{pfx}{desc}{sfx}")

    # 5. Join with connectors or plain commas
    connectors = STYLE_CONNECTORS.get(style, [])
    if desc_parts and connectors:
        offset = rng.randint(0, len(connectors) - 1) if rng else 0
        joined = desc_parts[0]
        for i, dp in enumerate(desc_parts[1:]):
            joined += connectors[(i + offset) % len(connectors)] + dp
        parts.append(joined)
    else:
        for dp in desc_parts:
            parts.append(dp + ",")

    # 6. Quality suffix
    quality = STYLE_QUALITY_SUFFIXES.get(style, "")
    if quality:
        parts.append(quality)

    # Assemble and clean
    prompt = " ".join(parts)
    prompt = re.sub(r",\s*,", ",", prompt)
    prompt = prompt.rstrip(",").strip()
    return prompt


def _build_negative(style, archetype=None, llm_negatives=None,
                    resolved_medium=None):
    """Build negative prompt from multiple sources.

    Combines:
    1. LLM-generated negatives (if available)
    2. Base template negatives for the style
    3. Archetype anti-affinity hints

    For Native style, avoids rejecting terms that describe the source medium
    (e.g., don't reject "illustration" if the medium IS illustration).
    """
    parts = set()

    # LLM negatives
    if llm_negatives:
        for term in llm_negatives.split(","):
            term = term.strip()
            if term:
                parts.add(term)

    # Base negatives for style
    base = BASE_NEGATIVES.get(style, BASE_NEGATIVES["Cinematic"])
    for term in base.split(","):
        term = term.strip()
        if term:
            parts.add(term)

    # Archetype anti-affinity hints
    if archetype and "negative_hints" in archetype:
        for hint in archetype["negative_hints"]:
            parts.add(hint)

    # For Native style, remove terms that conflict with the detected medium
    if style == "Native" and resolved_medium:
        medium_lower = resolved_medium.lower()
        medium_words = set(medium_lower.split())
        # Remove negative terms that match words in the medium name
        # e.g., don't reject "illustration" if medium is "digital illustration"
        conflicting = set()
        for term in parts:
            if term.lower() in medium_words or term.lower() in medium_lower:
                conflicting.add(term)
        if conflicting:
            log.debug("Native style: removing conflicting negatives %s for medium '%s'",
                      conflicting, resolved_medium)
            parts -= conflicting

    return ", ".join(sorted(parts))


def _display_style(name):
    """Add 'Photo-' prefix to photo styles for display (skip Any/Native)."""
    return name if name in ("Any", "Native") else f"Photo-{name}"


def _strip_style_prefix(name):
    """Strip 'Photo-' display prefix to recover the internal style name."""
    return name[6:] if name.startswith("Photo-") else name


def get_available_styles(nsfw=False):
    """Return list of available style/prompt-type names for display."""
    styles = sorted(STYLE_INSTRUCTIONS.keys())
    if not nsfw:
        styles = [s for s in styles if s not in ("Boudoir", "Erotica")]
    return ["Any"] + sorted(_display_style(s) for s in styles)


def get_available_moods(nsfw=False):
    """Return list of available mood names."""
    moods = sorted(MOOD_MODIFIERS.keys())
    if not nsfw:
        moods = [m for m in moods if m not in ("Sensual", "Provocative")]
    return ["Any"] + moods
