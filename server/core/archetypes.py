"""Archetype generation via photo-level agglomerative clustering.

Auto-generates archetypes from the element library by:
1. Grouping elements by source photo
2. Embedding focused photo profiles (environment + lighting — the most
   discriminating features across genres)
3. Agglomerative clustering with silhouette-based auto cluster count
4. Building archetype metadata from dominant traits per cluster

Photo-level clustering avoids the failure mode of element-level clustering,
where "lighting: soft window light" from a boudoir photo embeds near
"lighting: soft natural light" from an outdoor photo, chaining everything
into one giant cluster.
"""

import logging
import math
import re
from collections import Counter
from datetime import date

import numpy as np

from . import embeddings as emb_module

log = logging.getLogger("prompt808.archetypes")

# Minimum photos in a cluster before it gets merged into the nearest neighbor
MIN_CLUSTER_SIZE = 2

# Categories used to build the clustering feature text.
# These are the most discriminating across genres — pose, hair, clothing etc.
# are too subject-specific to distinguish scene types.
_CLUSTER_CATEGORIES = {"environment", "lighting"}

# Scoring priorities for distinctive naming — evocative categories rank higher
_DISTINCTIVE_CATS = {
    "mood": 1.5,
    "environment": 1.3,
    "lighting": 1.0,
    "palette": 0.8,
}

# Natural reading order for composing the final name
_NAME_ORDER = {
    "environment": 0, "mood": 1, "palette": 2,
    "lighting": 3,
}


def generate_archetypes(elements, use_llm_naming=True, model_manager=None):
    """Generate archetypes from the current element library.

    Clusters at the photo level (not individual element level) to produce
    meaningful scene-type groupings like "Studio Dramatic", "Outdoor Golden
    Hour", "Intimate Boudoir", etc.

    Args:
        elements: List of element dicts from the store.
        use_llm_naming: If True and model_manager is available, use LLM
                        to name clusters. Otherwise use trait-based names.
        model_manager: The model_manager module (for LLM naming).

    Returns:
        List of archetype dicts ready for store.archetypes.replace_all().
    """
    if not elements:
        return []

    # Group elements by source photo
    photo_groups = _group_by_photo(elements)

    # Too few photos for clustering
    if len(photo_groups) < 4:
        return _generate_flat_archetype(elements)

    # Create focused text profiles for each photo
    photo_keys = list(photo_groups.keys())
    photo_texts = [_photo_cluster_text(photo_groups[k]) for k in photo_keys]

    # Embed photo profiles
    photo_embeddings = emb_module.embed_texts(photo_texts)
    if photo_embeddings is None or len(photo_embeddings) < 4:
        return _generate_flat_archetype(elements)

    # Compute distance matrix
    sim_matrix = emb_module.cosine_similarity_matrix(photo_embeddings)
    dist_matrix = 1.0 - sim_matrix
    np.clip(dist_matrix, 0, None, out=dist_matrix)
    np.fill_diagonal(dist_matrix, 0)

    # Find optimal cluster count via silhouette score
    labels = _cluster_photos(dist_matrix, len(photo_keys))

    # Merge small clusters into nearest neighbor
    labels = _merge_small_clusters(labels, dist_matrix)

    # Build archetypes from photo clusters
    cluster_ids = sorted(set(labels))
    archetypes = []
    cluster_element_lists = []

    for cid in cluster_ids:
        photo_indices = [i for i, l in enumerate(labels) if l == cid]
        cluster_elements = []
        for idx in photo_indices:
            cluster_elements.extend(photo_groups[photo_keys[idx]])

        archetype = _build_archetype(
            cid, cluster_elements, len(photo_indices),
            False, None,  # naming handled in batch below
        )
        archetypes.append(archetype)
        cluster_element_lists.append(cluster_elements)

    # Batch rename using TF-IDF distinctive tags
    _rename_archetypes_distinctive(
        archetypes, cluster_element_lists,
        use_llm_naming, model_manager,
    )

    log.info("Generated %d archetypes from %d photos (%d elements)",
             len(archetypes), len(photo_keys), len(elements))
    return archetypes


# ---------------------------------------------------------------------------
# Photo grouping
# ---------------------------------------------------------------------------

def _group_by_photo(elements):
    """Group elements by their source photo.

    Uses thumbnail as key (stable across sessions), falling back to
    source_photo path.
    """
    groups = {}
    for e in elements:
        key = e.get("thumbnail") or e.get("source_photo") or "unknown"
        groups.setdefault(key, []).append(e)
    return groups


def _photo_cluster_text(elements):
    """Create a focused text profile for clustering.

    Only includes environment and lighting descriptions — the features
    that best distinguish scene types across genres.
    """
    parts = []
    for e in elements:
        cat = e.get("category", "")
        if cat in _CLUSTER_CATEGORIES:
            desc = e.get("desc", "")
            if desc:
                parts.append(desc)
    return ", ".join(parts) if parts else "general photograph"


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _cluster_photos(dist_matrix, n_photos):
    """Cluster photos using agglomerative clustering with auto cluster count.

    Tries a range of cluster counts and picks the one with the best
    silhouette score (a measure of how well-separated the clusters are).
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    # Search range: 3 to sqrt(n_photos), clamped
    min_k = 3
    max_k = max(min_k + 1, min(int(math.sqrt(n_photos)) + 1, 12))

    best_score = -1
    best_labels = None
    best_n = min_k

    for n in range(min_k, max_k + 1):
        if n >= n_photos:
            break
        try:
            agg = AgglomerativeClustering(
                n_clusters=n, metric="precomputed", linkage="average",
            )
            labels = agg.fit_predict(dist_matrix)
            score = silhouette_score(dist_matrix, labels, metric="precomputed")
            if score > best_score:
                best_score = score
                best_labels = labels
                best_n = n
        except Exception as e:
            log.warning("Clustering with n=%d failed: %s", n, e)

    if best_labels is None:
        # Fallback: everything in one cluster
        return np.zeros(n_photos, dtype=int)

    log.info("Auto-selected %d clusters (silhouette=%.3f)", best_n, best_score)
    return best_labels


def _merge_small_clusters(labels, dist_matrix):
    """Merge clusters with fewer than MIN_CLUSTER_SIZE photos into nearest.

    Finds the nearest cluster centroid and reassigns.
    """
    labels = labels.copy()
    cluster_ids = set(labels)

    # Compute cluster centroids (mean distance to all points in cluster)
    for cid in list(cluster_ids):
        members = [i for i, l in enumerate(labels) if l == cid]
        if len(members) >= MIN_CLUSTER_SIZE:
            continue

        # Find nearest cluster by mean distance
        best_target = None
        best_dist = float("inf")
        for other_cid in cluster_ids:
            if other_cid == cid:
                continue
            other_members = [i for i, l in enumerate(labels) if l == other_cid]
            if len(other_members) < MIN_CLUSTER_SIZE:
                continue
            # Mean distance from this cluster's members to the other cluster
            mean_dist = np.mean([
                dist_matrix[m][o]
                for m in members for o in other_members
            ])
            if mean_dist < best_dist:
                best_dist = mean_dist
                best_target = other_cid

        if best_target is not None:
            for m in members:
                labels[m] = best_target
            log.debug("Merged cluster %s (%d photos) into cluster %s",
                      cid, len(members), best_target)

    # Re-number clusters to be contiguous 0..N-1
    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique)}
    return np.array([remap[l] for l in labels])


# ---------------------------------------------------------------------------
# Archetype building
# ---------------------------------------------------------------------------

def _build_archetype(cluster_id, cluster_elements, n_photos,
                     use_llm_naming, model_manager):
    """Build an archetype dict from a cluster of elements."""
    categories = set()
    tag_map = {}
    element_ids = []
    cat_photos = {}

    for elem in cluster_elements:
        cat = elem.get("category", "unknown")
        categories.add(cat)
        tag_key = f"{cat}_tags"
        if tag_key not in tag_map:
            tag_map[tag_key] = set()
        for tag in elem.get("tags", []):
            tag_map[tag_key].add(tag)
        element_ids.append(elem.get("id", ""))
        # Track unique photos per category for frequency weights
        photo = elem.get("thumbnail") or elem.get("source_photo") or "unknown"
        cat_photos.setdefault(cat, set()).add(photo)

    compatible = {k: sorted(v) for k, v in tag_map.items()}

    # Category weights: fraction of photos with each category
    category_weights = {
        cat: round(len(photos) / max(n_photos, 1), 3)
        for cat, photos in cat_photos.items()
    }

    # Generate name
    if use_llm_naming and model_manager:
        name = _llm_name_cluster(cluster_elements, model_manager)
    else:
        name = _auto_name_cluster(cluster_elements)

    negative_hints = _generate_negative_hints(categories)
    archetype_id = re.sub(r'[^a-z0-9_]', '_', name.lower().replace(' ', '_'))
    # Deduplicate ID with cluster number suffix
    archetype_id = f"{archetype_id}_{cluster_id}"

    return {
        "id": archetype_id,
        "name": name,
        "compatible": compatible,
        "element_ids": element_ids,
        "category_weights": category_weights,
        "photo_count": n_photos,
        "negative_hints": negative_hints,
        "generated": date.today().isoformat(),
    }


def _auto_name_cluster(elements):
    """Generate a descriptive name from dominant environment and lighting traits.

    Strategy: "{Setting} {Light_Style}" — e.g., "Studio Dramatic",
    "Beach Golden Hour", "Forest Natural Light", "Bedroom Soft-Lit".
    """
    env_tags = Counter()
    light_tags = Counter()
    subject_types = Counter()

    for elem in elements:
        cat = elem.get("category", "")
        st = elem.get("subject_type")
        if st:
            subject_types[st] += 1
        for tag in elem.get("tags", []):
            if cat == "environment":
                env_tags[tag] += 1
            elif cat == "lighting":
                light_tags[tag] += 1

    # Pick environment (the setting word) — most distinctive
    setting = _pick_setting(env_tags)

    # Pick lighting modifier — complements the setting
    light_mod = _pick_light_modifier(light_tags)

    if setting and light_mod:
        return f"{setting} {light_mod}"
    elif setting:
        return setting
    elif light_mod:
        # No clear setting — use subject type + light
        top_subject = subject_types.most_common(1)
        subject = top_subject[0][0].replace("_", " ").title() if top_subject else "Mixed"
        return f"{subject} {light_mod}"
    else:
        top_subject = subject_types.most_common(1)
        return top_subject[0][0].replace("_", " ").title() if top_subject else "Mixed"


# Tags too generic/common to distinguish clusters in names
_SKIP_ENV = {
    "minimalist", "interior", "background", "blurred_background",
    "controlled_lighting", "natural_setting", "wall", "warm_light",
    "soft_light", "ambient_light", "bokeh",
}
_SKIP_LIGHT = {
    "soft_light", "natural_light", "even", "even_lighting", "diffused",
    "neutral", "warm_tones", "highlight_skin", "soft_shadows",
}


# Tags too generic or awkward-sounding in archetype names
_SKIP_NAMING = _SKIP_ENV | _SKIP_LIGHT | {
    "skin_tone", "natural_skin", "earth_tones", "natural_colors",
    "warm_palette", "cool_palette",
}

# Display name overrides for tags that read awkwardly in names
_TAG_DISPLAY = {
    # Environment
    "bed": "Bedroom",
    "dark_walls": "Dark Interior",
    "white_walls": "Bright Studio",
    "black_background": "Dark Backdrop",
    "dark_background": "Dark Backdrop",
    "blue_sheets": "Boudoir",
    "white_sheets": "Boudoir",
    "rocky_coast": "Coastal",
    "rock_cliff": "Coastal",
    # Lighting — avoid word overlap with common environment tags
    "studio_light": "Studio Lit",
    "window_light": "Window Lit",
    "string_lights": "String Lit",
}


def _pick_setting(env_tags):
    """Pick the best setting word from environment tags.

    Uses the most frequent non-generic tag. Specific locations
    (beach, forest, studio) are preferred over broad ones (indoor, outdoor).
    """
    if not env_tags:
        return None

    # Broad fallback tags — only use if nothing more specific is available
    broad = {"indoor", "outdoor"}

    # First pass: most frequent non-generic, non-broad tag
    for tag, count in env_tags.most_common(10):
        if tag not in _SKIP_ENV and tag not in broad:
            return _TAG_DISPLAY.get(tag, tag.replace("_", " ").title())

    # Second pass: allow broad tags
    for tag, count in env_tags.most_common(5):
        if tag not in _SKIP_ENV:
            return _TAG_DISPLAY.get(tag, tag.replace("_", " ").title())

    top = env_tags.most_common(1)[0][0] if env_tags else None
    return _TAG_DISPLAY.get(top, top.replace("_", " ").title()) if top else None


def _pick_light_modifier(light_tags):
    """Pick the best lighting descriptor for naming.

    Prefers distinctive lighting terms (golden_hour, chiaroscuro, dramatic)
    over common ones (soft_light, natural_light).
    """
    if not light_tags:
        return None

    for tag, count in light_tags.most_common(10):
        if tag not in _SKIP_LIGHT:
            return tag.replace("_", " ").title()

    return None


def _llm_name_cluster(elements, model_manager):
    """Use LLM to generate a descriptive name for a cluster."""
    element_summaries = []
    for elem in elements[:10]:
        element_summaries.append(
            f"- [{elem.get('category', '?')}] {elem.get('desc', elem.get('id', '?'))} "
            f"(tags: {', '.join(elem.get('tags', [])[:5])})"
        )

    prompt = (
        "Name this group of photographic elements in 3-5 words. "
        "The name should evoke the visual theme (e.g., 'Dramatic Mountain Sunset', "
        "'Intimate Nude Boudoir', 'Urban Street Night').\n\n"
        "Elements:\n" + "\n".join(element_summaries) + "\n\n"
        "Respond with ONLY the name, nothing else."
    )

    try:
        name = model_manager.generate_text(prompt, max_tokens=32, temperature=0.7, seed=42)
        name = name.strip().strip('"\'').strip('.').strip()
        if name and len(name) < 60:
            return name
    except Exception as e:
        log.warning("LLM naming failed: %s", e)

    return _auto_name_cluster(elements)


# ---------------------------------------------------------------------------
# Distinctive naming (TF-IDF based)
# ---------------------------------------------------------------------------


def _rename_archetypes_distinctive(archetypes, element_lists,
                                   use_llm_naming=False, model_manager=None):
    """Batch rename archetypes using TF-IDF distinctive tag scoring.

    Each archetype is named by the tags that best distinguish it from its
    siblings.  Tags are scored as TF × IDF × category_priority, where TF is
    the tag's frequency within the cluster, IDF penalises tags shared across
    many clusters, and category priority favours evocative categories
    (mood > environment > lighting > palette > composition > technique).
    """
    if not archetypes:
        return

    n_clusters = len(archetypes)

    # Per-cluster tag frequencies: Counter of (category, tag) → count
    cluster_tags = []
    for elems in element_lists:
        freq = Counter()
        for elem in elems:
            cat = elem.get("category", "")
            if cat not in _DISTINCTIVE_CATS:
                continue
            for tag in elem.get("tags", []):
                if tag not in _SKIP_NAMING:
                    freq[(cat, tag)] += 1
        cluster_tags.append(freq)

    # Document frequency: how many clusters contain each (cat, tag)
    doc_freq = Counter()
    for freq in cluster_tags:
        for key in freq:
            doc_freq[key] += 1

    # Score and name each archetype
    for i, arch in enumerate(archetypes):
        n_elems = max(len(element_lists[i]), 1)
        freq = cluster_tags[i]

        scored = []
        for (cat, tag), count in freq.items():
            tf = count / n_elems
            idf = 1.0 + math.log2(max(n_clusters, 2) / doc_freq.get((cat, tag), 1))
            priority = _DISTINCTIVE_CATS.get(cat, 0.5)
            scored.append((tf * idf * priority, tag, cat))

        scored.sort(reverse=True)

        # Pick top tags from different categories with no word overlap
        chosen = []   # [(display_name, category), ...]
        used_cats = set()
        for _, tag, cat in scored:
            if cat in used_cats:
                continue
            display = _TAG_DISPLAY.get(tag, tag.replace("_", " ").title())
            if _shares_word(display, [c[0] for c in chosen]):
                continue
            chosen.append((display, cat))
            used_cats.add(cat)
            if len(chosen) >= 3:
                break

        # Sort chosen tags by natural word order
        chosen.sort(key=lambda c: _NAME_ORDER.get(c[1], 99))

        # Try LLM naming with distinctive tags as context
        if use_llm_naming and model_manager:
            llm_name = _llm_name_from_tags(scored[:8], model_manager)
            if llm_name:
                _set_archetype_name(arch, llm_name, i)
                continue

        name = (" ".join(c[0] for c in chosen)
                if chosen else _auto_name_cluster(element_lists[i]))
        _set_archetype_name(arch, name, i)

    # Disambiguate any remaining duplicate names within this batch
    _disambiguate_names(archetypes, cluster_tags)


def _set_archetype_name(arch, name, cluster_id):
    """Assign a name and derive the archetype ID from it."""
    arch["name"] = name
    arch["id"] = re.sub(r'[^a-z0-9_]', '_', name.lower().replace(' ', '_'))
    arch["id"] = f"{arch['id']}_{cluster_id}"


def _shares_word(new_display, chosen_displays):
    """True if *new_display* shares a significant word with any chosen name."""
    _trivial = {"the", "a", "an", "of", "in", "on", "at", "to", "and", "with"}
    new_words = set(new_display.lower().split()) - _trivial
    if not new_words:
        return False
    for existing in chosen_displays:
        if new_words & (set(existing.lower().split()) - _trivial):
            return True
    return False


def _llm_name_from_tags(scored_tags, model_manager):
    """Use LLM to compose an evocative name from the most distinctive tags."""
    lines = []
    for _, tag, cat in scored_tags:
        lines.append(f"- {cat}: {tag.replace('_', ' ')}")

    prompt = (
        "Name this photo cluster in 3-5 evocative words that capture its "
        "visual essence.\n\n"
        "Distinctive traits (most important first):\n"
        + "\n".join(lines) + "\n\n"
        "Examples: 'Ethereal Golden Meadow', 'Dramatic Urban Night', "
        "'Intimate Chiaroscuro Studio'\n\n"
        "Respond with ONLY the name, nothing else."
    )

    try:
        name = model_manager.generate_text(
            prompt, max_tokens=32, temperature=0.7, seed=42,
        )
        name = name.strip().strip("\"'").rstrip(".").strip()
        if name and len(name) < 60:
            return name
    except Exception as e:
        log.warning("LLM naming failed: %s", e)

    return None


def _disambiguate_names(archetypes, cluster_tags):
    """Resolve duplicate names by appending a distinguishing tag or suffix."""
    # First pass: differentiate with unique tags
    name_indices = {}
    for i, arch in enumerate(archetypes):
        name_indices.setdefault(arch["name"], []).append(i)

    for name, indices in name_indices.items():
        if len(indices) <= 1:
            continue
        for idx in indices:
            freq = cluster_tags[idx]
            sibling_tags = set()
            for other_idx in indices:
                if other_idx != idx:
                    sibling_tags.update(cluster_tags[other_idx].keys())

            best = None
            for (cat, tag), _count in freq.most_common():
                if (cat, tag) in sibling_tags or tag in _SKIP_NAMING:
                    continue
                display = _TAG_DISPLAY.get(tag, tag.replace("_", " ").title())
                if not _shares_word(display, [archetypes[idx]["name"]]):
                    best = display
                    break

            if best:
                _set_archetype_name(
                    archetypes[idx], f"{archetypes[idx]['name']} {best}", idx,
                )

    # Second pass: numeric suffix for any still-duplicate names
    name_indices2 = {}
    for i, arch in enumerate(archetypes):
        name_indices2.setdefault(arch["name"], []).append(i)

    _NUMERALS = ["I", "II", "III", "IV", "V", "VI"]
    for name, indices in name_indices2.items():
        if len(indices) <= 1:
            continue
        for rank, idx in enumerate(indices):
            suffix = _NUMERALS[rank] if rank < len(_NUMERALS) else str(rank + 1)
            _set_archetype_name(archetypes[idx], f"{name} {suffix}", idx)


def _generate_negative_hints(present_categories):
    """Generate negative hints from categories NOT in this cluster."""
    anti_affinity = {
        "environment": ["studio"],
        "terrain": ["indoor", "studio", "portrait"],
        "sky": ["indoor", "studio"],
        "clothing": ["landscape", "wildlife"],
        "pose": ["landscape", "automotive"],
        "vehicle": ["portrait", "nude"],
        "animal": ["portrait", "fashion", "clothing"],
    }

    hints = set()
    for cat in present_categories:
        for anti_cat, anti_tags in anti_affinity.items():
            if cat != anti_cat and anti_cat not in present_categories:
                hints.update(anti_tags)

    return sorted(hints)[:10]


def _generate_flat_archetype(elements):
    """Single flat archetype for very small libraries."""
    categories = set()
    tag_map = {}
    element_ids = []

    for elem in elements:
        cat = elem.get("category", "unknown")
        categories.add(cat)
        tag_key = f"{cat}_tags"
        if tag_key not in tag_map:
            tag_map[tag_key] = set()
        for tag in elem.get("tags", []):
            tag_map[tag_key].add(tag)
        element_ids.append(elem.get("id", ""))

    compatible = {k: sorted(v) for k, v in tag_map.items()}

    return [{
        "id": "all_elements",
        "name": "All Elements",
        "compatible": compatible,
        "element_ids": element_ids,
        "photo_count": len(_group_by_photo(elements)),
        "negative_hints": [],
        "generated": date.today().isoformat(),
    }]
