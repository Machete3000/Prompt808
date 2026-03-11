"""Coherence checking utilities for element selection.

Color harmony validation used during element selection to prevent
clashing combinations (e.g., neon cyberpunk palette + pastoral meadow).
"""

import re


def has_tag_overlap(entry_tags, required_tags):
    """Check if an entry has at least one tag in common with required tags."""
    return bool(set(entry_tags) & set(required_tags))


# --- Color Harmony ---

# Color words grouped by temperature/family for cross-element clash detection.
_WARM_COLORS = {
    "red", "orange", "amber", "gold", "golden", "yellow", "rust", "burgundy",
    "rose", "pink", "magenta", "ochre", "sienna", "terracotta", "coral",
    "copper", "bronze", "scarlet", "crimson", "salmon", "vermillion",
    "warm", "fire", "flame", "sunlit", "sunbaked",
}
_COOL_COLORS = {
    "blue", "cyan", "teal", "turquoise", "purple", "lavender", "violet",
    "indigo", "emerald", "cobalt", "sapphire", "aqua", "ice", "icy",
    "frost", "cool", "arctic", "glacial", "cerulean", "periwinkle",
}
_NEON_COLORS = {
    "neon", "fluorescent", "electric", "vivid", "glowing", "rgb",
    "holographic", "laser", "prismatic", "rainbow",
}
_EARTH_COLORS = {
    "earth", "earthy", "ochre", "sienna", "umber", "khaki", "olive",
    "sage", "tan", "beige", "brown", "rustic", "muted", "faded",
    "sepia", "vintage",
}
_MONO_COLORS = {
    "monochrome", "grayscale", "desaturated", "muted", "black",
    "white", "gray", "grey", "charcoal", "silver",
}

# Color clash rules: (palette_signals, lighting_signals, environment_signals)
# A clash occurs when palette matches AND (lighting OR environment) matches.
_COLOR_CLASH_RULES = [
    # Monochrome palette + neon/colorful lighting
    {
        "palette": {"monochrome", "bw", "black_white", "grayscale", "desaturated"},
        "lighting": {"neon", "rainbow", "colorful", "rgb", "multicolor", "prismatic"},
        "environment": None,
    },
    # Urban/noir palette + pastoral/nature environment
    {
        "palette": {"noir", "urban", "neon", "cyberpunk", "gritty"},
        "lighting": None,
        "environment": {"pastoral", "meadow", "wildflower", "garden", "botanical",
                        "countryside", "rustic", "cottage", "farmland"},
    },
    # Desert/harsh warm palette + snow/ice environment
    {
        "palette": {"desert", "warm", "arid", "sand", "terracotta", "sunbaked"},
        "lighting": None,
        "environment": {"snow", "ice", "frozen", "glacier", "arctic", "winter",
                        "frost", "blizzard", "snow_ice", "cold"},
    },
    # Cold palette + desert environment
    {
        "palette": {"cold", "icy", "arctic", "frozen", "frost", "winter", "glacial"},
        "lighting": None,
        "environment": {"desert", "sand", "dune", "arid", "sahara", "oasis",
                        "canyon", "mesa", "badlands"},
    },
    # Magical/bioluminescent palette + mundane indoor environment
    {
        "palette": {"bioluminescent", "magical", "ethereal_glow", "enchanted",
                    "otherworldly", "fairy", "mystic"},
        "lighting": None,
        "environment": {"office", "loft", "apartment", "kitchen", "bathroom",
                        "modern", "corporate", "mundane", "domestic", "cubicle"},
    },
    # Vintage/sepia palette + neon/futuristic lighting
    {
        "palette": {"vintage", "sepia", "retro", "faded", "antique", "daguerreotype",
                    "old_film", "nostalgic"},
        "lighting": {"neon", "futuristic", "cyberpunk", "laser", "led", "holographic",
                     "rgb", "synth"},
        "environment": None,
    },
    # Tropical/bold palette + dark/noir environment
    {
        "palette": {"tropical", "bold", "vibrant", "saturated", "vivid", "carnival",
                    "fiesta", "pop"},
        "lighting": None,
        "environment": {"noir", "gritty", "dark_alley", "dystopian", "abandoned",
                        "derelict", "shadowy", "bleak", "grimy"},
    },
]

# Text-level clash rules between element pairs.
_TEXT_CLASH_RULES = [
    ("clothing", "neon", "palette", "mono"),
    ("clothing", "warm", "environment", "cool"),
    ("clothing", "cool", "environment", "warm"),
    ("clothing", "neon", "palette", "earth"),
    ("clothing", "earth", "lighting", "neon"),
    ("palette", "mono", "lighting", "neon"),
    ("palette", "warm", "environment", "cool"),
    ("palette", "cool", "environment", "warm"),
    ("environment", "neon", "palette", "earth"),
    ("clothing", "neon", "environment", "cool"),
    ("makeup", "warm", "palette", "mono"),
    ("makeup", "neon", "palette", "earth"),
]

_COLOR_WORD_RE = re.compile(r'\b[a-z]+\b')


def get_element_signals(element):
    """Extract a signal set from an element's tags, id, and cat/category."""
    if not element:
        return set()
    signals = set(element.get("tags", []))
    signals |= set(element["id"].split("_")) if "id" in element else set()
    if "cat" in element:
        signals.add(element["cat"])
    if "category" in element:
        signals.add(element["category"])
    return signals


def _extract_desc_colors(element):
    """Extract color family sets from an element's description text."""
    if not element:
        return set(), set(), set(), set(), set()
    desc = element.get("desc", "").lower()
    desc_bare = element.get("desc_bare", "").lower()
    text = f"{desc} {desc_bare}"

    words = set(_COLOR_WORD_RE.findall(text))
    word_list = _COLOR_WORD_RE.findall(text)
    for i in range(len(word_list) - 1):
        words.add(f"{word_list[i]}_{word_list[i+1]}")

    warm = words & _WARM_COLORS
    cool = words & _COOL_COLORS
    neon = words & _NEON_COLORS
    earth = words & _EARTH_COLORS
    mono = words & _MONO_COLORS
    return warm, cool, neon, earth, mono


def check_color_harmony(elements):
    """Check whether selected elements form a coherent color combination.

    Phase 1: Tag/signal-level checks against structural clash rules.
    Phase 2: Text-level checks extracting color words from descriptions.

    Args:
        elements: Dict mapping category names to element dicts.
                  Must include at least 'palette'. Other keys checked
                  if present: 'lighting', 'environment', 'clothing', 'makeup'.

    Returns:
        True if harmonious (no clashes), False if a clash is detected.
    """
    # Phase 1: Tag-level checks
    palette_signals = get_element_signals(elements.get("palette"))
    lighting_signals = get_element_signals(elements.get("lighting"))
    env_signals = get_element_signals(elements.get("environment"))

    for rule in _COLOR_CLASH_RULES:
        if not (palette_signals & rule["palette"]):
            continue
        if rule["lighting"] is not None and (lighting_signals & rule["lighting"]):
            return False
        if rule["environment"] is not None and (env_signals & rule["environment"]):
            return False

    # Phase 2: Text-level color checks
    color_cache = {}
    family_index = {"warm": 0, "cool": 1, "neon": 2, "earth": 3, "mono": 4}

    def _get_family(element_key, family_name):
        if element_key not in color_cache:
            color_cache[element_key] = _extract_desc_colors(elements.get(element_key))
        return color_cache[element_key][family_index[family_name]]

    for elem_a, fam_a, elem_b, fam_b in _TEXT_CLASH_RULES:
        colors_a = _get_family(elem_a, fam_a)
        colors_b = _get_family(elem_b, fam_b)
        if colors_a and colors_b:
            return False

    return True
