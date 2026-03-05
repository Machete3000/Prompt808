"""Prompt cache for seed determinism.

Caches generated prompts keyed by a hash of all inputs that affect output.
Same inputs = instant return, no LLM call needed. Invalidates when any
input changes or new data is ingested (library_version changes).

Backed by SQLite.
"""

import hashlib
import json
import logging

from . import database, library_manager

log = logging.getLogger("prompt808.prompt_cache")


def _make_key(seed, archetype_id, style, mood, model_name, quantization,
              library_version, enrichment="Vivid", temperature=0.7):
    """Create a deterministic hash key from all generation inputs."""
    raw = json.dumps({
        "seed": seed,
        "archetype_id": archetype_id,
        "style": style,
        "mood": mood,
        "model_name": model_name,
        "quantization": quantization,
        "library_version": library_version,
        "enrichment": enrichment,
        "temperature": temperature,
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get(seed, archetype_id, style, mood, model_name, quantization, library_version,
        enrichment="Vivid", temperature=0.7):
    """Look up a cached prompt. Returns (prompt, negative_prompt) or None."""
    key = _make_key(seed, archetype_id, style, mood, model_name, quantization,
                    library_version, enrichment, temperature)
    db = database.get_db()
    lib_id = library_manager.get_library_id()

    row = db.execute(
        "SELECT prompt, negative_prompt FROM prompt_cache WHERE library_id=? AND cache_key=?",
        (lib_id, key)
    ).fetchone()
    if row:
        log.info("Cache hit for key %s", key)
        return row["prompt"], row["negative_prompt"]
    return None


def put(seed, archetype_id, style, mood, model_name, quantization, library_version,
        prompt, negative_prompt, enrichment="Vivid", temperature=0.7):
    """Store a generated prompt in the cache."""
    key = _make_key(seed, archetype_id, style, mood, model_name, quantization,
                    library_version, enrichment, temperature)
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        db.execute(
            """INSERT OR REPLACE INTO prompt_cache
               (cache_key, library_id, prompt, negative_prompt, seed, archetype_id, style)
               VALUES (?,?,?,?,?,?,?)""",
            (key, lib_id, prompt, negative_prompt, seed, archetype_id, style)
        )
        db.commit()
    log.info("Cached prompt for key %s", key)


def invalidate():
    """Clear the entire prompt cache (called when library changes)."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        db.execute("DELETE FROM prompt_cache WHERE library_id=?", (lib_id,))
        db.commit()
    log.info("Prompt cache invalidated")


def size():
    """Return number of cached entries."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    return db.execute(
        "SELECT COUNT(*) as cnt FROM prompt_cache WHERE library_id=?", (lib_id,)
    ).fetchone()["cnt"]
