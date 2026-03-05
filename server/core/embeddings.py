"""Embedding layer for tag normalization and description deduplication.

Uses sentence-transformers (all-MiniLM-L6-v2, ~80MB) for lightweight
semantic similarity. Provides:
- Tag normalization: maps new tags to canonical forms at >0.85 similarity
- Description dedup: rejects elements whose descriptions are >0.90 similar
  to existing elements in the library
- Element embedding for DBSCAN clustering (archetype generation)

Embedding cache is backed by SQLite BLOB storage.
"""

import logging
import threading

import numpy as np

from . import database, library_manager

log = logging.getLogger("prompt808.embeddings")

# Module-level singleton for the embedding model
_model = None
_model_lock = threading.Lock()

# Thresholds (tunable)
TAG_SIMILARITY_THRESHOLD = 0.85
DEDUP_SIMILARITY_THRESHOLD = 0.90

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_model():
    """Lazy-load the sentence-transformer model (thread-safe)."""
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            log.info("Loading embedding model %s...", DEFAULT_MODEL_NAME)
            _model = SentenceTransformer(DEFAULT_MODEL_NAME)
            log.info("Embedding model loaded")
        return _model


def unload_model():
    """Free the embedding model from memory."""
    import gc

    global _model
    _model = None
    gc.collect()
    try:
        import comfy.model_management
        comfy.model_management.soft_empty_cache()
    except ImportError:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    log.info("Embedding model unloaded")


def embed_text(text):
    """Embed a single text string. Returns numpy array."""
    model = _get_model()
    return model.encode(text, convert_to_numpy=True)


def embed_texts(texts):
    """Embed a batch of text strings. Returns numpy array of shape (N, dim)."""
    if not texts:
        return np.array([])
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(embeddings):
    """Compute pairwise cosine similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    return np.dot(normalized, normalized.T)


# --- Tag Normalization ---


def normalize_tag(tag, existing_tags, existing_embeddings=None):
    """Check if a tag should map to an existing canonical form.

    Args:
        tag: The new tag to check.
        existing_tags: List of existing canonical tag strings.
        existing_embeddings: Optional pre-computed embeddings for existing_tags.

    Returns:
        (canonical_form, similarity) if a match is found above threshold,
        (tag, 0.0) if the tag is novel.
    """
    if not existing_tags:
        return tag, 0.0

    tag_embedding = embed_text(tag)
    if existing_embeddings is None:
        existing_embeddings = embed_texts(existing_tags)

    best_sim = 0.0
    best_match = tag
    for i, existing in enumerate(existing_tags):
        sim = cosine_similarity(tag_embedding, existing_embeddings[i])
        if sim > best_sim:
            best_sim = sim
            best_match = existing

    if best_sim >= TAG_SIMILARITY_THRESHOLD:
        log.debug("Tag '%s' normalized to '%s' (sim=%.3f)", tag, best_match, best_sim)
        return best_match, best_sim

    return tag, 0.0


def normalize_tags(tags, existing_tags, existing_embeddings=None):
    """Normalize a list of tags against existing vocabulary.

    Args:
        tags: List of new tags to normalize.
        existing_tags: List of existing canonical tag strings.
        existing_embeddings: Optional pre-computed embeddings for existing_tags.

    Returns list of (possibly remapped) tags.
    """
    if not existing_tags:
        return tags

    result = []
    if existing_embeddings is None:
        existing_embeddings = embed_texts(existing_tags)
    tag_embeddings = embed_texts(tags)

    for i, tag in enumerate(tags):
        best_sim = 0.0
        best_match = tag
        for j, existing in enumerate(existing_tags):
            sim = cosine_similarity(tag_embeddings[i], existing_embeddings[j])
            if sim > best_sim:
                best_sim = sim
                best_match = existing

        if best_sim >= TAG_SIMILARITY_THRESHOLD:
            result.append(best_match)
        else:
            result.append(tag)
    return result


# --- Description Deduplication ---


def is_duplicate(desc, existing_descs, threshold=None, existing_embeddings=None):
    """Check if a description is too similar to any existing description.

    Args:
        desc: New element description to check.
        existing_descs: List of existing description strings.
        threshold: Similarity threshold (default: DEDUP_SIMILARITY_THRESHOLD).
        existing_embeddings: Optional pre-computed embeddings for existing_descs.

    Returns:
        (is_dup, most_similar_desc, similarity) tuple.
    """
    if threshold is None:
        threshold = DEDUP_SIMILARITY_THRESHOLD
    if not existing_descs:
        return False, None, 0.0

    desc_embedding = embed_text(desc)
    if existing_embeddings is None:
        existing_embeddings = embed_texts(existing_descs)

    best_sim = 0.0
    best_match = None
    for i, existing in enumerate(existing_descs):
        sim = cosine_similarity(desc_embedding, existing_embeddings[i])
        if sim > best_sim:
            best_sim = sim
            best_match = existing

    is_dup = best_sim >= threshold
    if is_dup:
        log.debug("Duplicate detected: '%.50s...' similar to '%.50s...' (sim=%.3f)",
                  desc, best_match, best_sim)
    return is_dup, best_match, best_sim


# --- Embedding Cache (SQLite BLOB) ---


def clear_cache():
    """Delete all cached embeddings. Returns count removed."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        n = db.execute(
            "SELECT COUNT(*) as cnt FROM embeddings_cache WHERE library_id=?",
            (lib_id,)
        ).fetchone()["cnt"]
        db.execute("DELETE FROM embeddings_cache WHERE library_id=?", (lib_id,))
        db.commit()
    return n


def get_cached_embedding(element_id):
    """Get a cached embedding for an element by ID. Returns numpy array or None."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    row = db.execute(
        "SELECT embedding FROM embeddings_cache WHERE library_id=? AND element_id=?",
        (lib_id, element_id)
    ).fetchone()
    if row and row["embedding"]:
        return np.frombuffer(row["embedding"], dtype=np.float32).copy()
    return None


def cache_embedding(element_id, embedding):
    """Cache an element's embedding."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    blob = embedding.astype(np.float32).tobytes()
    with lock:
        db.execute(
            "INSERT OR REPLACE INTO embeddings_cache (element_id, library_id, embedding) VALUES (?,?,?)",
            (element_id, lib_id, blob)
        )
        db.commit()


def cache_embeddings(id_embedding_pairs):
    """Cache multiple embeddings at once."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        for element_id, embedding in id_embedding_pairs:
            blob = embedding.astype(np.float32).tobytes()
            db.execute(
                "INSERT OR REPLACE INTO embeddings_cache (element_id, library_id, embedding) VALUES (?,?,?)",
                (element_id, lib_id, blob)
            )
        db.commit()


def get_all_cached_embeddings():
    """Return dict of {element_id: numpy_array} for all cached embeddings."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    rows = db.execute(
        "SELECT element_id, embedding FROM embeddings_cache WHERE library_id=?",
        (lib_id,)
    ).fetchall()
    return {
        r["element_id"]: np.frombuffer(r["embedding"], dtype=np.float32).copy()
        for r in rows
    }
