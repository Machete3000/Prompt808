"""Image embedding layer for photo-level duplicate detection.

Uses CLIP (openai/clip-vit-base-patch32, ~350MB) to embed photos and detect
near-identical uploads before running expensive QwenVL inference.

Follows the same patterns as embeddings.py: lazy-loaded singleton model,
thread-safe. Cache backed by SQLite BLOB storage.
"""

import gc
import logging
import threading

import numpy as np

from . import database, library_manager

log = logging.getLogger("prompt808.image_embeddings")

# Module-level singletons
_model = None
_processor = None
_model_lock = threading.Lock()

# Similarity threshold — high to only catch near-identical photos
PHOTO_DEDUP_THRESHOLD = 0.95

DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"


def _get_model():
    """Lazy-load the CLIP model and processor (thread-safe)."""
    global _model, _processor
    with _model_lock:
        if _model is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            log.info("Loading image embedding model %s...", DEFAULT_MODEL_NAME)
            _processor = CLIPProcessor.from_pretrained(DEFAULT_MODEL_NAME, use_fast=True)
            _model = CLIPModel.from_pretrained(DEFAULT_MODEL_NAME)

            if torch.cuda.is_available():
                _model = _model.to("cuda")

            _model.eval()
            log.info("Image embedding model loaded")
        return _model, _processor


def unload_model():
    """Free the CLIP model from memory."""
    global _model, _processor
    _model = None
    _processor = None
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
    log.info("Image embedding model unloaded")


def embed_image(image_path):
    """Embed a single image. Returns normalized 512-dim numpy array."""
    import torch
    from PIL import Image

    model, processor = _get_model()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    # Normalize to unit vector
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten()


def is_duplicate_photo(image_path, content_hash, threshold=None):
    """Check if a photo is a duplicate of any previously analyzed photo.

    Args:
        image_path: Path to the image file.
        content_hash: MD5 hex digest of the image content.
        threshold: Similarity threshold (default: PHOTO_DEDUP_THRESHOLD).

    Returns:
        (is_dup, best_match_hash, similarity) tuple.
    """
    if threshold is None:
        threshold = PHOTO_DEDUP_THRESHOLD

    db = database.get_db()
    lib_id = library_manager.get_library_id()

    # Exact hash match — identical file
    row = db.execute(
        "SELECT content_hash FROM image_embeddings_cache WHERE library_id=? AND content_hash=?",
        (lib_id, content_hash)
    ).fetchone()
    if row:
        log.info("Exact duplicate photo detected (hash: %s)", content_hash)
        return True, content_hash, 1.0

    # Load all cached embeddings
    rows = db.execute(
        "SELECT content_hash, embedding FROM image_embeddings_cache WHERE library_id=?",
        (lib_id,)
    ).fetchall()
    if not rows:
        return False, None, 0.0

    # Compute embedding and compare against all cached
    new_embedding = embed_image(image_path)

    best_sim = 0.0
    best_hash = None
    for r in rows:
        cached_arr = np.frombuffer(r["embedding"], dtype=np.float32).copy()
        sim = float(np.dot(new_embedding, cached_arr))
        if sim > best_sim:
            best_sim = sim
            best_hash = r["content_hash"]

    is_dup = best_sim >= threshold
    if is_dup:
        log.info("Duplicate photo detected: similarity %.3f to %s", best_sim, best_hash)
    return is_dup, best_hash, best_sim


def register_photo(content_hash, image_path):
    """Register a photo's embedding in the cache after successful analysis."""
    embedding = embed_image(image_path)
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    blob = embedding.astype(np.float32).tobytes()
    with lock:
        db.execute(
            "INSERT OR REPLACE INTO image_embeddings_cache (content_hash, library_id, embedding) VALUES (?,?,?)",
            (content_hash, lib_id, blob)
        )
        db.commit()
    log.info("Photo registered: %s", content_hash)


# --- Cache management ---


def clear_cache():
    """Delete all cached image embeddings. Returns count removed."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        n = db.execute(
            "SELECT COUNT(*) as cnt FROM image_embeddings_cache WHERE library_id=?",
            (lib_id,)
        ).fetchone()["cnt"]
        db.execute("DELETE FROM image_embeddings_cache WHERE library_id=?", (lib_id,))
        db.commit()
    return n


def remove_by_hash_prefix(prefix):
    """Remove cached embeddings whose content hash starts with prefix.

    Returns number of entries removed.
    """
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        cursor = db.execute(
            "DELETE FROM image_embeddings_cache WHERE library_id=? AND content_hash LIKE ?",
            (lib_id, f"{prefix}%")
        )
        db.commit()
    return cursor.rowcount


def count():
    """Return number of cached photo embeddings."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    return db.execute(
        "SELECT COUNT(*) as cnt FROM image_embeddings_cache WHERE library_id=?",
        (lib_id,)
    ).fetchone()["cnt"]
