"""Element store — CRUD backed by SQLite.

Elements are the core data structure in Prompt808. Each element
represents a photographic attribute extracted from a user's reference
photo by a vision LLM.

Element schema:
{
    "id": "unique_slug",
    "desc": "flowing silk evening gown",
    "category": "clothing",
    "tags": ["elegant", "formal"],
    "attributes": {"level": "fashion"},
    "source_photo": "IMG_1234.jpg",
    "subject_type": "landscape",
    "thumbnail": "thumb_abc123.jpg",
    "medium": "photograph",
    "is_photograph": True,
    "extraction_type": "photo",
    "added": "2026-02-21"
}
"""

import json
import logging
from datetime import date

from ..core import database, library_manager

log = logging.getLogger("prompt808.store.elements")


def _row_to_dict(row):
    """Convert a sqlite3.Row to an element dict, parsing JSON columns."""
    d = dict(row)
    d["tags"] = json.loads(d["tags"]) if d.get("tags") else []
    d["attributes"] = json.loads(d["attributes"]) if d.get("attributes") else {}
    # Convert integer boolean back to Python bool
    if "is_photograph" in d and d["is_photograph"] is not None:
        d["is_photograph"] = bool(d["is_photograph"])
    # Remove library_id from the public dict
    d.pop("library_id", None)
    return d


def get_all():
    """Return all elements."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    rows = db.execute(
        "SELECT * FROM elements WHERE library_id=?", (lib_id,)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_by_category(category):
    """Return elements matching a specific category."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    rows = db.execute(
        "SELECT * FROM elements WHERE library_id=? AND category=?",
        (lib_id, category)
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_by_id(element_id):
    """Return a single element by ID, or None."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    row = db.execute(
        "SELECT * FROM elements WHERE library_id=? AND id=?",
        (lib_id, element_id)
    ).fetchone()
    return _row_to_dict(row) if row else None


def get_categories():
    """Return sorted list of unique category names in the library."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    rows = db.execute(
        "SELECT DISTINCT category FROM elements WHERE library_id=? AND category IS NOT NULL",
        (lib_id,)
    ).fetchall()
    return sorted(r["category"] for r in rows)


def add(element):
    """Add a single element. Auto-sets 'added' date if missing."""
    if "added" not in element:
        element["added"] = date.today().isoformat()

    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        db.execute(
            """INSERT INTO elements
               (id, library_id, desc, category, tags, attributes,
                source_photo, subject_type, thumbnail, medium,
                is_photograph, extraction_type, added)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                element.get("id"),
                lib_id,
                element.get("desc"),
                element.get("category"),
                json.dumps(element.get("tags", [])),
                json.dumps(element.get("attributes", {})),
                element.get("source_photo"),
                element.get("subject_type"),
                element.get("thumbnail"),
                element.get("medium"),
                1 if element.get("is_photograph", True) else 0,
                element.get("extraction_type", "photo"),
                element.get("added"),
            )
        )
        db.commit()
    return element


def add_many(new_elements):
    """Add multiple elements at once. Auto-sets 'added' date if missing."""
    today = date.today().isoformat()
    for e in new_elements:
        if "added" not in e:
            e["added"] = today

    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        db.executemany(
            """INSERT INTO elements
               (id, library_id, desc, category, tags, attributes,
                source_photo, subject_type, thumbnail, medium,
                is_photograph, extraction_type, added)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    e.get("id"),
                    lib_id,
                    e.get("desc"),
                    e.get("category"),
                    json.dumps(e.get("tags", [])),
                    json.dumps(e.get("attributes", {})),
                    e.get("source_photo"),
                    e.get("subject_type"),
                    e.get("thumbnail"),
                    e.get("medium"),
                    1 if e.get("is_photograph", True) else 0,
                    e.get("extraction_type", "photo"),
                    e.get("added"),
                )
                for e in new_elements
            ]
        )
        db.commit()
    return new_elements


def clear_all():
    """Delete all elements. Returns count of elements removed."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        n = db.execute(
            "SELECT COUNT(*) as cnt FROM elements WHERE library_id=?", (lib_id,)
        ).fetchone()["cnt"]
        db.execute("DELETE FROM elements WHERE library_id=?", (lib_id,))
        db.commit()
    return n


def delete(element_id):
    """Delete an element by ID. Also cleans up its embeddings_cache entry.

    Returns True if found and deleted.
    """
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        cursor = db.execute(
            "DELETE FROM elements WHERE library_id=? AND id=?",
            (lib_id, element_id)
        )
        if cursor.rowcount > 0:
            db.execute(
                "DELETE FROM embeddings_cache WHERE library_id=? AND element_id=?",
                (lib_id, element_id)
            )
        db.commit()
    return cursor.rowcount > 0


_UPDATABLE_COLUMNS = {"desc", "tags", "attributes", "is_photograph"}


def update(element_id, updates):
    """Update fields on an element. Returns updated element or None."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        # Build SET clause from updates dict
        set_parts = []
        params = []
        for key, value in updates.items():
            if key not in _UPDATABLE_COLUMNS:
                continue
            if key == "tags":
                set_parts.append(f"{key}=?")
                params.append(json.dumps(value))
            elif key == "attributes":
                set_parts.append(f"{key}=?")
                params.append(json.dumps(value))
            elif key == "is_photograph":
                set_parts.append(f"{key}=?")
                params.append(1 if value else 0)
            else:
                set_parts.append(f"{key}=?")
                params.append(value)

        if not set_parts:
            return None

        params.extend([lib_id, element_id])
        db.execute(
            f"UPDATE elements SET {', '.join(set_parts)} WHERE library_id=? AND id=?",
            params
        )
        db.commit()

        # Return the updated element
        row = db.execute(
            "SELECT * FROM elements WHERE library_id=? AND id=?",
            (lib_id, element_id)
        ).fetchone()

    return _row_to_dict(row) if row else None


def delete_by_thumbnail(thumbnail):
    """Delete all elements associated with a thumbnail. Also cleans up embeddings.

    Returns count of elements removed.
    """
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    lock = database.write_lock()

    with lock:
        # Collect element IDs before deleting so we can clean up embeddings
        rows = db.execute(
            "SELECT id FROM elements WHERE library_id=? AND thumbnail=?",
            (lib_id, thumbnail)
        ).fetchall()
        element_ids = [r["id"] for r in rows]

        cursor = db.execute(
            "DELETE FROM elements WHERE library_id=? AND thumbnail=?",
            (lib_id, thumbnail)
        )
        # Clean up embeddings for deleted elements
        for eid in element_ids:
            db.execute(
                "DELETE FROM embeddings_cache WHERE library_id=? AND element_id=?",
                (lib_id, eid)
            )
        db.commit()
    return cursor.rowcount


def count():
    """Return total number of elements."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    return db.execute(
        "SELECT COUNT(*) as cnt FROM elements WHERE library_id=?", (lib_id,)
    ).fetchone()["cnt"]


def count_by_category():
    """Return dict of {category: count}."""
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    rows = db.execute(
        "SELECT category, COUNT(*) as cnt FROM elements WHERE library_id=? GROUP BY category",
        (lib_id,)
    ).fetchall()
    return {r["category"]: r["cnt"] for r in rows}


def get_library_version():
    """Return a version string based on element count + max rowid.

    Used for prompt cache invalidation.
    """
    db = database.get_db()
    lib_id = library_manager.get_library_id()
    row = db.execute(
        "SELECT COUNT(*) as cnt, MAX(rowid) as max_rowid FROM elements WHERE library_id=?",
        (lib_id,)
    ).fetchone()
    cnt = row["cnt"]
    max_rowid = row["max_rowid"] or 0
    if cnt == 0:
        return "empty"
    return f"{cnt}_{max_rowid}"
