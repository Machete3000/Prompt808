"""Library manager — multi-library support for Prompt808.

Manages isolated libraries backed by SQLite. Each library has its own
elements, archetypes, vocabulary, embeddings, style profiles, and prompt
cache — all scoped by library_id in the database.

Thumbnails remain on disk as image files under:
    user_data/libraries/<name>/thumbnails/
"""

import logging
import re
import shutil
from contextvars import ContextVar
from pathlib import Path

from . import database

log = logging.getLogger("prompt808.library_manager")

_BASE_DIR = Path(__file__).resolve().parent.parent.parent / "user_data"
_LIBRARIES_DIR = _BASE_DIR / "libraries"

_active_library = None
_request_library: ContextVar[str | None] = ContextVar("request_library", default=None)
_request_library_id: ContextVar[tuple | None] = ContextVar("request_library_id", default=None)

# Valid library name: 1-50 chars, alphanumeric + spaces/dashes/underscores/parens
_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9 _()\-]{1,50}$")


def _validate_name(name):
    """Validate a library name. Returns trimmed name or raises ValueError."""
    if not isinstance(name, str):
        raise ValueError("Library name must be a string")
    name = name.strip()
    if not name:
        raise ValueError("Library name cannot be empty")
    if not _NAME_PATTERN.match(name):
        raise ValueError(
            "Library name must be 1-50 characters, using only letters, "
            "numbers, spaces, dashes, underscores, and parentheses"
        )
    return name


def _effective_library():
    """Return the library name for the current context (request-scoped or global)."""
    return _request_library.get() or _active_library


def get_library_id() -> int:
    """Return the integer PK for the current library context.

    Once resolved, the (name, ID) pair is cached in a ContextVar for the
    duration of the request. This makes in-flight requests resilient to
    concurrent library renames — after the first resolution, the cached
    integer ID is returned directly without a name lookup.  The cache is
    invalidated automatically when the effective library name changes.
    """
    name = _effective_library()
    cached = _request_library_id.get()
    if cached is not None:
        cached_name, cached_id = cached
        if cached_name == name:
            return cached_id
    db = database.get_db()
    row = db.execute(
        "SELECT id FROM libraries WHERE LOWER(name)=LOWER(?)", (name,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Library '{name}' not found in database")
    lib_id = row["id"]
    _request_library_id.set((name, lib_id))
    return lib_id


def get_data_dir():
    """Return the data directory for the active library, creating if needed."""
    path = _LIBRARIES_DIR / _effective_library()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_thumbnails_dir():
    """Return the thumbnails directory for the active library."""
    path = get_data_dir() / "thumbnails"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_active():
    """Return the name of the currently active library."""
    return _effective_library()


def set_request_library(name):
    """Set the library for the current request (context-scoped, not persisted).

    Only validates name format. DB existence is checked lazily by
    get_library_id() which runs inside asyncio.to_thread, keeping all
    database I/O off the event-loop thread.
    """
    name = _validate_name(name)
    _request_library.set(name)
    _request_library_id.set(None)  # Reset cached ID for new library scope


def set_active(name):
    """Switch the active library persistently. Validates that the library exists."""
    global _active_library
    name = _validate_name(name)
    db = database.get_db()
    lock = database.write_lock()

    with lock:
        row = db.execute(
            "SELECT id, name FROM libraries WHERE LOWER(name)=LOWER(?)", (name,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Library '{name}' does not exist")

        match = row["name"]
        db.execute("UPDATE libraries SET is_active=0")
        db.execute("UPDATE libraries SET is_active=1 WHERE id=?", (row["id"],))
        db.commit()
        _active_library = match

    log.info("Switched to library '%s'", match)


def list_libraries():
    """Return list of library info dicts with active flag and stats."""
    db = database.get_db()
    current = _effective_library()
    rows = db.execute("SELECT id, name, created_at FROM libraries ORDER BY id").fetchall()

    result = []
    for row in rows:
        lib_id = row["id"]
        name = row["name"]
        if not name:
            continue  # skip corrupted rows

        elem_count_row = db.execute(
            "SELECT COUNT(*) as cnt FROM elements WHERE library_id=?", (lib_id,)
        ).fetchone()
        element_count = elem_count_row["cnt"] if elem_count_row else 0

        # Count thumbnails on disk
        photo_count = 0
        try:
            thumbs_dir = _LIBRARIES_DIR / name / "thumbnails"
            if thumbs_dir.is_dir():
                photo_count = sum(1 for f in thumbs_dir.iterdir() if f.is_file())
        except OSError:
            pass

        result.append({
            "name": name,
            "active": name == current,
            "element_count": element_count,
            "photo_count": photo_count,
            "created_at": row["created_at"],
        })

    return result


def create_library(name):
    """Create a new library. Auto-activates if it's the first. Returns the validated name."""
    global _active_library
    name = _validate_name(name)
    db = database.get_db()
    lock = database.write_lock()

    with lock:
        existing = db.execute(
            "SELECT name FROM libraries WHERE LOWER(name)=LOWER(?)", (name,)
        ).fetchone()
        if existing is not None:
            raise ValueError(f"Library '{name}' already exists")

        # Auto-activate if this is the first library
        is_first = db.execute(
            "SELECT COUNT(*) as cnt FROM libraries"
        ).fetchone()["cnt"] == 0

        from datetime import datetime, timezone
        db.execute(
            "INSERT INTO libraries (name, is_active, created_at) VALUES (?, ?, ?)",
            (name, 1 if is_first else 0, datetime.now(timezone.utc).isoformat())
        )
        db.commit()

        if is_first:
            _active_library = name

    # Create directory for thumbnails
    lib_dir = _LIBRARIES_DIR / name
    lib_dir.mkdir(parents=True, exist_ok=True)

    log.info("Created library '%s'%s", name, " (auto-activated)" if is_first else "")
    return name


def rename_library(old_name, new_name):
    """Rename a library. Updates active reference if needed."""
    global _active_library
    old_name = _validate_name(old_name)
    new_name = _validate_name(new_name)

    if old_name.lower() == new_name.lower() and old_name == new_name:
        return new_name

    db = database.get_db()
    lock = database.write_lock()

    with lock:
        old_row = db.execute(
            "SELECT id, name FROM libraries WHERE LOWER(name)=LOWER(?)", (old_name,)
        ).fetchone()
        if old_row is None:
            raise ValueError(f"Library '{old_name}' does not exist")

        actual_old = old_row["name"]

        # Check new name doesn't conflict
        conflict = db.execute(
            "SELECT id FROM libraries WHERE LOWER(name)=LOWER(?) AND id!=?",
            (new_name, old_row["id"])
        ).fetchone()
        if conflict is not None:
            raise ValueError(f"Library '{new_name}' already exists")

        db.execute(
            "UPDATE libraries SET name=? WHERE id=?", (new_name, old_row["id"])
        )
        db.commit()

        # Rename directory on disk
        old_dir = _LIBRARIES_DIR / actual_old
        new_dir = _LIBRARIES_DIR / new_name
        if old_dir.exists():
            old_dir.rename(new_dir)

        if _active_library == actual_old:
            _active_library = new_name

    log.info("Renamed library '%s' -> '%s'", actual_old, new_name)
    return new_name


def delete_library(name):
    """Delete a library. If the last library is deleted, the system enters no-library state."""
    global _active_library
    name = _validate_name(name)
    db = database.get_db()
    lock = database.write_lock()

    with lock:
        row = db.execute(
            "SELECT id, name FROM libraries WHERE LOWER(name)=LOWER(?)", (name,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Library '{name}' does not exist")

        found = row["name"]
        lib_id = row["id"]

        # CASCADE deletes will clean up all related rows
        db.execute("DELETE FROM libraries WHERE id=?", (lib_id,))
        db.commit()

        # Remove directory on disk
        lib_dir = _LIBRARIES_DIR / found
        if lib_dir.exists():
            shutil.rmtree(lib_dir)

        # If deleted library was active, switch to first remaining or None
        if _active_library == found:
            first = db.execute("SELECT name FROM libraries ORDER BY id LIMIT 1").fetchone()
            if first:
                _active_library = first["name"]
                db.execute("UPDATE libraries SET is_active=0")
                db.execute(
                    "UPDATE libraries SET is_active=1 WHERE name=?",
                    (_active_library,)
                )
                db.commit()
            else:
                _active_library = None

    log.info("Deleted library '%s'", found)


def migrate_if_needed():
    """Handle startup: ensure DB schema exists and load active library if any."""
    global _active_library

    db = database.get_db()
    lock = database.write_lock()

    with lock:
        count = db.execute("SELECT COUNT(*) as cnt FROM libraries").fetchone()["cnt"]
        if count > 0:
            # Load active library
            active_row = db.execute(
                "SELECT name FROM libraries WHERE is_active=1"
            ).fetchone()
            if active_row:
                _active_library = active_row["name"]
            else:
                first = db.execute(
                    "SELECT name FROM libraries ORDER BY id LIMIT 1"
                ).fetchone()
                if first:
                    _active_library = first["name"]
            if _active_library:
                (_LIBRARIES_DIR / _active_library).mkdir(parents=True, exist_ok=True)
            log.info("Libraries ready (active: %s)", _active_library)
        else:
            _active_library = None
            log.info("No libraries — create one to get started")
