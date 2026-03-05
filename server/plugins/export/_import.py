"""Library import handler — always free.

Imports a .p808 zip file into a new or existing library.
The .p808 format contains:
- metadata.json: library name and export info
- {table}.json: JSON array of rows for each data table
- thumbnails/: directory of thumbnail image files
"""

import io
import json
import logging
import sqlite3
import zipfile

log = logging.getLogger("prompt808.plugins.export.import")

# Current format version — reject anything higher.
_FORMAT_VERSION = 1

# Valid column names per table — only these are accepted from imported data.
# Prevents SQL injection via crafted column names in untrusted .p808 files.
_VALID_COLUMNS = {
    "elements": {"id", "library_id", "desc", "category", "tags", "attributes",
                 "source_photo", "subject_type", "thumbnail", "medium",
                 "is_photograph", "extraction_type", "added"},
    "archetypes": {"id", "library_id", "name", "compatible", "element_ids",
                   "negative_hints", "generated"},
    "vocabulary": {"canonical", "library_id", "aliases", "count"},
    "style_profiles": {"genre", "library_id", "observations", "patterns",
                       "last_updated"},
    "embeddings_cache": {"element_id", "library_id", "embedding"},
    "image_embeddings_cache": {"content_hash", "library_id", "embedding"},
    "prompt_cache": {"cache_key", "library_id", "prompt", "negative_prompt",
                     "seed", "archetype_id", "style"},
}

# Required fields per table — rows missing these are skipped.
# Derived from schema NOT NULL + PRIMARY KEY constraints.
_REQUIRED_FIELDS = {
    "elements": ["id"],
    "archetypes": ["id"],
    "vocabulary": ["canonical"],
    "style_profiles": ["genre"],
    "prompt_cache": ["cache_key"],
}

# Primary key columns per table (excluding library_id which is always set).
_PRIMARY_KEYS = {
    "elements": ["id"],
    "archetypes": ["id"],
    "vocabulary": ["canonical"],
    "style_profiles": ["genre"],
    "prompt_cache": ["cache_key"],
    "embeddings_cache": ["element_id"],
    "image_embeddings_cache": ["content_hash"],
}

# Expected types for JSON TEXT columns — used for type validation/coercion.
_JSON_COLUMN_TYPES = {
    "elements": {"tags": list, "attributes": dict},
    "archetypes": {"compatible": dict, "element_ids": list, "negative_hints": list},
    "vocabulary": {"aliases": list},
    "style_profiles": {"patterns": dict},
}


def _deduplicate_name(name, library_manager):
    """If a library with *name* already exists, append (2), (3), … until unique.

    Respects the 50-char library name limit by truncating the base name if needed.
    """
    from ...core import database
    db = database.get_db()

    # Check if name is already free (case-insensitive, matching create_library)
    row = db.execute(
        "SELECT name FROM libraries WHERE LOWER(name)=LOWER(?)", (name,)
    ).fetchone()
    if row is None:
        return name

    # Collision — find the next available suffix
    for i in range(2, 100):
        suffix = f" ({i})"
        # Truncate base name to stay within 50-char limit
        max_base = 50 - len(suffix)
        candidate = name[:max_base] + suffix
        row = db.execute(
            "SELECT name FROM libraries WHERE LOWER(name)=LOWER(?)", (candidate,)
        ).fetchone()
        if row is None:
            return candidate

    # Extremely unlikely — 99 copies of the same name
    raise ValueError(f"Cannot deduplicate library name '{name}': too many copies")


def handle_import(file_data, target_library_name=None):
    """Import a .p808 zip file into a new or existing library.

    Args:
        file_data: Raw bytes of the .p808 zip file.
        target_library_name: Name for the imported library.
                            If None, uses the name from the export metadata.

    Returns:
        Dict with status, detailed counts, and warnings.
    """
    from ...core import database, library_manager

    try:
        zf = zipfile.ZipFile(io.BytesIO(file_data))
    except zipfile.BadZipFile:
        return {"status": "error", "message": "Invalid .p808 file (not a valid zip)"}

    with zf:
        # Read metadata
        try:
            meta_raw = zf.read("metadata.json")
            meta = json.loads(meta_raw)
        except (KeyError, json.JSONDecodeError):
            return {"status": "error", "message": "Invalid .p808 file (missing metadata)"}

        # Format version validation — backward compat: treat missing as version 1
        format_version = meta.get("format_version", 1)
        if format_version != _FORMAT_VERSION:
            return {
                "status": "error",
                "message": (f"Unsupported .p808 format version {format_version} "
                            f"(expected {_FORMAT_VERSION})"),
            }

        lib_name = target_library_name or meta.get("library_name", "imported")

        # Deduplicate: if a library with this name already exists, auto-suffix
        # with (2), (3), etc. to avoid merging unrelated data.
        lib_name = _deduplicate_name(lib_name, library_manager)

        # Create the new library
        try:
            library_manager.create_library(lib_name)
        except ValueError as e:
            return {"status": "error", "message": f"Failed to create library: {e}"}

        db = database.get_db()
        lock = database.write_lock()

        # Get the library ID
        row = db.execute("SELECT id FROM libraries WHERE name=?", (lib_name,)).fetchone()
        if row is None:
            return {"status": "error", "message": f"Failed to find/create library '{lib_name}'"}
        lib_id = row["id"]

        imported_counts = {}
        warnings = []

        with lock:
            # Import data tables from the zip
            for table_name in _VALID_COLUMNS:
                json_name = f"{table_name}.json"
                if json_name in zf.namelist():
                    try:
                        raw = json.loads(zf.read(json_name))
                        counts, table_warnings = _import_table_data(
                            db, lib_id, table_name, raw,
                        )
                        imported_counts[table_name] = counts
                        warnings.extend(table_warnings)
                    except Exception as e:
                        log.warning("Failed to import %s: %s", table_name, e)
                        warnings.append(f"{table_name}: import failed ({e})")

            # Referential integrity checks
            integrity_warnings = _check_referential_integrity(db, lib_id)
            warnings.extend(integrity_warnings)

            db.commit()

        # Import thumbnails with directory traversal protection
        thumb_count = 0
        thumbnails_dir = library_manager._LIBRARIES_DIR / lib_name / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)
        safe_prefix = str(thumbnails_dir.resolve())

        for name in zf.namelist():
            if name.startswith("thumbnails/") and not name.endswith("/"):
                filename = name.split("/", 1)[1]
                if not filename:
                    continue
                target_path = (thumbnails_dir / filename).resolve()
                if not target_path.is_relative_to(thumbnails_dir.resolve()):
                    log.warning("Skipping path traversal attempt in .p808 import: %s", name)
                    continue
                target_path.write_bytes(zf.read(name))
                thumb_count += 1

    log.info("Imported library '%s': %s, %d thumbnails, %d warnings",
             lib_name, imported_counts, thumb_count, len(warnings))
    result = {
        "status": "imported",
        "library_name": lib_name,
        "imported": imported_counts,
        "thumbnails": thumb_count,
    }
    if warnings:
        result["warnings"] = warnings
    return result


def _import_table_data(db, lib_id, table_name, rows):
    """Import rows into a table, scoped to lib_id.

    Only columns listed in _VALID_COLUMNS are accepted — unknown columns
    from the imported data are silently dropped. Uses INSERT OR REPLACE for
    conflict resolution (re-importing updates existing rows).

    Returns:
        Tuple of (counts_dict, warnings_list) where counts_dict has
        inserted/replaced/skipped keys.
    """
    counts = {"inserted": 0, "replaced": 0, "skipped": 0}
    table_warnings = []

    if not isinstance(rows, list):
        return counts, table_warnings

    valid_cols = _VALID_COLUMNS.get(table_name)
    if not valid_cols:
        return counts, table_warnings

    required = _REQUIRED_FIELDS.get(table_name, [])
    json_types = _JSON_COLUMN_TYPES.get(table_name, {})
    pk_cols = _PRIMARY_KEYS.get(table_name, [])

    # Load existing primary keys for this table+library to distinguish
    # inserts from replacements.
    existing_pks = set()
    if pk_cols:
        pk_col_str = ", ".join(pk_cols)
        try:
            existing_rows = db.execute(
                f"SELECT {pk_col_str} FROM {table_name} WHERE library_id=?",
                (lib_id,)
            ).fetchall()
            for r in existing_rows:
                existing_pks.add(tuple(r[c] for c in pk_cols))
        except sqlite3.Error:
            pass

    skip_reasons = {}

    for row in rows:
        if not isinstance(row, dict):
            counts["skipped"] += 1
            skip_reasons["not a dict"] = skip_reasons.get("not a dict", 0) + 1
            continue

        row["library_id"] = lib_id

        # Required field validation
        missing = [f for f in required if not row.get(f)]
        if missing:
            counts["skipped"] += 1
            reason = f"missing required field '{missing[0]}'"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

        # Filter to only valid columns
        cols = [c for c in row.keys() if c in valid_cols]
        if not cols:
            counts["skipped"] += 1
            continue

        # Re-serialize structured data (lists/dicts) to JSON strings for
        # SQLite TEXT columns, with type validation and coercion.
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (dict, list)):
                v = json.dumps(v)
            # Type validation for JSON columns
            if c in json_types:
                v = _validate_json_column(
                    v, json_types[c], table_name, c, table_warnings,
                )
            vals.append(v)

        # Check if this is an insert or replace
        pk_values = tuple(row.get(c) for c in pk_cols) if pk_cols else ()
        is_replace = pk_values in existing_pks

        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        try:
            db.execute(
                f"INSERT OR REPLACE INTO {table_name} ({col_names}) "
                f"VALUES ({placeholders})",
                vals,
            )
            if is_replace:
                counts["replaced"] += 1
            else:
                counts["inserted"] += 1
                if pk_values:
                    existing_pks.add(pk_values)
        except sqlite3.Error:
            counts["skipped"] += 1
            continue

    # Build summary warnings for skipped rows
    for reason, count in skip_reasons.items():
        table_warnings.append(f"{table_name}: {count} row(s) skipped ({reason})")

    return counts, table_warnings


def _validate_json_column(value, expected_type, table_name, col_name, warnings):
    """Validate and coerce a JSON column value to the expected type.

    Args:
        value: The current value (may be a JSON string or native Python object).
        expected_type: list or dict — the expected Python type after parsing.
        table_name: For warning messages.
        col_name: For warning messages.
        warnings: List to append warnings to (mutated in place).

    Returns:
        The value as a JSON string, coerced to the default if type is wrong.
    """
    default = [] if expected_type is list else {}

    # Parse if it's a string
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            warnings.append(
                f"{table_name}.{col_name}: invalid JSON, defaulted to "
                f"{json.dumps(default)}"
            )
            return json.dumps(default)

    # Check type
    if not isinstance(parsed, expected_type):
        warnings.append(
            f"{table_name}.{col_name}: expected {expected_type.__name__}, "
            f"got {type(parsed).__name__}, defaulted to {json.dumps(default)}"
        )
        return json.dumps(default)

    # Valid — ensure it's serialized as a string
    if isinstance(value, str):
        return value
    return json.dumps(parsed)


def _check_referential_integrity(db, lib_id):
    """Check referential integrity after import and fix issues.

    - archetypes.element_ids: remove references to non-existent elements
    - prompt_cache.archetype_id: warn about non-existent archetypes

    Returns:
        List of warning strings.
    """
    warnings = []

    # Load existing element IDs for this library
    element_rows = db.execute(
        "SELECT id FROM elements WHERE library_id=?", (lib_id,)
    ).fetchall()
    element_ids = {r["id"] for r in element_rows}

    # Check archetype element_ids
    archetypes = db.execute(
        "SELECT id, name, element_ids FROM archetypes WHERE library_id=?", (lib_id,)
    ).fetchall()

    total_cleaned = 0
    for arch in archetypes:
        raw_eids = arch["element_ids"]
        if not raw_eids:
            continue
        try:
            eids = json.loads(raw_eids) if isinstance(raw_eids, str) else raw_eids
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(eids, list):
            continue

        valid_eids = [eid for eid in eids if eid in element_ids]
        removed = len(eids) - len(valid_eids)
        if removed > 0:
            total_cleaned += removed
            db.execute(
                "UPDATE archetypes SET element_ids=? WHERE library_id=? AND id=?",
                (json.dumps(valid_eids), lib_id, arch["id"]),
            )

    if total_cleaned:
        warnings.append(
            f"archetypes: {total_cleaned} element_ids reference(s) cleaned "
            f"(elements not found)"
        )

    # Check prompt_cache archetype_id (warn only, don't modify)
    archetype_rows = db.execute(
        "SELECT id FROM archetypes WHERE library_id=?", (lib_id,)
    ).fetchall()
    archetype_ids = {r["id"] for r in archetype_rows}

    cache_rows = db.execute(
        "SELECT cache_key, archetype_id FROM prompt_cache WHERE library_id=?",
        (lib_id,),
    ).fetchall()

    dangling_cache = 0
    for cr in cache_rows:
        aid = cr["archetype_id"]
        if aid and aid != "Any" and aid not in archetype_ids:
            dangling_cache += 1

    if dangling_cache:
        warnings.append(
            f"prompt_cache: {dangling_cache} row(s) reference non-existent "
            f"archetype_id"
        )

    return warnings
