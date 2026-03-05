"""Aiohttp routes for ComfyUI's PromptServer.

Registers all Prompt808 API endpoints under /prompt808/api/*.
This replaces the FastAPI routers when running inside ComfyUI.
"""

import atexit
import asyncio
import hashlib
import json
import logging
import shutil
import tempfile
import threading
from pathlib import Path

from aiohttp import web

log = logging.getLogger("prompt808.routes")

# Cancellation signal — set on server shutdown so long-running analysis
# threads abort between pipeline stages instead of blocking exit.
_shutdown_event = threading.Event()
atexit.register(_shutdown_event.set)

# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

try:
    from server import PromptServer  # ComfyUI's server module
    routes = PromptServer.instance.routes
    _HAS_PROMPT_SERVER = True

    # Register shutdown hook so Ctrl+C sets the cancellation flag immediately,
    # before atexit runs (atexit fires too late — threads are already joining).
    async def _on_shutdown(_app):
        _shutdown_event.set()
        log.info("Prompt808: shutdown signal sent to analysis threads")

    PromptServer.instance.app.on_shutdown.append(_on_shutdown)
except Exception:
    # Not running inside ComfyUI — skip route registration
    routes = None
    _HAS_PROMPT_SERVER = False
    log.info("PromptServer not available — skipping aiohttp route registration")


def _get_library(request):
    """Read X-Library header and scope the request to that library.

    Raises HTTPBadRequest if X-Library header is invalid.
    Raises HTTPServiceUnavailable if no library exists (fresh install).
    """
    from .core import library_manager
    name = request.headers.get("X-Library")
    if name:
        try:
            library_manager.set_request_library(name)
        except ValueError as e:
            raise web.HTTPBadRequest(text=f"Invalid X-Library header: {e}")
    if library_manager.get_active() is None:
        raise web.HTTPServiceUnavailable(
            text="No library exists — open the Prompt808 sidebar to create one")
    return name


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

if _HAS_PROMPT_SERVER:

    @routes.get("/prompt808/api/health")
    async def health(request):
        # Health works even without a library — don't use _get_library()
        from .core import library_manager
        name = request.headers.get("X-Library")
        if name:
            try:
                library_manager.set_request_library(name)
            except ValueError:
                pass

        def _sync():
            active = library_manager.get_active()
            if active is None:
                return {"status": "ok", "elements": 0, "archetypes": 0,
                        "vocabulary": 0, "library": None}
            from .store import archetypes, elements, vocabulary
            return {
                "status": "ok",
                "elements": elements.count(),
                "archetypes": archetypes.count(),
                "vocabulary": vocabulary.count(),
                "library": active,
            }

        return web.json_response(await asyncio.to_thread(_sync))

    # -----------------------------------------------------------------------
    # Generation — options for node dropdowns
    # -----------------------------------------------------------------------

    @routes.get("/prompt808/api/generate/options")
    async def generation_options(request):
        _get_library(request)

        def _sync():
            from .core import database
            from .core.generator import get_available_moods, get_available_styles
            from .core.model_manager import get_model_names
            from .store import archetypes

            # Read NSFW setting to filter adult content from dropdowns
            nsfw = False
            try:
                db = database.get_db()
                row = db.execute(
                    "SELECT value FROM generate_settings WHERE key='app'"
                ).fetchone()
                if row and row["value"]:
                    nsfw = json.loads(row["value"]).get("nsfw", False)
            except Exception as e:
                log.warning("Failed to read NSFW setting: %s", e)

            return {
                "prompt_types": get_available_styles(nsfw=nsfw),
                "moods": get_available_moods(nsfw=nsfw),
                "archetypes": ["Any"] + archetypes.get_names(),
                "models": get_model_names(),
            }

        return web.json_response(await asyncio.to_thread(_sync))

    # -----------------------------------------------------------------------
    # App-wide settings (ComfyUI settings dialog → SQLite)
    # -----------------------------------------------------------------------

    @routes.get("/prompt808/api/settings")
    async def get_app_settings(request):
        """Return app-wide settings (NSFW, debug, etc.)."""
        def _sync():
            from .core import database
            db = database.get_db()
            row = db.execute(
                "SELECT value FROM generate_settings WHERE key='app'"
            ).fetchone()
            if row and row["value"]:
                try:
                    return json.loads(row["value"])
                except Exception as e:
                    log.warning("Failed to parse app settings: %s", e)
            return {}

        return web.json_response(await asyncio.to_thread(_sync))

    @routes.put("/prompt808/api/settings")
    async def save_app_settings(request):
        """Persist app-wide settings to DB (merge with existing)."""
        body = await request.json()

        def _sync():
            from .core import database
            db = database.get_db()
            lock = database.write_lock()
            with lock:
                # Merge with existing settings so partial updates work
                row = db.execute(
                    "SELECT value FROM generate_settings WHERE key='app'"
                ).fetchone()
                existing = {}
                if row and row["value"]:
                    try:
                        existing = json.loads(row["value"])
                    except Exception:
                        pass
                existing.update(body)
                db.execute(
                    "INSERT OR REPLACE INTO generate_settings (key, value) VALUES ('app', ?)",
                    (json.dumps(existing),)
                )
                db.commit()

        await asyncio.to_thread(_sync)
        return web.json_response({"status": "saved"})

    # -----------------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------------

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif"}

    @routes.get("/prompt808/api/analyze/options")
    async def analyze_options(request):
        from .core.model_manager import get_vision_model_names
        return web.json_response({
            "vision_models": get_vision_model_names(),
            "quantizations": ["FP16", "FP8", "8-bit", "4-bit"],
        })

    @routes.post("/prompt808/api/analyze")
    async def analyze_photo(request):
        """Analyze a photo and extract photographic elements (SSE streaming)."""
        _get_library(request)
        from .core import library_manager

        reader = await request.multipart()
        image_data = None
        image_filename = "photo.jpg"
        vision_model = "Qwen3-VL-8B-Instruct"
        quantization = "FP8"
        device = "auto"
        attention_mode = "auto"
        max_tokens = 2048
        force = False

        while True:
            field = await reader.next()
            if field is None:
                break
            name = field.name
            if name == "image":
                image_filename = field.filename or image_filename
                image_data = await field.read()
            else:
                val = (await field.read()).decode()
                if name == "vision_model":
                    vision_model = val
                elif name == "quantization":
                    quantization = val
                elif name == "device":
                    device = val
                elif name == "attention_mode":
                    attention_mode = val
                elif name == "max_tokens":
                    try:
                        max_tokens = int(val)
                    except (ValueError, TypeError):
                        return web.Response(status=400, text="max_tokens must be an integer")
                    if not (256 <= max_tokens <= 4096):
                        return web.Response(status=400, text="max_tokens must be between 256 and 4096")
                elif name == "force":
                    force = val.lower() in ("true", "1", "yes")

        if image_data is None:
            return web.Response(status=400, text="No image uploaded")

        suffix = Path(image_filename).suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            return web.Response(status=400, text=f"Unsupported format '{suffix}'. Use: {SUPPORTED_FORMATS}")

        thumbnails_dir = library_manager.get_thumbnails_dir()

        # Sidebar runs outside ComfyUI's workflow pipeline — aggressively
        # free VRAM so analysis models can load after a workflow run.
        # IMPORTANT: Call synchronously on the event-loop thread — ComfyUI's
        # model patching internals are not thread-safe. Dispatching to a
        # thread pool (asyncio.to_thread) corrupts tensor state and causes
        # "Cannot set version_counter for inference tensor" on the next
        # workflow execution.
        _unload_comfy_models()

        resp = web.StreamResponse(
            status=200, reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)

        progress_queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def progress_cb(message):
            loop.call_soon_threadsafe(progress_queue.put_nowait, message)

        try:
            task = asyncio.create_task(asyncio.to_thread(
                _run_analysis, image_data, image_filename, suffix,
                thumbnails_dir, vision_model, quantization, device,
                attention_mode, max_tokens, force, progress_cb,
            ))

            while not task.done():
                try:
                    msg = await asyncio.wait_for(progress_queue.get(), timeout=0.3)
                    await _sse_write(resp, "progress", {"message": msg})
                except asyncio.TimeoutError:
                    continue

            # Drain any remaining progress messages
            while not progress_queue.empty():
                msg = progress_queue.get_nowait()
                await _sse_write(resp, "progress", {"message": msg})

            result = task.result()
            await _sse_write(resp, "result", result)
        except Exception as e:
            log.error("Analysis failed: %s", e, exc_info=True)
            await _sse_write(resp, "error", {"message": str(e)})

        await resp.write_eof()
        return resp

    @routes.post("/prompt808/api/analyze/cleanup")
    async def analysis_cleanup(request):
        result = await asyncio.to_thread(_run_analysis_cleanup)
        return web.json_response(result)

    # -----------------------------------------------------------------------
    # Library — Elements
    # -----------------------------------------------------------------------

    @routes.get("/prompt808/api/library/elements")
    async def list_elements(request):
        _get_library(request)
        category = request.query.get("category")
        try:
            limit = int(request.query.get("limit", "100"))
            offset = int(request.query.get("offset", "0"))
        except (ValueError, TypeError) as e:
            return web.Response(status=400, text=f"Invalid query parameter: {e}")

        def _sync():
            from .store import elements
            if category:
                all_elems = elements.get_by_category(category)
            else:
                all_elems = elements.get_all()
            total = len(all_elems)
            page = all_elems[offset:offset + limit]
            return {
                "elements": page, "total": total,
                "offset": offset, "limit": limit,
            }

        return web.json_response(await asyncio.to_thread(_sync))

    @routes.get("/prompt808/api/library/elements/{element_id}")
    async def get_element(request):
        _get_library(request)
        element_id = request.match_info["element_id"]

        def _sync():
            from .store import elements
            return elements.get_by_id(element_id)

        elem = await asyncio.to_thread(_sync)
        if not elem:
            return web.Response(status=404, text=f"Element '{element_id}' not found")
        return web.json_response(elem)

    @routes.delete("/prompt808/api/library/elements/{element_id}")
    async def delete_element(request):
        _get_library(request)
        element_id = request.match_info["element_id"]

        def _sync():
            from .store import elements
            if not elements.delete(element_id):
                return False
            try:
                from .core import prompt_cache
                prompt_cache.invalidate()
            except Exception as e:
                log.warning("Prompt cache invalidation failed: %s", e)
            _regen_archetypes_and_styles()
            return True

        found = await asyncio.to_thread(_sync)
        if not found:
            return web.Response(status=404, text=f"Element '{element_id}' not found")
        return web.json_response({"status": "deleted", "id": element_id})

    @routes.patch("/prompt808/api/library/elements/{element_id}")
    async def update_element(request):
        _get_library(request)
        element_id = request.match_info["element_id"]
        body = await request.json()
        updates = {k: v for k, v in body.items() if v is not None and k in ("desc", "tags", "attributes")}
        if not updates:
            return web.Response(status=400, text="No fields to update")

        def _sync():
            from .store import elements
            result = elements.update(element_id, updates)
            if not result:
                return None
            try:
                from .core import prompt_cache
                prompt_cache.invalidate()
            except Exception as e:
                log.warning("Prompt cache invalidation failed: %s", e)
            if "tags" in updates:
                _regen_archetypes_and_styles()
            return result

        result = await asyncio.to_thread(_sync)
        if result is None:
            return web.Response(status=404, text=f"Element '{element_id}' not found")
        return web.json_response(result)

    @routes.get("/prompt808/api/library/categories")
    async def list_categories(request):
        _get_library(request)

        def _sync():
            from .store import elements
            return {
                "categories": elements.get_categories(),
                "counts": elements.count_by_category(),
            }

        try:
            return web.json_response(await asyncio.to_thread(_sync))
        except ValueError:
            return web.json_response({"categories": [], "counts": {}})

    # -----------------------------------------------------------------------
    # Library — Archetypes
    # -----------------------------------------------------------------------

    @routes.get("/prompt808/api/library/archetypes")
    async def list_archetypes(request):
        _get_library(request)

        def _sync():
            from .store import archetypes
            return {"archetypes": archetypes.get_all()}

        return web.json_response(await asyncio.to_thread(_sync))

    @routes.get("/prompt808/api/library/archetypes/{archetype_id}")
    async def get_archetype(request):
        _get_library(request)
        archetype_id = request.match_info["archetype_id"]

        def _sync():
            from .store import archetypes
            return archetypes.get_by_id(archetype_id)

        arch = await asyncio.to_thread(_sync)
        if not arch:
            return web.Response(status=404, text=f"Archetype '{archetype_id}' not found")
        return web.json_response(arch)

    @routes.delete("/prompt808/api/library/archetypes/{archetype_id}")
    async def delete_archetype(request):
        _get_library(request)
        archetype_id = request.match_info["archetype_id"]

        def _sync():
            from .store import archetypes
            return archetypes.delete(archetype_id)

        found = await asyncio.to_thread(_sync)
        if not found:
            return web.Response(status=404, text=f"Archetype '{archetype_id}' not found")
        return web.json_response({"status": "deleted", "id": archetype_id})

    @routes.post("/prompt808/api/library/archetypes/regenerate")
    async def regenerate_archetypes(request):
        _get_library(request)

        def _sync():
            from .core import archetypes as archetype_gen, model_manager
            from .store import archetypes as archetype_store, elements

            all_elements = elements.get_all()
            if not all_elements:
                return None

            new = archetype_gen.generate_archetypes(
                all_elements, use_llm_naming=True, model_manager=model_manager,
            )
            archetype_store.replace_all(new)
            return new

        new_archetypes = await asyncio.to_thread(_sync)
        if new_archetypes is None:
            return web.Response(status=400, text="Library is empty — add elements first")
        return web.json_response({
            "status": "regenerated",
            "count": len(new_archetypes),
            "archetypes": [a.get("name") for a in new_archetypes],
        })

    # -----------------------------------------------------------------------
    # Library — Photos
    # -----------------------------------------------------------------------

    @routes.get("/prompt808/api/library/photos")
    async def list_photos(request):
        _get_library(request)

        def _sync():
            from .store import elements
            all_elements = elements.get_all()
            photos = {}
            for elem in all_elements:
                thumb = elem.get("thumbnail")
                if not thumb:
                    continue
                if thumb not in photos:
                    photos[thumb] = {
                        "thumbnail": thumb,
                        "source_photo": elem.get("source_photo"),
                        "subject_type": elem.get("subject_type"),
                        "added": elem.get("added"),
                        "element_count": 0,
                        "categories": [],
                    }
                photos[thumb]["element_count"] += 1
                cat = elem.get("category")
                if cat and cat not in photos[thumb]["categories"]:
                    photos[thumb]["categories"].append(cat)
            result = sorted(photos.values(), key=lambda p: p.get("added") or "", reverse=True)
            return {"photos": result, "total": len(result)}

        return web.json_response(await asyncio.to_thread(_sync))

    @routes.get("/prompt808/api/library/photos/{thumbnail}/elements")
    async def get_photo_elements(request):
        _get_library(request)
        thumbnail = request.match_info["thumbnail"]

        def _sync():
            from .store import elements
            all_elements = elements.get_all()
            return [e for e in all_elements if e.get("thumbnail") == thumbnail]

        photo_elements = await asyncio.to_thread(_sync)
        if not photo_elements:
            return web.Response(status=404, text=f"No elements found for thumbnail '{thumbnail}'")
        return web.json_response({"elements": photo_elements, "total": len(photo_elements)})

    @routes.delete("/prompt808/api/library/photos/{thumbnail}")
    async def delete_photo(request):
        _get_library(request)
        thumbnail = request.match_info["thumbnail"]

        def _sync():
            from .core import image_embeddings, library_manager, prompt_cache
            from .store import elements

            removed_count = elements.delete_by_thumbnail(thumbnail)
            if removed_count == 0:
                return None

            thumb_path = library_manager.get_thumbnails_dir() / thumbnail
            if thumb_path.is_file():
                thumb_path.unlink()

            stem = Path(thumbnail).stem
            hash_prefix = stem.rsplit("_", 1)[-1] if "_" in stem else ""
            img_emb_removed = 0
            if hash_prefix:
                img_emb_removed = image_embeddings.remove_by_hash_prefix(hash_prefix)

            _regen_archetypes_and_styles()

            try:
                prompt_cache.invalidate()
            except Exception as e:
                log.warning("Prompt cache invalidation failed: %s", e)

            log.info("Photo deleted: %s (%d elements removed)", thumbnail, removed_count)
            return {
                "status": "deleted", "thumbnail": thumbnail,
                "elements_removed": removed_count,
                "image_embeddings_removed": img_emb_removed,
            }

        result = await asyncio.to_thread(_sync)
        if result is None:
            return web.Response(status=404, text=f"No elements found for thumbnail '{thumbnail}'")
        return web.json_response(result)

    # -----------------------------------------------------------------------
    # Library — Stats & Reset
    # -----------------------------------------------------------------------

    @routes.get("/prompt808/api/library/stats")
    async def library_stats(request):
        _get_library(request)

        def _sync():
            from .core import prompt_cache
            from .store import archetypes, elements, vocabulary
            return {
                "elements": elements.count(),
                "categories": len(elements.get_categories()),
                "category_counts": elements.count_by_category(),
                "archetypes": archetypes.count(),
                "vocabulary_size": vocabulary.count(),
                "cached_prompts": prompt_cache.size(),
                "library_version": elements.get_library_version(),
            }

        try:
            return web.json_response(await asyncio.to_thread(_sync))
        except ValueError:
            # Library was just created or hasn't been committed yet — return
            # empty stats so the UI renders cleanly instead of a 500 error.
            return web.json_response({
                "elements": 0, "categories": 0, "category_counts": {},
                "archetypes": 0, "vocabulary_size": 0, "cached_prompts": 0,
                "library_version": 0,
            })

    @routes.delete("/prompt808/api/library/reset")
    async def reset_all_data(request):
        _get_library(request)
        result = await asyncio.to_thread(_run_reset_all)
        return web.json_response(result)

    # -----------------------------------------------------------------------
    # Style Profiles
    # -----------------------------------------------------------------------

    @routes.get("/prompt808/api/style/profiles")
    async def list_profiles(request):
        _get_library(request)

        def _sync():
            from .core import style_profile
            return {
                "genres": style_profile.get_all_genres(),
                "summary": style_profile.get_summary(),
            }

        return web.json_response(await asyncio.to_thread(_sync))

    @routes.get("/prompt808/api/style/profiles/{genre}")
    async def get_profile(request):
        _get_library(request)
        genre = request.match_info["genre"]

        def _sync():
            from .core import style_profile
            all_genres = style_profile.get_all_genres()
            if genre not in all_genres:
                return None
            profile = style_profile.get_genre_profile(genre)
            context = style_profile.get_style_context(genre)
            return {
                "genre": genre, "profile": profile, "context_text": context,
            }

        result = await asyncio.to_thread(_sync)
        if result is None:
            return web.Response(status=404, text=f"No style profile for genre '{genre}'")
        return web.json_response(result)

    @routes.post("/prompt808/api/style/profiles/{genre}/reset")
    async def reset_genre_profile(request):
        _get_library(request)
        from .core import style_profile
        await asyncio.to_thread(style_profile.rebuild)
        return web.json_response({"status": "rebuilt", "genre": request.match_info["genre"]})

    @routes.post("/prompt808/api/style/profiles/reset")
    async def reset_all_profiles(request):
        _get_library(request)
        from .core import style_profile
        await asyncio.to_thread(style_profile.rebuild)
        return web.json_response({"status": "all_profiles_rebuilt"})

    # -----------------------------------------------------------------------
    # Libraries (multi-library CRUD)
    # -----------------------------------------------------------------------

    @routes.get("/prompt808/api/libraries")
    async def list_libraries(request):
        from .core import library_manager

        data = await asyncio.to_thread(library_manager.list_libraries)
        return web.json_response({"libraries": data})

    @routes.post("/prompt808/api/libraries")
    async def create_library(request):
        body = await request.json()
        name = body.get("name", "").strip()
        if not name:
            return web.Response(status=400, text="Library name required")

        def _sync():
            from .core import library_manager
            return library_manager.create_library(name)

        try:
            created_name = await asyncio.to_thread(_sync)
        except ValueError as e:
            return web.Response(status=400, text=str(e))
        return web.json_response({"status": "created", "name": created_name}, status=201)

    @routes.put("/prompt808/api/libraries/active")
    async def switch_library(request):
        body = await request.json()
        name = body.get("name", "").strip()

        def _sync():
            from .core import library_manager
            library_manager.set_active(name)
            return library_manager.get_active()

        try:
            active = await asyncio.to_thread(_sync)
        except ValueError as e:
            return web.Response(status=400, text=str(e))
        return web.json_response({"status": "switched", "active": active})

    @routes.patch("/prompt808/api/libraries/{name}")
    async def rename_library(request):
        old_name = request.match_info["name"]
        body = await request.json()
        new_name = body.get("name", "").strip()

        def _sync():
            from .core import library_manager
            return library_manager.rename_library(old_name, new_name)

        try:
            renamed = await asyncio.to_thread(_sync)
        except ValueError as e:
            return web.Response(status=400, text=str(e))
        return web.json_response({"status": "renamed", "old_name": old_name, "new_name": renamed})

    @routes.delete("/prompt808/api/libraries/{name}")
    async def delete_library(request):
        name = request.match_info["name"]

        def _sync():
            from .core import library_manager
            library_manager.delete_library(name)

        try:
            await asyncio.to_thread(_sync)
        except ValueError as e:
            return web.Response(status=400, text=str(e))
        return web.json_response({"status": "deleted", "name": name})

    # -----------------------------------------------------------------------
    # Export / Import / Pro status
    # -----------------------------------------------------------------------

    @routes.post("/prompt808/api/library/export")
    async def export_library(request):
        _get_library(request)

        body = {}
        if request.content_type == "application/json":
            try:
                body = await request.json()
            except Exception:
                pass
        include_thumbnails = body.get("include_thumbnails", True)

        from .plugins.export import handle_export

        def _sync():
            from .core import library_manager
            name = library_manager.get_active()
            result = handle_export(name, include_thumbnails=include_thumbnails)
            return name, result

        name, result = await asyncio.to_thread(_sync)
        if isinstance(result, dict):
            return web.json_response(result)
        # result is bytes — send as file download
        return web.Response(
            body=result,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Disposition": f'attachment; filename="{name}.p808"',
            },
        )

    @routes.post("/prompt808/api/library/import")
    async def import_library(request):
        from .plugins.export import handle_import
        try:
            reader = await request.multipart()
            file_data = None
            file_name = None
            target_name = None
            while True:
                field = await reader.next()
                if field is None:
                    break
                if field.name == "file":
                    file_data = await field.read()
                    file_name = field.filename
                elif field.name == "name":
                    target_name = (await field.read()).decode()
        except Exception as e:
            log.warning("Failed to read multipart upload: %s", e)
            return web.Response(status=400, text=f"Invalid upload: {e}")
        if file_data is None:
            return web.Response(status=400, text="No file uploaded")
        # Derive library name from the uploaded filename (strip .p808 extension).
        # An explicit target_name (from form data) takes priority, then filename,
        # then the handler falls back to metadata inside the file.
        if not target_name and file_name:
            name = file_name
            if name.lower().endswith(".p808"):
                name = name[:-5]
            if name.strip():
                target_name = name.strip()
        result = await asyncio.to_thread(handle_import, file_data, target_name)
        return web.json_response(result)

    # -----------------------------------------------------------------------
    # Thumbnails
    # -----------------------------------------------------------------------

    @routes.get("/prompt808/img/{filename}")
    async def serve_static_image(request):
        filename = request.match_info["filename"]
        img_dir = (Path(__file__).resolve().parent.parent / "docs" / "img").resolve()
        img_path = (img_dir / filename).resolve()
        if not img_path.is_relative_to(img_dir):
            return web.Response(status=403, text="Forbidden")
        if not img_path.is_file():
            return web.Response(status=404, text="Not found")
        return web.FileResponse(img_path)

    @routes.get("/prompt808/thumbnails/{filename}")
    async def serve_thumbnail(request):
        _get_library(request)
        from .core import library_manager
        filename = request.match_info["filename"]
        thumbs_dir = library_manager.get_thumbnails_dir()
        thumb_path = (thumbs_dir / filename).resolve()
        if not thumb_path.is_relative_to(thumbs_dir.resolve()):
            return web.Response(status=403, text="Forbidden")
        if not thumb_path.is_file():
            return web.Response(status=404, text="Thumbnail not found")
        return web.FileResponse(thumb_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _sse_write(resp, event, data):
    """Write a single SSE event to a StreamResponse."""
    payload = f"event:{event}\ndata:{json.dumps(data)}\n\n"
    await resp.write(payload.encode())


def _unload_comfy_models():
    """Force-unload all ComfyUI-tracked models and clear CUDA cache.

    Call this from sidebar endpoints (analysis, generation) before loading
    our own models.  Sidebar actions run outside ComfyUI's workflow pipeline,
    so the normal free_memory(N) may not reclaim enough VRAM after a workflow
    has loaded diffusion models.  ComfyUI will reload them on-demand when the
    next workflow runs.
    """
    try:
        import comfy.model_management
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        log.info("Unloaded all ComfyUI models (sidebar VRAM reclaim)")
    except ImportError:
        pass


def _check_shutdown():
    """Raise if the server is shutting down so analysis aborts promptly."""
    if _shutdown_event.is_set():
        raise InterruptedError("Server shutting down — analysis cancelled")


def _run_analysis(image_data, image_filename, suffix, thumbnails_dir,
                  vision_model, quantization, device, attention_mode,
                  max_tokens, force, progress_cb=None):
    """Run the full analysis pipeline (in thread pool)."""
    _progress = progress_cb or (lambda msg: None)
    tmp_path = None
    try:
        _check_shutdown()
        _progress("Preparing...")
        with tempfile.NamedTemporaryFile(
            dir=thumbnails_dir, suffix=suffix, delete=False
        ) as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name

        md5 = hashlib.md5()
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        content_hash = md5.hexdigest()

        _check_shutdown()
        from .core import image_embeddings
        if not force:
            _progress("Checking for duplicates...")
            is_dup, match_hash, similarity = image_embeddings.is_duplicate_photo(
                tmp_path, content_hash
            )
            if is_dup:
                log.info("Duplicate photo rejected (similarity: %.3f to %s)", similarity, match_hash)
                return {
                    "subject_type": None,
                    "elements_added": 0,
                    "duplicates_rejected": 0,
                    "status": f"duplicate_photo (similarity: {similarity:.3f})",
                }

        _check_shutdown()
        from .api.analysis import _create_thumbnail, _get_vision_manager
        from .core import analyzer, archetypes as archetype_gen, model_manager
        from .store import archetypes as archetype_store, elements, vocabulary

        analysis_result = analyzer.analyze_photo(
            image_path=tmp_path,
            vision_model_manager=_get_vision_manager(vision_model, quantization, device, attention_mode),
            quantization=quantization,
            device=device,
            attention_mode=attention_mode,
            max_tokens=max_tokens,
            progress_cb=_progress,
        )

        _check_shutdown()
        _progress("Saving results...")
        thumbnail_name = _create_thumbnail(tmp_path, image_filename, content_hash)
        analysis_result["thumbnail"] = thumbnail_name

        commit_result = analyzer.process_and_commit(
            analysis_result,
            element_store=elements,
            vocabulary_store=vocabulary,
            archetype_generator=archetype_gen,
            model_manager=model_manager,
        )

        if commit_result.get("added"):
            try:
                image_embeddings.register_photo(content_hash, tmp_path)
            except Exception as e:
                log.warning("Photo registration failed: %s", e)

        try:
            from .core import style_profile
            style_profile.update_from_analysis(analysis_result)
        except Exception as e:
            log.warning("Style profile update failed: %s", e)

        return {
            "subject_type": analysis_result.get("subject_type"),
            "elements_added": len(commit_result.get("added", [])),
            "duplicates_rejected": len(commit_result.get("duplicates", [])),
            "medium": analysis_result.get("medium"),
            "is_photograph": analysis_result.get("is_photograph", True),
            "status": commit_result.get("status", "unknown"),
        }
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _run_analysis_cleanup():
    """Unload all analysis models (in thread pool)."""
    from .api import analysis as analysis_mod
    freed = []

    if analysis_mod._vision_manager is not None:
        analysis_mod._vision_manager.unload()
        analysis_mod._vision_manager = None
        freed.append("vision_model")

    try:
        from .core import embeddings
        if embeddings._model is not None:
            embeddings.unload_model()
            freed.append("sentence_transformer")
    except Exception as e:
        log.warning("Failed to unload sentence_transformer: %s", e)

    try:
        from .core import image_embeddings
        image_embeddings.unload_model()
        freed.append("clip_model")
    except Exception as e:
        log.warning("Failed to unload clip_model: %s", e)

    # Final combined cleanup pass after all models freed
    if freed:
        import gc
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

    log.info("Analysis cleanup: freed %s", freed or "nothing")
    return {"status": "cleaned_up", "models_freed": freed}


def _regen_archetypes_and_styles():
    """Regenerate archetypes and rebuild style profiles (in thread pool)."""
    try:
        from .core import archetypes as archetype_gen, style_profile
        from .store import archetypes as archetype_store, elements
        remaining = elements.get_all()
        if remaining:
            new_archetypes = archetype_gen.generate_archetypes(
                remaining, use_llm_naming=False, model_manager=None,
            )
            archetype_store.replace_all(new_archetypes)
        else:
            archetype_store.clear_all()
        style_profile.rebuild()
    except Exception as e:
        log.warning("Post-operation regeneration failed: %s", e)


def _run_reset_all():
    """Clear all library data (in thread pool)."""
    from .core import embeddings, image_embeddings, library_manager, prompt_cache, style_profile
    from .store import archetypes, elements, vocabulary

    el = elements.clear_all()
    ar = archetypes.clear_all()
    vo = vocabulary.clear_all()
    em = embeddings.clear_cache()
    im = image_embeddings.clear_cache()
    style_profile.reset()
    prompt_cache.invalidate()

    thumbnails_dir = library_manager.get_thumbnails_dir()
    thumb_count = 0
    if thumbnails_dir.is_dir():
        for f in thumbnails_dir.iterdir():
            if f.is_file():
                f.unlink()
                thumb_count += 1

    log.warning("All data reset: %d elements, %d archetypes, %d vocab, %d embeddings, %d img embeddings, %d thumbnails",
                 el, ar, vo, em, im, thumb_count)
    return {
        "status": "all_data_cleared",
        "elements_removed": el, "archetypes_removed": ar,
        "vocabulary_removed": vo, "embeddings_removed": em,
        "image_embeddings_removed": im, "thumbnails_removed": thumb_count,
    }


