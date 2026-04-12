"""Prompt808 Generate Node — ComfyUI native prompt generator.

Calls the generation pipeline directly via Python. All generation settings
are exposed as node inputs so they are visible in the workflow graph.
"""

import json
import logging
import random

log = logging.getLogger("prompt808.node")

PROMPT_TYPES = ["Any", "Native", "Photo-Architectural", "Photo-Boudoir", "Photo-Cinematic",
                "Photo-Documentary", "Photo-Erotica", "Photo-Fashion", "Photo-Fine Art",
                "Photo-Portrait", "Photo-Street"]
MOODS = ["Any", "None", "Dramatic", "Elegant", "Ethereal", "Gritty", "Melancholic",
         "Mysterious", "Provocative", "Romantic", "Sensual", "Serene"]
QUANTIZATIONS = ["FP16", "FP8", "8-bit", "4-bit"]
ENRICHMENTS = ["Any", "Baseline", "Vivid", "Expressive", "Poetic", "Lyrical", "Freeform"]


class Prompt808Generate:
    """ComfyUI node that generates prompts via Prompt808."""

    CATEGORY = "Prompt808"
    DESCRIPTION = "Generates prompts from your Prompt808 library. All settings are exposed as node inputs. Debug mode is in ComfyUI Settings > Prompt808."
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt", "status")
    OUTPUT_TOOLTIPS = (
        "Generated prompt",
        "Negative prompt (terms to avoid)",
        "Generation status and archetype used",
    )

    @classmethod
    def INPUT_TYPES(cls):
        library_list = ["(no libraries)"]
        archetype_list = ["Any", "None"]
        model_list = ["None"]

        try:
            from .server.core import library_manager
            libs = library_manager.list_libraries()
        except Exception:
            try:
                from server.core import library_manager
                libs = library_manager.list_libraries()
            except Exception:
                libs = []

        if libs:
            library_list = ["All"] + [lib["name"] for lib in libs]
            active_lib = next((lib["name"] for lib in libs if lib.get("active")), library_list[1])
        else:
            library_list = ["(no libraries)"]
            active_lib = library_list[0]

        try:
            from .server.store import archetypes as arch_store
            archetype_list = ["Any", "None"] + arch_store.get_names()
        except Exception:
            try:
                from server.store import archetypes as arch_store
                archetype_list = ["Any", "None"] + arch_store.get_names()
            except Exception:
                pass

        try:
            from .server.core.model_manager import get_model_names
            model_list = get_model_names()
        except Exception:
            try:
                from server.core.model_manager import get_model_names
                model_list = get_model_names()
            except Exception:
                pass

        # Read NSFW setting from database to filter adult content options
        nsfw = False
        try:
            try:
                from .server.core import database
            except ImportError:
                from server.core import database
            db = database.get_db()
            row = db.execute(
                "SELECT value FROM generate_settings WHERE key='app'"
            ).fetchone()
            if row and row["value"]:
                nsfw = json.loads(row["value"]).get("nsfw", False)
        except Exception:
            pass

        prompt_types = PROMPT_TYPES if nsfw else [t for t in PROMPT_TYPES if t not in ("Photo-Boudoir", "Photo-Erotica")]
        moods = MOODS if nsfw else [m for m in MOODS if m not in ("Sensual", "Provocative")]

        return {
            "required": {},
            "optional": {
                "libraries": ("P808_LIBRARIES", {
                    "tooltip": "Connect a Library Select node for multi-library generation",
                }),
                "library": (library_list, {
                    "default": active_lib,
                    "tooltip": "Library to generate from" if libs else "No libraries — open the Prompt808 sidebar (camera icon) to create one",
                }),
                "prompt_type": (prompt_types, {
                    "default": "Any",
                    "tooltip": "Prompt style (Cinematic, Documentary, etc.)",
                }),
                "archetype": (archetype_list, {
                    "default": "Any",
                    "tooltip": "Archetype to filter elements by",
                }),
                "archetype_influence": ("INT", {
                    "default": 70,
                    "min": 0,
                    "max": 100,
                    "step": 5,
                    "tooltip": "How strongly the selected archetype biases element selection (0-100%). "
                               "Controls two things: (1) per-category probability of picking an "
                               "archetype-matched element vs. a random pool element, and (2) minimum "
                               "inclusion rate for categories the archetype covers -- e.g. a pose-only "
                               "archetype at 80% guarantees poses appear in at least 80% of prompts. "
                               "At 100%, ONLY archetype-matched categories are included. "
                               "Has no effect when archetype is None.",
                }),
                "mood": (moods, {
                    "default": "Any",
                    "tooltip": "Mood modifier for the generated prompt",
                }),
                "llm_model": (model_list, {
                    "default": model_list[0],
                    "tooltip": "LLM model for prompt composition. "
                               "None = simple mode (no LLM). "
                               "API = use an OpenAI-compatible server (LM Studio, Ollama, etc.) at the api_url below.",
                }),
                "enrichment": (ENRICHMENTS, {
                    "default": "Any",
                    "tooltip": "Creative enrichment level for LLM composition",
                }),
                "quantization": (QUANTIZATIONS, {
                    "default": "FP16",
                    "tooltip": "LLM quantization (FP16, FP8, 8-bit, 4-bit)",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.5,
                    "step": 0.05,
                    "tooltip": "LLM sampling temperature (higher = more creative)",
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 128,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Maximum tokens for LLM generation",
                }),
                "api_url": ("STRING", {
                    "default": "http://127.0.0.1:1234",
                    "tooltip": "Server URL for API mode (used when llm_model is set to API). "
                               "Works with LM Studio, Ollama, or any OpenAI-compatible endpoint. "
                               "Ignored when using a local HF model.",
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep LLM offloaded to CPU RAM after generation (faster next run)",
                }),
                "prefix": ("STRING", {
                    "default": "",
                    "tooltip": "Text prepended to the generated prompt (e.g. LoRA trigger word)",
                }),
                "suffix": ("STRING", {
                    "default": "",
                    "tooltip": "Text appended to the generated prompt (e.g. quality tags)",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFF,
                    "tooltip": "Random seed for deterministic generation",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Always re-execute

    def generate(self, seed=0, libraries=None, library="(no libraries)",
                 prompt_type="Any", archetype="Any", archetype_influence=70,
                 mood="Any", llm_model="None", enrichment="Any",
                 quantization="FP16", temperature=0.7, max_tokens=1024,
                 api_url="", keep_model_loaded=False, prefix="", suffix=""):
        """Generate a prompt using node inputs."""
        try:
            from .server.core import library_manager
        except ImportError:
            from server.core import library_manager

        # Resolve which libraries to use
        selected_libraries = self._resolve_libraries(
            libraries, library, library_manager,
        )
        if isinstance(selected_libraries, tuple):
            raise RuntimeError(selected_libraries[2])  # halt workflow

        multi = len(selected_libraries) > 1
        library_display = ", ".join(selected_libraries) if multi else selected_libraries[0]

        # Scope to first library for single-library path and cache/style fallback
        token = library_manager._request_library.set(selected_libraries[0])

        try:
            # Pre-flight: check element count (for single-library; multi checks inside)
            if not multi:
                try:
                    try:
                        from .server.store import elements
                    except ImportError:
                        from server.store import elements
                    count = elements.count()
                except Exception as e:
                    log.warning("Failed to check element count: %s", e)
                    count = -1
                if count == 0:
                    raise RuntimeError(
                        "Library is empty \u2014 open the Prompt808 sidebar "
                        "and analyze some images first"
                    )

            result = self._generate_native(
                seed=seed,
                prompt_type=prompt_type,
                archetype=archetype,
                archetype_influence=archetype_influence,
                mood=mood,
                llm_model=llm_model,
                enrichment=enrichment,
                quantization=quantization,
                temperature=temperature,
                max_tokens=max_tokens,
                api_url=api_url,
                keep_model_loaded=keep_model_loaded,
                prefix=prefix,
                suffix=suffix,
                multi_libraries=selected_libraries if multi else None,
            )

            # Build status line
            if llm_model == "API":
                model_display = f"API ({api_url.strip()})" if api_url else "API"
            elif llm_model and llm_model != "None":
                model_display = llm_model
            else:
                model_display = "None"

            # Archetype display: "library: archetype" format
            arch_name = result.get("archetype_used", "unknown")
            arch_lib = result.get("archetype_library", selected_libraries[0])
            if arch_name in ("None", "none", "unknown", "None (fallback)"):
                arch_display = arch_name
            else:
                arch_display = f"{arch_lib}: {arch_name}"

            status = "\n".join([
                f"library: {library_display}",
                f"prompt type: {_display_prompt_type(result.get('style_used', prompt_type))}",
                f"archetype: {arch_display}",
                f"archetype influence: {archetype_influence}%",
                f"mood: {result.get('mood_used', 'unknown')}",
                f"model: {model_display}",
                f"enrichment: {result.get('enrichment_used', 'unknown')}",
                f"elements: {len(result.get('elements_used', []))}",
                f"seed: {result.get('seed', seed)}",
            ])

            return (result.get("prompt", ""), result.get("negative_prompt", ""), status)
        except RuntimeError:
            raise  # expected errors (empty library, bad status) pass through
        except Exception as e:
            log.error("Prompt808 generation failed: %s", e, exc_info=True)
            raise RuntimeError(f"Prompt808 generation failed: {e}") from e
        finally:
            library_manager._request_library.reset(token)

    # ------------------------------------------------------------------
    # Library resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_libraries(libraries, library, library_manager):
        """Return a list of library names to generate from.

        ``libraries`` comes from the Library Select node (or ``None``).
        ``library`` comes from the dropdown widget.
        Returns a list of names, or a 3-tuple error to return early.
        """
        if library_manager.get_active() is None:
            msg = ("No library exists \u2014 open the Prompt808 sidebar "
                   "(camera icon) and create one first")
            log.warning(msg)
            return ("", "", msg)

        # Library Select node connected — takes priority
        if libraries is not None:
            valid = []
            all_libs = {lib["name"] for lib in library_manager.list_libraries()}
            for name in libraries:
                if name in all_libs:
                    valid.append(name)
                else:
                    log.warning("Library '%s' not found, skipping", name)
            if not valid:
                return ("", "", "All selected libraries are invalid or empty")
            return valid

        # "All" option in dropdown
        if library == "All":
            all_libs = library_manager.list_libraries()
            names = [lib["name"] for lib in all_libs]
            if not names:
                return ("", "", "No libraries exist")
            return names

        # Single library from dropdown
        if library and library != "(no libraries)":
            return [library]

        return ("", "", "No library selected")

    def _generate_native(self, seed, prompt_type, archetype, archetype_influence,
                         mood, llm_model, enrichment, quantization, temperature,
                         max_tokens, api_url, keep_model_loaded, prefix, suffix,
                         multi_libraries=None):
        """Direct Python call into the generation pipeline."""
        try:
            from .server.core import generator, model_manager, style_profile
            from .server.store import archetypes, elements
        except ImportError:
            from server.core import generator, model_manager, style_profile
            from server.store import archetypes, elements

        pbar = None
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(4)
        except ImportError:
            pass

        if pbar:
            pbar.update_absolute(0, 4)

        # Read NSFW setting so generator excludes adult styles from "Any"
        nsfw = False
        try:
            from .server.core import database
        except ImportError:
            from server.core import database
        try:
            db = database.get_db()
            row = db.execute(
                "SELECT value FROM generate_settings WHERE key='app'"
            ).fetchone()
            if row and row["value"]:
                app_settings = json.loads(row["value"])
                nsfw = app_settings.get("nsfw", False)
            else:
                app_settings = {}
        except Exception as e:
            log.warning("Failed to read app settings: %s", e)
            app_settings = {}

        # Node input is 0-100, generator expects 0.0-1.0
        archetype_influence = archetype_influence / 100.0
        balance_libraries = app_settings.get("balance_libraries", True)

        # Strip display prefix (e.g. "Photo-Cinematic" → "Cinematic")
        style = prompt_type[6:] if prompt_type.startswith("Photo-") else prompt_type

        # Parse prefixed archetype (e.g. "LibA: Studio Portrait" → library + name)
        archetype_library = None
        if archetype not in ("Any", "None") and ": " in archetype:
            archetype_library, archetype = archetype.split(": ", 1)

        # Resolve stores — merged wrappers for multi-library, real modules otherwise
        if multi_libraries:
            elem_store, arch_store, style_mod = _gather_multi_library_data(
                multi_libraries, elements, archetypes, style_profile, seed,
            )
        else:
            elem_store = elements
            arch_store = archetypes
            style_mod = style_profile

        # Resolve library-scoped archetype to its unique ID.
        # If the target library isn't in the selected set, fall back to "None"
        # rather than accidentally matching a same-named archetype elsewhere.
        if archetype_library and hasattr(arch_store, "get_by_name_and_library"):
            resolved = arch_store.get_by_name_and_library(archetype, archetype_library)
            if resolved:
                archetype = resolved.get("id", archetype)
            else:
                archetype = "None"

        # API mode: only active when llm_model is explicitly set to "API"
        effective_api_url = api_url.strip() if llm_model == "API" and api_url else None

        result = generator.generate_prompt(
            seed=seed,
            archetype_id=archetype,
            style=style,
            mood=mood,
            model_name=llm_model,
            quantization=quantization,
            enrichment=enrichment,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            model_manager=model_manager,
            element_store=elem_store,
            archetype_store=arch_store,
            style_profile_module=style_mod,
            debug=False,
            nsfw=nsfw,
            archetype_influence=archetype_influence,
            balance_libraries=balance_libraries,
            api_url=effective_api_url,
        )

        status = result.get("status", "")
        if status not in ("", "ok", "cache_hit"):
            raise RuntimeError(f"Generation failed ({status})")

        result["seed"] = seed

        # Resolve which library the archetype belongs to
        arch_used = result.get("archetype_used", "None")
        if arch_used not in ("None", "none", "unknown", "None (fallback)"):
            found = arch_store.get_by_name(arch_used)
            if found and found.get("_library"):
                result["archetype_library"] = found["_library"]

        # Apply prefix/suffix (even if prompt is empty, so LoRA triggers etc. survive)
        if prefix or suffix:
            parts = []
            if prefix:
                parts.append(prefix.strip())
            if result.get("prompt"):
                parts.append(result["prompt"])
            if suffix:
                parts.append(suffix.strip())
            result["prompt"] = " ".join(parts)

        # Post-generation model lifecycle (only for local HF models, not API)
        if llm_model not in (None, "None", "API"):
            if not keep_model_loaded:
                model_manager.unload_model()
            else:
                model_manager.offload_model()

        if pbar:
            pbar.update_absolute(4, 4)

        return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _display_prompt_type(style):
    """Map internal style name to display prompt type with Photo- prefix."""
    try:
        from .server.core.generator import _display_style
    except ImportError:
        from server.core.generator import _display_style
    return _display_style(style)


# Multi-library merging
# ------------------------------------------------------------------

class _MergedElementStore:
    """Duck-typed element store backed by pre-merged data."""

    def __init__(self, all_elements, version_str):
        self._elements = all_elements
        self._version = version_str

    def get_all(self):
        return self._elements

    def count(self):
        return len(self._elements)

    def get_library_version(self):
        return self._version


class _MergedArchetypeStore:
    """Duck-typed archetype store backed by pre-merged data."""

    def __init__(self, all_archetypes):
        self._archetypes = all_archetypes

    def get_all(self):
        return self._archetypes

    def get_by_id(self, archetype_id):
        return next(
            (a for a in self._archetypes if a["id"] == archetype_id), None,
        )

    def get_by_name(self, name):
        return next(
            (a for a in self._archetypes if a.get("name") == name), None,
        )

    def get_by_name_and_library(self, name, library_name):
        """Find an archetype by name scoped to a specific library."""
        return next(
            (a for a in self._archetypes
             if a.get("name") == name and a.get("_library") == library_name),
            None,
        )

    def get_names(self):
        return [a.get("name") or a.get("id") for a in self._archetypes]


class _MergedStyleProfile:
    """Duck-typed style profile module that picks randomly from all libraries."""

    def __init__(self, contexts, seed):
        self._contexts = contexts  # {genre: [context_str, ...]}
        self._seed = seed

    def get_all_genres(self):
        return list(self._contexts.keys())

    def get_style_context(self, genre, max_traits=5):
        available = [c for c in self._contexts.get(genre, []) if c]
        if not available:
            return ""
        rng = random.Random(self._seed + 4)
        return rng.choice(available)


def _gather_multi_library_data(library_names, elements_mod, archetypes_mod,
                                style_profile_mod, seed):
    """Gather and merge data from multiple libraries.

    Returns (element_store, archetype_store, style_profile_module) wrappers.
    """
    try:
        from .server.core import library_manager
    except ImportError:
        from server.core import library_manager

    all_elements = []
    all_archetypes = []
    version_parts = []
    style_contexts = {}  # genre -> [context_str, ...]

    for lib_name in sorted(library_names):
        token = library_manager._request_library.set(lib_name)
        try:
            lib_elements = elements_mod.get_all()
            for elem in lib_elements:
                tagged = dict(elem)
                tagged["_library"] = lib_name
                all_elements.append(tagged)

            lib_archetypes = archetypes_mod.get_all()
            for arch in lib_archetypes:
                tagged = dict(arch)
                tagged["_library"] = lib_name
                all_archetypes.append(tagged)

            lib_version = elements_mod.get_library_version()
            version_parts.append(f"{lib_name}={lib_version}")

            # Collect style contexts for all genres in this library
            for genre in style_profile_mod.get_all_genres():
                ctx = style_profile_mod.get_style_context(genre)
                if ctx:
                    style_contexts.setdefault(genre, []).append(ctx)
        finally:
            library_manager._request_library.reset(token)

    version_str = "multi:" + "|".join(version_parts)

    return (
        _MergedElementStore(all_elements, version_str),
        _MergedArchetypeStore(all_archetypes),
        _MergedStyleProfile(style_contexts, seed),
    )
