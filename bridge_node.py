"""Prompt808 Generate Node — ComfyUI native prompt generator.

Calls the generation pipeline directly via Python. All generation settings
are exposed as node inputs so they are visible in the workflow graph.
"""

import json
import logging

log = logging.getLogger("prompt808.node")

PROMPT_TYPES = ["Any", "Native", "Photo-Architectural", "Photo-Boudoir", "Photo-Cinematic",
                "Photo-Documentary", "Photo-Erotica", "Photo-Fashion", "Photo-Fine Art",
                "Photo-Portrait", "Photo-Street"]
MOODS = ["Any", "Dramatic", "Elegant", "Ethereal", "Gritty", "Melancholic",
         "Mysterious", "Provocative", "Romantic", "Sensual", "Serene"]
QUANTIZATIONS = ["FP16", "FP8", "8-bit", "4-bit"]
ENRICHMENTS = ["Baseline", "Vivid", "Expressive", "Poetic", "Lyrical", "Freeform"]


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
        archetype_list = ["Any"]
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
            library_list = [lib["name"] for lib in libs]
            active_lib = next((lib["name"] for lib in libs if lib.get("active")), library_list[0])
        else:
            library_list = ["(no libraries)"]
            active_lib = library_list[0]

        try:
            from .server.store import archetypes as arch_store
            archetype_list = ["Any"] + arch_store.get_names()
        except Exception:
            try:
                from server.store import archetypes as arch_store
                archetype_list = ["Any"] + arch_store.get_names()
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
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFF,
                    "tooltip": "Random seed for deterministic generation",
                }),
            },
            "optional": {
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
                "mood": (moods, {
                    "default": "Any",
                    "tooltip": "Mood modifier for the generated prompt",
                }),
                "llm_model": (model_list, {
                    "default": model_list[0],
                    "tooltip": "LLM model for prompt composition (None = simple mode)",
                }),
                "enrichment": (ENRICHMENTS, {
                    "default": "Vivid",
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
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Always re-execute

    def generate(self, seed, library="(no libraries)", prompt_type="Any",
                 archetype="Any", mood="Any", llm_model="None",
                 enrichment="Vivid", quantization="FP16",
                 temperature=0.7, max_tokens=1024, keep_model_loaded=False,
                 prefix="", suffix=""):
        """Generate a prompt using node inputs."""
        # Scope to the selected library
        token = None
        if library and library != "(no libraries)":
            try:
                try:
                    from .server.core import library_manager
                except ImportError:
                    from server.core import library_manager
                token = library_manager._request_library.set(library)
            except Exception as e:
                log.warning("Failed to scope library '%s': %s", library, e)

        # Check if any library exists
        try:
            try:
                from .server.core import library_manager as _lm
            except ImportError:
                from server.core import library_manager as _lm
            if _lm.get_active() is None:
                msg = "No library exists \u2014 open the Prompt808 sidebar (camera icon) and create one first"
                log.warning(msg)
                return ("", "", msg)
        except Exception as e:
            log.warning("Failed to check library state: %s", e)

        try:
            # Pre-flight: check element count
            try:
                try:
                    from .server.store import elements
                except ImportError:
                    from server.store import elements
                count = elements.count()
            except Exception as e:
                log.warning("Failed to check element count: %s", e)
                count = -1  # can't determine — let generation proceed

            if count == 0:
                msg = "Library is empty \u2014 open the Prompt808 sidebar and analyze some images first"
                log.warning(msg)
                prompt = prefix.strip() if prefix else ""
                return (prompt, "", msg)

            result = self._generate_native(
                seed=seed,
                prompt_type=prompt_type,
                archetype=archetype,
                mood=mood,
                llm_model=llm_model,
                enrichment=enrichment,
                quantization=quantization,
                temperature=temperature,
                max_tokens=max_tokens,
                keep_model_loaded=keep_model_loaded,
                prefix=prefix,
                suffix=suffix,
            )

            # Build status line
            status_parts = [result.get("status", "ok")]
            status_parts.append(f"archetype: {result.get('archetype_used', 'unknown')}")
            status_parts.append(f"elements: {len(result.get('elements_used', []))}")
            status_parts.append(f"seed: {result.get('seed', seed)}")
            status = " | ".join(status_parts)

            return (result.get("prompt", ""), result.get("negative_prompt", ""), status)
        except Exception as e:
            error_msg = f"Prompt808 generation failed: {e}"
            log.error(error_msg, exc_info=True)
            return ("", "", f"ERROR: {error_msg}")
        finally:
            if token is not None:
                try:
                    try:
                        from .server.core import library_manager
                    except ImportError:
                        from server.core import library_manager
                    library_manager._request_library.reset(token)
                except Exception as e:
                    log.warning("Failed to reset library scope: %s", e)

    def _generate_native(self, seed, prompt_type, archetype, mood, llm_model,
                         enrichment, quantization, temperature, max_tokens,
                         keep_model_loaded, prefix, suffix):
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
                nsfw = json.loads(row["value"]).get("nsfw", False)
        except Exception as e:
            log.warning("Failed to read NSFW setting: %s", e)

        # Strip display prefix (e.g. "Photo-Cinematic" → "Cinematic")
        style = prompt_type[6:] if prompt_type.startswith("Photo-") else prompt_type

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
            element_store=elements,
            archetype_store=archetypes,
            style_profile_module=style_profile,
            debug=False,
            nsfw=nsfw,
        )
        result["seed"] = seed

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

        # Post-generation model lifecycle
        if llm_model and llm_model != "None":
            if not keep_model_loaded:
                model_manager.unload_model()
            else:
                model_manager.offload_model()

        if pbar:
            pbar.update_absolute(4, 4)

        return result
