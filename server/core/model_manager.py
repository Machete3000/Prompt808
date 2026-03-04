"""Local LLM model manager for Prompt808.

Downloads HuggingFace text models on first use, caches them in memory,
and provides text generation with structured output support. Falls back
gracefully on any error with structured status reporting.
"""

import gc
import json
import logging
import re
import time
from pathlib import Path

log = logging.getLogger("prompt808.model_manager")

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_PATH = PROJECT_DIR / "models.json"
CUSTOM_MODELS_PATH = PROJECT_DIR / "custom_models.json"

# Attention mode choices
ATTENTION_MODES = ["auto", "flash_attention_2", "sdpa", "eager"]

# Enrichment level configurations for LLM prompt rewriting.
# Each level progressively loosens constraints to allow more creative freedom.
ENRICHMENT_LEVELS = {
    "Baseline": {
        "task_instruction": "Rewrite each element description below into a vivid photographic phrase.",
        "word_range": "5-20 words",
        "fidelity_rule": "Preserve the semantic meaning of the tags",
        "creative_rules": [],
        "temperature_nudge": 0.0,
        "omit_original_desc": False,
        "needs_scene_context": False,
    },
    "Vivid": {
        "task_instruction": "Enrich each element description with precise sensory and textural detail.",
        "word_range": "8-22 words",
        "fidelity_rule": "Stay close to the original meaning — vivid language only, no new concepts",
        "creative_rules": [
            "Use specific adjectives for light quality, texture, surface, and material behavior",
        ],
        "temperature_nudge": 0.1,
        "omit_original_desc": False,
        "needs_scene_context": False,
    },
    "Expressive": {
        "task_instruction": "Transform each element description into an evocative photographic phrase that captures mood and atmosphere.",
        "word_range": "10-25 words",
        "fidelity_rule": "The original tags are a starting point — you may reinterpret them with imagery that serves the mood",
        "creative_rules": [
            "Draw on sensory contrasts, movement, light quality, and emotional tone",
            "Specific photography or art references are welcome when they serve precision",
        ],
        "temperature_nudge": 0.15,
        "omit_original_desc": False,
        "needs_scene_context": False,
    },
    "Poetic": {
        "task_instruction": "Craft each element description as a fragment of heightened photographic poetry.",
        "word_range": "12-28 words",
        "fidelity_rule": "Use the original tags as raw material — metaphor, sensory transference, and art movement references are encouraged",
        "creative_rules": [
            "Reference specific photographers, painters, or film movements when they add precision",
            "Unexpected but exact language is preferred over safe generalities",
        ],
        "temperature_nudge": 0.2,
        "omit_original_desc": False,
        "needs_scene_context": False,
    },
    "Lyrical": {
        "task_instruction": "Generate a photographic description for each element using only its tags as inspiration.",
        "word_range": "12-28 words",
        "fidelity_rule": "Interpret the tags freely — create an original phrase, not a rewrite",
        "creative_rules": [
            "Each description should feel like it belongs in a curated photography exhibition caption",
            "Surprise and specificity over predictability",
        ],
        "temperature_nudge": 0.3,
        "omit_original_desc": True,
        "needs_scene_context": False,
    },
    "Freeform": {
        "task_instruction": "You are a photography director. Describe each element as a concrete, actionable direction for a photoshoot.",
        "word_range": "8-25 words",
        "fidelity_rule": "Ground every description in the scene context provided — archetype mood, environment, and visual coherence matter most",
        "creative_rules": [
            "Be specific about materials, light behavior, spatial relationships, and body mechanics",
            "Describe what the camera will see, not what the viewer should feel",
        ],
        "temperature_nudge": 0.15,
        "omit_original_desc": True,
        "needs_scene_context": True,
    },
}

# Module-level singleton state for loaded model
_loaded_model = None
_loaded_tokenizer = None
_loaded_signature = None  # (repo_id, quantization, device, attn_impl, use_torch_compile)
_loaded_is_bnb = False  # True if loaded with BnB — cannot CPU offload (NF4/int8 dtypes)
_offloaded_to_ram = False  # True if model was CPU-offloaded and needs full reload

# Status tracking for error surfacing
_last_status = "idle"  # "idle", "ok", "load_error: ...", "generation_error: ...", "parse_error: ..."


def get_last_status():
    """Return the status from the most recent operation."""
    return _last_status


def load_models_registry():
    """Load text_models from models.json + custom_models.json, return merged dict."""
    raw = {}
    try:
        with open(MODELS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        log.warning("Failed to load models.json: %s", e)

    models = dict(raw.get("text_models", {}))

    if CUSTOM_MODELS_PATH.exists():
        try:
            with open(CUSTOM_MODELS_PATH, "r", encoding="utf-8") as f:
                custom = json.load(f)
            # Custom models can have text_models key or be flat
            if "text_models" in custom:
                models.update(custom["text_models"])
            else:
                # Flat format: each key is a model name (exclude meta keys)
                models.update({k: v for k, v in custom.items()
                               if isinstance(v, dict) and "repo_id" in v})
        except Exception as e:
            log.warning("Failed to load custom_models.json: %s", e)

    return models


def _gpu_supports_fp8():
    """Check if the GPU supports native FP8 (compute capability >= 8.9)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability() >= (8, 9)
    except Exception:
        return False


def get_model_names():
    """Return ['None'] + sorted model names for UI dropdown.

    Pre-quantized FP8 models are excluded on GPUs with compute < 8.9.
    """
    registry = load_models_registry()
    fp8_ok = _gpu_supports_fp8()
    names = [
        name for name, info in registry.items()
        if fp8_ok or not info.get("quantized", False)
    ]
    return ["None"] + sorted(names)


def get_vision_model_names():
    """Return sorted vision model names from models.json.

    Pre-quantized FP8 models are excluded on GPUs with compute < 8.9.
    """
    try:
        with open(MODELS_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        fp8_ok = _gpu_supports_fp8()
        return sorted(
            name for name, info in raw.get("vision_models", {}).items()
            if fp8_ok or not info.get("quantized", False)
        )
    except Exception as e:
        log.warning("Failed to load vision models: %s", e)
        return []


def _resolve_device(device_choice):
    """Resolve 'auto' to the best available device."""
    if device_choice != "auto":
        return device_choice
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _comfy_kitchen_loaded():
    """Check if comfy_kitchen is loaded in the current process.

    comfy_kitchen's custom CUDA kernels are incompatible with bitsandbytes
    int8 matmul, causing segfaults.  This is used to auto-downgrade 8-bit
    to 4-bit quantization when running inside ComfyUI.
    """
    import sys
    return "comfy_kitchen" in sys.modules


def _flash_attn_available():
    """Check if flash-attn v2 is installed and GPU supports it (Ampere+)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            return False
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def resolve_attention(mode):
    """Resolve attention mode string to the attn_implementation value for transformers."""
    if mode == "sage":
        log.warning("SageAttention incompatible with text generation, using SDPA")
        return "sdpa"

    if mode == "flash_attention_2":
        if _flash_attn_available():
            return "flash_attention_2"
        log.warning("Flash Attention 2 unavailable, falling back to SDPA")
        return "sdpa"

    if mode == "sdpa":
        return "sdpa"

    if mode == "eager":
        return "eager"

    # Auto: try flash -> sdpa
    if _flash_attn_available():
        return "flash_attention_2"
    return "sdpa"


def load_model(model_name, quantization="FP16", device="auto", attention_mode="auto",
               use_torch_compile=False):
    """Load model + tokenizer. Auto-downloads from HuggingFace on first use.

    Skips reload if the same model/quantization/device/attention is already
    loaded (signature caching). Performs pre-flight memory check.
    """
    global _loaded_model, _loaded_tokenizer, _loaded_signature, _loaded_is_bnb

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    registry = load_models_registry()
    info = registry.get(model_name)
    if not info:
        raise ValueError(f"Unknown model: {model_name}")

    repo_id = info["repo_id"]
    device = _resolve_device(device)
    attn_impl = resolve_attention(attention_mode)
    is_pre_quantized = info.get("quantized", False)

    # Pre-quantized FP8 models need hardware FP8 support (compute >= 8.9)
    if is_pre_quantized and device == "cuda" and not _gpu_supports_fp8():
        raise ValueError(
            f"Model '{model_name}' has pre-quantized FP8 weights which require "
            f"a GPU with compute capability >= 8.9 (RTX 4090/H100+). "
            f"Your GPU has compute {'.'.join(str(x) for x in torch.cuda.get_device_capability())}. "
            f"Use a non-FP8 model instead (e.g. the same model with 4-bit or 8-bit quantization)."
        )

    # On-the-fly FP8 quantization also needs compute >= 8.9
    if quantization == "FP8" and not is_pre_quantized and device == "cuda" and not _gpu_supports_fp8():
        cc = torch.cuda.get_device_capability()
        log.warning("GPU compute capability %d.%d < 8.9 — FP8 not supported, "
                    "falling back to FP16", cc[0], cc[1])
        quantization = "FP16"

    # comfy_kitchen's custom CUDA kernels corrupt bitsandbytes int8 matmul,
    # causing segfaults during generation. 4-bit (NF4) is unaffected because
    # it dequantizes to fp16 before matmul.  Auto-downgrade 8-bit → 4-bit
    # when comfy_kitchen is present (i.e. running inside ComfyUI).
    if quantization == "8-bit" and _comfy_kitchen_loaded():
        log.warning("8-bit quantization is incompatible with ComfyUI's CUDA "
                    "kernels (comfy_kitchen) — auto-switching to 4-bit")
        quantization = "4-bit"

    signature = (repo_id, quantization, device, attn_impl, use_torch_compile)

    # Already loaded with same config — skip
    if _loaded_model is not None and _loaded_signature == signature:
        return

    # Unload previous model
    unload_model()

    # Ask ComfyUI to free VRAM before loading our model.
    # NOTE: sidebar callers (routes.py) should call unload_all_models()
    # before invoking generation — free_memory() alone may not reclaim
    # enough after a workflow run.  We still call free_memory() here as
    # a safety net for the node path (where ComfyUI manages memory).
    if device == "cuda":
        try:
            import comfy.model_management
            vram = info.get("vram_requirement", {})
            if quantization == "4-bit":
                est_gb = vram.get("4bit", 0)
            elif quantization in ("8-bit", "FP8"):
                est_gb = vram.get("8bit", 0)
            else:
                est_gb = vram.get("full", 0)
            if est_gb > 0:
                comfy.model_management.free_memory(
                    int(est_gb * 1024**3),
                    comfy.model_management.get_torch_device(),
                )
                log.info("Asked ComfyUI to free ~%.1fGB VRAM", est_gb)
        except ImportError:
            pass

    # Build loading kwargs
    load_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": attn_impl,
        "low_cpu_mem_usage": True,
    }

    # Determine quantization path and BnB flag
    use_bnb = False

    if is_pre_quantized:
        load_kwargs["dtype"] = "auto"
        # Pre-import compressed_tensors so it's in sys.modules before
        # transformers tries to find it. ComfyUI's runtime can interfere
        # with late imports during model deserialization.
        try:
            import compressed_tensors  # noqa: F401
        except ImportError:
            log.warning("compressed_tensors not installed — pre-quantized FP8 "
                        "models may fail to load")
    elif quantization == "FP8":
        from transformers import FineGrainedFP8Config
        load_kwargs["quantization_config"] = FineGrainedFP8Config()
    elif quantization == "4-bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        use_bnb = True
    elif quantization == "8-bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        use_bnb = True
    else:
        load_kwargs["dtype"] = torch.float16 if device == "cuda" else torch.float32

    _loaded_is_bnb = use_bnb

    if device == "cuda":
        load_kwargs["device_map"] = {"": "cuda:0"}

    log.info("Loading model %s (%s, %s) on %s...", model_name, quantization, attn_impl, device)

    _loaded_tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)

    _loaded_model = AutoModelForCausalLM.from_pretrained(
        repo_id, **load_kwargs
    ).eval()

    # Only move to device manually if device_map wasn't used (CPU-only mode)
    if "device_map" not in load_kwargs:
        _loaded_model.to(device)

    # Enable KV cache for generation
    _loaded_model.config.use_cache = True
    if hasattr(_loaded_model, "generation_config"):
        _loaded_model.generation_config.use_cache = True

    # torch.compile optimization (CUDA only, optional)
    if use_torch_compile and device == "cuda":
        try:
            log.info("Compiling model with torch.compile (first run will be slow)...")
            _loaded_model = torch.compile(_loaded_model, mode="reduce-overhead")
            log.info("torch.compile applied successfully")
        except Exception as e:
            log.warning("torch.compile failed, using eager mode: %s", e)

    _loaded_signature = signature
    log.info("Model %s loaded successfully", model_name)


def unload_model():
    """Free model from memory and clear CUDA cache."""
    global _loaded_model, _loaded_tokenizer, _loaded_signature, _loaded_is_bnb, _offloaded_to_ram

    had_model = _loaded_model is not None

    if _loaded_model is not None:
        try:
            _loaded_model.cpu()
        except Exception:
            pass

    _loaded_model = None
    _loaded_tokenizer = None
    _loaded_signature = None
    _loaded_is_bnb = False
    _offloaded_to_ram = False

    gc.collect()

    try:
        import comfy.model_management
        comfy.model_management.soft_empty_cache()
    except ImportError:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    if had_model:
        log.info("Model unloaded, VRAM freed")


def offload_model():
    """Offload model to free VRAM.

    BnB models are fully unloaded (NF4/int8 dtypes cannot exist on CPU).
    All other models (FP16, FP8, device_map) are fully unloaded too, but
    the signature is preserved so _ensure_on_device() can reload them.
    """
    global _offloaded_to_ram, _loaded_signature

    if _loaded_model is None:
        return

    if _loaded_is_bnb:
        log.info("BnB quantized model cannot offload to RAM, unloading fully")
        unload_model()
        return

    # For device_map-loaded models, .cpu() is unsafe (breaks dispatch hooks).
    # Instead, fully unload but preserve the signature so we can reload later.
    _offloaded_to_ram = True
    sig_backup = _loaded_signature
    unload_model()
    # Restore signature so _ensure_on_device() knows what to reload
    _loaded_signature = sig_backup

    log.info("Model offloaded (VRAM freed, will reload on next use)")


def _ensure_on_device():
    """Reload model to GPU if it was offloaded.

    After offload_model(), the model is fully unloaded but the signature
    is preserved. This function reloads it from the HuggingFace cache.
    """
    global _offloaded_to_ram

    if not _offloaded_to_ram:
        return
    if _loaded_signature is None:
        return

    _offloaded_to_ram = False
    repo_id, quantization, target_device, attn_impl, use_torch_compile = _loaded_signature

    log.info("Reloading offloaded model to GPU...")
    # Look up model name from repo_id
    registry = load_models_registry()
    model_name = None
    for name, info in registry.items():
        if info["repo_id"] == repo_id:
            model_name = name
            break

    if model_name:
        load_model(model_name, quantization=quantization, device=target_device,
                   attention_mode=attn_impl, use_torch_compile=use_torch_compile)
    else:
        log.error("Cannot reload: no model found for repo_id %s", repo_id)


def generate_text(prompt, max_tokens=1024, temperature=0.9, seed=42, debug=False):
    """Run inference on the loaded model. Returns generated text.

    Applies the tokenizer's chat template if available. Falls back to
    raw tokenization for base models.
    """
    import torch

    if _loaded_model is None or _loaded_tokenizer is None:
        raise RuntimeError("No model loaded")

    _ensure_on_device()
    log.info("Generating LLM text...")
    torch.manual_seed(seed)
    device = next(_loaded_model.parameters()).device

    # Format as chat message if the tokenizer supports it
    if hasattr(_loaded_tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            text = _loaded_tokenizer.apply_chat_template(
                messages, enable_thinking=False, **template_kwargs
            )
        except TypeError:
            text = _loaded_tokenizer.apply_chat_template(
                messages, **template_kwargs
            )
        inputs = _loaded_tokenizer(text, return_tensors="pt").to(device)
    else:
        inputs = _loaded_tokenizer(prompt, return_tensors="pt").to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = _loaded_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=_loaded_tokenizer.eos_token_id,
            pad_token_id=_loaded_tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = _loaded_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    num_new_tokens = len(new_tokens)
    tok_s = num_new_tokens / elapsed if elapsed > 0 else 0
    log.info("Generated %d tokens in %.1fs (%.1f tok/s)", num_new_tokens, elapsed, tok_s)

    if debug:
        raw_with_special = _loaded_tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
        log.info("LLM DEBUG seed=%d | input_tokens=%d | new_tokens=%d | "
                 "decoded_len=%d | elapsed=%.2fs | tok/s=%.1f",
                 seed, inputs["input_ids"].shape[1], num_new_tokens,
                 len(decoded), elapsed, tok_s)
        log.info("LLM DEBUG output:\n%s", decoded)
        log.info("LLM DEBUG raw (with special tokens):\n%s", raw_with_special)

    return decoded
