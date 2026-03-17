"""Photo analysis API — upload + QwenVL element extraction.

POST /api/analyze — accepts image upload, runs QwenVL extraction pipeline,
commits elements to store, regenerates archetypes.
"""

import hashlib
import logging
import shutil
import tempfile
import threading
from pathlib import Path

try:
    from fastapi import APIRouter, File, Form, HTTPException, UploadFile
    from pydantic import BaseModel
except ImportError:
    # Running inside ComfyUI — FastAPI not needed, routes.py handles endpoints
    APIRouter = None

    class BaseModel:
        """Stub so model classes can still be defined."""

if APIRouter is not None:
    router = APIRouter()
else:
    class _NoOpRouter:
        """Stub router whose decorators are identity functions."""
        def _noop(self, *a, **kw):
            return lambda fn: fn
        get = post = put = patch = delete = _noop

    router = _NoOpRouter()

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

log = logging.getLogger("prompt808.api.analysis")

# Supported image formats
SUPPORTED_FORMATS = {
    ".jpg", ".jpeg", ".jfif", ".jpe",
    ".png", ".webp", ".avif", ".avifs",
    ".bmp", ".dib", ".tiff", ".tif",
    ".heic", ".heif", ".gif",
    ".jp2", ".j2k", ".j2c", ".jpf", ".jpx", ".jpc",
    ".psd", ".tga", ".icb", ".vda", ".vst",
}

from ..core import library_manager


class AnalysisResponse(BaseModel):
    subject_type: str | None
    elements_added: int
    duplicates_rejected: int
    status: str


@router.get("/analyze/options")
async def analyze_options():
    """Return available options for the analysis UI."""
    from ..core.model_manager import get_vision_model_names

    return {
        "vision_models": get_vision_model_names(),
        "quantizations": ["FP16", "FP8", "8-bit", "4-bit"],
    }


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_photo(
    image: UploadFile = File(...),
    vision_model: str = Form("Qwen3-VL-8B-Instruct"),
    quantization: str = Form("FP8"),
    device: str = Form("auto"),
    attention_mode: str = Form("auto"),
    max_tokens: int = Form(2048),
    force: bool = Form(False),
):
    """Analyze a photo and extract photographic elements.

    Accepts an image upload, runs QwenVL extraction, normalizes tags,
    deduplicates descriptions, commits elements to the library, and
    regenerates archetypes.
    """
    # Validate file type
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise HTTPException(400, f"Unsupported format '{suffix}'. Use: {SUPPORTED_FORMATS}")

    # Stream uploaded image to temp file (avoids loading entire file into RAM)
    thumbnails_dir = library_manager.get_thumbnails_dir()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=thumbnails_dir, suffix=suffix, delete=False
        ) as tmp:
            shutil.copyfileobj(image.file, tmp)
            tmp_path = tmp.name

        # Compute content hash from temp file in chunks (no RAM spike)
        md5 = hashlib.md5()
        with open(tmp_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        content_hash = md5.hexdigest()

        # Check for duplicate photo before expensive QwenVL inference
        from ..core import image_embeddings

        if not force:
            is_dup, match_hash, similarity = image_embeddings.is_duplicate_photo(
                tmp_path, content_hash
            )
            if is_dup:
                log.info("Duplicate photo rejected (similarity: %.3f to %s)", similarity, match_hash)
                return AnalysisResponse(
                    subject_type=None,
                    elements_added=0,
                    duplicates_rejected=0,
                    status=f"duplicate_photo (similarity: {similarity:.3f})",
                )

        # Run analysis pipeline
        from ..core import analyzer, archetypes as archetype_gen, model_manager
        from ..store import archetypes as archetype_store, elements, vocabulary

        analysis_result = analyzer.analyze_photo(
            image_path=tmp_path,
            vision_model_manager=_get_vision_manager(vision_model, quantization, device, attention_mode),
            quantization=quantization,
            device=device,
            attention_mode=attention_mode,
            max_tokens=max_tokens,
        )

        # Generate a proper thumbnail from the uploaded image
        thumbnail_name = _create_thumbnail(tmp_path, image.filename or "photo", content_hash)
        analysis_result["thumbnail"] = thumbnail_name

        # Commit to store
        commit_result = analyzer.process_and_commit(
            analysis_result,
            element_store=elements,
            vocabulary_store=vocabulary,
            archetype_generator=archetype_gen,
            model_manager=model_manager,
        )

        # Register photo embedding for future dedup checks (only if elements were added)
        if commit_result.get("added"):
            try:
                image_embeddings.register_photo(content_hash, tmp_path)
            except Exception as e:
                log.warning("Photo registration failed: %s", e)

        # Update style profile
        try:
            from ..core import style_profile
            style_profile.update_from_analysis(analysis_result)
        except Exception as e:
            log.warning("Style profile update failed: %s", e)

        return AnalysisResponse(
            subject_type=analysis_result.get("subject_type"),
            elements_added=len(commit_result.get("added", [])),
            duplicates_rejected=len(commit_result.get("duplicates", [])),
            status=commit_result.get("status", "unknown"),
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("Analysis failed: %s", e, exc_info=True)
        raise HTTPException(500, f"Analysis failed: {e}")
    finally:
        # Always clean up the full-size temp file
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _create_thumbnail(image_path, original_filename, content_hash):
    """Create a 512px max JPEG thumbnail and return its filename."""
    from PIL import Image

    # Generate stable filename from content hash only (no original filename)
    hash8 = content_hash[:8]
    thumb_name = f"{hash8}.jpg"
    thumb_path = library_manager.get_thumbnails_dir() / thumb_name

    try:
        img = Image.open(image_path)
        img.thumbnail((512, 512))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(thumb_path, "JPEG", quality=80)
        log.info("Thumbnail created: %s (%dx%d)", thumb_name, img.width, img.height)
    except Exception as e:
        log.warning("Thumbnail creation failed: %s", e)
        thumb_name = None

    return thumb_name


class VisionModelManager:
    """Minimal wrapper for QwenVL vision model inference.

    Loads a Qwen2-VL or Qwen3-VL model using the transformers
    AutoModelForImageTextToText + AutoProcessor pipeline.
    """

    def __init__(self, model_name, quantization, device, attention_mode):
        self.model_name = model_name
        self.requested_quantization = quantization
        self.quantization = quantization
        self.device = device
        self.attention_mode = attention_mode
        self._model = None
        self._processor = None
        self._load_error = None

    def _load(self):
        if self._model is not None:
            return
        # If a previous load failed, raise immediately instead of
        # re-hitting HuggingFace API on every retry/image.
        if self._load_error is not None:
            raise self._load_error

        try:
            self._load_inner()
        except Exception as e:
            # Cache the error so subsequent calls fail fast
            if self._load_error is None:
                self._load_error = e
            raise

    def _load_inner(self):
        import json

        import torch
        from transformers import AutoProcessor

        # Resolve device
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # FP8 requires compute capability >= 8.9 (Ada Lovelace / RTX 40xx+)
        if self.quantization == "FP8" and device == "cuda":
            cc = torch.cuda.get_device_capability()
            if cc < (8, 9):
                log.warning("GPU compute capability %d.%d < 8.9 — FP8 not supported, "
                            "falling back to FP16", cc[0], cc[1])
                self.quantization = "FP16"

        # Look up repo_id from models.json
        models_path = Path(__file__).resolve().parent.parent.parent / "models.json"
        with open(models_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        vision_models = registry.get("vision_models", {})
        info = vision_models.get(self.model_name)
        if not info:
            raise ValueError(f"Unknown vision model: {self.model_name}")

        repo_id = info["repo_id"]
        is_pre_quantized = info.get("quantized", False)

        try:
            from transformers import AutoModelForImageTextToText as VLAutoModel
        except ImportError:
            from transformers import AutoModelForVision2Seq as VLAutoModel

        # Build loading kwargs
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if is_pre_quantized:
            load_kwargs["dtype"] = "auto"
            try:
                import compressed_tensors  # noqa: F401
            except ImportError:
                log.warning("compressed_tensors not installed — pre-quantized "
                            "FP8 models may fail to load")
        elif self.quantization == "FP8":
            from transformers import FineGrainedFP8Config
            load_kwargs["quantization_config"] = FineGrainedFP8Config()
        elif self.quantization == "4-bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "8-bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            load_kwargs["dtype"] = torch.float16 if device == "cuda" else torch.float32

        if device == "cuda":
            load_kwargs["device_map"] = {"": "cuda:0"}

        # Apply attention implementation
        from ..core.model_manager import resolve_attention
        attn_impl = resolve_attention(self.attention_mode)
        load_kwargs["attn_implementation"] = attn_impl

        log.info("Loading vision model %s (%s, attn=%s)...",
                 self.model_name, self.quantization, attn_impl)
        self._processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)

        try:
            self._model = VLAutoModel.from_pretrained(repo_id, **load_kwargs).eval()
        except Exception as e:
            error_msg = str(e).lower()
            # Flash attention failure — fall back to SDPA and retry
            if "flash_attn" in error_msg or "flash attention" in error_msg:
                log.warning("flash_attn failed (%s), falling back to SDPA", e)
                load_kwargs["attn_implementation"] = "sdpa"
                self._model = VLAutoModel.from_pretrained(repo_id, **load_kwargs).eval()
            elif self.quantization == "FP8":
                log.warning("FP8 loading failed for %s (%s), falling back to FP16",
                            self.model_name, e)
                load_kwargs.pop("quantization_config", None)
                load_kwargs["dtype"] = torch.float16 if device == "cuda" else torch.float32
                self.quantization = "FP16"
                self._model = VLAutoModel.from_pretrained(repo_id, **load_kwargs).eval()
            else:
                raise

        log.info("Vision model loaded")

    def generate_with_image(self, image_path, prompt, max_tokens=2048,
                            temperature=0.3, seed=42):
        """Run vision model inference on an image with a text prompt."""
        import torch
        from PIL import Image

        self._load()
        torch.manual_seed(seed)

        image = Image.open(image_path).convert("RGB")

        # Downscale large images to reduce CPU preprocessing time.
        # The Qwen VL processor slices images into 28x28 patches — a 21 MB
        # PNG at 4K+ resolution generates massive token counts and pegs the
        # CPU for minutes.  1536px on the longest side is plenty for element
        # extraction and keeps preprocessing under a few seconds.
        MAX_SIDE = 1536
        w, h = image.size
        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
            image = image.resize(
                (round(w * scale), round(h * scale)), Image.LANCZOS
            )

        # Build chat messages with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        )

        # Move to model device
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
            )

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        decoded = self._processor.decode(new_tokens, skip_special_tokens=True).strip()
        return decoded

    def unload(self):
        """Free vision model from memory."""
        import gc

        self._model = None
        self._processor = None
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


# Module-level vision model manager singleton
_vision_manager = None
_vision_lock = threading.Lock()


def _get_vision_manager(model_name, quantization, device, attention_mode):
    global _vision_manager
    with _vision_lock:
        if (_vision_manager is None
                or _vision_manager.model_name != model_name
                or _vision_manager.requested_quantization != quantization
                or getattr(_vision_manager, "device", None) != device
                or getattr(_vision_manager, "attention_mode", None) != attention_mode):
            if _vision_manager is not None:
                _vision_manager.unload()
            _vision_manager = VisionModelManager(model_name, quantization, device, attention_mode)
        return _vision_manager


@router.post("/analyze/cleanup")
async def analysis_cleanup():
    """Unload analysis models to free VRAM/RAM.

    Call this after a batch of photos has been analyzed.
    Unloads the vision model, sentence-transformer, and CLIP model.
    """
    global _vision_manager
    freed = []

    if _vision_manager is not None:
        _vision_manager.unload()
        _vision_manager = None
        freed.append("vision_model")

    try:
        from ..core import embeddings
        if embeddings._model is not None:
            embeddings.unload_model()
            freed.append("sentence_transformer")
    except Exception:
        pass

    try:
        from ..core import image_embeddings
        image_embeddings.unload_model()
        freed.append("clip_model")
    except Exception:
        pass

    log.info("Analysis cleanup: freed %s", freed or "nothing")
    return {"status": "cleaned_up", "models_freed": freed}
