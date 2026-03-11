"""Prompt808 — ComfyUI custom node package.

Registers the Prompt808 Generate node and all API routes on PromptServer.
"""

import logging
import re
from pathlib import Path

_pyproject = (Path(__file__).parent / "pyproject.toml").read_text(encoding="utf-8")
__version__ = re.search(r'^version\s*=\s*"([^"]+)"', _pyproject, re.MULTILINE).group(1)

log = logging.getLogger("prompt808")

# --- Node registration ---
from .bridge_node import Prompt808Generate  # noqa: E402
from .library_select_node import Prompt808LibrarySelect  # noqa: E402

NODE_CLASS_MAPPINGS = {
    "Prompt808 Generate": Prompt808Generate,
    "Prompt808 Library Select": Prompt808LibrarySelect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Prompt808 Generate": "Generate Prompt",
    "Prompt808 Library Select": "Select Libraries",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# --- Route registration (side-effect import) ---
try:
    from .server import routes  # noqa: F401
    log.info("Prompt808 API routes registered on PromptServer")
except Exception as e:
    log.warning("Prompt808 route registration skipped: %s", e)

# --- Database + library init on startup ---
try:
    from .server.core import library_manager
    library_manager.migrate_if_needed()
    log.info("Prompt808 ready (library: %s)", library_manager.get_active())
except Exception as e:
    log.warning("Prompt808 library init failed: %s", e)
