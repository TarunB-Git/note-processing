import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
if PROJECT_DIR.name == "desktop" and (PROJECT_DIR.parent / "android").is_dir():
    PROJECT_DIR = PROJECT_DIR.parent
DEFAULT_HOME = PROJECT_DIR / ".local" if ((PROJECT_DIR / "pyproject.toml").exists() or (PROJECT_DIR / "desktop/pyproject.toml").exists()) else Path.home() / ".local/share/note-processing"
APP_HOME = Path(os.environ.get("NOTE_PROCESSING_HOME", DEFAULT_HOME)).expanduser().resolve()
MODEL_DIR = APP_HOME / "models/minilm"
