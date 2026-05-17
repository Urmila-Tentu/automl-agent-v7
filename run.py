"""
AutoML Agent – Entry Point
Run with:  python run.py
Or:        uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from automl.config import settings

if __name__ == "__main__":
    ROOT = Path(__file__).parent

    # Only watch source code — never experiments/, data/, models/, logs/
    # This prevents uvicorn --reload from restarting mid-request when
    # the deploy function writes files into experiments/
    reload_dirs = [
        str(ROOT / "api"),
        str(ROOT / "automl"),
    ]
    reload_excludes = [
        str(ROOT / "experiments"),
        str(ROOT / "data"),
        str(ROOT / "models"),
        str(ROOT / "logs"),
        str(ROOT / "mlruns"),
    ]

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        reload_dirs=reload_dirs if settings.api_reload else None,
        reload_excludes=reload_excludes if settings.api_reload else None,
        log_level="info",
    )
