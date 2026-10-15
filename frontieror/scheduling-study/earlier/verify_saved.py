"""Sanitized-release entry point; original byte checks remain private."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[2]/"tools/verify_public_release.py"),run_name="__main__")
