from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    return yaml.safe_load(path.read_text())


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
