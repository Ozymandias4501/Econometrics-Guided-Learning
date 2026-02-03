from __future__ import annotations

import sys
from pathlib import Path

# Ensure `import src` works even when pytest is invoked via the installed console script,
# where `sys.path[0]` can be the script location rather than the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

