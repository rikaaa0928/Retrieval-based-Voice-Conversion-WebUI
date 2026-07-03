from __future__ import annotations

import sys
from pathlib import Path


TOOL_SRC = Path(__file__).resolve().parent / "rvc_training_data" / "src"
if str(TOOL_SRC) not in sys.path:
    sys.path.insert(0, str(TOOL_SRC))

from rvc_data_tools.generate_dataset import main


if __name__ == "__main__":
    main()
