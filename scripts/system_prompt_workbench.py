#!/usr/bin/env python3
"""System Prompt Workbench - Streamlit UI for prompt QA.

Launch:
  streamlit run scripts/system_prompt_workbench.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.system_prompts.workbench.app import main

if __name__ == "__main__":
    main()

