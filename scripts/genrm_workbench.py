#!/usr/bin/env python3
"""GenRM Judge Workbench - Streamlit UI for debugging and configuration.

Launch:
    streamlit run scripts/genrm_workbench.py

Prerequisites:
    1. vLLM server running on EC2 with SSH tunnel:
       ssh -L 8000:localhost:8000 -i ~/.ssh/whisperx-key-east1.pem ubuntu@<EC2_IP>

    2. On EC2, start vLLM:
       ~/start_genrm_bg.sh

Features:
    - Server status indicator with health check
    - Config panel for threshold/weight editing
    - Principle prompt editor
    - Single call step-by-step debugger
    - Batch processing with live statistics
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.genrm.workbench.app import main

if __name__ == "__main__":
    main()
