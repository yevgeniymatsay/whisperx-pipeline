"""Main Streamlit application for GenRM Judge Workbench.

Launch:
    streamlit run scripts/genrm_workbench.py
"""

import streamlit as st

from .state import init_session_state, get_state
from .components import (
    render_server_status,
    render_config_panel,
    render_principle_editor,
    render_call_debugger,
    render_batch_runner,
)


def main() -> None:
    """Main entry point for the GenRM Workbench."""
    st.set_page_config(
        page_title="GenRM Workbench",
        page_icon="🔬",
        layout="wide",
    )

    # Initialize session state
    init_session_state()

    # Header
    st.title("GenRM Judge Workbench")
    st.caption("Debug, configure, and test the GenRM judge pipeline")

    # Sidebar components
    render_server_status()
    render_config_panel()
    render_principle_editor()

    # Main content tabs
    tab1, tab2 = st.tabs(["Single Call Debugger", "Batch Processing"])

    with tab1:
        render_call_debugger()

    with tab2:
        render_batch_runner()


if __name__ == "__main__":
    main()
