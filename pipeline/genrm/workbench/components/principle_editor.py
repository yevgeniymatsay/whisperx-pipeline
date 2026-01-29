"""Principle editor component - edit evaluation prompts."""

import streamlit as st

from ...principles import PRINCIPLES
from ..state import get_state


def render_principle_editor() -> None:
    """Render principle prompt editor in sidebar."""
    state = get_state()
    config = state.config

    st.sidebar.markdown("### Principle Prompts")

    for principle_name, weight in config.principle_weights.items():
        with st.sidebar.expander(f"{principle_name} ({weight:.2f})"):
            current_text = state.principles.get(principle_name, "")

            new_text = st.text_area(
                f"Edit {principle_name} prompt",
                value=current_text,
                height=150,
                key=f"principle_text_{principle_name}",
                label_visibility="collapsed",
            )

            if new_text != current_text:
                state.principles[principle_name] = new_text

            # Show diff indicator
            original = PRINCIPLES.get(principle_name, "")
            if new_text != original:
                st.caption(":orange[Modified from default]")

    # Reset all button
    if st.sidebar.button("Reset All Prompts", key="reset_principles"):
        state.principles = PRINCIPLES.copy()
        st.rerun()

    st.sidebar.divider()
