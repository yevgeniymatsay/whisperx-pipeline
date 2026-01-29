"""Session state management for GenRM workbench."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import streamlit as st

from ..config import GenRMConfig
from ..principles import PRINCIPLES


@dataclass
class WorkbenchState:
    """Session state for the workbench."""

    # Server connection
    vllm_connected: bool = False
    vllm_model_id: Optional[str] = None
    vllm_latency_ms: Optional[float] = None

    # Config (editable copy for experimentation)
    config: GenRMConfig = field(default_factory=GenRMConfig)

    # Principle prompts (editable)
    principles: Dict[str, str] = field(default_factory=lambda: PRINCIPLES.copy())

    # Loaded call data
    current_call_key: Optional[str] = None
    current_call_data: Optional[Dict[str, Any]] = None
    current_role_prediction: Optional[Any] = None

    # Evaluation results
    formatted_messages: Optional[List[Dict[str, str]]] = None
    format_metadata: Optional[Dict[str, Any]] = None
    principle_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    routing_result: Optional[Dict[str, Any]] = None

    # Batch processing
    batch_results: List[Dict[str, Any]] = field(default_factory=list)
    batch_stats: Dict[str, int] = field(default_factory=dict)


def init_session_state() -> None:
    """Initialize session state with defaults."""
    if "workbench" not in st.session_state:
        st.session_state.workbench = WorkbenchState()


def get_state() -> WorkbenchState:
    """Get current workbench state."""
    init_session_state()
    return st.session_state.workbench


def reset_evaluation_state() -> None:
    """Clear evaluation results for a fresh run."""
    state = get_state()
    state.formatted_messages = None
    state.format_metadata = None
    state.principle_results = {}
    state.routing_result = None


def reset_config() -> None:
    """Reset config and principles to defaults."""
    state = get_state()
    state.config = GenRMConfig()
    state.principles = PRINCIPLES.copy()
