"""Server status component - vLLM health and model info."""

import time
from urllib.parse import urlparse

import streamlit as st

from ...vllm_client import VLLMClient
from ...config import GenRMConfig
from ..state import get_state


def is_safe_url(url: str) -> tuple[bool, str]:
    """
    Validate URL is safe for internal use (prevents SSRF).

    Only allows localhost, 127.0.0.1, and private network IPs.

    Returns:
        (is_safe, error_message)
    """
    try:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return False, "Invalid scheme. Use http or https."

        hostname = parsed.hostname
        if hostname is None:
            return False, "Invalid URL: no hostname"

        # Allow localhost variants
        if hostname in ("localhost", "127.0.0.1", "::1"):
            return True, ""

        # Allow private network ranges (10.x.x.x, 172.16-31.x.x, 192.168.x.x)
        if hostname.startswith("10.") or hostname.startswith("192.168."):
            return True, ""
        if hostname.startswith("172."):
            try:
                second_octet = int(hostname.split(".")[1])
                if 16 <= second_octet <= 31:
                    return True, ""
            except (ValueError, IndexError):
                pass

        return False, "Only localhost and private IPs allowed for security."

    except Exception as e:
        return False, f"Invalid URL: {e}"


def check_server_health(config: GenRMConfig) -> tuple[bool, str | None, float | None]:
    """
    Check vLLM server health and measure latency.

    Returns:
        (is_healthy, model_id, latency_ms)
    """
    client = VLLMClient(config)

    start = time.time()
    is_healthy = client.health_check()
    latency_ms = (time.time() - start) * 1000

    model_id = client.get_model_id() if is_healthy else None

    return is_healthy, model_id, latency_ms


def render_server_status() -> None:
    """Render server status indicator in sidebar."""
    state = get_state()
    config = state.config

    st.sidebar.markdown("### vLLM Server")

    col1, col2 = st.sidebar.columns([3, 1])

    with col2:
        if st.button("Refresh", key="refresh_server", use_container_width=True):
            is_healthy, model_id, latency_ms = check_server_health(config)
            state.vllm_connected = is_healthy
            state.vllm_model_id = model_id
            state.vllm_latency_ms = latency_ms

    with col1:
        if state.vllm_connected:
            st.markdown("**Status:** :green[Connected]")
        elif state.vllm_connected is False and state.vllm_latency_ms is not None:
            st.markdown("**Status:** :red[Disconnected]")
        else:
            st.markdown("**Status:** :gray[Unknown]")

    if state.vllm_connected:
        st.sidebar.markdown(f"**Model:** `{state.vllm_model_id or 'unknown'}`")
        if state.vllm_latency_ms:
            st.sidebar.markdown(f"**Latency:** {state.vllm_latency_ms:.0f}ms")
    elif state.vllm_latency_ms is not None:
        st.sidebar.error(f"Server not responding at {config.vllm_base_url}")
        st.sidebar.caption(
            "Make sure vLLM is running. SSH tunnel command:\n"
            "`ssh -L 8000:localhost:8000 -i ~/.ssh/whisperx-key-east1.pem ubuntu@<EC2_IP>`"
        )

    # URL editor
    with st.sidebar.expander("Server URL"):
        new_url = st.text_input(
            "vLLM Base URL",
            value=config.vllm_base_url,
            key="vllm_url_input",
        )
        if new_url != config.vllm_base_url:
            # Validate URL for security (prevent SSRF)
            is_safe, error_msg = is_safe_url(new_url)
            if is_safe:
                state.config.vllm_base_url = new_url
                state.vllm_connected = False
                state.vllm_model_id = None
                state.vllm_latency_ms = None
            else:
                st.error(error_msg)

    st.sidebar.divider()
