"""Config panel component - threshold and weight sliders."""

import streamlit as st

from ..state import get_state, reset_config


def render_config_panel() -> None:
    """Render config editing panel in sidebar."""
    state = get_state()
    config = state.config

    st.sidebar.markdown("### Routing Thresholds")

    # Accept threshold slider
    accept = st.sidebar.slider(
        "Accept",
        min_value=0.0,
        max_value=1.0,
        value=config.accept_threshold,
        step=0.05,
        help="Calls with aggregate score >= this value are ACCEPTED",
        key="accept_threshold_slider",
    )
    if accept != config.accept_threshold:
        # Ensure accept > review
        if accept <= config.review_threshold:
            st.sidebar.warning("Accept must be > Review")
        else:
            state.config.accept_threshold = accept

    # Review threshold slider
    review = st.sidebar.slider(
        "Review",
        min_value=0.0,
        max_value=1.0,
        value=config.review_threshold,
        step=0.05,
        help="Calls with score between Review and Accept go to REVIEW",
        key="review_threshold_slider",
    )
    if review != config.review_threshold:
        if review >= config.accept_threshold:
            st.sidebar.warning("Review must be < Accept")
        else:
            state.config.review_threshold = review

    st.sidebar.caption(f"REJECT: < {config.review_threshold:.2f}")

    st.sidebar.divider()

    st.sidebar.markdown("### Principle Weights")

    weights = dict(config.principle_weights)
    updated_weights = {}
    weight_sum = 0.0

    for principle, weight in weights.items():
        new_weight = st.sidebar.slider(
            principle,
            min_value=0.0,
            max_value=1.0,
            value=weight,
            step=0.05,
            key=f"weight_{principle}",
        )
        updated_weights[principle] = new_weight
        weight_sum += new_weight

    # Show total and validation
    if abs(weight_sum - 1.0) < 0.001:
        st.sidebar.success(f"Total: {weight_sum:.2f}")
    else:
        st.sidebar.error(f"Total: {weight_sum:.2f} (must be 1.0)")

    # Apply weights if valid
    if abs(weight_sum - 1.0) < 0.001 and updated_weights != config.principle_weights:
        state.config.principle_weights = updated_weights

    st.sidebar.divider()

    st.sidebar.markdown("### Generation")

    max_tokens = st.sidebar.number_input(
        "max_tokens",
        min_value=256,
        max_value=4096,
        value=config.max_tokens,
        step=256,
        help="Maximum tokens for GenRM response",
        key="max_tokens_input",
    )
    if max_tokens != config.max_tokens:
        state.config.max_tokens = int(max_tokens)

    # Temperature slider
    temperature = st.sidebar.slider(
        "temperature",
        min_value=0.0,
        max_value=2.0,
        value=config.temperature,
        step=0.1,
        help="0.0 = deterministic, higher = more random",
        key="temperature_slider",
    )
    if temperature != config.temperature:
        state.config.temperature = temperature

    # Top logprobs
    top_logprobs = st.sidebar.number_input(
        "top_logprobs",
        min_value=1,
        max_value=20,
        value=config.top_logprobs,
        help="Number of top logprobs to return (vLLM max is 20)",
        key="top_logprobs_input",
    )
    if top_logprobs != config.top_logprobs:
        state.config.top_logprobs = int(top_logprobs)

    st.sidebar.divider()

    # S3 Output Prefixes (in expander to reduce clutter)
    with st.sidebar.expander("S3 Output Prefixes"):
        st.caption("Where routed calls are saved in S3")

        s3_accepted = st.text_input(
            "Accepted",
            value=config.s3_prefix_accepted,
            key="s3_prefix_accepted_input",
        )
        if s3_accepted != config.s3_prefix_accepted:
            state.config.s3_prefix_accepted = s3_accepted

        s3_review = st.text_input(
            "Review",
            value=config.s3_prefix_review,
            key="s3_prefix_review_input",
        )
        if s3_review != config.s3_prefix_review:
            state.config.s3_prefix_review = s3_review

        s3_rejected = st.text_input(
            "Rejected",
            value=config.s3_prefix_rejected,
            key="s3_prefix_rejected_input",
        )
        if s3_rejected != config.s3_prefix_rejected:
            state.config.s3_prefix_rejected = s3_rejected

        s3_fallback = st.text_input(
            "Role Fallback",
            value=config.s3_prefix_role_fallback,
            key="s3_prefix_role_fallback_input",
        )
        if s3_fallback != config.s3_prefix_role_fallback:
            state.config.s3_prefix_role_fallback = s3_fallback

    st.sidebar.divider()

    # Reset button
    if st.sidebar.button("Reset to Defaults", use_container_width=True):
        reset_config()
        st.rerun()
