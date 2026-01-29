"""Call debugger component - step-by-step single call evaluation."""

import time
from typing import Any, Dict, List, Optional

import streamlit as st

from ...cli import load_call_from_s3, simple_role_heuristic, extract_call_id
from ...vllm_client import VLLMClient
from ...prompt_formatter import (
    format_conversation_for_genrm,
    merge_consecutive_same_role,
    validate_conversation,
)
from ...scorer import judgment_to_score, aggregate_scores
from ...router import GenRMRouter, RoutingDecision
from ..state import get_state, reset_evaluation_state


def render_call_debugger() -> None:
    """Render single call debugger tab."""
    state = get_state()

    st.markdown("## Single Call Debugger")
    st.caption("Step through the GenRM pipeline for a single call")

    # Step 1: Load Call
    render_load_step()

    # Only show subsequent steps if call is loaded
    if state.current_call_data:
        st.divider()
        render_role_step()

        if state.current_role_prediction:
            st.divider()
            render_format_step()

            if state.formatted_messages:
                st.divider()
                render_evaluate_step()

                if state.principle_results:
                    st.divider()
                    render_routing_step()


def render_load_step() -> None:
    """Render Step 1: Load call from S3."""
    state = get_state()

    st.markdown("### 1. Load Call")

    col1, col2 = st.columns([4, 1])

    with col1:
        s3_key = st.text_input(
            "S3 Key",
            value=state.current_call_key or "",
            placeholder="runs/VIDEO_ID/RUN_ID/calls/CALL_ID/spk_turns.json",
            key="s3_key_input",
        )

    with col2:
        st.write("")  # Spacer for alignment
        load_clicked = st.button("Load", type="primary", use_container_width=True)

    if load_clicked and s3_key:
        with st.spinner("Loading from S3..."):
            try:
                call_data = load_call_from_s3(s3_key)
                state.current_call_key = s3_key
                state.current_call_data = call_data
                reset_evaluation_state()
                state.current_call_data = call_data  # Re-set after reset
                st.success(f"Loaded {len(call_data.get('turns', []))} turns")
            except Exception as e:
                st.error(f"Failed to load: {e}")
                state.current_call_data = None

    # Show loaded call info
    if state.current_call_data:
        turns = state.current_call_data.get("turns", [])
        call_id = extract_call_id(state.current_call_key or "")

        with st.expander(f"Call Data: {call_id}", expanded=False):
            st.json({
                "call_id": call_id,
                "num_turns": len(turns),
                "speakers": list(set(
                    t.get("spk") or t.get("speaker", "")
                    for t in turns
                )),
                "first_3_turns": turns[:3],
            })


def render_role_step() -> None:
    """Render Step 2: Role assignment."""
    state = get_state()

    st.markdown("### 2. Role Assignment")

    if not state.current_role_prediction:
        if st.button("Assign Roles (Heuristic)", key="assign_roles"):
            turns = state.current_call_data.get("turns", [])
            role_prediction = simple_role_heuristic(turns)
            state.current_role_prediction = role_prediction
            st.rerun()
    else:
        role_pred = state.current_role_prediction

        # Show role mapping
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Speaker Roles:**")
            for speaker, role in role_pred.speaker_roles.items():
                role_color = ":blue[agent]" if role == "agent" else ":green[user]"
                st.markdown(f"- `{speaker}` -> {role_color}")

        with col2:
            st.markdown("**Confidence:**")
            for speaker, conf in role_pred.confidence_scores.items():
                st.markdown(f"- `{speaker}`: {conf:.2f}")

        st.markdown(f"**Route:** `{role_pred.route_to}`")

        if st.button("Re-assign Roles", key="reassign_roles"):
            state.current_role_prediction = None
            state.formatted_messages = None
            state.format_metadata = None
            state.principle_results = {}
            state.routing_result = None
            st.rerun()


def render_format_step() -> None:
    """Render Step 3: Format conversation."""
    state = get_state()

    st.markdown("### 3. Format Conversation")

    if not state.formatted_messages:
        if st.button("Format for GenRM", key="format_conv"):
            turns = state.current_call_data.get("turns", [])
            messages, metadata = format_conversation_for_genrm(
                turns, state.current_role_prediction
            )
            messages = merge_consecutive_same_role(messages)
            is_valid, issues = validate_conversation(messages)

            state.formatted_messages = messages
            state.format_metadata = {
                **metadata,
                "is_valid": is_valid,
                "validation_issues": issues,
            }
            st.rerun()
    else:
        messages = state.formatted_messages
        metadata = state.format_metadata or {}

        # Validation status
        if metadata.get("is_valid"):
            st.success(f"Valid conversation with {len(messages)} turns")
        else:
            st.warning(f"Validation issues: {metadata.get('validation_issues', [])}")

        # Show metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Included", metadata.get("included_turns", 0))
        with col2:
            st.metric("Filtered Narrator", metadata.get("filtered_narrator_turns", 0))
        with col3:
            st.metric("Filtered Empty", metadata.get("filtered_empty_turns", 0))

        # Show formatted messages
        with st.expander("Formatted Messages", expanded=False):
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                color = "blue" if role == "assistant" else "green"
                content = msg.get("content", "") or ""
                preview = content[:200] + "..." if len(content) > 200 else content
                st.markdown(f"**[{i}] :{color}[{role}]:** {preview}")


def render_evaluate_step() -> None:
    """Render Step 4: Evaluate principles."""
    state = get_state()

    st.markdown("### 4. Evaluate Principles")

    if not state.vllm_connected:
        st.warning("Connect to vLLM server first (check sidebar)")
        return

    # Evaluate buttons
    col1, col2 = st.columns([1, 3])

    with col1:
        evaluate_all = st.button("Evaluate All", type="primary", key="eval_all")

    with col2:
        # Individual principle buttons
        principle_cols = st.columns(len(state.principles))
        for i, principle_name in enumerate(state.principles.keys()):
            with principle_cols[i]:
                already_done = principle_name in state.principle_results
                label = f"{principle_name}"
                if already_done:
                    judgment = state.principle_results[principle_name].get("judgment", "?")
                    label = f"{principle_name} ({judgment})"
                if st.button(label, key=f"eval_{principle_name}", disabled=already_done):
                    evaluate_single_principle(principle_name)
                    st.rerun()

    if evaluate_all:
        evaluate_all_principles()
        st.rerun()

    # Show results
    if state.principle_results:
        st.markdown("**Results:**")

        for principle_name, result in state.principle_results.items():
            judgment = result.get("judgment", "UNKNOWN")
            score = result.get("score", 0.5)
            latency = result.get("latency_ms", 0)
            reasoning = result.get("reasoning", "")

            # Badge color
            if judgment == "Yes":
                badge = ":green[YES]"
            elif judgment == "No":
                badge = ":red[NO]"
            else:
                badge = ":gray[UNKNOWN]"

            with st.expander(f"{principle_name}: {badge} (score: {score:.2f}, {latency:.0f}ms)"):
                st.markdown("**Reasoning:**")
                if reasoning:
                    st.markdown(f"```\n{reasoning}\n```")
                else:
                    st.caption("No reasoning extracted")


def evaluate_single_principle(principle_name: str) -> None:
    """Evaluate a single principle and store result."""
    state = get_state()
    config = state.config

    client = VLLMClient(config)
    principle_text = state.principles[principle_name]
    messages = state.formatted_messages

    start = time.time()
    response = client.evaluate_principle(
        conversation_messages=messages,
        principle_text=principle_text,
        principle_name=principle_name,
    )
    latency = (time.time() - start) * 1000

    state.principle_results[principle_name] = {
        "judgment": response.judgment,
        "score": judgment_to_score(response.judgment),
        "reasoning": response.reasoning,
        "latency_ms": latency,
        "raw_text": response.generated_text,
    }


def evaluate_all_principles() -> None:
    """Evaluate all principles sequentially."""
    state = get_state()

    progress = st.progress(0)
    status = st.empty()

    principles = list(state.principles.keys())

    for i, principle_name in enumerate(principles):
        status.text(f"Evaluating {principle_name}...")
        evaluate_single_principle(principle_name)
        progress.progress((i + 1) / len(principles))

    status.text("Done!")


def render_routing_step() -> None:
    """Render Step 5: Routing decision."""
    state = get_state()
    config = state.config

    st.markdown("### 5. Routing Decision")

    if not state.routing_result:
        if st.button("Compute Routing", key="compute_route"):
            # Build principle_scores in expected format
            principle_scores = {}
            for name, result in state.principle_results.items():
                principle_scores[name] = {
                    "judgment": result.get("judgment"),
                    "score": result.get("score", 0.5),
                }

            # Compute aggregate
            aggregate = aggregate_scores(principle_scores, config.principle_weights)

            # Determine decision
            if aggregate >= config.accept_threshold:
                decision = "ACCEPT"
            elif aggregate >= config.review_threshold:
                decision = "REVIEW"
            else:
                decision = "REJECT"

            state.routing_result = {
                "decision": decision,
                "aggregate_score": aggregate,
                "thresholds": {
                    "accept": config.accept_threshold,
                    "review": config.review_threshold,
                },
            }
            st.rerun()
    else:
        result = state.routing_result
        decision = result["decision"]
        aggregate = result["aggregate_score"]
        thresholds = result["thresholds"]

        # Decision badge
        if decision == "ACCEPT":
            st.success(f"**Decision: ACCEPT** (score: {aggregate:.3f})")
        elif decision == "REVIEW":
            st.warning(f"**Decision: REVIEW** (score: {aggregate:.3f})")
        else:
            st.error(f"**Decision: REJECT** (score: {aggregate:.3f})")

        # Threshold visualization
        st.markdown("**Score vs Thresholds:**")

        # Simple bar visualization
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score", f"{aggregate:.3f}")
        with col2:
            st.metric("Accept >=", f"{thresholds['accept']:.2f}")
        with col3:
            st.metric("Review >=", f"{thresholds['review']:.2f}")

        # Score breakdown
        st.markdown("**Weighted Contributions:**")
        weights = config.principle_weights
        for principle_name, weight in weights.items():
            if principle_name in state.principle_results:
                score = state.principle_results[principle_name].get("score", 0.5)
                contribution = weight * score
                st.markdown(
                    f"- {principle_name}: {score:.2f} x {weight:.2f} = {contribution:.3f}"
                )

        if st.button("Re-compute Routing", key="recompute_route"):
            state.routing_result = None
            st.rerun()
