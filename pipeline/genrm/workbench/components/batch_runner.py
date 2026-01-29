"""Batch runner component - process multiple calls with progress tracking."""

from typing import List

import streamlit as st
import boto3

from ...cli import judge_single_call
from ...config import GenRMConfig
from ....config import S3_BUCKET, AWS_REGION
from ..state import get_state


def list_spk_turns_files(prefix: str, limit: int = 500) -> List[str]:
    """List spk_turns.json files under an S3 prefix."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    paginator = s3.get_paginator("list_objects_v2")

    call_keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("spk_turns.json"):
                call_keys.append(key)
                if len(call_keys) >= limit:
                    return call_keys

    return call_keys


def render_batch_runner() -> None:
    """Render batch processing tab."""
    state = get_state()
    config = state.config

    st.markdown("## Batch Processing")
    st.caption("Process multiple calls and view aggregate statistics")

    # S3 prefix input
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        prefix = st.text_input(
            "S3 Prefix",
            value="runs/",
            placeholder="runs/VIDEO_ID/",
            key="batch_prefix",
        )

    with col2:
        limit = st.number_input(
            "Limit",
            min_value=1,
            max_value=500,
            value=100,
            key="batch_limit",
        )

    with col3:
        dry_run = st.checkbox("Dry Run", value=True, key="batch_dry_run")

    # Server check
    if not state.vllm_connected:
        st.warning("Connect to vLLM server first (check sidebar)")
        return

    # Scan for files
    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("Scan S3", key="scan_s3"):
            with st.spinner("Scanning..."):
                keys = list_spk_turns_files(prefix, int(limit))
                st.session_state.batch_keys = keys

    # Show scan results
    batch_keys = st.session_state.get("batch_keys", [])

    if batch_keys:
        st.info(f"Found {len(batch_keys)} calls to process")

        with st.expander("Files to process"):
            for key in batch_keys[:20]:
                st.markdown(f"- `{key}`")
            if len(batch_keys) > 20:
                st.caption(f"... and {len(batch_keys) - 20} more")

        # Start batch button
        if st.button("Start Batch", type="primary", key="start_batch"):
            run_batch(batch_keys, config, dry_run)

    # Show batch results
    if state.batch_results:
        st.divider()
        render_batch_results()


def run_batch(keys: List[str], config: GenRMConfig, dry_run: bool) -> None:
    """Run batch processing with progress updates."""
    state = get_state()

    # Reset results
    state.batch_results = []
    state.batch_stats = {
        "total": len(keys),
        "accept": 0,
        "review": 0,
        "reject": 0,
        "role_fallback": 0,
        "skipped": 0,
        "error": 0,
    }

    progress = st.progress(0)
    status = st.empty()
    stats_container = st.empty()

    for i, s3_key in enumerate(keys):
        # Extract call name safely (handle malformed paths)
        path_parts = s3_key.split("/")
        call_name = path_parts[-2] if len(path_parts) >= 2 else s3_key
        status.text(f"Processing {i + 1}/{len(keys)}: {call_name}...")

        try:
            result = judge_single_call(
                s3_key=s3_key,
                model_dir=None,  # Use heuristic
                config=config,
                dry_run=dry_run,
            )

            state.batch_results.append(result)

            # Update stats
            if result.get("status") == "skipped":
                state.batch_stats["skipped"] += 1
            else:
                decision = result.get("decision", "").lower()
                if decision in state.batch_stats:
                    state.batch_stats[decision] += 1

        except Exception as e:
            state.batch_results.append({
                "s3_key": s3_key,
                "status": "error",
                "error": str(e),
            })
            state.batch_stats["error"] += 1

        # Update progress
        progress.progress((i + 1) / len(keys))

        # Update live stats
        with stats_container.container():
            render_live_stats(state.batch_stats)

    status.text("Batch complete!")


def render_live_stats(stats: dict) -> None:
    """Render live statistics during batch processing."""
    total = stats.get("total", 1)
    processed = sum(v for k, v in stats.items() if k != "total")

    cols = st.columns(6)

    with cols[0]:
        st.metric("Accept", stats.get("accept", 0))
    with cols[1]:
        st.metric("Review", stats.get("review", 0))
    with cols[2]:
        st.metric("Reject", stats.get("reject", 0))
    with cols[3]:
        st.metric("Fallback", stats.get("role_fallback", 0))
    with cols[4]:
        st.metric("Skipped", stats.get("skipped", 0))
    with cols[5]:
        st.metric("Errors", stats.get("error", 0))


def render_batch_results() -> None:
    """Render batch results summary."""
    state = get_state()
    stats = state.batch_stats
    results = state.batch_results

    st.markdown("### Batch Results")

    # Stats summary
    total = stats.get("total", 0)
    accept = stats.get("accept", 0)
    review = stats.get("review", 0)
    reject = stats.get("reject", 0)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pct = (accept / total * 100) if total else 0
        st.metric("ACCEPT", f"{accept} ({pct:.0f}%)")

    with col2:
        pct = (review / total * 100) if total else 0
        st.metric("REVIEW", f"{review} ({pct:.0f}%)")

    with col3:
        pct = (reject / total * 100) if total else 0
        st.metric("REJECT", f"{reject} ({pct:.0f}%)")

    with col4:
        errors = stats.get("error", 0) + stats.get("skipped", 0)
        pct = (errors / total * 100) if total else 0
        st.metric("Skip/Error", f"{errors} ({pct:.0f}%)")

    # Distribution bar
    if total > 0:
        st.markdown("**Distribution:**")

        # Simple text-based bar
        bar_width = 50
        accept_width = int(accept / total * bar_width)
        review_width = int(review / total * bar_width)
        reject_width = int(reject / total * bar_width)

        bar = (
            ":green[" + "=" * accept_width + "]"
            + ":orange[" + "=" * review_width + "]"
            + ":red[" + "=" * reject_width + "]"
        )
        st.markdown(bar)

    # Results table
    with st.expander("Detailed Results"):
        for result in results:
            decision = result.get("decision", result.get("status", "unknown"))
            score = result.get("aggregate_score")
            call_id = result.get("call_id", result.get("s3_key", "?"))

            if decision == "ACCEPT":
                color = ":green"
            elif decision == "REVIEW":
                color = ":orange"
            elif decision == "REJECT":
                color = ":red"
            else:
                color = ":gray"

            score_str = f" ({score:.3f})" if score else ""
            st.markdown(f"{color}[{decision}]{score_str} - `{call_id}`")

    # Clear results button
    if st.button("Clear Results", key="clear_batch"):
        state.batch_results = []
        state.batch_stats = {}
        st.session_state.batch_keys = []
        st.rerun()
