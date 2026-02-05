"""System Prompt Workbench - Streamlit UI for QA and preference labeling.

Launch:
  streamlit run scripts/system_prompt_workbench.py
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st

from ..azure_client import AzureGPTClient
from ..constants import SCHEMA_VERSION, STYLES, PROMPT_TEMPLATE_VERSION, ANALYSIS_TEMPLATE_VERSION
from ..schema import analysis_response_format, prompt_response_format
from ..templates import analysis_messages, generation_messages, repair_messages
from ..generator import (
    compute_persona_fact_count,
    compute_persona_strength,
    recommend_level,
)
from ..s3_io import default_bucket, load_json, object_exists, write_json
from ..text_metrics import compute_topic_clarity
from ..validator import validate_system_prompt
from .parsing import parse_messages_json, normalize_transcript_messages, parse_request_json


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("bucket", default_bucket())
    ss.setdefault("accepted_prefix", "genrm/accepted/")
    ss.setdefault("output_prefix", "synthetic_system_prompts/v1/")
    ss.setdefault("input_mode", "S3 call_id")  # or "Manual JSON"
    ss.setdefault("call_id", "")
    ss.setdefault("manual_messages_json", "")
    ss.setdefault("accepted_obj", None)
    ss.setdefault("prompt_obj", None)
    ss.setdefault("status_msg", "")
    ss.setdefault("deployment_override", "")
    ss.setdefault("analysis_temperature", 0.0)
    ss.setdefault("analysis_max_tokens", 600)
    ss.setdefault("base_temperature", 0.0)
    ss.setdefault("base_max_tokens", 400)
    ss.setdefault("candidate_max_tokens", 400)
    ss.setdefault("request_lab_request_json", "")
    ss.setdefault("request_lab_last_schema", None)
    ss.setdefault("request_lab_last_response", None)
    ss.setdefault(
        "debug",
        {
            "last_request_json": None,
            "last_response_json": None,
            "last_schema": None,
            "call_log": [],
        },
    )


def _prompt_key(output_prefix: str, call_id: str) -> str:
    return f"{output_prefix}{call_id}.json"


def _accepted_key(accepted_prefix: str, call_id: str) -> str:
    return f"{accepted_prefix}{call_id}.json"


def _ensure_prompt_obj_skeleton(
    *,
    bucket: str,
    accepted_key: str,
    call_id: str,
    turns: List[Dict[str, str]],
) -> Dict[str, Any]:
    now = _utc_now_iso()
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {"bucket": bucket, "key": accepted_key, "call_id": call_id},
        "turns": turns,
        "analysis": {},
        "prompts": {"minimal": "", "topic_specific": "", "highly_detailed": ""},
        "prompts_info": {},
        "qa": {
            "minimal": {"candidates": [], "preferences": []},
            "topic_specific": {"candidates": [], "preferences": []},
            "highly_detailed": {"candidates": [], "preferences": []},
        },
        "generator_meta": {
            "provider": "azure_openai",
            "deployment": "",
            "api_version": "",
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "analysis_template_version": ANALYSIS_TEMPLATE_VERSION,
            "created_at": now,
            "updated_at": now,
        },
    }


def _recompute_analysis_fields(obj: Dict[str, Any]) -> None:
    turns = obj.get("turns") or []
    analysis = obj.get("analysis") or {}
    analysis["topic_clarity"] = compute_topic_clarity(turns)
    persona_facts = analysis.get("persona_facts") or []
    analysis["persona_fact_count"] = compute_persona_fact_count(persona_facts)
    analysis["persona_strength"] = compute_persona_strength(analysis["persona_fact_count"])
    analysis["recommended_level"] = recommend_level(
        persona_fact_count=analysis["persona_fact_count"],
        topic_clarity=analysis["topic_clarity"],
    )
    obj["analysis"] = analysis


def _sync_prompt_widgets_from_obj(obj: Dict[str, Any]) -> None:
    prompts = obj.get("prompts") or {}
    for style in STYLES:
        st.session_state[f"base_prompt_{style}"] = prompts.get(style) or ""


def _copy_to_request_lab(req_obj: Dict[str, Any]) -> None:
    st.session_state["request_lab_request_json"] = json.dumps(req_obj, indent=2, ensure_ascii=False)
    st.session_state["request_lab_last_schema"] = None
    st.session_state["request_lab_last_response"] = None


def _set_last_azure_call(
    *,
    schema_type: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    response_format: Dict[str, Any],
    response_json: Dict[str, Any],
    label: Optional[str] = None,
    style: Optional[str] = None,
    client_meta: Optional[Dict[str, str]] = None,
) -> None:
    dbg = st.session_state.setdefault("debug", {})
    dbg.setdefault("call_log", [])

    req_obj: Dict[str, Any] = {
        "schema_type": schema_type,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if client_meta:
        req_obj["client"] = client_meta

    dbg["last_request_json"] = req_obj
    dbg["last_response_json"] = response_json
    dbg["last_schema"] = response_format

    dbg["call_log"].append(
        {
            "ts": _utc_now_iso(),
            "label": label,
            "style": style,
            "request": req_obj,
            "schema": response_format,
            "response": response_json,
        }
    )
    # Keep memory bounded in long QA sessions.
    if len(dbg["call_log"]) > 200:
        dbg["call_log"] = dbg["call_log"][-200:]


def render_sidebar() -> None:
    ss = st.session_state
    st.sidebar.header("Config")
    ss.bucket = st.sidebar.text_input("S3 bucket", value=ss.bucket)
    ss.accepted_prefix = st.sidebar.text_input("Accepted prefix", value=ss.accepted_prefix)
    ss.output_prefix = st.sidebar.text_input("Output prefix", value=ss.output_prefix)

    st.sidebar.divider()
    st.sidebar.header("Azure GPT-4.1")
    st.sidebar.caption("Requires env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_GPT41")
    ss.deployment_override = st.sidebar.text_input(
        "Deployment override (optional)",
        value=ss.deployment_override,
        placeholder="my-gpt41-deployment",
    )

    with st.sidebar.expander("Azure defaults", expanded=False):
        st.caption("Used by Overview buttons (Request Lab overrides these).")
        ss.analysis_temperature = st.slider(
            "analysis temperature",
            min_value=0.0,
            max_value=1.5,
            value=float(ss.analysis_temperature),
            step=0.1,
        )
        ss.analysis_max_tokens = st.number_input(
            "analysis max_tokens",
            min_value=1,
            max_value=4096,
            value=int(ss.analysis_max_tokens),
            step=10,
        )
        st.divider()
        ss.base_temperature = st.slider(
            "base prompt temperature",
            min_value=0.0,
            max_value=1.5,
            value=float(ss.base_temperature),
            step=0.1,
        )
        ss.base_max_tokens = st.number_input(
            "base prompt max_tokens",
            min_value=1,
            max_value=4096,
            value=int(ss.base_max_tokens),
            step=10,
        )
        st.divider()
        ss.candidate_max_tokens = st.number_input(
            "candidate max_tokens",
            min_value=1,
            max_value=4096,
            value=int(ss.candidate_max_tokens),
            step=10,
        )
    if st.sidebar.button("Check env"):
        try:
            client = AzureGPTClient(deployment_override=ss.deployment_override or None)
            st.sidebar.success(
                "Configured: "
                f"endpoint={client.config.endpoint}, "
                f"deployment={client.config.deployment}, "
                f"api_version={client.config.api_version}"
            )
        except Exception as e:  # noqa: BLE001
            st.sidebar.error(str(e))

    if st.sidebar.button("Smoke test (Azure)"):
        try:
            client = AzureGPTClient(deployment_override=ss.deployment_override or None)
            msgs = [
                {"role": "system", "content": "Return a JSON object strictly matching the schema."},
                {"role": "user", "content": "Return {\"system_prompt\": \"ok\"}"},
            ]
            rf = prompt_response_format()
            resp = client.chat_json(
                messages=msgs,
                response_format=rf,
                temperature=0.0,
                max_tokens=50,
            )
            _set_last_azure_call(
                schema_type="prompt",
                messages=msgs,
                temperature=0.0,
                max_tokens=50,
                response_format=rf,
                response_json=resp,
                label="Smoke test (Azure)",
                client_meta={
                    "endpoint": client.config.endpoint,
                    "deployment": client.config.deployment,
                    "api_version": client.config.api_version,
                },
            )
            st.sidebar.success("Smoke test OK (see 'Last Azure Call' for details).")
        except Exception as e:  # noqa: BLE001
            st.sidebar.error(str(e))


def render_load_section() -> None:
    """Load call by call_id from S3."""
    ss = st.session_state
    st.subheader("Load call")

    col1, col2 = st.columns([3, 1])
    with col1:
        ss.call_id = st.text_input("call_id", value=ss.call_id, placeholder="VIDEOID_startMs_endMs")
    with col2:
        st.write("")
        if st.button("Load", type="primary", use_container_width=True):
            if not ss.call_id.strip():
                st.error("Enter a call_id")
                return

            call_id = ss.call_id.strip()
            prompt_key = _prompt_key(ss.output_prefix, call_id)
            accepted_key = _accepted_key(ss.accepted_prefix, call_id)

            ss.accepted_obj = None
            ss.prompt_obj = None

            if object_exists(ss.bucket, prompt_key):
                ss.prompt_obj = load_json(ss.bucket, prompt_key)
                ss.status_msg = f"Loaded prompt object: s3://{ss.bucket}/{prompt_key}"
            else:
                if not object_exists(ss.bucket, accepted_key):
                    st.error(f"Not found: s3://{ss.bucket}/{accepted_key}")
                    return
                ss.accepted_obj = load_json(ss.bucket, accepted_key)
                turns = ss.accepted_obj.get("turns") or []
                if not turns:
                    st.error("Accepted object has no turns")
                    return
                ss.prompt_obj = _ensure_prompt_obj_skeleton(
                    bucket=ss.bucket,
                    accepted_key=accepted_key,
                    call_id=call_id,
                    turns=turns,
                )
                _recompute_analysis_fields(ss.prompt_obj)
                ss.status_msg = f"Loaded accepted call (no prompt object yet): s3://{ss.bucket}/{accepted_key}"

            if ss.prompt_obj:
                _sync_prompt_widgets_from_obj(ss.prompt_obj)

    if ss.status_msg:
        st.info(ss.status_msg)

def render_manual_load_section() -> None:
    """Load a manual conversation from pasted JSON (local only)."""
    ss = st.session_state
    st.subheader("Manual conversation (local only)")

    st.caption(
        "Paste either: "
        "a JSON array of {role, content}, "
        "or an object with {messages:[...]} (SFT format), "
        "or an object with {turns:[...]} (GenRM accepted format)."
    )
    st.caption("Tip: if you pasted comma-separated message objects, wrap them in `[...]`.")
    ss.manual_messages_json = st.text_area(
        "Manual messages JSON",
        value=ss.manual_messages_json,
        height=180,
        placeholder='[{"role":"assistant","content":"..."},{"role":"user","content":"..."}]',
    )
    if st.button("Load manual conversation", type="primary"):
        try:
            messages = parse_messages_json(ss.manual_messages_json)
            system_count = sum(1 for m in messages if m.get("role") == "system")
            turns = normalize_transcript_messages(messages)
            if system_count:
                st.warning(f"Dropped {system_count} system message(s) from the manual input transcript.")

            manual_id = f"manual_{uuid.uuid4()}"
            ss.call_id = manual_id
            ss.accepted_obj = None
            ss.prompt_obj = _ensure_prompt_obj_skeleton(
                bucket=ss.bucket,
                accepted_key="manual",
                call_id=manual_id,
                turns=turns,
            )
            # Override source key explicitly for manual mode
            ss.prompt_obj["source"]["key"] = "manual"
            _recompute_analysis_fields(ss.prompt_obj)
            _sync_prompt_widgets_from_obj(ss.prompt_obj)
            ss.status_msg = "Loaded manual conversation (not saved to S3)."
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            st.code(
                '[{"role":"assistant","content":"..."},{"role":"user","content":"..."}]',
                language="json",
            )


def render_source_section() -> None:
    ss = st.session_state
    ss.input_mode = st.radio(
        "Input mode",
        options=["S3 call_id", "Manual JSON"],
        horizontal=True,
        index=0 if ss.input_mode == "S3 call_id" else 1,
    )
    if ss.input_mode == "S3 call_id":
        render_load_section()
    else:
        render_manual_load_section()


def render_transcript(obj: Dict[str, Any]) -> None:
    st.subheader("Transcript")
    turns = obj.get("turns") or []
    with st.expander(f"Turns ({len(turns)})", expanded=False):
        for i, t in enumerate(turns):
            role = t.get("role", "?")
            content = (t.get("content") or "").strip()
            st.markdown(f"**[{i}] {role}:** {content}")


def render_analysis(obj: Dict[str, Any]) -> None:
    st.subheader("Analysis")
    analysis = obj.get("analysis") or {}
    col1, col2, col3 = st.columns(3)
    col1.metric("topic_clarity", f"{float(analysis.get('topic_clarity') or 0.0):.2f}")
    col2.metric("persona_fact_count", int(analysis.get("persona_fact_count") or 0))
    col3.metric("recommended_level", str(analysis.get("recommended_level") or ""))

    topic = analysis.get("topic") or ""
    st.text_input("topic", value=topic, key="analysis_topic", disabled=True)

    persona_facts = analysis.get("persona_facts") or []
    with st.expander(f"persona_facts ({len(persona_facts)})", expanded=False):
        st.json(persona_facts)


def render_generate_buttons(obj: Dict[str, Any], *, allow_s3_save: bool) -> None:
    st.subheader("Generate / Edit base prompts")
    ss = st.session_state
    analysis = obj.get("analysis") or {}
    topic = (analysis.get("topic") or "").strip() or "general conversation"
    persona_facts = analysis.get("persona_facts") or []

    # Quick status table
    prompts_info = obj.get("prompts_info") or {}
    rows = []
    for style in STYLES:
        info = prompts_info.get(style) or {}
        rows.append({
            "style": style,
            "valid": bool(info.get("valid")) if info else False,
            "violations": len(info.get("violations") or []) if info else 0,
            "length": len((obj.get("prompts") or {}).get(style) or ""),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Analyze (Azure)", use_container_width=True):
            try:
                client = AzureGPTClient(deployment_override=ss.deployment_override or None)
                turns = obj.get("turns") or []
                msgs = analysis_messages(turns)
                rf = analysis_response_format()
                analyzed = client.chat_json(
                    messages=msgs,
                    response_format=rf,
                    temperature=float(ss.analysis_temperature),
                    max_tokens=int(ss.analysis_max_tokens),
                )
                _set_last_azure_call(
                    schema_type="analysis",
                    messages=msgs,
                    temperature=float(ss.analysis_temperature),
                    max_tokens=int(ss.analysis_max_tokens),
                    response_format=rf,
                    response_json=analyzed,
                    label="Analyze (Azure)",
                    client_meta={
                        "endpoint": client.config.endpoint,
                        "deployment": client.config.deployment,
                        "api_version": client.config.api_version,
                    },
                )
                analysis.update(analyzed)
                obj["analysis"] = analysis
                _recompute_analysis_fields(obj)
                obj.setdefault("generator_meta", {})["deployment"] = client.config.deployment
                obj.setdefault("generator_meta", {})["api_version"] = client.config.api_version
                st.success("Analysis updated")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

    with col2:
        if st.button("Generate all 3 (Azure)", type="primary", use_container_width=True):
            try:
                client = AzureGPTClient(deployment_override=ss.deployment_override or None)
                for style in STYLES:
                    turns = obj.get("turns") or []
                    msgs = generation_messages(
                        turns=turns,
                        style=style,
                        topic=topic,
                        allowed_persona_facts=persona_facts,
                    )
                    rf = prompt_response_format()
                    resp = client.chat_json(
                        messages=msgs,
                        response_format=rf,
                        temperature=float(ss.base_temperature),
                        max_tokens=int(ss.base_max_tokens),
                    )
                    _set_last_azure_call(
                        schema_type="prompt",
                        messages=msgs,
                        temperature=float(ss.base_temperature),
                        max_tokens=int(ss.base_max_tokens),
                        response_format=rf,
                        response_json=resp,
                        label="Generate base prompt (Azure)",
                        style=style,
                        client_meta={
                            "endpoint": client.config.endpoint,
                            "deployment": client.config.deployment,
                            "api_version": client.config.api_version,
                        },
                    )
                    p = (resp.get("system_prompt") or "").strip()
                    v = validate_system_prompt(prompt=p, style=style, topic=topic)
                    if not v.valid:
                        repair_msgs = repair_messages(
                            turns=turns,
                            style=style,
                            topic=topic,
                            allowed_persona_facts=persona_facts,
                            bad_prompt=p,
                            violations=v.violations,
                        )
                        repair_resp = client.chat_json(
                            messages=repair_msgs,
                            response_format=rf,
                            temperature=0.0,
                            max_tokens=int(ss.base_max_tokens),
                        )
                        _set_last_azure_call(
                            schema_type="prompt",
                            messages=repair_msgs,
                            temperature=0.0,
                            max_tokens=int(ss.base_max_tokens),
                            response_format=rf,
                            response_json=repair_resp,
                            label="Repair base prompt (Azure)",
                            style=style,
                            client_meta={
                                "endpoint": client.config.endpoint,
                                "deployment": client.config.deployment,
                                "api_version": client.config.api_version,
                            },
                        )
                        p = (repair_resp.get("system_prompt") or "").strip()
                        v = validate_system_prompt(prompt=p, style=style, topic=topic)
                    obj.setdefault("prompts", {})[style] = p
                    obj.setdefault("prompts_info", {})[style] = {
                        "valid": v.valid,
                        "violations": v.violations,
                        "stats": v.stats,
                    }
                obj.setdefault("generator_meta", {})["deployment"] = client.config.deployment
                obj.setdefault("generator_meta", {})["api_version"] = client.config.api_version
                _sync_prompt_widgets_from_obj(obj)
                st.success("Generated prompts")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

    with col3:
        save_clicked = st.button("Save to S3", use_container_width=True, disabled=not allow_s3_save)
        if not allow_s3_save:
            st.caption("Manual mode is local only (S3 save disabled).")
        if save_clicked and allow_s3_save:
            ss = st.session_state
            call_id = (obj.get("source") or {}).get("call_id") or ss.call_id.strip()
            key = _prompt_key(ss.output_prefix, call_id)
            obj.setdefault("generator_meta", {})["updated_at"] = _utc_now_iso()
            try:
                write_json(ss.bucket, key, obj, dry_run=False)
                st.success(f"Saved: s3://{ss.bucket}/{key}")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

    prompts = obj.get("prompts") or {}
    for style in STYLES:
        st.markdown(f"**{style}**")
        prompts[style] = st.text_area(
            f"{style} prompt",
            value=prompts.get(style) or "",
            height=120 if style != "highly_detailed" else 220,
            key=f"base_prompt_{style}",
        )
    obj["prompts"] = prompts


def render_request_previews(obj: Dict[str, Any]) -> None:
    st.subheader("Request previews (what we send to Azure)")
    ss = st.session_state

    analysis = obj.get("analysis") or {}
    topic = (analysis.get("topic") or "").strip() or "general conversation"
    persona_facts = analysis.get("persona_facts") or []
    turns = obj.get("turns") or []

    # Azure config (safe to show)
    client_meta: Dict[str, str] = {}
    try:
        client = AzureGPTClient(deployment_override=ss.deployment_override or None)
        client_meta = {
            "endpoint": client.config.endpoint,
            "deployment": client.config.deployment,
            "api_version": client.config.api_version,
        }
    except Exception:
        client_meta = {}

    if client_meta:
        st.info(f"Azure: endpoint={client_meta['endpoint']} | deployment={client_meta['deployment']} | api_version={client_meta['api_version']}")
    else:
        st.warning("Azure not configured (or missing env). You can still preview requests, but you can't run them.")

    tabs = st.tabs(["Analyze", "Generate (minimal)", "Generate (topic_specific)", "Generate (highly_detailed)", "Repair preview"])

    with tabs[0]:
        msgs = analysis_messages(turns)
        req_obj = {
            "schema_type": "analysis",
            "messages": msgs,
            "temperature": float(ss.analysis_temperature),
            "max_tokens": int(ss.analysis_max_tokens),
        }
        st.caption(f"temperature={req_obj['temperature']} | max_tokens={req_obj['max_tokens']} | template={ANALYSIS_TEMPLATE_VERSION}")
        if st.button("Copy to Request Lab", key="copy_preview_analysis"):
            _copy_to_request_lab(req_obj)
            st.success("Copied to Request Lab.")
        with st.expander("Messages (role view)", expanded=True):
            for m in req_obj["messages"]:
                st.markdown(f"**{m.get('role','')}**")
                st.code(m.get("content") or "")
        with st.expander("Messages (JSON)", expanded=False):
            st.json(req_obj["messages"])
        with st.expander("Schema (analysis)", expanded=False):
            st.json(analysis_response_format())

    def _render_gen_tab(style: str, key_suffix: str) -> None:
        msgs = generation_messages(
            turns=turns,
            style=style,
            topic=topic,
            allowed_persona_facts=persona_facts,
        )
        req_obj = {
            "schema_type": "prompt",
            "messages": msgs,
            "temperature": float(ss.base_temperature),
            "max_tokens": int(ss.base_max_tokens),
        }
        st.caption(f"style={style} | temperature={req_obj['temperature']} | max_tokens={req_obj['max_tokens']} | template={PROMPT_TEMPLATE_VERSION}")
        if st.button("Copy to Request Lab", key=f"copy_preview_gen_{key_suffix}"):
            _copy_to_request_lab(req_obj)
            st.success("Copied to Request Lab.")
        with st.expander("Messages (role view)", expanded=True):
            for m in req_obj["messages"]:
                st.markdown(f"**{m.get('role','')}**")
                st.code(m.get("content") or "")
        with st.expander("Messages (JSON)", expanded=False):
            st.json(req_obj["messages"])
        with st.expander("Schema (prompt)", expanded=False):
            st.json(prompt_response_format())

    with tabs[1]:
        _render_gen_tab("minimal", "minimal")
    with tabs[2]:
        _render_gen_tab("topic_specific", "topic")
    with tabs[3]:
        _render_gen_tab("highly_detailed", "detailed")

    with tabs[4]:
        style = st.selectbox("Style", options=list(STYLES), index=0, key="repair_preview_style")
        base_prompt = (obj.get("prompts") or {}).get(style) or ""
        v = validate_system_prompt(prompt=base_prompt, style=style, topic=topic)
        msgs = repair_messages(
            turns=turns,
            style=style,
            topic=topic,
            allowed_persona_facts=persona_facts,
            bad_prompt=base_prompt,
            violations=v.violations,
        )
        req_obj = {
            "schema_type": "prompt",
            "messages": msgs,
            "temperature": 0.0,
            "max_tokens": int(ss.base_max_tokens),
        }
        st.caption(f"style={style} | violations={len(v.violations)} | max_tokens={req_obj['max_tokens']}")
        if st.button("Copy to Request Lab", key="copy_preview_repair"):
            _copy_to_request_lab(req_obj)
            st.success("Copied to Request Lab.")
        with st.expander("Messages (role view)", expanded=True):
            for m in req_obj["messages"]:
                st.markdown(f"**{m.get('role','')}**")
                st.code(m.get("content") or "")
        with st.expander("Messages (JSON)", expanded=False):
            st.json(req_obj["messages"])


def render_qa(obj: Dict[str, Any]) -> None:
    st.subheader("QA (candidates + preference pairs)")
    ss = st.session_state
    qa = obj.setdefault("qa", {})

    style = st.selectbox("Style", options=list(STYLES), index=0)
    style_qa = qa.setdefault(style, {"candidates": [], "preferences": []})

    col1, col2, col3 = st.columns(3)
    with col1:
        n = st.number_input("N candidates", min_value=1, max_value=20, value=3, step=1)
    with col2:
        temperature = st.slider("temperature", min_value=0.0, max_value=1.5, value=0.9, step=0.1)
    with col3:
        max_tokens = st.number_input(
            "candidate max_tokens",
            min_value=1,
            max_value=4096,
            value=int(ss.candidate_max_tokens),
            step=10,
            key="qa_candidate_max_tokens",
        )

    if st.button("Generate candidates", use_container_width=True):
        try:
            client = AzureGPTClient(deployment_override=ss.deployment_override or None)
            analysis = obj.get("analysis") or {}
            topic = (analysis.get("topic") or "").strip() or "general conversation"
            persona_facts = analysis.get("persona_facts") or []
            for _ in range(int(n)):
                turns = obj.get("turns") or []
                msgs = generation_messages(
                    turns=turns,
                    style=style,
                    topic=topic,
                    allowed_persona_facts=persona_facts,
                )
                rf = prompt_response_format()
                resp = client.chat_json(
                    messages=msgs,
                    response_format=rf,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
                _set_last_azure_call(
                    schema_type="prompt",
                    messages=msgs,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    response_format=rf,
                    response_json=resp,
                    label="Generate QA candidate (Azure)",
                    style=style,
                    client_meta={
                        "endpoint": client.config.endpoint,
                        "deployment": client.config.deployment,
                        "api_version": client.config.api_version,
                    },
                )
                p = (resp.get("system_prompt") or "").strip()
                cand = {
                    "id": str(uuid.uuid4()),
                    "prompt": p,
                    "gen_params": {
                        "temperature": float(temperature),
                        "max_tokens": int(max_tokens),
                        "style": style,
                        "deployment": client.config.deployment,
                        "api_version": client.config.api_version,
                    },
                    "generated_at": _utc_now_iso(),
                }
                style_qa.setdefault("candidates", []).append(cand)
            st.success(f"Added {n} candidates")
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

    candidates = style_qa.get("candidates") or []
    if candidates:
        st.markdown(f"**Candidates ({len(candidates)})**")
        for c in candidates[-10:]:
            st.markdown(f"- `{c['id']}`: {c.get('prompt','')[:120]}{'...' if len(c.get('prompt',''))>120 else ''}")

        ids = [c["id"] for c in candidates]
        chosen_id = st.selectbox("Chosen candidate", options=ids, key=f"chosen_{style}")
        rejected_id = st.selectbox("Rejected candidate", options=ids, key=f"rejected_{style}")

        rater = st.text_input("Rater", value="", placeholder="yevgeniy", key=f"rater_{style}")
        notes = st.text_input("Notes (optional)", value="", key=f"notes_{style}")
        pass_fail = st.selectbox("Pass/Fail", options=["pass", "fail"], index=0, key=f"pf_{style}")

        if st.button("Save preference pair", type="primary"):
            if chosen_id == rejected_id:
                st.error("Chosen and rejected must differ")
            else:
                pref = {
                    "chosen_id": chosen_id,
                    "rejected_id": rejected_id,
                    "rater": rater.strip() or None,
                    "rated_at": _utc_now_iso(),
                    "notes": notes.strip() or None,
                    "pass_fail": pass_fail,
                }
                style_qa.setdefault("preferences", []).append(pref)
                st.success("Saved preference")

    prefs = style_qa.get("preferences") or []
    if prefs:
        with st.expander(f"Preferences ({len(prefs)})", expanded=False):
            st.json(prefs)

    qa[style] = style_qa
    obj["qa"] = qa


def render_request_lab(obj: Dict[str, Any]) -> None:
    st.subheader("Request Lab")
    st.caption("Build, inspect, edit, and run raw Azure GPT-4.1 requests; optionally apply outputs back into the object.")

    ss = st.session_state

    req_type = st.selectbox("Request type", options=["analysis", "generate_prompt", "repair_prompt"])
    style = None
    if req_type in {"generate_prompt", "repair_prompt"}:
        style = st.selectbox("Style", options=list(STYLES), index=0)

    col1, col2, col3 = st.columns(3)
    with col1:
        temperature = st.slider("temperature", min_value=0.0, max_value=1.5, value=0.0, step=0.1)
    with col2:
        max_tokens = st.number_input("max_tokens", min_value=1, max_value=4096, value=600 if req_type == "analysis" else 400, step=10)
    with col3:
        turns = obj.get("turns") or []
        cap_turns = st.number_input("cap_turns (0=full)", min_value=0, max_value=max(0, len(turns)), value=0, step=10)

    turns_subset = turns if int(cap_turns) == 0 else turns[: int(cap_turns)]

    # Build default request JSON
    if st.button("Build default request", type="primary"):
        analysis = obj.get("analysis") or {}
        topic = (analysis.get("topic") or "").strip() or "general conversation"
        persona_facts = analysis.get("persona_facts") or []

        if req_type == "analysis":
            schema_type = "analysis"
            messages = analysis_messages(turns_subset)
        elif req_type == "generate_prompt":
            schema_type = "prompt"
            messages = generation_messages(
                turns=turns_subset,
                style=style or "minimal",
                topic=topic,
                allowed_persona_facts=persona_facts,
            )
        else:
            schema_type = "prompt"
            base_prompt = (obj.get("prompts") or {}).get(style or "minimal") or ""
            v = validate_system_prompt(prompt=base_prompt, style=style or "minimal", topic=topic)
            messages = repair_messages(
                turns=turns_subset,
                style=style or "minimal",
                topic=topic,
                allowed_persona_facts=persona_facts,
                bad_prompt=base_prompt,
                violations=v.violations,
            )

        req_obj = {
            "schema_type": schema_type,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        ss.request_lab_request_json = json.dumps(req_obj, indent=2, ensure_ascii=False)

    ss.request_lab_request_json = st.text_area(
        "Raw request JSON",
        value=ss.request_lab_request_json,
        height=280,
    )

    run_col1, run_col2 = st.columns([1, 1])
    with run_col1:
        run_clicked = st.button("Run request", use_container_width=True)
    with run_col2:
        apply_clicked = st.button("Apply output to object", use_container_width=True)

    schema_to_show: Optional[Dict[str, Any]] = None

    if run_clicked:
        try:
            req = parse_request_json(ss.request_lab_request_json)
            schema_type = req["schema_type"]
            rf = analysis_response_format() if schema_type == "analysis" else prompt_response_format()
            schema_to_show = rf

            client = AzureGPTClient(deployment_override=ss.deployment_override or None)
            resp = client.chat_json(
                messages=req["messages"],
                response_format=rf,
                temperature=req["temperature"],
                max_tokens=req["max_tokens"],
            )

            ss.request_lab_last_schema = rf
            ss.request_lab_last_response = resp
            _set_last_azure_call(
                schema_type=schema_type,
                messages=req["messages"],
                temperature=req["temperature"],
                max_tokens=req["max_tokens"],
                response_format=rf,
                response_json=resp,
                label="Request Lab",
                style=style,
                client_meta={
                    "endpoint": client.config.endpoint,
                    "deployment": client.config.deployment,
                    "api_version": client.config.api_version,
                },
            )
            st.success("Request completed")
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

    if apply_clicked:
        try:
            req = parse_request_json(ss.request_lab_request_json)
            schema_type = req["schema_type"]
            resp = ss.request_lab_last_response
            if not isinstance(resp, dict):
                raise ValueError("No response to apply (run a request first).")

            if schema_type == "analysis":
                analysis = obj.get("analysis") or {}
                analysis.update(resp)
                obj["analysis"] = analysis
                _recompute_analysis_fields(obj)
                st.success("Applied analysis output")
            else:
                if not style:
                    raise ValueError("Select a style to apply the prompt output.")
                system_prompt = (resp.get("system_prompt") or "").strip()
                if not system_prompt:
                    raise ValueError("Response missing system_prompt")
                obj.setdefault("prompts", {})[style] = system_prompt
                analysis = obj.get("analysis") or {}
                topic = (analysis.get("topic") or "").strip() or "general conversation"
                v = validate_system_prompt(prompt=system_prompt, style=style, topic=topic)
                obj.setdefault("prompts_info", {})[style] = {
                    "valid": v.valid,
                    "violations": v.violations,
                    "stats": v.stats,
                }
                _sync_prompt_widgets_from_obj(obj)
                st.success(f"Applied prompt output to {style}")
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

    st.divider()
    st.markdown("### Inspectors")

    schema = ss.request_lab_last_schema or schema_to_show
    if schema:
        with st.expander("Schema (view-only)", expanded=False):
            st.json(schema)

    dbg = ss.get("debug") or {}
    last_req = dbg.get("last_request_json")
    last_resp = dbg.get("last_response_json")
    last_schema = dbg.get("last_schema")

    if last_schema:
        with st.expander("Last schema (from any Azure call)", expanded=False):
            st.json(last_schema)
    if last_req:
        with st.expander("Last request (from any Azure call)", expanded=False):
            st.json(last_req)
    if last_resp:
        with st.expander("Last response (from any Azure call)", expanded=False):
            st.json(last_resp)


def render_last_azure_call() -> None:
    ss = st.session_state
    dbg = ss.get("debug") or {}
    if not dbg.get("last_request_json"):
        return
    with st.expander("Last Azure Call", expanded=False):
        st.json({
            "request": dbg.get("last_request_json"),
            "schema": dbg.get("last_schema"),
            "response": dbg.get("last_response_json"),
        })


def render_azure_call_log(*, key: str) -> None:
    ss = st.session_state
    dbg = ss.get("debug") or {}
    calls = dbg.get("call_log") or []
    if not calls:
        return

    with st.expander(f"Azure Call Log ({len(calls)})", expanded=False):
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("Clear log", use_container_width=True, key=f"clear_log_{key}"):
                dbg["call_log"] = []
                st.success("Cleared Azure call log")
                return

        labels = []
        for i, c in enumerate(calls):
            req = (c.get("request") or {})
            client = req.get("client") or {}
            dep = client.get("deployment") or "?"
            temp = req.get("temperature")
            mt = req.get("max_tokens")
            stl = c.get("style") or "-"
            lbl = c.get("label") or "Azure call"
            ts = c.get("ts") or ""
            labels.append(f"[{i}] {ts} | {lbl} | style={stl} | temp={temp} | max_tokens={mt} | dep={dep}")

        selected = st.selectbox(
            "Select call",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
            index=len(labels) - 1,
            key=f"call_log_select_{key}",
        )
        entry = calls[int(selected)]
        st.json(entry)


def main() -> None:
    st.set_page_config(page_title="System Prompt Workbench", layout="wide")
    _init_state()

    st.title("System Prompt Workbench")
    st.caption("Generate synthetic system prompts + label preferences (DPO-ready pairs)")

    render_sidebar()
    st.divider()
    render_source_section()

    obj = st.session_state.get("prompt_obj")
    if not obj:
        st.stop()

    tab_overview, tab_lab = st.tabs(["Overview", "Request Lab"])

    allow_s3_save = st.session_state.get("input_mode") == "S3 call_id"

    with tab_overview:
        st.divider()
        col_a, col_b = st.columns([2, 1])
        with col_a:
            render_transcript(obj)
            render_generate_buttons(obj, allow_s3_save=allow_s3_save)
            render_request_previews(obj)
        with col_b:
            render_analysis(obj)
            render_qa(obj)

        render_last_azure_call()
        render_azure_call_log(key="overview")

    with tab_lab:
        render_request_lab(obj)
        render_azure_call_log(key="lab")

    # Persist changes in session state
    st.session_state.prompt_obj = obj


if __name__ == "__main__":
    main()
