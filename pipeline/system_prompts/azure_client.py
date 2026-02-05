"""Azure OpenAI client wrapper for GPT-4.1 system prompt generation."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from .schema import analysis_response_format, prompt_response_format
from .templates import analysis_messages, generation_messages, repair_messages


@dataclass(frozen=True)
class AzureOpenAIConfig:
    endpoint: str
    api_key: str
    api_version: str
    deployment: str


class AzureGPTClient:
    """Thin wrapper around the OpenAI Python SDK for Azure OpenAI."""

    def __init__(
        self,
        deployment_override: Optional[str] = None,
        config: Optional[AzureOpenAIConfig] = None,
    ):
        loaded = config or self._load_from_env()
        if config is None and deployment_override and deployment_override.strip():
            loaded = AzureOpenAIConfig(
                endpoint=loaded.endpoint,
                api_key=loaded.api_key,
                api_version=loaded.api_version,
                deployment=deployment_override.strip(),
            )
        self.config = loaded
        self._client = self._make_client(self.config)

    @staticmethod
    def _maybe_load_dotenv() -> None:
        """Load key=value pairs from .env into os.environ (best-effort).

        This is intentionally lightweight (no python-dotenv dependency) and only
        sets env vars that are not already defined.
        """
        path = os.path.join(os.getcwd(), ".env")
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[len("export ") :].strip()
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip("'").strip('"')
                    if not key:
                        continue
                    os.environ.setdefault(key, value)
        except OSError:
            return

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        raw = (endpoint or "").strip()
        if not raw:
            return ""
        # Allow values like:
        # - https://{resource}.openai.azure.com
        # - https://{resource}.openai.azure.com/openai/v1/
        # - https://{resource}.openai.azure.com/openai/deployments/{dep}/chat/completions?api-version=...
        if not raw.startswith(("http://", "https://")):
            raw = "https://" + raw
        parsed = urlparse(raw)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        return raw.rstrip("/")

    @staticmethod
    def _extract_api_version_and_deployment(raw_endpoint: str) -> tuple[Optional[str], Optional[str]]:
        raw = (raw_endpoint or "").strip()
        if not raw:
            return None, None
        if not raw.startswith(("http://", "https://")):
            raw = "https://" + raw
        parsed = urlparse(raw)

        api_version: Optional[str] = None
        if parsed.query:
            qs = parse_qs(parsed.query)
            v = qs.get("api-version")
            if v and v[0]:
                api_version = v[0]

        deployment: Optional[str] = None
        m = re.search(r"/openai/deployments/([^/]+)/", parsed.path)
        if m:
            deployment = m.group(1)

        return api_version, deployment

    @staticmethod
    def _load_from_env() -> AzureOpenAIConfig:
        def _read_env() -> tuple[str, str, str, str]:
            raw_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            endpoint = AzureGPTClient._normalize_endpoint(raw_endpoint)
            api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
            api_version = os.environ.get("OPENAI_API_VERSION", "").strip()
            deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_GPT41", "").strip()

            extracted_api_version, extracted_deployment = AzureGPTClient._extract_api_version_and_deployment(raw_endpoint)
            if not api_version and extracted_api_version:
                api_version = extracted_api_version
            if not deployment and extracted_deployment:
                deployment = extracted_deployment

            return endpoint, api_key, api_version, deployment

        endpoint, api_key, api_version, deployment = _read_env()

        missing_pre = [k for k, v in {
            "AZURE_OPENAI_ENDPOINT": endpoint,
            "AZURE_OPENAI_API_KEY": api_key,
            "OPENAI_API_VERSION": api_version,
            "AZURE_OPENAI_DEPLOYMENT_GPT41": deployment,
        }.items() if not v]

        if missing_pre:
            AzureGPTClient._maybe_load_dotenv()
            endpoint, api_key, api_version, deployment = _read_env()

        missing = [k for k, v in {
            "AZURE_OPENAI_ENDPOINT": endpoint,
            "AZURE_OPENAI_API_KEY": api_key,
            "OPENAI_API_VERSION": api_version,
            "AZURE_OPENAI_DEPLOYMENT_GPT41": deployment,
        }.items() if not v]
        if missing:
            raise RuntimeError(
                "Missing Azure OpenAI env vars: "
                + ", ".join(missing)
                + ". Set them before running generation/QA (shell env or a local .env file)."
            )

        return AzureOpenAIConfig(
            endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment=deployment,
        )

    @staticmethod
    def _make_client(config: AzureOpenAIConfig):
        try:
            from openai import AzureOpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "OpenAI SDK not installed. Install `openai` to use Azure GPT-4.1."
            ) from e

        return AzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=config.endpoint,
            api_version=config.api_version,
        )

    def _chat_json(
        self,
        *,
        messages: List[Dict[str, str]],
        response_format: Dict[str, Any],
        temperature: float,
        max_tokens: int,
        retries: int = 3,
    ) -> Dict[str, Any]:
        try:
            import openai
        except ImportError:  # pragma: no cover
            openai = None  # type: ignore

        last_err: Optional[BaseException] = None
        for attempt in range(retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.config.deployment,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )

                content = getattr(resp.choices[0].message, "content", None)
                if not content or not isinstance(content, str):
                    raise RuntimeError("Empty response content from Azure OpenAI")

                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"Response was not valid JSON (len={len(content)}): {content[:200]}"
                    ) from e

            except Exception as e:  # noqa: BLE001
                last_err = e

                # Retry common transient errors from the OpenAI SDK, but don't
                # hard-depend on exception classes if SDK changes.
                retryable = False
                if openai is not None:
                    retryable = isinstance(
                        e,
                        (
                            getattr(openai, "RateLimitError", Exception),
                            getattr(openai, "APIError", Exception),
                            getattr(openai, "APIConnectionError", Exception),
                            getattr(openai, "APITimeoutError", Exception),
                        ),
                    )

                if not retryable or attempt >= retries:
                    break

                backoff_s = 1.0 * (2 ** attempt)
                time.sleep(backoff_s)

        raise RuntimeError(f"Azure OpenAI request failed: {last_err}") from last_err

    def chat_json(
        self,
        *,
        messages: List[Dict[str, str]],
        response_format: Dict[str, Any],
        temperature: float,
        max_tokens: int,
        retries: int = 3,
    ) -> Dict[str, Any]:
        """Public wrapper for Request Lab-style calls."""
        return self._chat_json(
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
        )

    def analyze_transcript(self, turns: List[Dict[str, str]]) -> Dict[str, Any]:
        msgs = analysis_messages(turns)
        return self.chat_json(
            messages=msgs,
            response_format=analysis_response_format(),
            temperature=0.0,
            max_tokens=600,
        )

    def generate_system_prompt(
        self,
        *,
        turns: List[Dict[str, str]],
        style: str,
        topic: str,
        allowed_persona_facts: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 400,
    ) -> str:
        msgs = generation_messages(
            turns=turns,
            style=style,
            topic=topic,
            allowed_persona_facts=allowed_persona_facts,
        )
        data = self.chat_json(
            messages=msgs,
            response_format=prompt_response_format(),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (data.get("system_prompt") or "").strip()

    def repair_system_prompt(
        self,
        *,
        turns: List[Dict[str, str]],
        style: str,
        topic: str,
        allowed_persona_facts: List[Dict[str, Any]],
        bad_prompt: str,
        violations: List[str],
    ) -> str:
        msgs = repair_messages(
            turns=turns,
            style=style,
            topic=topic,
            allowed_persona_facts=allowed_persona_facts,
            bad_prompt=bad_prompt,
            violations=violations,
        )
        data = self.chat_json(
            messages=msgs,
            response_format=prompt_response_format(),
            temperature=0.0,
            max_tokens=400,
        )
        return (data.get("system_prompt") or "").strip()
