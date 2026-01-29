"""vLLM API client for GenRM inference.

GenRM-Principle models use a special format:
1. Messages include a "principle" role with the evaluation criterion
2. Model outputs chain-of-thought reasoning in <think> tags
3. Output ends with "Final Judgement: Yes" or "Final Judgement: No"
"""
import logging
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import requests

from .config import GenRMConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class GenRMResponse:
    """Response from GenRM API."""
    generated_text: str
    judgment: Optional[str]  # "Yes" or "No" parsed from output
    reasoning: str  # Content from <think> tags
    finish_reason: str
    raw_response: Dict[str, Any]


def parse_genrm_output(text: str) -> tuple[Optional[str], str]:
    """
    Parse GenRM output to extract judgment and reasoning.

    Args:
        text: Raw model output

    Returns:
        (judgment, reasoning) where judgment is "Yes", "No", or None
    """
    # Extract reasoning from <think> tags
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else ""

    # Find "Final Judgement: Yes/No" (case-insensitive)
    judgment_match = re.search(
        r"Final\s+Judgement:\s*(Yes|No)",
        text,
        re.IGNORECASE
    )
    judgment = judgment_match.group(1).capitalize() if judgment_match else None

    return judgment, reasoning


class VLLMClient:
    """Client for vLLM OpenAI-compatible API with GenRM support."""

    def __init__(self, config: GenRMConfig = DEFAULT_CONFIG):
        self.config = config
        self.base_url = config.vllm_base_url.rstrip("/")
        self.completions_url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models"

    def health_check(self) -> bool:
        """Check if vLLM server is responding."""
        try:
            response = requests.get(self.models_url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def get_model_id(self) -> Optional[str]:
        """Get the model ID from the server."""
        try:
            response = requests.get(self.models_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                if models:
                    return models[0].get("id")
        except requests.RequestException:
            pass
        return None

    def evaluate_principle(
        self,
        conversation_messages: List[Dict[str, str]],
        principle_text: str,
        principle_name: str = "",
    ) -> GenRMResponse:
        """
        Evaluate a conversation against a principle using GenRM format.

        GenRM expects messages in this format:
        - {"role": "principle", "content": "The principle to evaluate"}
        - {"role": "assistant", "content": "Agent turn 1"}
        - {"role": "user", "content": "Lead turn 1"}
        - ...

        Args:
            conversation_messages: The conversation turns (assistant/user roles)
            principle_text: The evaluation principle/criterion
            principle_name: Name of principle being evaluated (for logging)

        Returns:
            GenRMResponse with judgment, reasoning, and raw response
        """
        # Get actual model ID from server
        model_id = self.get_model_id() or self.config.model_name

        # Build messages with principle role first
        messages = [
            {"role": "principle", "content": principle_text},
            *conversation_messages,
        ]

        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        logger.debug(f"GenRM request for {principle_name}")

        try:
            response = requests.post(
                self.completions_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,  # Allow time for reasoning
            )
            response.raise_for_status()
            data = response.json()

        except requests.RequestException as e:
            logger.error(f"vLLM API error: {e}")
            raise RuntimeError(f"vLLM API request failed: {e}") from e

        # Parse response
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        generated_text = message.get("content", "").strip()
        finish_reason = choice.get("finish_reason", "unknown")

        # Parse judgment and reasoning
        judgment, reasoning = parse_genrm_output(generated_text)

        if judgment is None:
            logger.warning(
                f"Could not parse 'Final Judgement' from {principle_name} response. "
                f"Output ends with: ...{generated_text[-100:]}"
            )

        logger.info(
            f"GenRM {principle_name}: {judgment or 'UNKNOWN'} "
            f"(reasoning: {len(reasoning)} chars)"
        )

        return GenRMResponse(
            generated_text=generated_text,
            judgment=judgment,
            reasoning=reasoning,
            finish_reason=finish_reason,
            raw_response=data,
        )

    def evaluate_all_principles(
        self,
        conversation_messages: List[Dict[str, str]],
        principles: Dict[str, str],
    ) -> Dict[str, GenRMResponse]:
        """
        Evaluate a conversation against multiple principles.

        Args:
            conversation_messages: The conversation (assistant/user turns)
            principles: {principle_name: principle_text}

        Returns:
            {principle_name: GenRMResponse}
        """
        results = {}

        for principle_name, principle_text in principles.items():
            results[principle_name] = self.evaluate_principle(
                conversation_messages=conversation_messages,
                principle_text=principle_text,
                principle_name=principle_name,
            )

        return results
