"""
OpenAI-compatible local LLM client.

Works with any server exposing the OpenAI chat completions API,
including vLLM and SGLang.
"""
# pylint: disable=broad-exception-caught
import os
import time
import logging
import re
from dataclasses import dataclass
from typing import Dict, Union, Optional, Type, List
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
from dotenv import load_dotenv
from pydantic import BaseModel as PydanticBaseModel

from agentbrew.core.config import BaseConfig
from agentbrew.core.context import Context
from .base import BaseLLM

load_dotenv()


def _normalize_openai_base_url(base_url: str) -> str:
    """Ensure the base URL ends with exactly one /v1 for the OpenAI client."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


@dataclass
class LocalLLMConfig(BaseConfig):
    """
    Configuration for local LLM servers (vLLM, SGLang, or any OpenAI-compatible server).

    Attributes:
        model_name (str): The name of the model to use.
        api_key (str): The API key (default: environment variable LOCAL_LLM_API_KEY or VLLM_API_KEY).
        base_url (str): The base URL of the server (default: "http://localhost:2024").
        temperature (float): Controls randomness in output (default: 0.7).
        top_p (float): Controls diversity of output (default: 0.8).
        top_k (int): Top-k sampling parameter for vLLM (default: 20).
        repetition_penalty (float): Repetition penalty for vLLM (default: 1.05).
        frequency_penalty (float): Penalizes frequent token use (default: 0.0).
        presence_penalty (float): Penalizes repeated topics (default: 0.0).
        max_completion_tokens (int): Maximum number of tokens in the completion (default: 20000).
        seed (int): Random seed for reproducibility (default: 12345).
        reasoning (str): Reasoning level (default: "low").
        enable_thinking (bool): Enable thinking/reasoning mode for Qwen3 models (default: False).
    """
    model_name: str = "gpt-oss-120b"
    api_key: str = os.environ.get("LOCAL_LLM_API_KEY", os.environ.get("VLLM_API_KEY", ""))
    base_url: str = os.environ.get("LOCAL_LLM_BASE_URL", os.environ.get("VLLM_BASE_URL", "http://localhost:2024"))
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.05
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_completion_tokens: int = 20000
    seed: int = 12345
    reasoning: str = "low"
    enable_thinking: bool = False
    use_extra_body: bool = True
    use_max_completion_tokens: bool = False
    timeout: int = 120


class LocalLLMModel(BaseLLM):
    """
    OpenAI-compatible local LLM client using the chat completions API.

    Works with any server exposing the OpenAI /v1/chat/completions API,
    including vLLM and SGLang. Supports Qwen3-specific parameters
    (top_k, repetition_penalty, enable_thinking) via extra_body.

    Attributes:
        config_class: Configuration class for the model.
        alias: Aliases for the model — use "vllm_local", "sglang_local", or "local_llm".
    """
    config_class = LocalLLMConfig
    alias = ["vllm_local", "sglang_local", "local_llm"]
    env_vars = ["VLLM_API_KEY", "VLLM_BASE_URL"]

    def __init__(self, config: Optional[Union[Dict, str]] = None):
        super().__init__()
        self.config = LocalLLMModel.config_class.load(config)
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=_normalize_openai_base_url(self.config.base_url),
        )

    @staticmethod
    def _normalize_tool_call_response(content: str) -> str:
        """Extract JSON payload from model-emitted <tool_call> tags."""
        if not isinstance(content, str) or "<tool_call" not in content:
            return content

        match = re.search(r"<tool_call[^>]*>\s*(.*?)\s*</tool_call>", content, flags=re.DOTALL)
        if match:
            return match.group(1).strip()

        return re.sub(r"</?tool_call[^>]*>", "", content, flags=re.DOTALL).strip()

    def _generate(
            self,
            messages: List[dict[str, str]],
            response_format: Type[PydanticBaseModel] = None,
            **kwargs
    ):  # pylint: disable=too-many-return-statements
        """
        Generates content using a local LLM server (vLLM/SGLang) via chat completions API.

        Args:
            messages (List[dict[str, str]]): List of message dictionaries,
                each containing 'role' and 'content' keys.
            response_format (Type[PydanticBaseModel], optional): Pydantic model
                defining the structure of the desired output. If None, generates
                free-form text.
            **kwargs: Additional keyword arguments including:
                - max_retries (int): Maximum number of retry attempts (default: 5)
                - base_delay (float): Base delay in seconds for exponential backoff (default: 10.0)
                - max_tokens (int): Override max completion tokens for this call
                - timeout (int): Request timeout in seconds (default: 60)

        Returns:
            Union[str, PydanticBaseModel, None]: Generated content as a string
                if no response_format is provided, a Pydantic model instance if
                response_format is provided, or None if parsing structured output fails.
                Returns None if all retry attempts fail or non-retryable errors occur.
        """
        max_retries = kwargs.get("max_retries", 5)
        base_delay = kwargs.get("base_delay", 10.0)
        max_tokens = kwargs.get("max_tokens", self.config.max_completion_tokens)
        _ = response_format  # structured output not yet supported for local models

        for attempt in range(max_retries + 1):
            try:
                request_kwargs = {
                    "model": self.config.model_name,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "timeout": int(kwargs.get("timeout", self.config.timeout)),
                }
                token_key = (
                    "max_completion_tokens"
                    if self.config.use_max_completion_tokens
                    else "max_tokens"
                )
                request_kwargs[token_key] = max_tokens
                if self.config.use_extra_body:
                    request_kwargs["extra_body"] = {
                        "top_k": self.config.top_k,
                        "repetition_penalty": self.config.repetition_penalty,
                        "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking},
                    }
                response = self.client.chat.completions.create(**request_kwargs)
                return self._normalize_tool_call_response(response.choices[0].message.content)

            except (RateLimitError, APIError, APITimeoutError) as e:
                if attempt == max_retries:
                    logging.warning("All %d attempts failed. Last error: %s", max_retries + 1, e)
                    return None

                delay = base_delay * (2 ** attempt)
                logging.info(
                    "Attempt %d failed with error: %s. Retrying in %.1f seconds...",
                    attempt + 1,
                    e,
                    delay,
                )
                time.sleep(delay)

            except Exception as e:
                logging.error("Non-retryable error occurred: %s", e)
                return None

    def set_context(self, context: Context):
        """
        Set context, e.g., environment variables (API keys).
        """
        super().set_context(context)
        # LOCAL_LLM_* vars take explicit precedence; VLLM_* are legacy defaults and
        # must NOT override a base_url already set in the YAML config.
        api_key_override = context.env.get("LOCAL_LLM_API_KEY")
        base_url_override = context.env.get("LOCAL_LLM_BASE_URL")
        if api_key_override:
            self.config.api_key = api_key_override
        if base_url_override:
            self.config.base_url = base_url_override
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=_normalize_openai_base_url(self.config.base_url),
        )
