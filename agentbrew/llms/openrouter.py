"""
OpenRouter LLMs
"""
# pylint: disable=broad-exception-caught
import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, Union, Optional, Type, List
from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError
from dotenv import load_dotenv
from pydantic import BaseModel as PydanticBaseModel

import requests

from agentbrew.core.config import BaseConfig
from agentbrew.core.context import Context
from .base import BaseLLM

load_dotenv()

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

model_name_map = {
    "Qwen3Coder_OR": "qwen/qwen3-coder",
    "GrokCoderFast1_OR": "x-ai/grok-code-fast-1",
    "GPTOSS120B_OR": "openai/gpt-oss-120b",
    "GPTOSS20B_OR": "openai/gpt-oss-20b",
    "DeepSeekV3_1_OR": "deepseek/deepseek-chat-v3.1",
    "DeepSeekV3_2_OR": "deepseek/deepseek-v3.2",
    "DeepSeekV3_2_685B_OR": "deepseek/deepseek-v3.2",
    "GLM4_5_OR": "z-ai/glm-4.5",
    "GLM4_5_AIR_OR": "z-ai/glm-4.5-air",
    "GLM4_6_OR": "z-ai/glm-4.6",
    "GLM5_OR": "z-ai/glm-5",
    "KimiK2_OR": "moonshotai/kimi-k2",
    "Qwen3_235B_OR": "qwen/qwen3-235b-a22b",
    "Qwen3_235B_A22B_OR": "qwen/qwen3-235b-a22b",
    "Qwen3Max_OR": "qwen/qwen3-max",
    "KimiK2_0905_OR": "moonshotai/kimi-k2-0905",
    "DeepSeekV3_1_Terminus_OR": "deepseek/deepseek-v3.1-terminus",
    "DeepSeekV3_2_EXP_OR": "deepseek/deepseek-v3.2-exp",
    "ClaudeSonnet4_5_OR": "anthropic/claude-sonnet-4.5",
    "ClaudeSonnet45_OR": "anthropic/claude-sonnet-4.5",
    "ClaudeHaiku4_5_OR": "anthropic/claude-haiku-4.5",
    "Ling1T_OR": "inclusionai/ling-1t",
    "Gemini3Pro_OR": "google/gemini-3.1-pro-preview",
    "Gemini3ProPreview_OR": "google/gemini-3.1-pro-preview",
    "Gemini3_1Pro_OR": "google/gemini-3.1-pro-preview",
    "Gemini3_1ProPreview_OR": "google/gemini-3.1-pro-preview",
    "Grok4_1Fast_Thinking_OR": "x-ai/grok-4.1-fast",
    "GPT5_OR": "openai/gpt-5",
    "GPT5_1_OR": "openai/gpt-5.1",
    "GPT5_1_Medium_OR": "openai/gpt-5.1",
    "GPT5_Medium_OR": "openai/gpt-5",
}

reasoning_disable_unsupported_model_prefixes = (
    "google/gemini-3",
    "openai/gpt-5",
    "openai/gpt-oss-",
)

@dataclass
class OpenRouterConfig(BaseConfig):
    """
    Configuration for OpenRouter language models.

    Attributes:
        model_name (str): The name of the OpenRouter model to use (default: "GPTOSS120B_OR").
        api_key (str): The OpenRouter API key (default: environment variable OPENROUTER_API_KEY).
        base_url (str): The OpenRouter-compatible API base URL.
        temperature (float): Controls randomness in output (default: 1.0).
        top_p (float): Controls diversity of output (default: 1.0).
        frequency_penalty (float): Penalizes frequent token use (default: 0.0).
        presence_penalty (float): Penalizes repeated topics (default: 0.0).
        max_completion_tokens (int): Maximum number of tokens in the completion (default: 2048).
        seed (int): Random seed for reproducibility (default: 12345).
        reasoning (str): Reasoning level (default: "high").
        parallel_tool_calls (bool): Whether to enable parallel tool calls (default: True).
    """
    model_name: str = "GPT5_1_OR"
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    base_url: str = os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_completion_tokens: int = 20000
    seed: int = 12345
    reasoning: str = "low"
    timeout: int = 180
    parallel_tool_calls: bool = True


class OpenRouterModel(BaseLLM):
    """
    OpenRouter language models.

    This class provides methods to interact with OpenRouter's language models,
    including generating responses based on input messages.

    Attributes:
        config_class (Type[OpenRouterConfig]): Configuration class for the model.
        alias (str): Alias for the model, used for identification.
    """
    config_class = OpenRouterConfig
    alias = "openrouter"
    env_vars = ["OPENROUTER_API_KEY"]

    def __init__(self, config: Optional[Union[Dict, str]] = None):
        super().__init__()
        self.config = OpenRouterModel.config_class.load(config)

    async def generate_async(self, messages=None, tracer=None, callbacks=None, **kwargs):
        kwargs.setdefault("timeout", self.config.timeout)
        return await super().generate_async(
            messages=messages,
            tracer=tracer,
            callbacks=callbacks,
            **kwargs
        )

    def _get_model_endpoints(self, model_name: str) -> Optional[list[dict]]:
        """
        Fetches endpoints for the model from OpenRouter,
        with metadata including provider, quantization, etc.
        """
        if self.config.base_url.rstrip("/") != DEFAULT_OPENROUTER_BASE_URL:
            return None

        # model_name should be something like "qwen/qwen3-max"
        url = f"{DEFAULT_OPENROUTER_BASE_URL}/models/{model_name}/endpoints"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}"
        }
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            endpoints = data.get("endpoints", [])
            return endpoints
        except Exception as e:
            logging.warning("Failed to fetch endpoints for model %s: %s", model_name, e)
            return None

    def _choose_non_fp4_provider(self, model_name: str) -> Optional[list[str]]:
        """
        Returns a list of provider slugs (or endpoints) excluding those with 'fp4' quantization.
        """
        endpoints = self._get_model_endpoints(model_name)
        if not endpoints:
            return None

        # Filter out endpoints whose quantization mention "fp4"
        good = []
        for ep in endpoints:
            quant = ep.get("quantization", "")
            provider = ep.get("provider_name")
            name = ep.get("name", "")
            # if any of these contains "fp4", skip
            if quant and "fp4" in quant.lower():
                continue
            if name and "fp4" in name.lower():
                continue
            # provider_name might also include quant info in some deployments
            if provider and "fp4" in provider.lower():
                continue
            good.append(provider)
        # dedupe
        good_unique = list(dict.fromkeys([p for p in good if p]))
        return good_unique or None

    def _generate(
            self,
            messages: List[dict[str, str]],
            response_format: Type[PydanticBaseModel] = None,
            **kwargs
    ):
        """
        Generates content using the OpenRouter model.

        Args:
            messages (List[dict[str, str]]): List of message dictionaries,
                each containing 'role' and 'content' keys.
            response_format (Type[PydanticBaseModel], optional): Pydantic model
                defining the structure of the desired output. If None, generates
                free-form text.
            **kwargs: Additional keyword arguments including:
                - max_retries (int): Maximum number of retry attempts (default: 5)
                - base_delay (float): Base delay in seconds for exponential backoff (default: 10.0)
                - timeout (int): Request timeout in seconds (default: 60)

        Returns:
            Union[str, PydanticBaseModel, None]: Generated content as a string
                if no response_format is provided, a Pydantic model instance if
                response_format is provided, or None if parsing structured output fails.
                Returns None if all retry attempts fail or non-retryable errors occur.
        """
        max_retries = kwargs.get("max_retries", 5)
        base_delay = kwargs.get("base_delay", 10.0)
        # Map model name to OpenRouter model name
        model_name = model_name_map.get(self.config.model_name, self.config.model_name)

        # Attempt to filter out fp4 providers
        non_fp4 = self._choose_non_fp4_provider(model_name)
        provider_hint = {}
        if non_fp4:
            # instruct OpenRouter to ignore all other providers
            provider_hint["only"] = non_fp4
            provider_hint["allow_fallbacks"] = True
            logging.info("Using only non-fp4 providers: %s", non_fp4)

        # incorporate provider hints into kwargs
        call_kwargs = dict(kwargs)
        # If provider_hint is specified, incorporate it into extra_body
        if provider_hint:
            call_kwargs.setdefault("extra_body", {})["provider"] = provider_hint

        for attempt in range(max_retries + 1):
            try:
                client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )
                # Prepare common parameters, starting with call_kwargs
                common_params = {
                    "messages": messages,
                    "model": model_name,
                    "temperature": self.config.temperature,
                    "timeout": int(kwargs.get("timeout", self.config.timeout)),
                    "top_p": self.config.top_p,
                    "frequency_penalty": self.config.frequency_penalty,
                    "presence_penalty": self.config.presence_penalty,
                    "seed": self.config.seed,
                    "max_completion_tokens": self.config.max_completion_tokens,
                    "parallel_tool_calls": self.config.parallel_tool_calls,
                }

                # Merge call_kwargs into common_params
                # Handle extra_body specially to merge it properly
                if "extra_body" in call_kwargs:
                    common_params.setdefault("extra_body", {}).update(call_kwargs.pop("extra_body"))
                common_params.update(call_kwargs)

                if not model_name.startswith(reasoning_disable_unsupported_model_prefixes):
                    common_params.setdefault("extra_body", {})["reasoning"] = {
                        "enabled": False
                    }

                if response_format is None:
                    chat = client.chat.completions.create(**common_params)
                    # If tools are provided, return the entire response object
                    # so the caller can handle both content and tool_calls
                    if 'tools' in kwargs:
                        return chat
                    # For backward compatibility, return just content when no tools
                    return chat.choices[0].message.content

                # Add response_format for structured output
                parse_params = common_params.copy()
                parse_params["response_format"] = response_format

                chat = client.beta.chat.completions.parse(**parse_params)
                # If tools are provided, return the entire response object
                # so the caller can handle both content and tool_calls
                if 'tools' in kwargs:
                    return chat
                # For backward compatibility, return just parsed content when no tools
                return chat.choices[0].message.parsed

            except (RateLimitError, APIError, APITimeoutError) as e:
                if attempt == max_retries:
                    # Last attempt failed, return None instead of raising
                    logging.warning("All %d attempts failed. Last error: %s", max_retries + 1, e)
                    return None

                # Calculate delay with exponential backoff
                delay = base_delay * (2 ** attempt)
                logging.info("Attempt %d failed with error: %s. Retrying in %.1f seconds...",
                           attempt + 1, e, delay)
                time.sleep(delay)

            except Exception as e:
                # For non-retryable errors, return None instead of raising
                logging.error("Non-retryable error occurred: %s", e)
                return None

    def set_context(self, context: Context):
        """
        Set context, e.g., environment variables (API keys).
        """
        super().set_context(context)
        self.config.api_key = context.env.get("OPENROUTER_API_KEY", self.config.api_key)
        self.config.base_url = context.env.get("OPENROUTER_BASE_URL", self.config.base_url)
