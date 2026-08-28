"""
Provides the base class for LLMs.

This module defines the BaseLLM class, which serves as a foundation for
implementing various LLM models. It includes abstract methods and utility
functions for generating content, handling messages, and managing configurations.
"""
# pylint: disable=unused-argument,broad-exception-caught
from abc import abstractmethod
import asyncio
import copy
import re
import traceback
from typing import Any, Dict, List

from openai.types.responses import Response as OpenAIResponse
from pydantic import BaseModel
from agentbrew.core.misc import ComponentABCMeta, ExportConfigMixin
from agentbrew.core.logger import get_logger
from agentbrew.tracing import Tracer
from agentbrew.callbacks.base import (
    BaseCallback,
    CallbackMessage,
    MessageType,
    Status,
    Event,
    send_message
)
from agentbrew.core.context import Context


class BaseLLM(ExportConfigMixin, metaclass=ComponentABCMeta):
    """
    Base class for LLMs.

    This abstract base class defines the interface and common functionality
    for LLM implementations. It includes methods for generating content,
    handling messages, and managing configurations.

    Attributes:
        config: Configuration settings for the LLM.
    """

    def __init__(self):
        self.config: Any = None
        self.logger = get_logger(self.__class__.__name__)
        self._project_id: str = ""
        self._name: str = ""
        self._context: Context = Context()

    @abstractmethod
    def _generate(self, messages: List[dict[str, str]], **kwargs) -> Any:
        """Generates content based on formatted messages.

        This abstract method must be implemented by subclasses to define the
        specific content generation logic for each LLM type.

        Args:
            messages (List[dict[str, str]]): A list of message dictionaries,
                each containing 'role' and 'content' keys.
            **kwargs: Additional keyword arguments for model-specific parameters.

        Returns:
            Any: The generated content or model response.
        """
        raise NotImplementedError("`Generate` must be implemented by a subclass.")

    def generate(
            self,
            messages: List[dict[str, str]],
            tracer: Tracer = None,
            callbacks: BaseCallback | List[BaseCallback] = None,
            **kwargs
    ) -> Any:
        """
        Generates content based on formatted messages with tracing support.

        This method wraps the _generate method, adding tracing functionality
        and error handling.

        Args:
            messages (List[dict[str, str]]): A list of message dictionaries,
                each containing 'role' and 'content' keys.
            tracer (Tracer, optional): Tracer object for tracking model outputs.
                If None, a new Tracer will be created.
            callbacks (BaseCallback | List[BaseCallback], optional):
                Callbacks for recording LLM call status and responses
            **kwargs: Additional keyword arguments for model-specific parameters.

        Returns:
            Any: The generated content or model response.

        Raises:
            Exception: If an error occurs during content generation.
        """
        if not self.support_remote_mcp():
            kwargs.pop("remote_mcp", None)
        if not self.support_tool_call():
            kwargs.pop("callable_tools", None)
        tracer = tracer if tracer else Tracer()

        with tracer.sprout() as t:
            send_message(callbacks, message=CallbackMessage(
                source=self.id, type=MessageType.EVENT, data=Event.BEFORE_CALL,
                metadata={"method": "generate"}, project_id=self.project_id))
            send_message(callbacks, message=CallbackMessage(
                source=self.id, type=MessageType.STATUS, data=Status.RUNNING,
                project_id=self.project_id))
            try:
                response = self._generate(messages, **kwargs)
                messages_dict = copy.deepcopy(messages)
                # Handle different response types for tracing
                if hasattr(response, 'output') and isinstance(response, OpenAIResponse):
                    # Handle OpenAI-style response objects
                    response_data = response.model_dump()
                    messages_dict_tmp = []
                    for message in messages_dict:
                        if not isinstance(message, dict):
                            messages_dict_tmp.append(message.model_dump())
                        else:
                            messages_dict_tmp.append(message)
                    messages_dict = messages_dict_tmp
                elif isinstance(response, BaseModel):
                    response_data = response.model_dump(mode="json")
                elif hasattr(response, 'choices') and hasattr(response, 'model'):
                    # Handle OpenAI-style response objects
                    response_data = {
                        "model": getattr(response, 'model', ''),
                        "choices": [
                            {
                                "message": {
                                    "content": getattr(choice.message, 'content', None),
                                    "tool_calls": getattr(choice.message, 'tool_calls', None)
                                }
                            } for choice in response.choices
                        ],
                        "usage": getattr(response, 'usage', None)
                    }
                else:
                    response_data = response

                t.add({
                    "type": "llm",
                    "class": self.__class__.__name__,
                    "config": self.config.to_dict(),
                    "messages": messages_dict,
                    "response": response_data,
                    "error": ""
                })
            except Exception as e:
                tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                t.add({
                    "type": "llm",
                    "class": self.__class__.__name__,
                    "config": self.config.to_dict(),
                    "messages": messages_dict,
                    "response": str(e) + "\n" + tb_str,
                    "error": str(e) + "\n" + tb_str
                })
                send_message(callbacks, message=CallbackMessage(
                    source=self.id, type=MessageType.ERROR, data=str(e),
                    project_id=self.project_id))
                send_message(callbacks, message=CallbackMessage(
                    source=self.id, type=MessageType.EVENT, data=Event.AFTER_CALL,
                    metadata={"method": "generate"}, project_id=self.project_id))
                send_message(callbacks, message=CallbackMessage(
                    source=self.id, type=MessageType.STATUS, data=Status.FAILED,
                    project_id=self.project_id))
                raise e

        # Handle different response types for callbacks
        if isinstance(response, BaseModel):
            callback_data = response.model_dump(mode="json")
        elif hasattr(response, 'choices') and hasattr(response, 'model'):
            # Handle OpenAI-style response objects
            callback_data = {
                "model": getattr(response, 'model', ''),
                "choices": [
                    {
                        "message": {
                            "content": getattr(choice.message, 'content', None),
                            "tool_calls": getattr(choice.message, 'tool_calls', None)
                        }
                    } for choice in response.choices
                ]
            }
        else:
            # Ensure callback_data is never None to avoid CallbackMessage validation errors
            if hasattr(response, 'output') and isinstance(response, OpenAIResponse):
                callback_data = response.model_dump()
            else:
                callback_data = response if response is not None else ""

        send_message(callbacks, message=CallbackMessage(
            source=self.id, type=MessageType.RESPONSE,
            data=callback_data,
            project_id=self.project_id))
        send_message(callbacks, message=CallbackMessage(
            source=self.id, type=MessageType.EVENT, data=Event.AFTER_CALL,
            metadata={"method": "generate"}, project_id=self.project_id))
        send_message(callbacks, message=CallbackMessage(
            source=self.id, type=MessageType.STATUS, data=Status.SUCCEEDED,
            project_id=self.project_id))
        return response

    async def _call_generate(
            self,
            messages: List[dict[str, str]],
            tracer: Tracer = None,
            callbacks: BaseCallback | List[BaseCallback] = None,
            **kwargs
    ):
        """Use asyncio to run the blocking call."""
        loop = asyncio.get_event_loop()
        executor = kwargs.pop("_executor", None)
        return await loop.run_in_executor(
            executor,
            lambda: self.generate(
                messages=messages,
                tracer=tracer,
                callbacks=callbacks,
                **kwargs,
            ),
        )

    async def generate_async(
            self,
            messages: List[dict[str, str]] = None,
            tracer: Tracer = None,
            callbacks: BaseCallback | List[BaseCallback] = None,
            **kwargs
    ) -> Any:
        """
        Asynchronously generates content based on formatted messages with tracing support.

        Args:
            messages (List[dict[str, str]]): A list of message dictionaries,
                each containing 'role' and 'content' keys.
            tracer (Tracer, optional): Tracer object for tracking model outputs.
                If None, a new Tracer will be created.
            callbacks (BaseCallback | List[BaseCallback], optional):
                Callbacks for recording LLM call status and responses
            **kwargs: Additional keyword arguments for model-specific parameters.

        Returns:
            Any: The generated content or model response.
        """
        retries = kwargs.pop("retries", 3)
        retry_delay = kwargs.pop("retry_delay", 5)
        timeout = kwargs.pop("timeout", 120)

        for attempt in range(retries + 1):
            try:
                return await asyncio.wait_for(
                    self._call_generate(
                        messages=messages,
                        tracer=tracer,
                        callbacks=callbacks,
                        **kwargs
                    ),
                    timeout=timeout
                )
            except asyncio.TimeoutError as e:
                if attempt < retries:
                    self.logger.warning("Timeout on attempt %d/%d. Retrying...", attempt + 1, retries + 1)
                    await asyncio.sleep(retry_delay)
                else:
                    self.logger.error("All %d attempts failed with timeout", retries + 1)
                    raise e
            except Exception as e:
                overflow = self._parse_context_overflow_error(e)
                if overflow is not None:
                    # A long ReAct history can push the request over the server's
                    # context window even though max_tokens was configured for a
                    # short reply; retrying identically always fails the same way.
                    # Shrink max_tokens to whatever headroom is actually left.
                    max_context, input_tokens = overflow
                    safe_max_tokens = max_context - input_tokens - 64
                    if safe_max_tokens >= 64:
                        self.logger.warning(
                            "Requested max_tokens too large for %d input tokens "
                            "(context window %d). Retrying with max_tokens=%d.",
                            input_tokens, max_context, safe_max_tokens,
                        )
                        kwargs["max_tokens"] = safe_max_tokens
                        continue
                    self.logger.error(
                        "Conversation history (%d tokens) leaves no room for a "
                        "reply within the %d-token context window.",
                        input_tokens, max_context,
                    )
                    raise e
                if attempt < retries:
                    self.logger.warning("Error on attempt %d/%d: %s. Retrying...", attempt + 1, retries + 1, str(e))
                    await asyncio.sleep(retry_delay)
                else:
                    self.logger.error("All %d attempts failed with error: %s", retries + 1, str(e))
                    raise e

    @staticmethod
    def _parse_context_overflow_error(exc: Exception) -> tuple[int, int] | None:
        """Detect an OpenAI-compatible "max_tokens too large for context window"
        error and extract (max_context_length, input_tokens) from its message,
        or return None if this isn't that error."""
        message = str(exc)
        if "maximum context length" not in message:
            return None
        match = re.search(
            r"maximum context length is (\d+) tokens and your request has (\d+) input tokens",
            message,
        )
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    def get_response(
            self,
            system_message: str,
            user_message: str,
            tracer: Tracer = None,
            **kwargs
    ):
        """
        Generates content based on system and user messages.

        Args:
            system_message (str): The system message providing context or instructions.
            user_message (str): The user's input or query.
            tracer (Tracer, optional): Tracer object for tracking model outputs.
            **kwargs: Additional keyword arguments for model-specific parameters.

        Returns:
            Any: The generated content or model response.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        return self.generate(messages, tracer=tracer, **kwargs)

    async def get_response_async(
            self,
            system_message: str,
            user_message: str,
            tracer: Tracer = None,
            **kwargs
    ):
        """
        Generates content based on system and user messages.

        Args:
            system_message (str): The system message providing context or instructions.
            user_message (str): The user's input or query.
            tracer (Tracer, optional): Tracer object for tracking model outputs.
            **kwargs: Additional keyword arguments for model-specific parameters.

        Returns:
            Any: The generated content or model response.
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        return await self.generate_async(messages, tracer=tracer, **kwargs)

    def dump_config(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the LLM configuration.

        Returns:
            Dict[str, Any]: A dictionary containing the LLM class name and configuration.
        """
        return {
            "class": self.__class__.__name__,
            "config": self.config.to_dict()
        }

    @property
    def name(self) -> str:
        """Return LLM name."""
        return self._name

    def set_name(self, name: str):
        """Set a new name."""
        self._name = name

    @property
    def project_id(self) -> str:
        """Return the ID of the project using this LLM."""
        return self._project_id

    @project_id.setter
    def project_id(self, value: str):
        """Set the ID of the project using this LLM."""
        self._project_id = value

    @property
    def id(self) -> str:
        """Return the ID of this LLM."""
        name = self._name if self._name else self.config.model_name
        if self._project_id:
            return f"{self._project_id}:llm:{name}"
        return f"llm:{name}"

    def get_children_ids(self) -> List[str]:
        """Return the IDs of child components."""
        return []

    def list_undefined_env_vars(self, **kwargs) -> List[str]:
        """
        Return a list of undefined environment variables used in this LLM.

        Returns:
            List[str]: A list of undefined environment variables.
        """
        context = self._context
        env_vars = self.__class__.env_vars
        return [name for name in env_vars if context.get_env(name) == ""]

    def set_context(self, context: Context):
        """
        Set context, e.g., environment variables (API keys).
        """
        self._context = context

    def support_remote_mcp(self) -> bool:
        """
        Return a flag indicating if the model supports remote MCP servers.
        """
        return False

    def support_tool_call(self) -> bool:
        """
        Return a flag indicating if the model supports function/tool call API.
        """
        return False
