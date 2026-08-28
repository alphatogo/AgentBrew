"""
ReAct agent implementation with per-step summarization (no Mem0).
Uses the same VLLM model (or other provider) to summarize each step.
Injects:
- All past step summaries
- Last 3 steps' full traces
into the main ReAct prompt.
"""
import os
import json
import uuid
from typing import Optional, Union, Dict, List, Any
from collections import OrderedDict
from dataclasses import dataclass, field
from contextlib import contextmanager
import asyncio

from mcp.types import TextContent

from agentbrew.mcp.manager import MCPManager
from agentbrew.llms.base import BaseLLM
from agentbrew.core.logger import get_logger
from agentbrew.tracing import Tracer
from agentbrew.callbacks.base import (
    send_message,
    send_message_async,
    CallbackMessage,
    MessageType
)
from .base import BaseAgentConfig, BaseAgent
from agentbrew.browser.postprocess import postprocess_tool_output
from agentbrew.agents.notion_postprocess import postprocess_notion_tool_output
from .react import ReAct, ReActConfig
from .utils import build_system_prompt
from .types import AgentResponse

DEFAULT_CONFIG_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "prompts")


class RepeatedToolCallError(Exception):
    """Raised when the same tool (with identical arguments) is called consecutively too many times."""
    pass


@dataclass
class ReActSummaryConfig(ReActConfig):
    """
    Configuration for ReActSummary agent with per-step summarization.
    """
    # Main ReAct system prompt
    system_prompt: str = os.path.join(DEFAULT_CONFIG_FOLDER, "react_summary_prompt.j2")
    # Prompt template path used for the per-step summary (set in config)
    # The template receives: the question, the history summary, and the current step's
    # full content, and outputs a short summary of that step
    summary_prompt: str = os.path.join(DEFAULT_CONFIG_FOLDER, "react_step_summary_prompt.j2")

    # Prompt used by the LLM to decide whether to truncate history/tool output
    truncate_prompt: str = os.path.join(DEFAULT_CONFIG_FOLDER, "react_truncate_history_prompt.j2")

    # Short-term full-trace window (most recent N steps)
    history_window_steps: int = 3

    # Max tool output length (hard truncation limit)
    max_tool_output_length: int = 10000

    # Inference params for the summary call to vLLM (uses the same BaseLLM)
    summary_max_tokens: int = 1024
    truncate_max_tokens: int = 1024
    summary_temperature: float = 0.1

    # Inference params for the LLM used in the truncation decision
    truncate_temperature: float = 0.0

    task_prompt: str = ""


class HistoryManager:
    """Short-term history manager; keeps the full trace text of the most recent N steps."""

    def __init__(self, window_size: int, max_text_length: int):
        self.window_size = window_size
        self.max_text_length = max_text_length
        # Each element represents the multi-line log (list of strings) for one step
        self.history: List[List[str]] = []

    def add_step(self, content: str, is_new: bool = False):
        """Add content to the history."""
        if is_new:
            self.history.append([content])
        else:
            if not self.history:
                self.history.append([])
            self.history[-1].append(content)

        # Cap the window size, keeping only the most recent window_size steps
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def get_formatted_history(self) -> str:
        """Get the formatted history text (only the most recent window_size steps)."""
        if not self.history:
            return ""
        history_str = "\n\n".join(["\n".join(step) for step in self.history])
        return history_str

    # -------- Low-level truncation logic: execution only, no decision-making --------

    def truncate_with_strategy(
        self,
        text: str,
        strategy: str,
        keep_chars: int = None,
        keep_head: int = None,
        keep_tail: int = None,
    ) -> str:
        """
        Truncate text according to a strategy; this only executes the truncation,
        it does not decide whether truncation is needed:
        - strategy == "head": keep the first keep_chars
        - strategy == "tail": keep the last keep_chars
        - strategy == "head_tail": keep the first keep_head + last keep_tail,
          with "..." marking the omitted middle
        """
        length = len(text)
        if length == 0:
            return text

        if strategy == "head":
            if keep_chars is None or keep_chars >= length:
                return text
            hidden = length - keep_chars
            return (
                text[:keep_chars]
                + f"\n\n... [SYSTEM NOTE: Data truncated. {hidden} chars hidden from the end.]"
            )

        if strategy == "tail":
            if keep_chars is None or keep_chars >= length:
                return text
            hidden = length - keep_chars
            return (
                f"... [SYSTEM NOTE: Data truncated. {hidden} chars hidden from the beginning.]\n\n"
                + text[-keep_chars:]
            )

        if strategy == "head_tail":
            keep_head = keep_head or 0
            keep_tail = keep_tail or 0
            if keep_head + keep_tail >= length:
                return text
            hidden = length - keep_head - keep_tail
            return (
                text[:keep_head]
                + f"\n\n... [SYSTEM NOTE: Middle truncated. {hidden} chars hidden.]\n\n"
                + text[-keep_tail:]
            )

        # Unknown strategy -> don't truncate
        return text

    def truncate_text_hard_limit(self, text: str) -> str:
        """
        The original simple hard-truncation logic, used as a fallback:
        keep the first max_text_length characters and append a note at the end.
        """
        if len(text) <= self.max_text_length:
            return text

        truncated_len = len(text) - self.max_text_length
        return (
            text[:self.max_text_length] +
            f"\n\n... [SYSTEM NOTE: Data truncated. {truncated_len} chars hidden. Refine query to see more.]"
        )

    def clear(self):
        """Clear the history."""
        self.history.clear()


class ReActSummary(ReAct):
    """
    ReAct Agent with per-step summarization, no Mem0.
    - After each step, the same LLM (e.g. vLLM) is called to generate a summary of that step.
    - The prompt uses the template specified by config.summary_prompt.
    - The main prompt is injected with:
        * all step summaries (long-term, concise)
        * the full trace of the most recent N steps (detailed)
    """
    alias = ["react_summary"]
    config_class = ReActSummaryConfig

    def __init__(
        self,
        mcp_manager: MCPManager,
        llm: BaseLLM,
        config: Optional[Union[Dict, str]] = None
    ):
        super().__init__(mcp_manager=mcp_manager, llm=llm, config=config)
        self._logger = get_logger(f"{self.__class__.__name__}:{self._name}")

        self.history_manager = HistoryManager(
            self._config.history_window_steps,
            self._config.max_tool_output_length
        )

        # Stores the summary (short text) of every step
        self.step_summaries: List[str] = []

        # Repeated tool call detection
        self._last_tool_key: Optional[str] = None
        self._consecutive_same_calls: int = 0
        self._max_consecutive_same_calls: int = 10

    # -----------------------------
    # Helper: extract text from a tool result
    # -----------------------------
    def _extract_text_from_result(self, tool_result: Any) -> str:
        """Extract text from a tool result."""
        if hasattr(tool_result, 'content'):
            parts = []
            for item in tool_result.content:
                if item.type == "text" and item.text:
                    parts.append(item.text)
                elif item.type == "resource" and hasattr(item.resource, 'text'):
                    parts.append(item.resource.text)
            return "\n".join(parts)

        return str(tool_result)

    # -----------------------------
    # Per-step summary
    # -----------------------------
    async def _summarize_step(
        self,
        *,
        question: str,
        step_index: int,
        thought: str,
        action: Union[str, Dict, None],
        result: str,
    ) -> str:
        """
        Use the same BaseLLM to produce a short summary of the current step's trace.
        The single-step summary prompt is filled in with:
        - the question (QUESTION)
        - what's been completed so far (PAST_SUMMARIES)
        - everything from the current step (STEP_FULL_CONTENT)
        """
        try:
            # 1) Full content of the current step
            current_action_str = (
                json.dumps(action, ensure_ascii=False)
                if isinstance(action, dict)
                else (str(action) if action is not None else "")
            )
            step_full_content = "\n".join([
                f"Step {step_index}",
                f"Thought: {thought or ''}",
                f"Action: {current_action_str}",
                f"Result: {result or ''}",
            ])

            # 2) Summaries of all past steps
            if self.step_summaries:
                past_summaries_text = "\n".join(
                    f"- {s}" for s in self.step_summaries
                )
            else:
                past_summaries_text = ""

            params = {
                "QUESTION": question,
                "STEP_INDEX": step_index,
                "PAST_SUMMARIES": past_summaries_text,
                "STEP_FULL_CONTENT": step_full_content,
            }

            # 3) Build the single-step summary prompt
            summary_prompt = build_system_prompt(
                system_prompt_template=self._config.summary_prompt,
                tool_prompt_template="",
                tools=None,
                include_tool_description=False,
                **params,
            )

            summary_response = await self._llm.generate_async(
                messages=[{"role": "user", "content": summary_prompt}],
                tracer=None,
                callbacks=[],
            )

            summary_text = str(summary_response).strip()
            summary_text = summary_text.strip("`").strip()
            if summary_text.startswith("json"):
                summary_text = summary_text[4:].strip()
            return summary_text

        except Exception as e:
            self._logger.error(f"Step summarization failed at step {step_index}: {e}")
            return f"Step {step_index} summary unavailable due to error."

    # -----------------------------
    # Truncation decision (LLM-driven)
    # -----------------------------
    async def _decide_truncate(
        self,
        *,
        question: str,
        text: str,
        thought: str,
        action: str
    ) -> str:
        """
        Use the LLM (via truncate_prompt) to decide whether and how to truncate text.
        Returns a JSON string (the LLM's raw output), or an empty string on error.
        """
        try:
            max_len = self._config.max_tool_output_length

            params = {
                "QUESTION": question,
                "CURRENT_TEXT": text,
                "THOUGHT": thought,
                "ACTION": action,
            }

            truncate_prompt = build_system_prompt(
                system_prompt_template=self._config.truncate_prompt,
                tool_prompt_template="",
                tools=None,
                include_tool_description=False,
                **params,
            )


            resp = await self._llm.generate_async(
                messages=[{"role": "user", "content": truncate_prompt}],
                tracer=None,
                callbacks=[]
            )
            text_resp = str(resp).strip().strip("`").strip()
            if text_resp.startswith("json"):
                text_resp = text_resp[4:].strip()
            return text_resp
        except Exception as e:
            self._logger.error(f"Truncate decision failed: {e}")
            return ""


    async def _truncate_with_llm(
        self,
        *,
        question: str,
        raw_text: str,
        thought: str,
        action: str
    ) -> str:
        """
        Unified external entry point: first use the LLM to decide whether and how to
        truncate, then execute via HistoryManager. If the decision fails or returns
        invalid JSON, fall back to a simple hard limit.
        """
        max_len = self._config.max_tool_output_length

        if len(raw_text) <= max_len:
            return raw_text

        if len(raw_text) >= max_len*3:
            keep_head = max_len // 2
            keep_tail = max_len - keep_head
            return self.history_manager.truncate_with_strategy(
                raw_text,
                strategy="head_tail",
                keep_head=keep_head,
                keep_tail=keep_tail,
            )

        decision_str = await self._decide_truncate(question=question, text=raw_text, thought=thought, action=action)

        if not decision_str:
            keep_head = max_len // 2
            keep_tail = max_len - keep_head
            return self.history_manager.truncate_with_strategy(
                raw_text,
                strategy="head_tail",
                keep_head=keep_head,
                keep_tail=keep_tail,
            )

        try:
            decision = json.loads(decision_str)
        except json.JSONDecodeError as e:
            self._logger.error(
                f"Truncate decision JSON parse error: {e}, raw: {decision_str[:500]}"
            )
            keep_head = max_len // 2
            keep_tail = max_len - keep_head
            return self.history_manager.truncate_with_strategy(
                raw_text,
                strategy="head_tail",
                keep_head=keep_head,
                keep_tail=keep_tail,
            )

        if not decision.get("truncate", False):
            return raw_text

        strategy = decision.get("strategy", "head")

        if strategy in ("head", "tail"):
            keep_chars = max_len
            return self.history_manager.truncate_with_strategy(
                raw_text,
                strategy=strategy,
                keep_chars=keep_chars,
            )

        if strategy == "head_tail":
            keep_head = max_len // 2
            keep_tail = max_len - keep_head
            return self.history_manager.truncate_with_strategy(
                raw_text,
                strategy="head_tail",
                keep_head=keep_head,
                keep_tail=keep_tail,
            )

        keep_head = max_len // 2
        keep_tail = max_len - keep_head
        return self.history_manager.truncate_with_strategy(
            raw_text,
            strategy="head_tail",
            keep_head=keep_head,
            keep_tail=keep_tail,
        )


    def _build_prompt(self, question: str) -> str:

        params = {
            "INSTRUCTION": self._config.instruction,
            "QUESTION": question,
            "MAX_STEPS": self._config.max_iterations,
            "TASK_PROMPT": self._config.task_prompt
        }

        # All history summaries (long-term, condensed)
        if self.step_summaries:
            summaries_text = "\n".join(
                f"- {s}" for s in self.step_summaries
            )
            params["STEP_SUMMARIES"] = summaries_text

        # Full trace of the most recent N steps
        history = self.history_manager.get_formatted_history()
        if history:
            params["RECENT_HISTORY"] = history

        if self._config.context_examples:
            params["CONTEXT_EXAMPLES"] = self._config.context_examples

        params.update(self._config.template_vars)

        return build_system_prompt(
            system_prompt_template=self._config.system_prompt,
            tool_prompt_template=self._config.tools_prompt,
            tools=self._tools,
            **params
        )

    # -----------------------------
    # Single iteration
    # -----------------------------
    async def _process_iteration(
        self,
        message: str,
        task_session_id: str,
        iter_num: int,
        tracer: Tracer,
        callbacks: List
    ) -> Optional[AgentResponse]:
        """Process a single ReAct iteration."""

        # Phase 1: build the prompt & generate the ReAct response
        prompt = self._build_prompt(message)
        messages = [{"role": "user", "content": prompt}]

        response = await self._llm.generate_async(
            messages=messages,
            tracer=tracer,
            callbacks=callbacks
        )

        # Phase 2: parse and execute
        thought_for_summary: str = ""
        action_for_summary: Union[str, Dict, None] = None
        result_for_summary: str = ""

        try:
            try:
                parsed = self._parse_response(response)
            except Exception:
                tracer.record_llm(
                    {"text": str(response or "")},
                    messages=messages,
                    response=response,
                )
                raise
            tracer.record_llm(
                parsed,
                messages=messages,
                response=response,
            )

            # Start a new step (full trace)
            step_index = iter_num + 1
            self.history_manager.add_step(f"Step {step_index}", is_new=True)

            # Handle the final answer
            if "answer" in parsed:
                final_answer = parsed["answer"]
                thought_for_summary = parsed.get("thought", "")
                action_for_summary = None
                result_for_summary = final_answer

                self.history_manager.add_step(f"Thought: {thought_for_summary}")
                self.history_manager.add_step(f"Answer: {final_answer}")

                await self._send_callback_message(
                    callbacks, iter_num,
                    thought=thought_for_summary,
                    answer=final_answer
                )

                # Summarize this step before finishing
                step_summary = await self._summarize_step(
                    question=message,
                    step_index=step_index,
                    thought=thought_for_summary,
                    action=action_for_summary,
                    result=result_for_summary,
                )
                self.step_summaries.append(step_summary)

                return AgentResponse(
                    name=self._name,
                    class_name=self.__class__.__name__,
                    response=final_answer,
                    trace_id=tracer.trace_id
                )

            # Handle an Action
            if "action" in parsed:
                thought_for_summary = parsed.get("thought", "")
                action_for_summary = parsed["action"]

                await self._process_action(
                    parsed, iter_num, tracer, callbacks,
                    thought_for_summary=thought_for_summary,
                    question=message,
                )

            # Handle a direct result (no tool call)
            elif "result" in parsed:
                thought_for_summary = parsed.get("thought", "")
                # result_for_summary uses the raw result; truncation is handled inside _process_result
                result_for_summary = parsed["result"]
                action_for_summary = None

                await self._process_result(
                    parsed, iter_num, callbacks,
                    thought_for_summary=thought_for_summary,
                    question=message,
                )

        except RepeatedToolCallError as e:
            self._logger.warning(f"Repeated tool call detected, stopping task: {e}")
            self.history_manager.add_step(f"[STOPPED] {e}")
            return AgentResponse(
                name=self._name,
                class_name=self.__class__.__name__,
                response=str(e),
                trace_id=tracer.trace_id
            )
        except json.JSONDecodeError as e:
            self._logger.error(f"JSON Parse Error: {e}")
            self.history_manager.add_step(
                "Error: Failed to parse JSON response"
            )
        except Exception as e:
            self._logger.error(f"Iteration error: {e}")
            self.history_manager.add_step(f"Error: {str(e)[:500]}")

        return None

    def _parse_response(self, response: str) -> Dict:
        """Parse the LLM response (JSON)."""
        import re  # pylint: disable=import-outside-toplevel
        response = response.strip()
        # Strip Qwen3-style <think>...</think> reasoning tokens
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        response = response.strip('`').strip()
        if response.startswith("json"):
            response = response[4:].strip()
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            sanitized = self._sanitize_json_escapes(response)
            try:
                return json.loads(sanitized)
            except json.JSONDecodeError:
                # LLM commonly embeds multi-line file/code content in a JSON string
                # without escaping the newlines (e.g. answers containing a full file's
                # contents for "add file with exact content"-style tasks), which raises
                # "Invalid control character" rather than a plain syntax error. Escape
                # any raw control characters that appear inside string literals before
                # giving up on this attempt.
                control_escaped = self._escape_raw_control_chars_in_strings(sanitized)
                if control_escaped != sanitized:
                    try:
                        return json.loads(control_escaped)
                    except json.JSONDecodeError:
                        pass
                else:
                    control_escaped = sanitized
                # LLM may append extra text after the JSON object ("Extra data" error).
                # Fall back to extracting just the first complete {...} block.
                extracted = self._extract_first_json(control_escaped)
                if extracted:
                    try:
                        return json.loads(extracted)
                    except json.JSONDecodeError:
                        pass
                # Final fallback: use json_repair to fix structural issues
                # (missing commas, single quotes, etc.) that LLMs commonly produce.
                try:
                    from json_repair import repair_json  # pylint: disable=import-outside-toplevel
                    repaired = repair_json(control_escaped or response, return_objects=True)
                    if isinstance(repaired, dict):
                        return repaired
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
                raise

    @staticmethod
    def _escape_raw_control_chars_in_strings(text: str) -> str:
        """Escape literal control characters (newline, tab, CR, etc.) found inside
        JSON string literals. LLMs frequently paste multi-line file/code content
        straight into a JSON string without escaping the newlines, which json.loads
        rejects as an "Invalid control character" rather than a recoverable syntax
        error. Structural characters outside of strings are left untouched."""
        escapes = {
            "\n": "\\n",
            "\r": "\\r",
            "\t": "\\t",
            "\b": "\\b",
            "\f": "\\f",
        }
        out = []
        in_string = False
        escape_next = False
        for ch in text:
            if escape_next:
                out.append(ch)
                escape_next = False
                continue
            if ch == "\\" and in_string:
                out.append(ch)
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                out.append(ch)
                continue
            if in_string and ch in escapes:
                out.append(escapes[ch])
                continue
            if in_string and ord(ch) < 0x20:
                out.append(f"\\u{ord(ch):04x}")
                continue
            out.append(ch)
        return "".join(out)

    @staticmethod
    def _extract_first_json(text: str) -> str:
        """Extract the first complete {...} JSON object from text."""
        start = text.find('{')
        if start == -1:
            return ""
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start=start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return ""

    @staticmethod
    def _sanitize_json_escapes(text: str) -> str:
        """Fix invalid JSON escape sequences (e.g. \' from SQL) so json.loads can parse."""
        import re  # pylint: disable=import-outside-toplevel
        # Replace any \x where x is not a valid JSON escape character with just x.
        # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
        return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

    # -----------------------------
    # Action / Result handling
    # -----------------------------
    async def _process_action(
        self,
        parsed: Dict,
        iter_num: int,
        tracer: Tracer,
        callbacks: List,
        thought_for_summary: str = "",
        question: str = "",
    ):
        """Execute an Action."""
        thought = thought_for_summary or parsed.get("thought", "")
        self.history_manager.add_step(f"Thought: {thought}")

        action = parsed["action"]

        # Validate the Action format
        if not isinstance(action, dict) or "server" not in action or "tool" not in action:
            error_msg = f"Invalid action format: {action}"
            self.history_manager.add_step(f"Action: {error_msg}")
            self.history_manager.add_step("Result: Invalid action structure")

            await self._send_callback_message(
                callbacks, iter_num,
                thought=thought,
                action=str(action),
                result="Invalid action"
            )

            # Also generate a summary for this step
            step_index = iter_num + 1
            step_summary = await self._summarize_step(
                question=question,
                step_index=step_index,
                thought=thought,
                action=action,
                result="Invalid action",
            )
            self.step_summaries.append(step_summary)
            return

        # Record the Action
        self.history_manager.add_step(
            f"Action: Using `{action.get('tool')}` in `{action.get('server')}`"
        )
        self.history_manager.add_step(
            f"Action Input: {str(action.get('arguments'))}"
        )

        # Detect consecutive repeated tool calls
        try:
            args_str = json.dumps(action.get("arguments") or {}, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            args_str = str(action.get("arguments", ""))
        tool_key = f"{action.get('server')}::{action.get('tool')}::{args_str}"

        if tool_key == self._last_tool_key:
            self._consecutive_same_calls += 1
        else:
            self._consecutive_same_calls = 1
            self._last_tool_key = tool_key

        if self._consecutive_same_calls >= self._max_consecutive_same_calls:
            raise RepeatedToolCallError(
                f"Tool '{action.get('tool')}' on server '{action.get('server')}' "
                f"called with identical arguments {self._max_consecutive_same_calls} "
                f"times consecutively. Stopping task to prevent abuse."
            )

        # Execute the tool
        # Tools that trigger JS-driven rendering: wait for the page to settle before reading the result
        _JS_TRIGGER_TOOLS = {
            "browser_type",        # search box input / submit
            "browser_fill_form",   # form submission
            "browser_select_option",  # dropdown selection
            "browser_press_key",   # Enter triggers a search
            "browser_click",       # a click may trigger SPA routing/loading
            "browser_navigate",    # page navigation
        }
        _JS_WAIT_SECONDS = 2.0   # seconds to wait for JS rendering

        tool_recorded = False
        try:
            tool_result = await self.call_tool(
                action, tracer=tracer, callbacks=callbacks
            )

            # Tools that trigger dynamic JS loading: wait for rendering to finish before processing the result
            tool_name = action.get("tool", "")
            full_text = self._extract_text_from_result(tool_result)
            tracer.record_tool(
                server=action.get("server", ""),
                tool_name=tool_name,
                arguments=action.get("arguments") or {},
                content=full_text,
                is_error=bool(getattr(tool_result, "isError", False)),
            )
            tool_recorded = True
            if tool_name in _JS_TRIGGER_TOOLS:
                await asyncio.sleep(_JS_WAIT_SECONDS)

            # Extract and truncate the text
            if tool_name.startswith("browser_"):
                # Browser tools: rule-based filtering first, then LLM compression if needed
                truncated_text = postprocess_tool_output(tool_name, full_text)
                if len(truncated_text) > self._config.max_tool_output_length:
                    truncated_text = await self._truncate_with_llm(
                        question=question,
                        raw_text=truncated_text,
                        thought=thought,
                        action=action
                    )
            else:
                # Notion and similar tools: rule-based compression first, then LLM truncation if needed
                compressed_text = postprocess_notion_tool_output(tool_name, full_text)
                if len(compressed_text) > self._config.max_tool_output_length:
                    truncated_text = await self._truncate_with_llm(
                        question=question,
                        raw_text=compressed_text,
                        thought=thought,
                        action=action
                    )
                else:
                    truncated_text = compressed_text

            self.history_manager.add_step(f"Result: {truncated_text}")

            await self._send_callback_message(
                callbacks, iter_num,
                thought=thought,
                action=action,
                result=truncated_text
            )

            # Summarize at the end of every action step
            step_index = iter_num + 1
            step_summary = await self._summarize_step(
                question=question,
                step_index=step_index,
                thought=thought,
                action=action,
                result=truncated_text,
            )
            self.step_summaries.append(step_summary)

        except Exception as e:
            self._logger.error(f"Tool execution failed: {e}")
            error_msg = f"Tool Execution Error - {str(e)[:500]}"
            if not tool_recorded:
                tracer.record_tool(
                    server=action.get("server", ""),
                    tool_name=action.get("tool", ""),
                    arguments=action.get("arguments") or {},
                    content=error_msg,
                    is_error=True,
                )
            self.history_manager.add_step(f"Result: {error_msg}")

            await self._send_callback_message(
                callbacks, iter_num,
                thought=thought,
                action=action,
                result=error_msg
            )

            # Also summarize on error
            step_index = iter_num + 1
            step_summary = await self._summarize_step(
                question=question,
                step_index=step_index,
                thought=thought,
                action=action,
                result=error_msg,
            )
            self.step_summaries.append(step_summary)

    async def _process_result(
        self,
        parsed: Dict,
        iter_num: int,
        callbacks: List,
        thought_for_summary: str = "",
        question: str = "",
    ):
        """Handle a direct result (no tool call)."""
        thought = thought_for_summary or parsed.get("thought", "")
        self.history_manager.add_step(f"Thought: {thought}")

        # Use the LLM to decide whether to truncate
        raw_result = parsed["result"]
        result_content = await self._truncate_with_llm(
            question=question,
            raw_text=raw_result,
            thought=thought,
            action="",
        )
        self.history_manager.add_step(f"Result: {result_content}")

        await self._send_callback_message(
            callbacks, iter_num,
            thought=thought,
            result=result_content
        )

        # Generate the summary (using the truncated result here to avoid an overly long summary)
        step_index = iter_num + 1
        step_summary = await self._summarize_step(
            question=question,
            step_index=step_index,
            thought=thought,
            action=None,
            result=result_content,
        )
        self.step_summaries.append(step_summary)

    # -----------------------------
    # Main execution loop
    # -----------------------------
    async def _execute(
        self,
        message: Union[str, List[str]],
        output_format: Optional[Union[str, Dict]] = None,
        **kwargs
    ) -> AgentResponse:
        """Run the main execution loop."""

        # Normalize the input
        if isinstance(message, (list, tuple)):
            message = "\n".join(message)

        if output_format is not None:
            message = f"{message}\n\n{self._get_output_format_prompt(output_format)}"

        # Initialize
        task_session_id = kwargs.get("trace_id", str(uuid.uuid4()))
        tracer = kwargs.get("tracer", Tracer())
        callbacks = kwargs.get("callbacks", [])

        # Clear the history and summaries
        self.history_manager.clear()
        self.step_summaries.clear()

        # Reset the repeated-call detector
        self._last_tool_key = None
        self._consecutive_same_calls = 0

        # Main loop
        for iter_num in range(self._config.max_iterations):
            result = await self._process_iteration(
                message, task_session_id,
                iter_num, tracer, callbacks
            )

            if result:
                return result

        # Max iterations reached
        return AgentResponse(
            name=self._name,
            class_name=self.__class__.__name__,
            response="Max iterations reached without a final answer.",
            trace_id=tracer.trace_id
        )

    # -----------------------------
    # State management
    # -----------------------------
    def get_history(self) -> str:
        """Get the currently active history (full trace of the most recent N steps)."""
        return self.history_manager.get_formatted_history()

    def clear_history(self):
        """Clear the history and summaries."""
        self.history_manager.clear()
        self.step_summaries.clear()

    def reset(self):
        """Reset the agent."""
        self.clear_history()

    # -----------------------------
    # Logging callback
    # -----------------------------
    @staticmethod
    async def _send_callback_message(
        callbacks: List,
        iter_num: int,
        thought: str = "",
        action: Union[str, Dict] = "",
        result: str = "",
        answer: str = ""
    ):
        """Send a callback message (keeps the original logic)."""
        logs = []
        if thought:
            logs.append(("thought", thought))
        if action:
            logs.append(("action", str(action)))
        if result:
            logs.append(("result", result))
        if answer:
            logs.append(("answer", answer))

        data = OrderedDict({"Iteration": iter_num + 1})
        for tag, value in logs:
            data[tag] = value

        # Structured log
        send_message(callbacks, message=CallbackMessage(
            source=__file__,
            type=MessageType.LOG,
            data=data
        ))

        # Plain-text log
        data_text = [
            f"{'=' * 66}\n",
            f"Iteration: {iter_num + 1}\n",
            f"{'-' * 66}\n",
        ]
        for tag, value in logs:
            data_text.append(f"\033[32m{tag.capitalize()}: {value}\n\n\033[0m")

        await send_message_async(
            callbacks,
            message=CallbackMessage(
                source=__file__,
                type=MessageType.LOG,
                metadata={
                    "event": "plain_text",
                    "data": "".join(data_text)
                }
            )
        )
