# Air/agent.py
import os
import asyncio
import json
from typing import Any, Dict, List, Optional
import msgpack

from Air.logging import logging_instance as log
from Air.llm import LLM


class Agent:
    """
    🌬️ Air Agent — Ultra-lightweight LLM agent with tool support.

    Agents can use tools via JSON. The LLM returns:
    {
        "action": "use_tool",
        "tool": "tool_name",
        "params": {...}
    }

    Agent loops until LLM stops requesting tools.
    """

    def __init__(
        self,
        name: str,
        role: str,
        description: str,
        goal: str,
        tools: Optional[List[Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = "fast_agents_cache/agents",
        preload: bool = False,
        verbose: bool = True,
    ):
        self.name = name
        self.role = role
        self.description = description
        self.goal = goal
        self.tools = tools or []
        self.llm_config = llm_config or {}
        self.verbose = verbose
        self.agent_obj: Optional[LLM] = None

        # === Disk Caching ===
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._config_path = os.path.join(self.cache_dir, f"{self.name}.msgpack")
        else:
            self._config_path = None

        # === System Prompt with Tools Info ===
        self._system_prompt = self._build_system_prompt()

        # === Load cached config if possible ===
        if self.cache_dir:
            self._load_config()
        else:
            self._save_config = lambda: None

        if self.verbose:
            log.dim(f"🌬️ Air Agent '{self.name}' ready")
            log.dim(f"Role: {self.role}")
            log.info(f"Goal: {self.goal}")
            if self.tools:
                log.info(f"Tools: {', '.join([t.name for t in self.tools])}")

    def _build_system_prompt(self) -> str:
        """Build system prompt including tool information."""
        base_prompt = (
            f"You are {self.name}. Role: {self.role}\n"
            f"Description: {self.description}\n"
            f"Goal: {self.goal}\n"
            f"Provide direct, clear analysis without verbose preamble."
        )

        if not self.tools:
            return base_prompt

        # Add tools section
        tools_section = "\n\n=== AVAILABLE TOOLS ===\n"
        for tool in self.tools:
            schema = tool.get_schema()
            tools_section += f"\nTool: {tool.name}\n"
            tools_section += f"Description: {schema['description']}\n"
            tools_section += (
                f"Parameters: {json.dumps(schema['parameters'], indent=2)}\n"
            )

        tools_section += (
            "\n=== OPTIONAL TOOL USAGE ===\n"
            "You MAY use tools if you think they will help. It's optional.\n"
            "To use a tool, return JSON in this format:\n"
            '{"action": "use_tool", "tool": "tool_name", "params": {"param1": "value1", ...}}\n'
            "The tool will execute and return results. You can then use those results in your answer.\n"
            "You can also provide your answer directly without using any tools.\n"
        )

        return base_prompt + tools_section

    # =====================================================================
    # Disk Caching
    # =====================================================================
    def _load_config(self):
        """Load agent configuration from MessagePack cache."""
        try:
            if os.path.exists(self._config_path):
                with open(self._config_path, "rb") as f:
                    data = msgpack.unpackb(f.read(), strict_map_key=False)
                self.llm_config = data.get("llm_config", self.llm_config)
                if self.verbose:
                    log.fade(f"Loaded cache for agent '{self.name}'.")
            else:
                self._save_config()
        except Exception as e:
            if self.verbose:
                log.warn(f"Cache load failed for '{self.name}': {e}")

    def _save_config(self):
        """Persist agent configuration using MessagePack."""
        try:
            config = {
                "name": self.name,
                "role": self.role,
                "description": self.description,
                "goal": self.goal,
                "llm_config": self.llm_config,
            }
            with open(self._config_path, "wb") as f:
                f.write(msgpack.packb(config, use_bin_type=True))
            if self.verbose:
                log.dim(f"Saved agent '{self.name}' configuration.")
        except Exception as e:
            if self.verbose:
                log.warn(f"Failed to save '{self.name}' config: {e}")

    # =====================================================================
    # LLM Initialization
    # =====================================================================
    async def _ensure_llm(self):
        """Initialize LLM on first use."""
        if not self.agent_obj:
            self.agent_obj = LLM(**self.llm_config)
            if self.verbose:
                log.fade(f"LLM for '{self.name}' initialized.")

    # =====================================================================
    # Tool Execution
    # =====================================================================
    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract tool call JSON from LLM response."""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                data = json.loads(json_str)
                if data.get("action") == "use_tool":
                    return data
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Execute a tool by name with given parameters."""
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    if self.verbose:
                        log.info(f"  ⚙ Tool input: {tool_name}")
                        log.dim(f"     params: {json.dumps(params, indent=6)}")

                    result = tool.run(**params)

                    if self.verbose:
                        log.success(f"  ⚙ Tool output: {tool_name}")
                        log.dim(
                            f"     result: {result[:200]}{'...' if len(result) > 200 else ''}"
                        )

                    return result
                except Exception as e:
                    error_msg = json.dumps({"error": f"Tool execution failed: {e}"})
                    if self.verbose:
                        log.error(f"  ⚙ Tool error: {tool_name}")
                        log.error(f"     {str(e)}")
                    return error_msg

        error_msg = json.dumps({"error": f"Tool '{tool_name}' not found"})
        if self.verbose:
            log.error(f"  ⚙ Tool not found: {tool_name}")
        return error_msg

    # =====================================================================
    # Execution
    # =====================================================================
    async def run_async(self, input_data: Optional[str] = None) -> str:
        """
        Run agent asynchronously with multi-turn tool support.

        Keeps looping until LLM stops requesting tools or max turns reached.
        """
        if not self.agent_obj:
            await self._ensure_llm()

        messages = [{"role": "system", "content": self._system_prompt}]

        if input_data:
            messages.append({"role": "user", "content": input_data})

        if self.verbose:
            log.agent_thinking(self.name)

        max_turns = 10
        turn = 0

        while turn < max_turns:
            turn += 1

            try:
                # Get response from LLM
                if hasattr(self.agent_obj, "predict_async"):
                    response = await self.agent_obj.predict_async(messages)
                else:
                    response = self.agent_obj.predict(messages)

                text = getattr(response, "content", str(response))
                text = text.strip() if isinstance(text, str) else str(text).strip()

                # Check if LLM wants to use a tool
                tool_call = self._parse_tool_call(text)

                if tool_call:
                    # Execute tool
                    tool_name = tool_call.get("tool", "")
                    params = tool_call.get("params", {})

                    if self.verbose:
                        log.agent_tool_use(self.name, tool_name)

                    tool_result = self._execute_tool(tool_name, params)

                    # Add LLM response and tool result to conversation
                    messages.append({"role": "assistant", "content": text})
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Tool '{tool_name}' returned: {tool_result}\n"
                            "Use this information to answer the original question. "
                            "If you have enough information, provide your final answer. "
                            "If you need more data, you can use other tools or provide your analysis.",
                        }
                    )
                    continue

                # LLM decided it's done (no tool call)
                if self.verbose:
                    log.agent_done(self.name, text)

                return text

            except Exception as e:
                if self.verbose:
                    log.agent_error(self.name, str(e))
                return f"__error__: {e}"

        # Timeout after max_turns
        if self.verbose:
            log.agent_error(self.name, "Max tool turns exceeded")
        return "__error__: Max tool turns exceeded"

    def run_sync(self, input_data: Optional[str] = None) -> str:
        """Run agent synchronously (blocking)."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.run_async(input_data))
