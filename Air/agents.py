# agent.py
import os
import pickle
import threading
import time
from typing import Any, Dict, List, Optional

from Air.llm import LLMAgent


class Agent:
    CACHE_DIR = "fast_agents_cache/agents"
    _llm_cache: Dict[str, LLMAgent] = {}

    def __init__(
        self,
        name: str,
        role: str,
        description: str,
        goal: str,
        tools: Optional[List[Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        preload: bool = True,
    ):
        self.name = name
        self.role = role
        self.description = description
        self.goal = goal
        self.tools = tools or []
        self.llm_config = llm_config or {}
        self.agent_obj: Optional[LLMAgent] = None
        self._load_thread: Optional[threading.Thread] = None

        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self._pickle_path = os.path.join(self.CACHE_DIR, f"{self.name}.pkl")

        # Try to load from pickle (safe fallback)
        if os.path.exists(self._pickle_path):
            try:
                with open(self._pickle_path, "rb") as f:
                    saved = pickle.load(f)
                self.__dict__.update(saved.__dict__)
                print(f"[Agent] Loaded '{self.name}' from cache.")
            except Exception:
                print(f"[Agent] Cache for '{self.name}' corrupted — ignoring.")
        else:
            self._save_config()

        if preload:
            self._ensure_llm_threaded()

    def _save_config(self):
        with open(self._pickle_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Agent] Saved '{self.name}' configuration to cache.")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["agent_obj"] = None
        state["_load_thread"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.agent_obj = None
        self._load_thread = None

    def _load_llm(self):
        """Initialize or reuse cached LLMAgent."""
        key = str(sorted(self.llm_config.items()))
        if key not in Agent._llm_cache:
            start = time.time()
            llm = LLMAgent(**self.llm_config)
            llm.preload_instructions = {
                "role": self.role,
                "description": self.description,
                "goal": self.goal,
                "tools": self.tools,
            }
            Agent._llm_cache[key] = llm
            print(
                f"[Agent] Loaded LLM for '{self.name}' in {round(time.time() - start, 2)}s."
            )
        self.agent_obj = Agent._llm_cache[key]

    def _ensure_llm_threaded(self):
        if not self.agent_obj and not self._load_thread:
            self._load_thread = threading.Thread(target=self._load_llm, daemon=True)
            self._load_thread.start()

    def run(self, input_data: Optional[str] = None) -> str:
        if not self.agent_obj:
            if self._load_thread and self._load_thread.is_alive():
                print(f"[Agent:{self.name}] Waiting for LLM preload...")
                self._load_thread.join()
            if not self.agent_obj:
                self._load_llm()

        content = (
            f"Description: {self.description}\n"
            f"Goal: {self.goal}\n"
            f"Tools: {', '.join(map(str, self.tools)) or 'None'}"
        )
        if input_data:
            content += f"\nInput: {input_data}"

        messages = [{"role": "system", "content": content}]
        return self.agent_obj.predict(messages)
