# airbubble.py
import logging
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from Air.agents import Agent
from Air.llm import LLMAgent


class AirBubble:
    """
    Orchestrates multiple agents working toward one unified goal.
    Executes in parallel for maximum speed.
    """

    CACHE_DIR = "fast_agents_cache/air_bubble"
    logger = logging.getLogger("AirBubble")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("\n%(asctime)s | ▶▶ %(message)s ◀◀\n", datefmt="%H:%M:%S")
    )
    if not logger.hasHandlers():
        logger.addHandler(handler)
    logger.propagate = False

    def __init__(
        self,
        name: str,
        goal: str,
        agents: Optional[List[Agent]] = None,
        preload: bool = True,
        verbose: bool = True,
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.goal = goal
        self.agents: Dict[str, Any] = {}
        self.verbose = verbose
        self.final_llm = LLMAgent(**(llm_config or {}))
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self._pickle_path = os.path.join(self.CACHE_DIR, f"{self.name}.pkl")

        for agent in agents or []:
            self.register_agent(agent)

        if preload:
            self._preload_agents()

    def register_agent(self, agent: Any):
        self.agents[agent.name] = agent
        agent._ensure_llm_threaded()

    def _preload_agents(self):
        with ThreadPoolExecutor(max_workers=len(self.agents) or 1) as pool:
            for _ in pool.map(lambda a: a._ensure_llm_threaded(), self.agents.values()):
                pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["final_llm"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.final_llm = None

    def save(self):
        with open(self._pickle_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.info(f"Saved AirBubble '{self.name}' to cache.")

    @classmethod
    def load(cls, name: str) -> "AirBubble":
        path = os.path.join(cls.CACHE_DIR, f"{name}.pkl")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        cls.logger.info(f"Loaded AirBubble '{name}' from cache.")
        return obj

    def run(self, user_input: str = "") -> str:
        """
        Run all agents collaboratively toward one unified goal.
        Visible multi-agent reasoning and refinement.
        """

        if self.verbose:
            print(f"\n[AirBubble:{self.name}] Running orchestrated goal alignment...")

        shared_context = f"GLOBAL GOAL: {self.goal}\nUSER INPUT: {user_input}\n"
        step_outputs = []

        # --- Phase 1: Individual contributions ---
        for agent_name, agent in self.agents.items():
            if self.verbose:
                print(f"[AirBubble] Gathering expertise from {agent_name}...")
            prompt = (
                f"You are {agent.role}. Your task contributes to the overall goal:\n"
                f"{self.goal}\n\n"
                f"Agent Description: {agent.description}\n"
                f"Specific Agent Goal: {agent.goal}\n"
                f"User Input: {user_input}\n"
                f"Shared Context So Far:\n{shared_context}\n\n"
                f"Please respond ONLY with your unique insight or contribution, "
                f"not a final answer."
            )
            result = agent.run(prompt)
            shared_context += f"\n[{agent_name} contribution]: {result}\n"
            step_outputs.append((agent_name, result))

        # --- Phase 2: Cooperative refinement ---
        if self.verbose:
            print("\n[AirBubble] Refining combined knowledge...")

        refine_prompt = (
            f"Here are the contributions from all agents working toward the goal:\n\n"
            f"{shared_context}\n\n"
            f"Please synthesize these insights into a coherent, step-by-step unified strategy "
            f"that integrates each contribution while keeping the central goal in focus: "
            f"'{self.goal}'. Be explicit about which agent’s insight influenced which part."
        )

        refinement = self.final_llm.predict(
            [{"role": "system", "content": refine_prompt}]
        )

        # --- Phase 3: Final synthesis ---
        final_prompt = (
            f"The team of agents has refined a shared understanding:\n{refinement}\n\n"
            f"Now, as the final reasoning step, produce ONE unified response that solves "
            f"the goal '{self.goal}' based on all agent contributions and refinements. "
            f"Ensure the reasoning feels collaborative, not repetitive."
        )

        final_output = self.final_llm.predict(
            [{"role": "system", "content": final_prompt}]
        )

        if self.verbose:
            print(f"\n[AirBubble:{self.name}] FINAL OUTPUT:\n{final_output}")

        return final_output
