import os
import asyncio
import pickle
from typing import Any, Dict, List, Optional, Sequence, Literal, Callable

from Air.agents import Agent
from Air.llm import LLM
from Air.logging import logging_instance as log


class AirBubble:
    """
    AirBubble — orchestrates multiple agents toward one unified goal.
    Supports: async, sync, chain, discussion, hierarchy modes.
    """

    CACHE_DIR = "fast_agents_cache/air_bubble"

    def __init__(
        self,
        name: str,
        goal: str,
        agents: Optional[Sequence[Agent]] = None,
        verbose: bool = True,
        llm_config: Optional[Dict[str, Any]] = None,
        mode: Literal["sync", "chain", "discussion", "hierarchy"] = "async",
    ):
        self.name = name
        self.goal = goal
        self.verbose = verbose
        self._agents: Dict[str, Agent] = {}
        self._final_llm = LLM(**(llm_config or {}))
        self.mode = mode

        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self._pickle_path = os.path.join(self.CACHE_DIR, f"{self.name}.pkl")

        for agent in agents or []:
            self.register_agent(agent)

        if self.verbose:
            log.init_bubble(self.name, len(self._agents), self.mode)

    # ====================================================================
    # Helper: Extract text from response objects
    # ====================================================================
    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract and clean text from response object."""
        text = getattr(response, "content", str(response))
        return text.strip() if isinstance(text, str) else str(text).strip()

    # ====================================================================
    # Helper: Build formatted prompt
    # ====================================================================
    def _build_prompt(self, instruction: str, context: str = "") -> str:
        """Build a standard prompt with goal and context."""
        prompt = f"GLOBAL GOAL: {self.goal}\n{instruction}"
        if context:
            prompt += f"\n{context}"
        return prompt

    # ====================================================================
    # Agent Management
    # ====================================================================
    def register_agent(self, agent: Agent):
        """Register an agent into the bubble."""
        self._agents[agent.name] = agent
        if self.verbose:
            log.agent_register(agent.name, agent.role)

    def unregister_agent(self, name: str):
        """Remove an agent from the bubble."""
        if name in self._agents:
            del self._agents[name]
            if self.verbose:
                log.agent_unregister(name)

    # ====================================================================
    # Persistence
    # ====================================================================
    def save(self):
        """Persist AirBubble state to disk."""
        with open(self._pickle_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        if self.verbose:
            log.bubble_save(self.name)

    @classmethod
    def load(cls, name: str) -> "AirBubble":
        """Load AirBubble from disk."""
        path = os.path.join(cls.CACHE_DIR, f"{name}.pkl")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if obj.verbose:
            log.bubble_load(name)
        return obj

    # ====================================================================
    # Execution: Helper for running single agent
    # ====================================================================
    def _run_agent_sync(self, agent: Agent, prompt: str) -> Dict[str, str]:
        """Run a single agent synchronously."""
        try:
            response = agent.run_sync(prompt)
            text = self._extract_text(response)
            return {"name": agent.name, "output": text, "error": None}
        except Exception as e:
            return {"name": agent.name, "output": "", "error": str(e)}

    async def _run_agent_async(self, agent: Agent, prompt: str) -> Dict[str, str]:
        """Run a single agent asynchronously."""
        try:
            response = await agent.run_async(prompt)
            text = self._extract_text(response)
            return {"name": agent.name, "output": text, "error": None}
        except Exception as e:
            return {"name": agent.name, "output": "", "error": str(e)}

    # ====================================================================
    # Execution: ASYNC (parallel)
    # ====================================================================
    async def _run_all_agents_async(self, prompt: str) -> List[Dict[str, str]]:
        """Run all agents concurrently."""
        tasks = [
            self._run_agent_async(agent, prompt) for agent in self._agents.values()
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)

    # ====================================================================
    # Execution: SYNC (sequential)
    # ====================================================================
    def _run_all_agents_sync(self, prompt: str) -> List[Dict[str, str]]:
        """Run all agents sequentially."""
        return [self._run_agent_sync(agent, prompt) for agent in self._agents.values()]

    # ====================================================================
    # Execution: CHAIN (output feeds to next)
    # ====================================================================
    def _run_chain_mode(self, user_input: str) -> List[Dict[str, str]]:
        """Chain mode: each agent's output feeds to the next agent."""
        results = []
        current_input = user_input

        for agent in self._agents.values():
            try:
                if self.verbose:
                    log.agent_thinking(agent.name)

                prompt = self._build_prompt(
                    "Analyze and build upon the above.",
                    f"PREVIOUS OUTPUT:\n{current_input}",
                )

                response = agent.run_sync(prompt)
                text = self._extract_text(response)

                results.append({"name": agent.name, "output": text, "error": None})
                current_input = text

                if self.verbose:
                    log.agent_done(agent.name, text)

            except Exception as e:
                results.append({"name": agent.name, "output": "", "error": str(e)})
                if self.verbose:
                    log.agent_error(agent.name, str(e))

        return results

    # ====================================================================
    # Execution: DISCUSSION (agents discuss, main LLM moderates)
    # ====================================================================
    def _run_discussion_mode(self, user_input: str) -> str:
        """Discussion mode: agents provide viewpoints, main LLM synthesizes."""
        if self.verbose:
            log.discussion_start()

        agent_prompt = self._build_prompt(
            "Provide your perspective and analysis.", f"USER QUESTION: {user_input}"
        )

        # Collect initial perspectives
        perspectives = []
        for agent in self._agents.values():
            try:
                if self.verbose:
                    log.agent_thinking(agent.name)

                response = agent.run_sync(agent_prompt)
                text = self._extract_text(response)[:300]  # Limit token bloat

                perspectives.append({"agent": agent.name, "view": text})

                if self.verbose:
                    log.agent_done(agent.name, text)

            except Exception as e:
                if self.verbose:
                    log.agent_error(agent.name, str(e))

        # Main LLM moderates and synthesizes
        return self._synthesize_with_llm(
            f"As a moderator, synthesize these perspectives into a cohesive response. "
            f"Identify areas of agreement, highlight key disagreements, and provide a balanced conclusion.",
            "You are a discussion moderator synthesizing multiple expert perspectives.",
            perspectives_text="\n\n".join(
                [f"[{p['agent']}]: {p['view']}" for p in perspectives]
            ),
            user_input=user_input,
        )

    # ====================================================================
    # Execution: HIERARCHY (tree of agents, bottom-up synthesis)
    # ====================================================================
    def _run_hierarchy_mode(self, user_input: str) -> str:
        """Hierarchy mode: all agents think independently, then synthesize up."""
        if self.verbose:
            log.hierarchy_start()

        agent_list = list(self._agents.values())

        # Level 1: All agents think
        level_outputs = {}
        for agent in agent_list:
            try:
                if self.verbose:
                    log.agent_thinking(agent.name)

                prompt = self._build_prompt(
                    "Provide detailed analysis.", f"USER INPUT: {user_input}"
                )

                response = agent.run_sync(prompt)
                text = self._extract_text(response)[:500]

                level_outputs[agent.name] = text

                if self.verbose:
                    log.agent_done(agent.name, text)

            except Exception as e:
                if self.verbose:
                    log.agent_error(agent.name, str(e))

        # Level 2+: Synthesize layer by layer
        current_level = level_outputs
        level_num = 2

        while len(current_level) > 1:
            if self.verbose:
                log.hierarchy_level(level_num, len(current_level))

            combined = "\n\n".join(
                [f"[{name}]: {output}" for name, output in current_level.items()]
            )

            try:
                result = self._synthesize_with_llm(
                    "Summarize the key points and insights in 200 words maximum.",
                    "Summarize the following insights concisely.",
                    agent_contributions=combined,
                    user_input=user_input,
                )
                current_level = {f"Level_{level_num}": result}
                level_num += 1

            except Exception as e:
                if self.verbose:
                    log.synthesis_error(str(e))
                return f"HIERARCHY ERROR: {e}"

        return list(current_level.values())[0] if current_level else ""

    # ====================================================================
    # Synthesis: LLM helper (consolidates all synthesis calls)
    # ====================================================================
    def _synthesize_with_llm(
        self,
        task_instruction: str,
        system_role: str,
        agent_contributions: str = "",
        perspectives_text: str = "",
        user_input: str = "",
    ) -> str:
        """Generic LLM synthesis helper."""
        content_parts = [f"GOAL: {self.goal}"]

        if user_input:
            content_parts.append(f"USER INPUT: {user_input}")

        if agent_contributions:
            content_parts.append(f"AGENT CONTRIBUTIONS:\n{agent_contributions}")

        if perspectives_text:
            content_parts.append(f"AGENT PERSPECTIVES:\n{perspectives_text}")

        content_parts.append(task_instruction)

        synthesis_prompt = "\n\n".join(content_parts)

        try:
            messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": synthesis_prompt},
            ]
            response = self._final_llm.predict(messages)
            return self._extract_text(response)

        except Exception as e:
            if self.verbose:
                log.synthesis_error(str(e))
            return f"SYNTHESIS ERROR: {e}"

    # ====================================================================
    # Main Execution Router
    # ====================================================================
    def run(self, user_input: str = "") -> str:
        """Route to appropriate execution mode."""
        if self.verbose:
            log.bubble_start(self.name, self.goal, user_input, self.mode)

        match self.mode:
            case "async":
                return self._run_mode_async(user_input)
            case "sync":
                return self._run_mode_sync(user_input)
            case "chain":
                return self._run_mode_chain(user_input)
            case "discussion":
                return self._run_mode_discussion(user_input)
            case "hierarchy":
                return self._run_mode_hierarchy(user_input)
            case _:
                raise ValueError(f"Unknown mode: {self.mode}")

    def _run_mode_async(self, user_input: str) -> str:
        """ASYNC mode: all agents run in parallel."""
        agent_prompt = self._build_prompt(
            "Provide your analysis and insights.", f"USER INPUT: {user_input}"
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(self._run_all_agents_async(agent_prompt))

        return self._synthesize_results(results, user_input)

    def _run_mode_sync(self, user_input: str) -> str:
        """SYNC mode: all agents run sequentially."""
        agent_prompt = self._build_prompt(
            "Provide your analysis and insights.", f"USER INPUT: {user_input}"
        )

        results = self._run_all_agents_sync(agent_prompt)
        return self._synthesize_results(results, user_input)

    def _run_mode_chain(self, user_input: str) -> str:
        """CHAIN mode: agent output feeds to next agent."""
        results = self._run_chain_mode(user_input)
        final_output = results[-1]["output"] if results else ""

        if self.verbose:
            log.bubble_done(final_output)

        return final_output

    def _run_mode_discussion(self, user_input: str) -> str:
        """DISCUSSION mode: agents discuss, main LLM moderates."""
        result = self._run_discussion_mode(user_input)

        if self.verbose:
            log.bubble_done(result)

        return result

    def _run_mode_hierarchy(self, user_input: str) -> str:
        """HIERARCHY mode: bottom-up synthesis."""
        result = self._run_hierarchy_mode(user_input)

        if self.verbose:
            log.bubble_done(result)

        return result

    # ====================================================================
    # Synthesis (used by async/sync modes)
    # ====================================================================
    def _synthesize_results(
        self, results: List[Dict[str, str]], user_input: str
    ) -> str:
        """Synthesize parallel agent outputs."""
        if self.verbose:
            for res in results:
                if res["error"]:
                    log.agent_error(res["name"], res["error"])
                else:
                    log.agent_result(res["name"], res["output"])

        agent_contributions = "\n\n".join(
            [
                (
                    f"[{res['name']}]:\n{res['output']}"
                    if not res["error"]
                    else f"[{res['name']}]: ERROR — {res['error']}"
                )
                for res in results
            ]
        )

        final_text = self._synthesize_with_llm(
            "Synthesize these contributions into a clear, actionable response. "
            "Highlight key insights, resolve conflicts, provide next steps.",
            "You are a synthesis agent combining insights from multiple specialized agents.",
            agent_contributions=agent_contributions,
            user_input=user_input,
        )

        if self.verbose:
            log.bubble_done(final_text)

        return final_text
