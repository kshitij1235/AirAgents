# Air/airbubble.py
import os
import asyncio
import pickle
from typing import Any, Dict, List, Optional, Sequence, Literal

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
    # Execution: ASYNC (parallel, default)
    # ====================================================================
    async def _run_agent_task(self, agent: Agent, prompt: str) -> Dict[str, str]:
        """Run a single agent and return its output."""
        try:
            response = await agent.run_async(prompt)
            text = getattr(response, "content", str(response))
            text = text.strip() if isinstance(text, str) else str(text).strip()
            return {"name": agent.name, "output": text, "error": None}
        except Exception as e:
            return {"name": agent.name, "output": "", "error": str(e)}

    async def _run_all_agents_async(self, prompt: str) -> List[Dict[str, str]]:
        """Run all agents concurrently."""
        tasks = [self._run_agent_task(agent, prompt) for agent in self._agents.values()]
        return await asyncio.gather(*tasks, return_exceptions=False)

    # ====================================================================
    # Execution: SYNC (sequential)
    # ====================================================================
    def _run_all_agents_sync(self, prompt: str) -> List[Dict[str, str]]:
        """Run all agents sequentially."""
        results = []
        for agent in self._agents.values():
            try:
                response = agent.run_sync(prompt)
                text = (
                    getattr(response, "content", str(response))
                    if hasattr(response, "content")
                    else str(response)
                )
                results.append({"name": agent.name, "output": text, "error": None})
            except Exception as e:
                results.append({"name": agent.name, "output": "", "error": str(e)})
        return results

    # ====================================================================
    # Execution: CHAIN (output of one agent feeds to next)
    # ====================================================================
    def _run_chain_mode(self, user_input: str) -> List[Dict[str, str]]:
        """Chain mode: each agent's output feeds to the next agent."""
        results = []
        current_input = user_input

        for agent in self._agents.values():
            try:
                if self.verbose:
                    log.agent_thinking(agent.name)

                prompt = (
                    f"GLOBAL GOAL: {self.goal}\n"
                    f"PREVIOUS OUTPUT:\n{current_input}\n"
                    f"Now analyze and build upon the above."
                )

                response = agent.run_sync(prompt)
                text = getattr(response, "content", str(response))
                text = text.strip() if isinstance(text, str) else str(text).strip()

                results.append({"name": agent.name, "output": text, "error": None})
                current_input = text  # Pass output to next agent

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
        """
        Discussion mode: agents provide viewpoints, main LLM synthesizes.
        Token-efficient: stores only summaries, not full conversation history.
        """
        if self.verbose:
            log.discussion_start()

        agent_prompt = (
            f"GLOBAL GOAL: {self.goal}\n"
            f"USER QUESTION: {user_input}\n"
            f"Provide your perspective and analysis."
        )

        # Round 1: Collect initial perspectives
        perspectives = []
        for agent in self._agents.values():
            try:
                if self.verbose:
                    log.agent_thinking(agent.name)

                response = agent.run_sync(agent_prompt)
                text = getattr(response, "content", str(response))
                text = text.strip()[:300]  # Limit to prevent token bloat

                perspectives.append({"agent": agent.name, "view": text})

                if self.verbose:
                    log.agent_done(agent.name, text)

            except Exception as e:
                if self.verbose:
                    log.agent_error(agent.name, str(e))

        # Main LLM moderates and synthesizes
        perspective_text = "\n\n".join(
            [f"[{p['agent']}]: {p['view']}" for p in perspectives]
        )

        moderation_prompt = (
            f"DISCUSSION TOPIC: {self.goal}\n"
            f"USER QUESTION: {user_input}\n\n"
            f"AGENT PERSPECTIVES:\n{perspective_text}\n\n"
            f"As a moderator, synthesize these perspectives into a cohesive response. "
            f"Identify areas of agreement, highlight key disagreements, and provide a balanced conclusion."
        )

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a discussion moderator synthesizing multiple expert perspectives.",
                },
                {"role": "user", "content": moderation_prompt},
            ]
            response = self._final_llm.predict(messages)
            result = getattr(response, "content", str(response)).strip()

        except Exception as e:
            result = f"DISCUSSION ERROR: {e}"
            if self.verbose:
                log.synthesis_error(str(e))

        return result

    # ====================================================================
    # Execution: HIERARCHY (tree of agents, bottom-up synthesis)
    # ====================================================================
    def _run_hierarchy_mode(self, user_input: str) -> str:
        """
        Hierarchy mode: all agents think independently, then layer by layer
        synthesize up the tree. Each level summarizes before passing up.
        """
        if self.verbose:
            log.hierarchy_start()

        agent_list = list(self._agents.values())

        # Level 1: All agents think
        level_outputs = {}
        for agent in agent_list:
            try:
                if self.verbose:
                    log.agent_thinking(agent.name)

                prompt = (
                    f"GLOBAL GOAL: {self.goal}\n"
                    f"USER INPUT: {user_input}\n"
                    f"Provide detailed analysis."
                )

                response = agent.run_sync(prompt)
                text = getattr(response, "content", str(response)).strip()[:500]

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

            # Group outputs and summarize
            combined = "\n\n".join(
                [f"[{name}]: {output}" for name, output in current_level.items()]
            )

            summary_prompt = (
                f"GOAL: {self.goal}\n"
                f"INPUT: {user_input}\n\n"
                f"AGENT CONTRIBUTIONS:\n{combined}\n\n"
                f"Summarize the key points and insights in 200 words maximum."
            )

            try:
                messages = [
                    {
                        "role": "system",
                        "content": "Summarize the following insights concisely.",
                    },
                    {"role": "user", "content": summary_prompt},
                ]
                response = self._final_llm.predict(messages)
                summary = getattr(response, "content", str(response)).strip()

                # Move to next level
                current_level = {f"Level_{level_num}": summary}
                level_num += 1

            except Exception as e:
                if self.verbose:
                    log.synthesis_error(str(e))
                return f"HIERARCHY ERROR: {e}"

        # Return final result
        final_result = list(current_level.values())[0] if current_level else ""
        return final_result

    # ====================================================================
    # Main Execution Router
    # ====================================================================
    def run(self, user_input: str = "") -> str:
        """Route to appropriate execution mode."""
        if self.verbose:
            log.bubble_start(self.name, self.goal, user_input, self.mode)

        if self.mode == "async":
            return self._run_mode_async(user_input)
        elif self.mode == "sync":
            return self._run_mode_sync(user_input)
        elif self.mode == "chain":
            return self._run_mode_chain(user_input)
        elif self.mode == "discussion":
            return self._run_mode_discussion(user_input)
        elif self.mode == "hierarchy":
            return self._run_mode_hierarchy(user_input)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _run_mode_async(self, user_input: str) -> str:
        """ASYNC mode: all agents run in parallel."""
        agent_prompt = (
            f"GLOBAL GOAL: {self.goal}\n"
            f"USER INPUT: {user_input}\n"
            f"Provide your analysis and insights."
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(self._run_all_agents_async(agent_prompt))

        return self._synthesize_results(results, user_input)

    def _run_mode_sync(self, user_input: str) -> str:
        """SYNC mode: all agents run sequentially."""
        agent_prompt = (
            f"GLOBAL GOAL: {self.goal}\n"
            f"USER INPUT: {user_input}\n"
            f"Provide your analysis and insights."
        )

        results = self._run_all_agents_sync(agent_prompt)
        return self._synthesize_results(results, user_input)

    def _run_mode_chain(self, user_input: str) -> str:
        """CHAIN mode: agent output feeds to next agent."""
        results = self._run_chain_mode(user_input)

        # Last agent output is the final answer
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

        # Build synthesis prompt
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

        synthesis_prompt = (
            f"GLOBAL GOAL: {self.goal}\n"
            f"USER INPUT: {user_input}\n\n"
            f"AGENT CONTRIBUTIONS:\n{agent_contributions}\n\n"
            f"Synthesize these contributions into a clear, actionable response. "
            f"Highlight key insights, resolve conflicts, provide next steps."
        )

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a synthesis agent combining insights from multiple specialized agents.",
                },
                {"role": "user", "content": synthesis_prompt},
            ]
            final_response = self._final_llm.predict(messages)
            final_text = getattr(final_response, "content", str(final_response))
            final_text = final_text.strip()
        except Exception as e:
            if self.verbose:
                log.synthesis_error(str(e))
            final_text = f"SYNTHESIS ERROR: {e}"

        if self.verbose:
            log.bubble_done(final_text)

        return final_text
