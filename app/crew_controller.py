"""
CrewController orchestrates multi-agent requests in PrepAI using Crew AI.
"""

import logging
from config.settings import settings
# Hypothetical import for Crew AI
from crewai.core import CrewManager

from agents.agent_dsa import DSAAgent
from agents.agent_lld import LLDAgent
from agents.agent_hld import HLDAgent
from agents.agent_behavioral import BehavioralAgent


class CrewController:
    """Central multi-agent controller for PrepAI via Crew AI."""

    def __init__(self) -> None:
        """Initialize the CrewController by setting up Crew AI and all agents."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.crew_manager = CrewManager(settings.CREWAI_ENDPOINT)

        llm_client = settings.get_llm_client()

        self.agents = {
            "dsa": DSAAgent(llm_client),
            "lld": LLDAgent(llm_client),
            "hld": HLDAgent(llm_client),
            "behavioral": BehavioralAgent(llm_client),
        }

    def route_request(self, agent_type: str, user_input: str) -> tuple[str, str]:
        """Routes the request to the appropriate agent.

        Args:
            agent_type: One of ('dsa', 'lld', 'hld', 'behavioral').
            user_input: The question or request string.

        Returns:
            A tuple of (response_text, source_text).
        """
        self.crew_manager.log_request(agent_type, user_input)  # Hypothetical logging

        agent = self.agents.get(agent_type)
        if not agent:
            return f"No agent for '{agent_type}'.", "N/A"

        if agent_type == "dsa":
            response = agent.solve_problem(user_input)
        elif agent_type == "lld":
            response = agent.propose_lld_solution(user_input)
        elif agent_type == "hld":
            response = agent.propose_hld_architecture(user_input)
        elif agent_type == "behavioral":
            response = agent.handle_behavioral_question(user_input)
        else:
            response = f"Unknown agent type: {agent_type}"

        return response, f"Agent: {agent_type}"
