import logging
from crewai import Crew
from app.dsa_agent import dsa_agent
from app.lld_agent import lld_agent
from app.hld_agent import hld_agent
from app.behavioral_agent import behavioral_agent

logging.basicConfig(level=logging.INFO)


class CrewManager:
    def __init__(self):
        """Manages the multi-agent CrewAI system for Prep AI."""
        self.crew = Crew(agents=[dsa_agent.agent, lld_agent.agent, hld_agent.agent, behavioral_agent.agent])

    def execute_query(self, query):
        """Delegates queries to the best agent based on the query type."""
        logging.info(f"Executing query with CrewAI: {query}")
        response = self.crew.kickoff(query)
        return response


crew_manager = CrewManager()
