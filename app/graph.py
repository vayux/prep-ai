"""
Defines a LangGraph pipeline for PrepAI that uses the CrewController
to route requests to the correct agent.
"""

import logging
from langgraph import Graph, Node, Transition
from app.crew_controller import CrewController

logging.basicConfig(level=logging.INFO)

crew_controller = CrewController()


def route_to_agent(data: dict) -> dict:
    """Routes user input to the correct agent using CrewController.

    Args:
        data: A dictionary containing 'agent_type' and 'user_input' keys.

    Returns:
        A dictionary with 'response' and 'source'.
    """
    agent_type = data.get("agent_type")
    user_input = data.get("user_input")
    if not agent_type or not user_input:
        return {"error": "Missing agent_type or user_input"}

    response, source = crew_controller.route_request(agent_type, user_input)
    return {"response": response, "source": source}


node_start = Node(name="start", func=lambda d: d)
node_route = Node(name="route_agent", func=route_to_agent)
node_end = Node(name="end", func=lambda d: d)

prep_ai_graph = Graph(
    nodes=[node_start, node_route, node_end],
    transitions=[
        Transition(node_start, node_route, condition=lambda d: True),
        Transition(node_route, node_end, condition=lambda d: True),
    ]
)
