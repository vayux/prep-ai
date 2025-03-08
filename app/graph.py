from langchain.schema import AIMessage, HumanMessage
from langchain.chat_models import ChatOllama
from app.dsa_agent import DSA
from langgraph.graph import StateGraph
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrepAIGraph:
    def __init__(self):
        """Initializes the LangGraph StateGraph for Prep AI with correct state handling."""
        logger.info("Initializing PrepAIGraph.")
        self.llm = ChatOllama(model="llama3:latest")  # Use Llama 3 latest
        self.dsa_agent = DSA()
        self.graph = StateGraph(state_schema=Dict)
        self.graph.add_node("query_rag", self.query_rag)
        self.graph.add_node("query_llm", self.query_llm)
        self.graph.add_node("finish", self.finish)
        self.graph.set_entry_point("query_rag")
        self.graph.add_conditional_edges(
            "query_rag",
            self.should_use_llm,
            {"use_llm": "query_llm", "finish": "finish"},
        )
        self.graph.add_edge("query_llm", "finish")
        self.graph.set_finish_point("finish")
        self.executor = self.graph.compile()
        logger.info("PrepAIGraph initialized.")

    def query_rag(self, state: Dict) -> Dict:
        """Handles retrieval-augmented generation (RAG) queries."""
        logger.info("Entering query_rag node")
        logger.debug(f"Current state: {state}")
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], HumanMessage):
            logger.error("query_rag: Last message is not HumanMessage")
            raise ValueError("Last message must be a HumanMessage")

        question = messages[-1].content
        logger.debug(f"Extracted question: {question}")
        response, is_sufficient = self.dsa_agent.query_rag(question)
        use_llm = not is_sufficient
        messages.append(AIMessage(content=response))

        logger.debug(f"DSA response: {response}, is_sufficient: {is_sufficient}, use_llm: {use_llm}")

        logger.info("Exiting query_rag node")
        return {"messages": messages, "source": "query_rag", "use_llm": use_llm}

    def should_use_llm(self, state: Dict) -> str:
        """Determines whether to use the LLM based on RAG result."""
        if state.get("use_llm"):
            logger.info("RAG result insufficient, using LLM.")
            return "use_llm"
        else:
            logger.info("RAG result sufficient, skipping LLM.")
            return "finish"

    def query_llm(self, state: Dict) -> Dict:
        """Handles LLM-based queries with Llama 3."""
        logger.info("Entering query_llm node")
        logger.debug(f"Current state: {state}")
        messages = state.get("messages", [])

        if not messages:
            logger.error("query_llm: Messages list is empty")
            raise ValueError("Messages list is empty")

        prompt = messages
        prompt.append(HumanMessage(content="Provide the python code for the given problem, and explain the code."))
        logger.debug(f"LLM Prompt: {prompt}")
        response = self.llm(messages=prompt)
        logger.debug(f"LLM response: {response}")
        messages.append(AIMessage(content=response.content))
        logger.info("Exiting query_llm node")
        return {"messages": messages, "source": "query_llm"}

    def finish(self, state: Dict) -> Dict:
        """Simply returns the current state."""
        logger.info("Entering finish node.")
        return state

    def run(self, question: str) -> tuple[str, str]:
        """Executes the Prep AI graph and returns result with source."""
        logger.info("Starting PrepAIGraph run")
        logger.debug(f"Input question: {question}")
        input_state = {"messages": [HumanMessage(content=question)]}
        logger.debug(f"Initial input state: {input_state}")
        try:
            result = self.executor.invoke(input_state)
            logger.debug(f"Graph execution result: {result}")
            source = result.get("source", "unknown")
            logger.debug(f"Extracted source: {source}")
            logger.info("PrepAIGraph run completed successfully.")
            return result["messages"][-1].content, source
        except Exception as e:
            logger.error(f"Error during PrepAIGraph run: {e}", exc_info=True)
            return "An error occurred during processing.", "error"


prep_ai_graph = PrepAIGraph()
