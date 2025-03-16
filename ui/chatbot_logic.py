"""
Implements chatbot logic for PrepAI, including multi-threaded chat
handling, user input parsing, and calls to the LangGraph pipeline.
"""

import concurrent.futures
import logging
import streamlit as st
from app.graph import prep_ai_graph
from ui.thread import Thread

logging.basicConfig(level=logging.INFO)


class PrepAIChatbotLogic:
    """Handles the core logic for the PrepAI chatbot, including thread
    management and pipeline execution.
    """

    def __init__(self, graph_provider=prep_ai_graph) -> None:
        """
        Args:
            graph_provider: A LangGraph Graph object that processes user queries.
        """
        self.graph_provider = graph_provider
        self.initialize_session_state()

    def initialize_session_state(self) -> None:
        """Initializes Streamlit session state variables for threads, messages, etc."""
        if "threads" not in st.session_state:
            st.session_state["threads"] = {}
        if "current_thread" not in st.session_state:
            st.session_state["current_thread"] = None
        if "processing" not in st.session_state:
            st.session_state["processing"] = False
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

    def create_new_thread(self) -> None:
        """Creates a new chat thread and updates session state."""
        thread = Thread()
        st.session_state["threads"][thread.id] = thread
        st.session_state["current_thread"] = thread.id
        st.session_state["messages"] = []

    def delete_thread(self, thread_id: str) -> None:
        """Deletes a specified thread.

        Args:
            thread_id: The ID of the thread to delete.
        """
        del st.session_state["threads"][thread_id]
        if st.session_state["current_thread"] == thread_id:
            # Reassign current_thread if possible
            if st.session_state["threads"]:
                new_thread_id = next(iter(st.session_state["threads"]))
                st.session_state["current_thread"] = new_thread_id
                st.session_state["messages"] = (
                    st.session_state["threads"][new_thread_id].get_messages()
                )
            else:
                st.session_state["current_thread"] = None
                st.session_state["messages"] = []

    def display_thread_selection(self) -> None:
        """Displays available threads in the Streamlit sidebar."""
        st.sidebar.header("💬 Threads")
        thread_ids = list(st.session_state["threads"].keys())

        for idx, t_id in enumerate(thread_ids, start=1):
            thread = st.session_state["threads"][t_id]
            thread_msgs = thread.get_messages()

            # Use the first user message or a fallback as the thread name
            if thread_msgs:
                possible_name = thread_msgs[0]["content"]
                thread_name = (possible_name[:20] + "...") if len(possible_name) > 20 else possible_name
            else:
                thread_name = f"Thread {idx}"

            col1, col2 = st.sidebar.columns([0.8, 0.2])
            with col1:
                if st.session_state["current_thread"] == t_id:
                    st.button(f"**{thread_name}**", key=f"thread_{t_id}", use_container_width=True)
                else:
                    if st.button(thread_name, key=f"thread_select_{t_id}", use_container_width=True):
                        st.session_state["current_thread"] = t_id
                        st.session_state["messages"] = thread_msgs
                        st.experimental_rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{t_id}"):
                    self.delete_thread(t_id)
                    st.experimental_rerun()

    def handle_user_input(self) -> None:
        """Captures user input and processes it through the LangGraph pipeline."""
        user_input = st.chat_input(
            "Ask me anything...",
            disabled=st.session_state["processing"]
        )
        if user_input:
            current_thread_id = st.session_state["current_thread"]
            if not current_thread_id:
                self.create_new_thread()
                current_thread_id = st.session_state["current_thread"]

            thread = st.session_state["threads"][current_thread_id]

            thread.add_message({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["processing"] = True

            agent_type, actual_query = self.parse_agent_type(user_input)

            def run_graph():
                """Runs the pipeline to get a response."""
                data = {"agent_type": agent_type, "user_input": actual_query}
                return self.graph_provider.run(data)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_graph)
                with st.spinner("Generating response..."):
                    try:
                        result = future.result()
                        response_text = result.get("response", "")
                        source_text = result.get("source", "N/A")
                        answer_content = f"{response_text}\n\n(Source: {source_text})"

                        thread.add_message({"role": "assistant", "content": answer_content})
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": answer_content}
                        )
                    except Exception as exc:
                        logging.error(f"Error generating response: {exc}")
                        st.error(f"Error generating response: {exc}")

            st.session_state["processing"] = False
            st.experimental_rerun()

    def parse_agent_type(self, user_input: str) -> tuple[str, str]:
        """Parses the user's input to determine which agent to use.

        For example, if the user types "dsa: how to reverse a linked list",
        the agent_type is 'dsa' and actual_query is 'how to reverse a linked list'.

        If no prefix is found, defaults to 'behavioral'.

        Args:
            user_input: The raw text the user typed.

        Returns:
            A tuple (agent_type, actual_query).
        """
        default_agent = "behavioral"
        if ":" in user_input:
            parts = user_input.split(":", 1)
            agent = parts[0].strip().lower()
            query = parts[1].strip()
            return agent, query
        return default_agent, user_input

    def get_current_thread_messages(self) -> list[dict]:
        """Returns messages for the current thread.

        Returns:
            A list of message dicts for the current thread.
        """
        return st.session_state["messages"]

    def set_graph_provider(self, graph_provider) -> None:
        """Allows changing the pipeline/graph at runtime.

        Args:
            graph_provider: A new Graph or similar pipeline object.
        """
        self.graph_provider = graph_provider
