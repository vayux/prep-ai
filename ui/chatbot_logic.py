import concurrent.futures
import logging
import streamlit as st
from app.graph import prep_ai_graph
from ui.chatbot_ui import display_messages
from ui.thread import Thread

logging.basicConfig(level=logging.INFO)

class PrepAIChatbotLogic:
    """Handles the logic of the PrepAI chatbot."""

    def __init__(self, graph_provider=prep_ai_graph):
        """Initializes the chatbot logic."""
        self.graph_provider = graph_provider
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initializes session state variables."""
        if "threads" not in st.session_state:
            st.session_state["threads"] = {}
        if "current_thread" not in st.session_state:
            st.session_state["current_thread"] = None
        if "processing" not in st.session_state:
            st.session_state["processing"] = False
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

    def create_new_thread(self):
        """Creates a new chat thread."""
        thread = Thread()
        st.session_state["threads"][thread.id] = thread
        st.session_state["current_thread"] = thread.id
        st.session_state["messages"] = []

    def delete_thread(self, thread_id):
        """Deletes a chat thread."""
        del st.session_state["threads"][thread_id]
        if st.session_state["current_thread"] == thread_id:
            st.session_state["current_thread"] = next(iter(st.session_state["threads"]), None)
            if st.session_state["current_thread"]:
                st.session_state["messages"] = st.session_state["threads"][st.session_state["current_thread"]].get_messages()
            else:
                st.session_state["messages"] = []

    def display_thread_selection(self):
        """Displays the thread selection sidebar."""
        st.sidebar.header("💬 Threads")
        thread_ids = list(st.session_state["threads"].keys())

        if thread_ids:
            for thread_id in thread_ids:
                thread = st.session_state["threads"][thread_id]
                thread_messages = thread.get_messages()
                thread_name = (
                    thread_messages[0].get("content", f"Thread {thread_ids.index(thread_id) + 1}")
                    if thread_messages
                    else f"Thread {thread_ids.index(thread_id) + 1}"
                )

                col1, col2 = st.sidebar.columns([0.8, 0.2])  # Added spec here

                with col1:
                    if st.session_state["current_thread"] == thread_id:
                        st.button(f"**{thread_name}**", key=f"thread_{thread_id}", use_container_width=True)
                    else:
                        st.button(thread_name, key=f"thread_{thread_id}", use_container_width=True)

                with col2:
                    if st.button("🗑️", key=f"delete_{thread_id}"):
                        self.delete_thread(thread_id)
                        st.rerun()

    def handle_user_input(self):
        """Handles user input and chatbot response."""
        user_input = st.chat_input("Ask me anything:", disabled=st.session_state["processing"])
        if user_input:
            current_thread_id = st.session_state["current_thread"]
            thread = st.session_state["threads"][current_thread_id]

            thread.add_message({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "user", "content": user_input})

            st.session_state["processing"] = True

            with concurrent.futures.ThreadPoolExecutor() as executor:
                graph_future = executor.submit(self.graph_provider.run, user_input)
                with st.spinner("Generating Response..."):
                    try:
                        graph_response, source = graph_future.result()
                        thread.add_message({"role": "assistant", "content": f"{graph_response}Source: {source}"})
                        st.session_state.messages.append({"role": "assistant", "content": f"{graph_response}Source: {source}"})
                    except Exception as e:
                        logging.error(f"Error generating response: {e}")
                        st.error(f"Error generating response: {e}")
            st.session_state["processing"] = False
            # Force a rerun after message update
            st.rerun()

    def get_current_thread_messages(self):
        """Gets messages from the current thread."""
        return st.session_state["messages"]

    def set_graph_provider(self, graph_provider):
        """Dynamically sets the graph provider."""
        self.graph_provider = graph_provider