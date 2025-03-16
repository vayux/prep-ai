"""
Main entry point for the PrepAI Streamlit chatbot application.
"""

import streamlit as st
from ui.chatbot_ui import setup_ui, display_messages
from ui.chatbot_logic import PrepAIChatbotLogic


def main() -> None:
    """Starts the PrepAI Streamlit application."""
    setup_ui()

    logic = PrepAIChatbotLogic()

    # Sidebar button to create a new thread
    if st.sidebar.button("➕ New Thread"):
        logic.create_new_thread()
        st.session_state["messages"] = []

    # Ensure at least one thread is active
    if st.session_state["current_thread"] is None:
        logic.create_new_thread()

    # Display existing threads in sidebar
    logic.display_thread_selection()

    # Process user input in the main area
    logic.handle_user_input()

    # Display messages from the current thread
    display_messages(st.session_state["messages"])


if __name__ == "__main__":
    main()
