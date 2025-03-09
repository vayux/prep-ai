import streamlit as st
from ui.chatbot_ui import setup_ui, display_messages
from ui.chatbot_logic import PrepAIChatbotLogic

def main():
    """Main function to run the chatbot."""
    setup_ui()
    logic = PrepAIChatbotLogic()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.sidebar.button("➕ New Thread"):
        logic.create_new_thread()
        st.session_state.messages = [] #clear messages on new thread.

    if st.session_state["current_thread"] is None:
        logic.create_new_thread()

    logic.display_thread_selection()
    logic.handle_user_input()
    display_messages(st.session_state.messages)

if __name__ == "__main__":
    main()