"""
Streamlit UI components for the PrepAI chatbot, including styling and
message display functionality.
"""

import streamlit as st

CHAT_MESSAGE_STYLE = """
<style>
.stApp {
   background: linear-gradient(to bottom, #f8f8f8, #ffffff);
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    width: auto;
    max-width: 60%;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    display: inline-block;
}
.user {
    background-color: #e0f7fa;
    margin-left: auto;
    margin-right: 0;
}
.assistant {
    background-color: #f0f0f0;
    margin-right: auto;
    margin-left: 0;
}
div[data-baseweb="input"] > div {
    max-width: 70%;
    margin: 0 auto;
}
.message-divider {
    border-bottom: 1px solid #eee;
    margin: 1rem 0;
}
.centered-title {
    text-align: center;
    margin-bottom: 2rem;
}
</style>
"""


def setup_ui() -> None:
    """Sets up the Streamlit UI configuration and base styling."""
    st.set_page_config(page_title="PrepAI", page_icon="🤖", layout="wide")
    st.markdown(CHAT_MESSAGE_STYLE, unsafe_allow_html=True)
    st.markdown("<div class='centered-title'><h1>PrepAI 🤖</h1></div>",
                unsafe_allow_html=True)


def display_messages(messages: list[dict]) -> None:
    """Displays chat messages in the UI, with code block handling.

    Args:
        messages: A list of dicts containing message data, e.g.
                  [{"role": "user", "content": "Hello"}].
    """
    message_container = st.empty()
    with message_container.container():
        for i, msg in enumerate(messages):
            message_class = "user" if msg["role"] == "user" else "assistant"
            content = msg["content"]
            if "```" in content:
                parts = content.split("```")
                st.markdown(
                    f'<div class="chat-message {message_class}">{parts[0]}</div>',
                    unsafe_allow_html=True
                )
                if len(parts) > 1:
                    st.code(parts[1], language="python")
                if len(parts) > 2:
                    st.markdown(
                        f'<div class="chat-message {message_class}">'
                        f'{parts[2]}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    f'<div class="chat-message {message_class}">{content}</div>',
                    unsafe_allow_html=True
                )
            st.markdown("<div class='message-divider'></div>", unsafe_allow_html=True)
