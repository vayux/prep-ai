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
    margin-right:0;
}
.assistant {
    background-color: #f0f0f0;
    margin-right: auto;
    margin-left:0;
}
div[data-baseweb="input"] > div {
    max-width: 70%; /* Adjust this value to your liking */
    margin: 0 auto; /* Center the input */
}
.thread-button {
    background-color: #4caf50;
    color: white;
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 5px;
    margin-bottom: 0.5rem;
    cursor: pointer;
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.thread-button:hover {
    background-color: #45a049;
}
.source-info {
    font-style: italic;
    color: #777;
    margin-top: 0.5rem;
}
.thread-title {
    font-weight: bold;
    margin-bottom: 0.5rem;
}
.message-divider {
    border-bottom: 1px solid #eee;
    margin: 1rem 0;
}
.hide-streamlit-sidebar > div {
    display: none;
}
.centered-title {
    text-align: center;
    margin-bottom: 2rem;
}
.content-container {
    max-width: 800px;
    margin: 0 auto;
}
</style>
"""

def setup_ui():
    """Sets up the UI for the chatbot."""
    st.set_page_config(page_title="PrepAI", page_icon="🤖", layout="wide")
    st.markdown(CHAT_MESSAGE_STYLE, unsafe_allow_html=True)
    st.markdown("<style>.hide-streamlit-sidebar > div {display: none;}</style>", unsafe_allow_html=True)
    st.markdown("<div class='centered-title'><h1>PrepAI 🤖</h1></div>", unsafe_allow_html=True)

def display_messages(messages):
    """Displays all chat messages."""
    message_container = st.empty()  # Create an empty container

    with message_container.container():
        for i, message in enumerate(messages):
            with st.container(key=f"message_{i}"):
                message_class = "user" if message["role"] == "user" else "assistant"
                content = message["content"]
                if message["role"] == "assistant" and "Source:" in content:
                    parts = content.split("Source:", 1)
                    content = f"{parts[0]}<div class='source-info'>Source: {parts[1]}</div>"
                if message["role"] == "assistant" and "```" in content:
                    parts = content.split("```")
                    st.markdown(f'<div class="chat-message {message_class}">{parts[0]}</div>', unsafe_allow_html=True)
                    if len(parts) > 1:
                        st.code(parts[1], language="python")
                    if len(parts) > 2:
                        st.markdown(f'<div class="chat-message {message_class}">{parts[2]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message {message_class}">{content}</div>', unsafe_allow_html=True)
                st.markdown("<div class='message-divider'></div>", unsafe_allow_html=True)