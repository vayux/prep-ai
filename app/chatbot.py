import concurrent.futures
import streamlit as st
from app.graph import prep_ai_graph

st.set_page_config(page_title="PrepAI", page_icon="🤖")
st.title("PrepAI 🤖")

# Initialize session state for messages and processing status
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "processing" not in st.session_state:
    st.session_state["processing"] = False

# Display chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask me anything:", disabled=st.session_state["processing"])

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.session_state["processing"] = True  # Disable input during processing

    # Display user's question immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Use a container to group assistant responses for better layout
    assistant_container = st.container()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        graph_future = executor.submit(prep_ai_graph.run, user_input)

        with assistant_container.chat_message("assistant"):
            with st.spinner("Generating Response..."):
                graph_response, source = graph_future.result()  # Get source
                st.markdown(f"**Generated Answer (Source: {source}):**\n\n{graph_response}")

                # Append assistant's response to session state
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": f"**Generated Answer (Source: {source}):**\n\n{graph_response}",
                    }
                )

    st.session_state["processing"] = False  # Re-enable input
