import streamlit as st
import time
from app.dsa import query_rag, query_llm_directly
import concurrent.futures

# Configure the page
st.set_page_config(page_title="PrepAI", page_icon="🤖", layout="centered")
st.title("PrepAI 🤖")

# Initialize session state for chat history if not present
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field
user_input = st.chat_input("Ask me anything related to LeetCode:")

if user_input:
    # Append user message to session state
    st.session_state["messages"].append({"role": "user", "content": user_input})
    
    # Display user input in chat
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Run both queries in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        llm_future = executor.submit(query_llm_directly, user_input)
        vector_db_future = executor.submit(query_rag, user_input)

                
        # LLM response
        with st.chat_message("assistant"):
            with st.spinner("Generating answer from LLM..."):
                llm_answer = llm_future.result()
                st.markdown(f"**LLM Answer:**\n\n{llm_answer}")
            
            # Append response to session state
            st.session_state["messages"].append({"role": "assistant", "content": f"**LLM Answer:**\n\n{llm_answer}"})

        # Vector DB response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving answer from Vector DB..."):
                vector_db_answer = vector_db_future.result()
                st.markdown(f"**Vector DB Answer:**\n\n{vector_db_answer}")
            
            # Append response to session state
            st.session_state["messages"].append({"role": "assistant", "content": f"**Vector DB Answer:**\n\n{vector_db_answer}"})
