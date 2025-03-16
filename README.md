# PrepAI Chatbot

**PrepAI Chatbot** is an open-source, **multi-agent**, AI-powered interview preparation tool designed to help users practice coding, design, and behavioral questions, receiving detailed AI-generated feedback.

This project employs a **config-driven LLM** approach—so you can seamlessly switch between **Ollama**, **OpenAI**, or other LLMs in the future. It also integrates a **RAG (Retrieval-Augmented Generation)** pipeline with **FAISS**, orchestrates specialized **AI agents** via **Crew AI**, and organizes them under a **LangGraph** pipeline for efficient, context-aware question handling.

---

## 🚀 Overview

- Built using **Streamlit** for an interactive UI.
- Incorporates **LangGraph** for pipeline orchestration.
- Employs **FAISS** for knowledge-base retrieval (RAG).
- Automatically selects specialized agents for **DSA**, **LLD**, **HLD**, or **Behavioral** interview questions.
- Remembers conversation history to maintain context across queries.

When a user’s question matches the knowledge base, the system responds with a **retrieved** answer. Otherwise, it gracefully falls back to a **Large Language Model (LLM)**—either local (via **Ollama**) or via **OpenAI**—for a best-effort solution.

---

## 🎯 Objective

**Prep AI** aims to be a next-generation, **modular** interview prep assistant that can handle multiple question domains:

- **Data Structures & Algorithms (DSA)**
- **Low-Level Design (LLD)**
- **High-Level Design (HLD)**
- **Behavioral Interviews**

Each domain is handled by a dedicated AI agent, coordinated by a **Crew AI** super-agent. The overarching pipeline uses **LangGraph** for advanced orchestration and **FAISS** for RAG-based context retrieval.

---

## 🏗 Project Summary

### Key Technologies

1. **Local LLM (Ollama)** or **OpenAI**: Configurable LLM backend.  
2. **FAISS**: Vector database for RAG queries.  
3. **LangGraph**: Node-based pipeline orchestration for controlling the flow of requests.  
4. **Crew AI**: Multi-agent orchestration to route user queries to the correct agent.  
5. **Streamlit**: Web-based UI framework, supporting multi-threaded chat sessions.

#### Architectural Highlights

- **Config-Driven LLM**: Switch between Ollama or OpenAI by changing environment variables.
- **RAG Pipeline**: A vector store (FAISS) to fetch context from user-provided PDFs/text files.
- **Multi-Agent System**: Dedicated agents for DSA, LLD, HLD, and Behavioral interviews.
- **Multi-Threaded Chat**: Users can create multiple “threads” for separate conversations.
- **Data Indexing Script**: A standalone script to embed PDF/TXT documents into FAISS.

---

### 🔮 Future Enhancements

- **Voice-Based Interactions**: Add voice-driven interview simulations.
- **Real Interviewer Mode**: AI-driven interviewer with randomized follow-up questions.
- **Enterprise Integrations**: Customizable domain-specific embeddings for corporate interview prep.

---

## 🔥 Key Features

- ✅ **AI-Powered** responses for coding, system design, and behavioral queries  
- ✅ **Vector Store** (FAISS) for context retrieval, ensuring accurate, context-rich answers  
- ✅ **Automatic Agent Selection** via Crew AI (DSA, LLD, HLD, Behavioral)  
- ✅ **Config-Driven LLM** (Ollama or OpenAI, easily extended to other backends)  
- ✅ **Multi-Threaded Chat** with session management in Streamlit  
- ✅ **Mock Interview Simulations**: Ideal for real-time practice and iterative feedback  

---

## 📽 Video Overview

Check out our **video walkthrough** to see Prep AI in action:

[![PrepAI Chatbot Overview](https://img.youtube.com/vi/Zg_cl5VYHJ0/0.jpg)](https://youtu.be/9yugsgBXoOg)

This video demonstrates:
- How the **DSA AI Agent** solves Data Structures & Algorithm problems  
- **Step-by-step** solutions and code optimizations  
- Interactive **Q&A** with the system  

*Watch now to experience Prep AI!* 

---

## 🛠 Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
```
### 2. Navigate to the Project Directory
```bash
cd prep-ai
```
### 3. Install Dependencies
We use Poetry for dependency management:
```bash
poetry install
```
### 4. (Optional) Index Your Data
If you have PDFs or TXT files for context retrieval, place them in ./data (or any directory you like), then run:
```bash
poetry run python scripts/index_data.py --data_dir ./data
```
This command will create a FAISS index in ./faiss_index by default.

### 5. Launch the Chatbot
```bash
poetry run streamlit run main.py
```
Then open your browser at the provided URL (e.g., http://localhost:8501).

## 🎯 Usage
1. Select or Create a Thread in the sidebar (optional).

2. Ask a question in the Streamlit chat input.

- For a DSA question, type:

    ```dsa: How to reverse a linked list?```
- For LLD, type:

    ```lld: Design a URL shortener.```

- (If no prefix is provided, the system defaults to the Behavioral agent.)
3. If a relevant answer is retrieved from FAISS, it’s shown immediately. Otherwise, the LLM (Ollama or OpenAI) is queried.

4. Repeat to refine your question or create additional threads for separate conversations.

## 🤝 Contributing
We welcome contributions from the community! To get started, please:

1. Read our [`CONTRIBUTING.md`](CONTRIBUTING.md) guidelines for details on the workflow.
2. Fork the repo, make changes on a feature branch, and open a pull request.
3. Ensure your code follows PEP 8 guidelines, has docstrings, and passes any existing tests.
## 📜 License
This project is open-source under the MIT [`License`](License). See LICENSE for full details.
##
💡 Join us in redefining interview preparation with AI! 🚀