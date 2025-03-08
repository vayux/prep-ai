# PrepAI Chatbot

PrepAI Chatbot is an open-source, AI-powered interview preparation tool designed to help users practice coding questions and receive AI-generated feedback. It leverages cutting-edge AI models to provide structured, insightful, and optimized answers.

## 🚀 Overview
This chatbot application is built using **Streamlit** and **LangGraph**. It employs a **Retrieval-Augmented Generation (RAG) agent** to answer user queries by prioritizing responses from a knowledge base. If a suitable answer is not found, it falls back to a **Large Language Model (LLM)**. The conversation history is preserved, ensuring both user questions and AI responses are accessible.

## 🎯 Objective
**Prep AI** is a next-generation interview preparation assistant that integrates multiple AI models and RAG-based retrieval systems. It specializes in **technical and behavioral interview coaching**, featuring dedicated AI agents for different aspects of interview preparation.

## 🏗 Project Summary
Prep AI combines **specialized AI models** for:
- **Data Structures & Algorithms (DSA)**
- **Low-Level Design (LLD)**
- **High-Level Design (HLD)**
- **Behavioral Interviews**

These models are orchestrated under a **super-agent chatbot**, which intelligently selects the most relevant model based on user input. The chatbot leverages RAG pipelines for enhanced accuracy and ensures a seamless, interactive experience through **Streamlit**. 

Technologies used:
- **Ollama** for local LLM inference
- **LangChain & LangGraph** for orchestration
- **FAISS** for vector-based knowledge retrieval
- **Microservices architecture** for scalability and modular deployment
- **Cloud-based GPU instances** for performance optimization

### 🔮 Future Enhancements
- **Voice-Based Interactions**: Enabling voice-driven interview simulations
- **Real Interviewer Mode**: AI-driven interviewer simulation for a realistic experience
- **Enterprise Integrations**: Custom AI training for corporate interview preparation

---

## 🔥 Key Features
✅ AI-powered responses for coding and design questions  
✅ Integration with **Vector Databases** for context-aware responses  
✅ **Automatic Model Selection** for optimized query handling  
✅ **Mock Interview Simulations** for real-world practice  
✅ **User Feedback System** to enhance model accuracy  
✅ **Voice Interaction Agent (Upcoming Feature)**  

## 📽 Video Overview

Check out our **video walkthrough** to see Prep AI in action:

[![PrepAI Chatbot Overview](https://img.youtube.com/vi/Zg_cl5VYHJ0/0.jpg)](https://youtu.be/9yugsgBXoOg)

This video highlights the **DSA AI Agent**, showcasing its ability to:
- Solve **Data Structures & Algorithm** problems
- Provide **step-by-step explanations**
- Suggest **code optimizations**
- Execute **code snippets for validation**

📌 *Watch now to experience Prep AI in action!*

---

## 🛠 Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone <repository-url>
```

### 2️⃣ Navigate to the Project Directory
```bash
cd prep-ai
```

### 3️⃣ Install Dependencies
```bash
poetry install
```

### 4️⃣ Format Code using Black
```bash
poetry run black .
```

### 5️⃣ Run the Chatbot
```bash
streamlit run ui/chatbot.py
```

---

## 🎯 Usage
- Type any **coding, design, or behavioral interview question** in the chat interface.
- The AI intelligently selects the **best agent** to respond.
- If the knowledge base lacks an answer, the system falls back to an **LLM**.

---

## 🤝 Contributing
We **welcome contributions!** Please check out [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines on how to get involved.

---

## 📜 License
This project is **open-source** under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

💡 *Join us in redefining interview preparation with AI!* 🚀
