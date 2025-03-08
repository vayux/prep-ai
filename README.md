# PrepAI Chatbot

PrepAI Chatbot is an open-source AI-powered interview preparation tool designed to help users practice coding questions and receive AI-generated feedback. It leverages advanced language models to provide precise, well-structured answers.

## Objective
Prep AI is an advanced interview preparation assistant that leverages multiple AI models and Retrieval-Augmented Generation (RAG) systems to help users prepare for technical and behavioral interviews. It consists of specialized AI models for different aspects of interviews and a super-agent chatbot that integrates them seamlessly.

## Project Summary
Prep AI is an AI-powered interview preparation assistant that integrates specialized models for Data Structures & Algorithms (DSA), Low-Level Design (LLD), High-Level Design (HLD), and Behavioral interviews. It leverages a super-agent chatbot that intelligently selects the most suitable model based on user queries. The system employs Retrieval-Augmented Generation (RAG) for enhanced accuracy and offers an interactive UI using Streamlit. Open-source technologies like Ollama, LangChain, FAISS, and LangGraph enable efficient AI model execution. The project follows a microservices architecture, ensuring scalability and modular deployment with cloud-based GPU instances. Future enhancements include voice-based interactions, real interviewer mode, and enterprise integrations.

## Project Requirements
1. **Core Functionalities**
   - **DSA AI Agent**: Solves data structures and algorithm problems, provides step-by-step explanations, suggests optimizations, and executes code snippets for validation.
   - **LLD AI Agent**: Helps with low-level design questions, providing class diagrams, design patterns, and best practices using tools like Mermaid.js.
   - **HLD AI Agent**: Assists with high-level system design, discussing architectures, scalability, and trade-offs, and generating system component diagrams.
   - **Behavioral AI Agent**: Guides users through behavioral interviews, offering STAR framework responses, analyzing responses, and providing AI-driven feedback.
   - **Super-Agent Chatbot**: Integrates all specialized models to provide a seamless interview prep experience by intelligently routing queries.
   - **Automatic Model Selection**: The chatbot automatically determines the most suitable AI model (DSA, LLD, HLD, or Behavioral) based on the user's question using a rule-based or ML classifier.
   - **Custom RAG Pipelines**: Enhances accuracy by retrieving relevant information from interview question-answer datasets using FAISS or Milvus.
   - **User Feedback System**: Collects feedback to refine AI responses and improve model accuracy over time.
   - **Interactive Mock Interviews**: Simulates real interview environments with AI-driven question generation and evaluation.
   - **Voice Interaction Agent (Future Scope)**: Enables voice-based interview practice using Whisper for transcription and LLMs for response generation.
   - **User Interface (UI)**: Provides a user-friendly interface using Streamlit or similar frameworks for seamless interaction with AI models.

## Features
- AI-powered responses to coding questions
- Integration with Vector DB for context-based answers
- Parallel processing of queries for efficiency

## Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd prep-ai
   ```
3. Install dependencies:
   ```bash
   poetry install
   ```
4. Run the chatbot:
   ```bash
   streamlit run ui/chatbot.py
   ```

## Usage
- Ask any coding-related question in the chat interface.
- Receive AI-generated answers based on the context from the Vector DB and LLM.

## Contributing
We welcome contributions! Please see `CONTRIBUTING.md` for more details.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
