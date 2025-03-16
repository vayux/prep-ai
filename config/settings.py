"""
Module for handling configuration and environment-driven settings
for the PrepAI project.
"""

import os


class Settings:
    """Encapsulates configuration for PrepAI, including LLM selection
    and various file paths.
    """

    # Data paths
    DATA_PATH = os.getenv("DATA_PATH", "./data")
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./faiss_index")

    # LLM backend can be: "ollama", "openai", etc.
    LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()

    # Ollama configuration
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2-7b")

    # OpenAI configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # Crew AI endpoint
    CREWAI_ENDPOINT = os.getenv("CREWAI_ENDPOINT", "http://localhost:8080")

    def get_llm_client(self):
        """Return an LLM client object based on the LLM_BACKEND setting.

        If LLM_BACKEND is 'ollama', instantiate an Ollama client.
        If LLM_BACKEND is 'openai', instantiate an OpenAI client.
        Raises:
            ValueError: If the specified LLM_BACKEND is not supported.

        Returns:
            Callable[[str], str]: A callable object that takes a string prompt
            and returns a string response.
        """
        if self.LLM_BACKEND == "ollama":
            from ollama import Ollama
            return Ollama(model=self.OLLAMA_MODEL)

        if self.LLM_BACKEND == "openai":
            import openai
            openai.api_key = self.OPENAI_API_KEY
            return OpenAIWrapper(model=self.OPENAI_MODEL)

        raise ValueError(f"Unsupported LLM_BACKEND: {self.LLM_BACKEND}")


class OpenAIWrapper:
    """A minimal wrapper for OpenAI's ChatCompletion to mimic a callable
    interface similar to Ollama.
    """

    def __init__(self, model: str):
        """
        Args:
            model: The name of the OpenAI model (e.g., 'gpt-3.5-turbo').
        """
        self.model = model

    def __call__(self, prompt: str) -> str:
        """Call the OpenAI API to get a response for the given prompt.

        Args:
            prompt: User query or any textual input to the model.

        Returns:
            The text content from OpenAI's model response.
        """
        import openai
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"]


settings = Settings()
