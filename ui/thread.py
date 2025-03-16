"""
Represents a single chat thread for PrepAI, including a list of messages.
"""

import uuid


class Thread:
    """Stores messages for a particular chat thread."""

    def __init__(self, thread_id: str = None, messages: list[dict] = None) -> None:
        """
        Args:
            thread_id: Optional custom thread ID.
            messages: Optional list of existing messages (role/content dicts).
        """
        self.id = thread_id or str(uuid.uuid4())
        self.messages = messages or []

    def add_message(self, message: dict) -> None:
        """Adds a message dict (role, content) to the thread.

        Args:
            message: A dict with keys such as {"role": "user", "content": "..."}.
        """
        self.messages.append(message)

    def get_messages(self) -> list[dict]:
        """Returns all messages in this thread.

        Returns:
            A list of message dicts.
        """
        return self.messages
