import uuid

class Thread:
    """Represents a chat thread."""

    def __init__(self, thread_id=None, messages=None):
        """Initializes a thread."""
        self.id = thread_id or str(uuid.uuid4())
        self.messages = messages or []

    def add_message(self, message):
        """Adds a message to the thread."""
        self.messages.append(message)

    def get_messages(self):
        """Returns the messages in the thread."""
        return self.messages