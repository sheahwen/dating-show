"""Message management for the dating show chat."""

from typing import List
from autogen_core.models import LLMMessage
from autogen_agentchat.messages import TextMessage

from common.text import extract_content_without_thoughts


class MessageManager:
    """Handles message storage and retrieval for the dating show chat."""

    def __init__(self):
        self.messages: List[LLMMessage] = []
        self.sanitized_messages: List[TextMessage] = []

    def add_message(self, message: LLMMessage) -> None:
        """
        Add a message to both original and sanitized message lists.

        Args:
            message: The original message to add
        """
        self.messages.append(message)

        sanitized_message = TextMessage(
            content=extract_content_without_thoughts(message.content),
            source=message.source,
        )
        self.sanitized_messages.append(sanitized_message)

    def get_original_messages(self) -> List[LLMMessage]:
        """Get the original messages with thoughts included."""
        return self.messages.copy()

    def get_sanitized_messages(self) -> List[TextMessage]:
        """Get the sanitized messages without thoughts."""
        return self.sanitized_messages.copy()

    def get_recent_messages(self, limit: int = 8) -> List[TextMessage]:
        """Get the last N sanitized messages."""
        return (
            self.sanitized_messages[-limit:]
            if len(self.sanitized_messages) > limit
            else self.sanitized_messages.copy()
        )

    def get_message_count(self) -> int:
        """Get the total number of messages."""
        return len(self.messages)
