"""Manages conversation history and memory for LLM interactions."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import copy


@dataclass
class ConversationHistory:
    """Container for conversation messages with metadata."""

    messages: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_compactified_at: Optional[datetime] = None
    compactification_count: int = 0

    def add_message(self, message: Dict[str, Any], tokens: int = 0) -> None:
        """Add a message to the history."""
        self.messages.append(copy.deepcopy(message))
        self.total_tokens += tokens

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get a copy of all messages."""
        return copy.deepcopy(self.messages)

    def clear(self) -> None:
        """Clear all messages and reset token count."""
        self.messages.clear()
        self.total_tokens = 0

    def replace_messages(
        self, new_messages: List[Dict[str, Any]], new_token_count: int
    ) -> None:
        """Replace all messages with new ones."""
        self.messages = copy.deepcopy(new_messages)
        self.total_tokens = new_token_count
        self.last_compactified_at = datetime.now()
        self.compactification_count += 1


class MemoryManager:
    """Manages conversation memory with automatic compactification."""

    def __init__(
        self,
        token_threshold: int = 50000,  # Default to ~100k tokens before compactifying
        preserve_last_n_messages: int = 10,
        summary_max_tokens: int = 4000,
        enabled: bool = True,
    ):
        """
        Initialize the memory manager.

        Args:
            token_threshold: Number of tokens before triggering compactification
            preserve_last_n_messages: Number of recent messages to always preserve
            summary_max_tokens: Maximum tokens for the summary
            enabled: Whether memory management is enabled
        """
        self.token_threshold = token_threshold
        self.preserve_last_n_messages = preserve_last_n_messages
        self.summary_max_tokens = summary_max_tokens
        self.enabled = enabled
        self.history = ConversationHistory()

    def should_compactify(self) -> bool:
        """Check if memory should be compactified based on token count."""
        if not self.enabled:
            return False
        return self.history.total_tokens >= self.token_threshold

    def add_messages(self, messages: List[Dict[str, Any]], tokens: int) -> None:
        """Add multiple messages to history."""
        for message in messages:
            self.history.add_message(
                message, tokens // len(messages) if messages else 0
            )

    def get_messages_for_compactification(
        self,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split messages into system messages, messages to summarize, and messages to preserve.

        System messages are always preserved at the beginning.
        Only user/assistant messages are eligible for summarization.

        Returns:
            Tuple of (system_messages, messages_to_summarize, messages_to_preserve)
        """
        all_messages = self.history.get_messages()

        # Separate system messages from others
        system_messages = []
        other_messages = []

        for msg in all_messages:
            if msg.get("role") == "system":
                system_messages.append(msg)
            else:
                other_messages.append(msg)

        # Apply preservation logic only to non-system messages
        if len(other_messages) <= self.preserve_last_n_messages:
            return system_messages, [], other_messages

        split_index = len(other_messages) - self.preserve_last_n_messages
        return (
            system_messages,
            other_messages[:split_index],
            other_messages[split_index:],
        )

    def update_after_compactification(
        self,
        system_messages: List[Dict[str, Any]],
        summary_message: Dict[str, Any],
        preserved_messages: List[Dict[str, Any]],
        new_token_count: int,
    ) -> None:
        """Update history after compactification."""
        # Combine: system messages + summary + preserved messages
        new_messages = system_messages + [summary_message] + preserved_messages
        self.history.replace_messages(new_messages, new_token_count)

    def get_current_messages(self) -> List[Dict[str, Any]]:
        """Get current conversation messages."""
        return self.history.get_messages()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_tokens": self.history.total_tokens,
            "message_count": len(self.history.messages),
            "compactification_count": self.history.compactification_count,
            "last_compactified_at": (
                self.history.last_compactified_at.isoformat()
                if self.history.last_compactified_at
                else None
            ),
            "enabled": self.enabled,
            "token_threshold": self.token_threshold,
        }
