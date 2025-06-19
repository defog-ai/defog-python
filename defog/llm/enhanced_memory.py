"""Enhanced memory management with cross-agent sharing and context summarization."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field

from .memory.history_manager import MemoryManager
from .shared_context import SharedContextStore, ArtifactType
from .utils import chat_async

logger = logging.getLogger(__name__)


@dataclass
class SharedMemoryEntry:
    """Entry in shared memory with agent attribution."""

    agent_id: str
    messages: List[Dict[str, Any]]
    summary: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    tokens: int = 0
    tags: List[str] = field(default_factory=list)


class EnhancedMemoryManager(MemoryManager):
    """
    Extended memory manager with cross-agent memory sharing and advanced summarization.

    Features:
    - Cross-agent memory sharing through SharedContextStore
    - Automatic context summarization when approaching limits
    - Ability to spawn fresh agents with summarized context
    - Memory tagging and search capabilities
    """

    def __init__(
        self,
        token_threshold: int = 50000,
        preserve_last_n_messages: int = 10,
        summary_max_tokens: int = 4000,
        enabled: bool = True,
        shared_context_store: Optional[SharedContextStore] = None,
        agent_id: Optional[str] = None,
        summarization_provider: str = "anthropic",
        summarization_model: str = "claude-sonnet-4-20250514",
        cross_agent_sharing: bool = True,
        max_shared_memory_entries: int = 50,
    ):
        """
        Initialize enhanced memory manager.

        Args:
            token_threshold: Tokens before triggering compactification
            preserve_last_n_messages: Number of recent messages to preserve
            summary_max_tokens: Maximum tokens for summary
            enabled: Whether memory management is enabled
            shared_context_store: Optional shared context store for cross-agent memory
            agent_id: ID of the agent using this memory manager
            summarization_provider: LLM provider for summarization
            summarization_model: Model to use for summarization
            cross_agent_sharing: Whether to enable cross-agent memory sharing
            max_shared_memory_entries: Maximum shared memory entries to keep
        """
        super().__init__(
            token_threshold=token_threshold,
            preserve_last_n_messages=preserve_last_n_messages,
            summary_max_tokens=summary_max_tokens,
            enabled=enabled,
        )

        self.shared_context = shared_context_store
        self.agent_id = agent_id or "unknown"
        self.summarization_provider = summarization_provider
        self.summarization_model = summarization_model
        self.cross_agent_sharing = cross_agent_sharing
        self.max_shared_memory_entries = max_shared_memory_entries

        # Track shared memory keys
        self._shared_memory_keys: Set[str] = set()

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        logger.info(f"EnhancedMemoryManager initialized for agent {self.agent_id}")

    async def add_messages_with_sharing(
        self,
        messages: List[Dict[str, Any]],
        tokens: int,
        tags: Optional[List[str]] = None,
        share: bool = True,
    ) -> None:
        """
        Add messages to history with optional cross-agent sharing.

        Args:
            messages: Messages to add
            tokens: Token count for the messages
            tags: Optional tags for categorizing the memory
            share: Whether to share this memory with other agents
        """
        # Add to local history
        self.add_messages(messages, tokens)

        # Share with other agents if enabled
        if share and self.cross_agent_sharing and self.shared_context:
            await self._share_memory(messages, tokens, tags)

    async def _share_memory(
        self,
        messages: List[Dict[str, Any]],
        tokens: int,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Share memory entry with other agents through shared context."""
        async with self._lock:
            # Create memory key
            timestamp = datetime.now()
            memory_key = f"memory/{self.agent_id}/{timestamp.isoformat()}"

            # Create shared memory entry
            entry = SharedMemoryEntry(
                agent_id=self.agent_id,
                messages=messages,
                timestamp=timestamp,
                tokens=tokens,
                tags=tags or [],
            )

            # Write to shared context
            await self.shared_context.write_artifact(
                agent_id=self.agent_id,
                key=memory_key,
                content=entry.__dict__,
                artifact_type=ArtifactType.TEXT,
                metadata={
                    "tokens": tokens,
                    "tags": tags,
                    "message_count": len(messages),
                },
            )

            self._shared_memory_keys.add(memory_key)

            # Clean up old entries if needed
            await self._cleanup_old_shared_memories()

    async def get_cross_agent_context(
        self,
        other_agent_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        max_entries: int = 10,
    ) -> List[SharedMemoryEntry]:
        """
        Retrieve relevant context from other agents.

        Args:
            other_agent_ids: Specific agent IDs to get context from (None = all)
            tags: Filter by tags
            since: Only get entries after this time
            max_entries: Maximum number of entries to retrieve

        Returns:
            List of shared memory entries from other agents
        """
        if not self.shared_context or not self.cross_agent_sharing:
            return []

        # Get memory artifacts from shared context
        artifacts = await self.shared_context.list_artifacts(
            pattern="memory/*", since=since
        )

        entries = []
        for artifact in artifacts[:max_entries]:
            # Skip own memories
            if artifact.agent_id == self.agent_id:
                continue

            # Apply agent filter
            if other_agent_ids and artifact.agent_id not in other_agent_ids:
                continue

            # Apply tag filter
            if tags and artifact.metadata:
                artifact_tags = artifact.metadata.get("tags", [])
                if not any(tag in artifact_tags for tag in tags):
                    continue

            # Convert to SharedMemoryEntry
            try:
                entry = SharedMemoryEntry(**artifact.content)
                entries.append(entry)
            except Exception as e:
                logger.warning(f"Failed to parse shared memory entry: {e}")

        return entries

    async def summarize_context(
        self, messages_to_summarize: List[Dict[str, Any]], context: Optional[str] = None
    ) -> str:
        """
        Create an intelligent summary of messages using LLM.

        Args:
            messages_to_summarize: Messages to summarize
            context: Optional context about what to focus on

        Returns:
            Summary text
        """
        if not messages_to_summarize:
            return "No messages to summarize."

        # Prepare summarization prompt
        summary_prompt = f"""Please provide a concise summary of the following conversation.
Focus on key decisions, important information, and action items.
Maximum length: {self.summary_max_tokens // 4} words.

{"Context: " + context if context else ""}

Conversation to summarize:"""

        # Create messages for summarization
        summarization_messages = [{"role": "system", "content": summary_prompt}]

        # Add the messages to summarize
        summarization_messages.extend(messages_to_summarize)

        # Add final instruction
        summarization_messages.append(
            {"role": "user", "content": "Please provide the summary now."}
        )

        try:
            # Call LLM for summarization
            response = await chat_async(
                provider=self.summarization_provider,
                model=self.summarization_model,
                messages=summarization_messages,
                max_completion_tokens=self.summary_max_tokens,
                temperature=0.3,
                reasoning_effort="medium",
            )

            summary = response.content

            # Store summary in shared context if available
            if self.shared_context and self.cross_agent_sharing:
                summary_key = f"summary/{self.agent_id}/{datetime.now().isoformat()}"
                await self.shared_context.write_artifact(
                    agent_id=self.agent_id,
                    key=summary_key,
                    content=summary,
                    artifact_type=ArtifactType.SUMMARY,
                    metadata={
                        "original_messages": len(messages_to_summarize),
                        "context": context,
                    },
                )

            return summary

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback to simple extraction
            return self._simple_summary_fallback(messages_to_summarize)

    def _simple_summary_fallback(self, messages: List[Dict[str, Any]]) -> str:
        """Simple fallback summary when LLM summarization fails."""
        summary_parts = ["Previous conversation summary:"]

        for msg in messages[-5:]:  # Last 5 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."
            summary_parts.append(f"- {role}: {content}")

        return "\n".join(summary_parts)

    async def prepare_context_for_new_agent(
        self,
        focus: Optional[str] = None,
        include_cross_agent: bool = True,
        max_context_tokens: int = 10000,
    ) -> Dict[str, Any]:
        """
        Prepare a summarized context for spawning a new agent.

        Args:
            focus: Optional focus area for the context
            include_cross_agent: Whether to include cross-agent memories
            max_context_tokens: Maximum tokens for the context

        Returns:
            Dictionary with prepared context for new agent
        """
        context = {
            "parent_agent_id": self.agent_id,
            "timestamp": datetime.now().isoformat(),
            "focus": focus,
        }

        # Get and summarize current conversation
        if self.history.messages:
            current_summary = await self.summarize_context(
                self.history.messages, context=focus
            )
            context["current_conversation_summary"] = current_summary

        # Include relevant cross-agent context
        if include_cross_agent and self.shared_context:
            cross_agent_entries = await self.get_cross_agent_context(max_entries=5)

            if cross_agent_entries:
                cross_agent_summaries = []
                for entry in cross_agent_entries:
                    if entry.summary:
                        cross_agent_summaries.append(
                            {
                                "agent_id": entry.agent_id,
                                "summary": entry.summary,
                                "timestamp": entry.timestamp.isoformat(),
                            }
                        )
                context["cross_agent_context"] = cross_agent_summaries

        # Get recent shared artifacts if any
        if self.shared_context:
            recent_artifacts = await self.shared_context.get_recent_artifacts(
                limit=5, artifact_type=ArtifactType.RESULT
            )

            if recent_artifacts:
                context["recent_results"] = [
                    {
                        "key": artifact.key,
                        "agent_id": artifact.agent_id,
                        "content": (
                            str(artifact.content)[:200] + "..."
                            if len(str(artifact.content)) > 200
                            else artifact.content
                        ),
                    }
                    for artifact in recent_artifacts
                ]

        return context

    async def _cleanup_old_shared_memories(self) -> None:
        """Clean up old shared memory entries to prevent unbounded growth."""
        if len(self._shared_memory_keys) > self.max_shared_memory_entries:
            # Get oldest keys to remove
            sorted_keys = sorted(self._shared_memory_keys)
            keys_to_remove = sorted_keys[
                : len(sorted_keys) - self.max_shared_memory_entries
            ]

            for key in keys_to_remove:
                try:
                    await self.shared_context.delete_artifact(key, self.agent_id)
                    self._shared_memory_keys.remove(key)
                except Exception as e:
                    logger.warning(f"Failed to cleanup old memory {key}: {e}")

    async def compact_with_enhanced_summary(self) -> None:
        """
        Perform memory compactification with enhanced summarization.

        This method extends the base compactification with:
        - Intelligent LLM-based summarization
        - Cross-agent memory sharing of summaries
        - Preservation of important context
        """
        if not self.should_compactify():
            return

        # Get messages to compact
        system_msgs, to_summarize, to_preserve = (
            self.get_messages_for_compactification()
        )

        if not to_summarize:
            return

        # Create enhanced summary
        summary = await self.summarize_context(to_summarize)

        # Create summary message
        summary_message = {
            "role": "system",
            "content": f"[Previous conversation summary]\n{summary}\n[End of summary]",
        }

        # Share the summary if cross-agent sharing is enabled
        if self.cross_agent_sharing and self.shared_context:
            await self._share_memory(
                [summary_message],
                tokens=len(summary) // 4,  # Rough estimate
                tags=["summary", "compactified"],
            )

        # Update history
        new_token_count = self._estimate_tokens(
            system_msgs + [summary_message] + to_preserve
        )
        self.update_after_compactification(
            system_msgs, summary_message, to_preserve, new_token_count
        )

        logger.info(f"Memory compactified for agent {self.agent_id}")

    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count for messages."""
        # Rough estimation: 1 token per 4 characters
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return total_chars // 4
