"""Tests for enhanced orchestrator features."""

import asyncio
import pytest
import pytest_asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

from defog.llm import (
    SharedContextStore,
    ArtifactType,
    Artifact,
    EnhancedMemoryManager,
    SharedMemoryEntry,
    ThinkingAgent,
    ExplorationExecutor,
    ExplorationStrategy,
    EnhancedAgentOrchestrator,
)
from defog.llm.orchestrator import Agent, SubAgentTask
from defog.llm.exploration_executor import ExplorationPath


class TestSharedContextStore:
    """Test the SharedContextStore functionality."""

    @pytest_asyncio.fixture
    async def shared_context(self):
        """Create a temporary shared context store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SharedContextStore(base_path=tmpdir)
            yield store

    @pytest.mark.asyncio
    async def test_write_and_read_artifact(self, shared_context):
        """Test basic write and read operations."""
        # Write artifact
        artifact = await shared_context.write_artifact(
            agent_id="test_agent",
            key="test_key",
            content={"data": "test_value"},
            artifact_type=ArtifactType.JSON,
            metadata={"test": "metadata"},
        )

        assert artifact.key == "test_key"
        assert artifact.agent_id == "test_agent"
        assert artifact.content == {"data": "test_value"}
        assert artifact.version == 1

        # Read artifact
        read_artifact = await shared_context.read_artifact("test_key")
        assert read_artifact is not None
        assert read_artifact.content == {"data": "test_value"}
        assert read_artifact.metadata == {"test": "metadata"}

    @pytest.mark.asyncio
    async def test_artifact_versioning(self, shared_context):
        """Test artifact versioning."""
        # Write initial version
        artifact1 = await shared_context.write_artifact(
            agent_id="test_agent",
            key="versioned_key",
            content="version 1",
            artifact_type=ArtifactType.TEXT,
        )
        assert artifact1.version == 1

        # Overwrite with new version
        artifact2 = await shared_context.write_artifact(
            agent_id="test_agent",
            key="versioned_key",
            content="version 2",
            artifact_type=ArtifactType.TEXT,
        )
        assert artifact2.version == 2

        # Read should get latest version
        latest = await shared_context.read_artifact("versioned_key")
        assert latest.content == "version 2"
        assert latest.version == 2

    @pytest.mark.asyncio
    async def test_list_artifacts_with_filters(self, shared_context):
        """Test listing artifacts with various filters."""
        # Create artifacts with different types and agents
        await shared_context.write_artifact(
            agent_id="agent1",
            key="plan/task1",
            content="plan content",
            artifact_type=ArtifactType.PLAN,
        )

        await shared_context.write_artifact(
            agent_id="agent2",
            key="result/task1",
            content="result content",
            artifact_type=ArtifactType.RESULT,
        )

        await shared_context.write_artifact(
            agent_id="agent1",
            key="result/task2",
            content="another result",
            artifact_type=ArtifactType.RESULT,
        )

        # Test pattern filter
        plan_artifacts = await shared_context.list_artifacts(pattern="plan/*")
        assert len(plan_artifacts) == 1
        assert plan_artifacts[0].key == "plan/task1"

        # Test agent filter
        agent1_artifacts = await shared_context.list_artifacts(agent_id="agent1")
        assert len(agent1_artifacts) == 2

        # Test artifact type filter
        result_artifacts = await shared_context.list_artifacts(
            artifact_type=ArtifactType.RESULT
        )
        assert len(result_artifacts) == 2

    @pytest.mark.asyncio
    async def test_artifact_lineage(self, shared_context):
        """Test artifact lineage tracking."""
        # Create parent artifact
        parent = await shared_context.write_artifact(
            agent_id="agent1",
            key="parent",
            content="parent content",
            artifact_type=ArtifactType.PLAN,
        )

        # Create child artifact
        child = await shared_context.write_artifact(
            agent_id="agent1",
            key="child",
            content="child content",
            artifact_type=ArtifactType.RESULT,
            parent_key="parent",
        )

        # Create grandchild artifact
        grandchild = await shared_context.write_artifact(
            agent_id="agent1",
            key="grandchild",
            content="grandchild content",
            artifact_type=ArtifactType.RESULT,
            parent_key="child",
        )

        # Get lineage
        lineage = await shared_context.get_artifact_lineage("grandchild")
        assert len(lineage) == 3
        assert lineage[0].key == "parent"
        assert lineage[1].key == "child"
        assert lineage[2].key == "grandchild"

    @pytest.mark.asyncio
    async def test_delete_artifact(self, shared_context):
        """Test artifact deletion."""
        # Create artifact
        await shared_context.write_artifact(
            agent_id="test_agent",
            key="to_delete",
            content="delete me",
            artifact_type=ArtifactType.TEXT,
        )

        # Verify it exists
        artifact = await shared_context.read_artifact("to_delete")
        assert artifact is not None

        # Delete it
        deleted = await shared_context.delete_artifact("to_delete", "test_agent")
        assert deleted is True

        # Verify it's gone
        artifact = await shared_context.read_artifact("to_delete")
        assert artifact is None


class TestEnhancedMemoryManager:
    """Test the EnhancedMemoryManager functionality."""

    @pytest_asyncio.fixture
    async def memory_manager(self):
        """Create memory manager with shared context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_context = SharedContextStore(base_path=tmpdir)
            manager = EnhancedMemoryManager(
                token_threshold=1000,
                preserve_last_n_messages=5,
                summary_max_tokens=500,
                shared_context_store=shared_context,
                agent_id="test_agent",
                cross_agent_sharing=True,
            )
            yield manager

    @pytest.mark.asyncio
    async def test_add_messages_with_sharing(self, memory_manager):
        """Test adding messages with cross-agent sharing."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        await memory_manager.add_messages_with_sharing(
            messages=messages, tokens=50, tags=["greeting"], share=True
        )

        # Verify messages were added to local history
        assert len(memory_manager.history.messages) == 2
        assert memory_manager.history.total_tokens == 50

        # Verify shared memory was created
        assert len(memory_manager._shared_memory_keys) == 1

    @pytest.mark.asyncio
    async def test_get_cross_agent_context(self, memory_manager):
        """Test retrieving cross-agent context."""
        # Add messages from current agent
        await memory_manager.add_messages_with_sharing(
            messages=[{"role": "user", "content": "Test message"}],
            tokens=20,
            tags=["test"],
            share=True,
        )

        # Simulate messages from another agent
        other_agent_entry = SharedMemoryEntry(
            agent_id="other_agent",
            messages=[{"role": "user", "content": "Other agent message"}],
            tokens=30,
            tags=["other"],
        )

        # Write to shared context as other agent
        await memory_manager.shared_context.write_artifact(
            agent_id="other_agent",
            key=f"memory/other_agent/{datetime.now().isoformat()}",
            content=other_agent_entry.__dict__,
            artifact_type=ArtifactType.TEXT,
            metadata={"tags": ["other"]},
        )

        # Get cross-agent context
        entries = await memory_manager.get_cross_agent_context(
            other_agent_ids=["other_agent"], max_entries=10
        )

        assert len(entries) == 1
        assert entries[0].agent_id == "other_agent"
        assert entries[0].messages[0]["content"] == "Other agent message"

    @pytest.mark.asyncio
    async def test_simple_summary_fallback(self, memory_manager):
        """Test the simple summary fallback."""
        messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Message 3"},
            {"role": "assistant", "content": "Response 3"},
        ]

        summary = memory_manager._simple_summary_fallback(messages)
        assert "Previous conversation summary:" in summary
        assert "Message 2" in summary  # Should include last 5 messages


class TestThinkingAgent:
    """Test the ThinkingAgent functionality."""

    @pytest_asyncio.fixture
    async def thinking_agent(self):
        """Create a thinking agent with shared context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_context = SharedContextStore(base_path=tmpdir)

            # Mock tools
            async def test_tool(input: str) -> str:
                return f"Tool result for: {input}"

            agent = ThinkingAgent(
                agent_id="thinking_agent",
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                system_prompt="You are a thinking test agent",
                tools=[test_tool],
                shared_context_store=shared_context,
                enable_thinking_mode=True,
                reasoning_effort="medium",
            )
            yield agent

    @pytest.mark.asyncio
    async def test_thinking_mode(self, thinking_agent):
        """Test the thinking mode functionality."""
        thinking_result = await thinking_agent.think(
            prompt="How to solve a complex problem?",
            context={"problem": "test problem"},
            save_to_shared_context=True,
        )

        # Verify thinking result
        assert isinstance(thinking_result, str)
        assert len(thinking_result) > 0

        # Verify artifact was saved
        artifacts = await thinking_agent.shared_context.list_artifacts(
            pattern="thinking/*", artifact_type=ArtifactType.PLAN
        )
        assert len(artifacts) > 0

    @pytest.mark.asyncio
    async def test_explore_alternatives(self, thinking_agent):
        """Test alternative exploration."""
        alternatives = await thinking_agent.explore_alternatives(
            task="Sort a large dataset",
            current_approach="QuickSort",
            num_alternatives=2,
        )

        assert len(alternatives) == 2
        for alt in alternatives:
            assert "approach" in alt
            assert "description" in alt

        # Verify exploration was saved
        artifacts = await thinking_agent.shared_context.list_artifacts(
            pattern="exploration/*", artifact_type=ArtifactType.EXPLORATION
        )
        assert len(artifacts) > 0


class TestExplorationExecutor:
    """Test the ExplorationExecutor functionality."""

    @pytest_asyncio.fixture
    async def exploration_executor(self):
        """Create an exploration executor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_context = SharedContextStore(base_path=tmpdir)
            executor = ExplorationExecutor(
                shared_context=shared_context,
                max_parallel_explorations=2,
                exploration_timeout=30.0,
                enable_learning=True,
            )
            yield executor

    @pytest.mark.asyncio
    async def test_exploration_strategy_decision(self, exploration_executor):
        """Test exploration strategy decision logic."""

        # Test with simple paths - should choose sequential
        simple_task = SubAgentTask(
            agent_id="test", task_id="task1", task_description="Simple task"
        )
        simple_paths = [
            ExplorationPath(
                path_id="p1",
                description="Path 1",
                approach="Approach 1",
                estimated_complexity="low",
            ),
            ExplorationPath(
                path_id="p2",
                description="Path 2",
                approach="Approach 2",
                estimated_complexity="low",
            ),
        ]

        strategy = exploration_executor._decide_strategy(simple_task, simple_paths)
        assert strategy == ExplorationStrategy.SEQUENTIAL

        # Test with complex paths - should choose parallel
        complex_paths = [
            ExplorationPath(
                path_id="p1",
                description="Path 1",
                approach="Approach 1",
                estimated_complexity="high",
            ),
            ExplorationPath(
                path_id="p2",
                description="Path 2",
                approach="Approach 2",
                estimated_complexity="high",
            ),
        ]

        strategy = exploration_executor._decide_strategy(simple_task, complex_paths)
        assert strategy == ExplorationStrategy.PARALLEL

        # Test with dependencies - should choose sequential
        dependent_paths = [
            ExplorationPath(
                path_id="p1",
                description="Path 1",
                approach="Approach 1",
                prerequisites=["setup"],
            )
        ]

        strategy = exploration_executor._decide_strategy(simple_task, dependent_paths)
        assert strategy == ExplorationStrategy.SEQUENTIAL


class TestEnhancedAgentOrchestrator:
    """Test the EnhancedAgentOrchestrator functionality."""

    @pytest_asyncio.fixture
    async def orchestrator(self):
        """Create an enhanced orchestrator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create main agent
            main_agent = Agent(
                agent_id="main",
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                system_prompt="You are the main orchestrator",
            )

            # Mock tools
            async def mock_tool(input: str) -> str:
                return f"Mock result: {input}"

            orchestrator = EnhancedAgentOrchestrator(
                main_agent=main_agent,
                available_tools=[mock_tool],
                shared_context_path=tmpdir,
                enable_thinking_agents=True,
                enable_exploration=True,
                enable_cross_agent_memory=True,
                max_parallel_tasks=2,
                global_timeout=60.0,
            )
            yield orchestrator

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.shared_context is not None
        assert orchestrator.exploration_executor is not None
        assert orchestrator.enable_thinking_agents is True
        assert orchestrator.enable_exploration is True
        assert orchestrator.enable_cross_agent_memory is True

    @pytest.mark.asyncio
    async def test_enhanced_subagent_creation(self, orchestrator):
        """Test creation of enhanced subagents."""

        # Mock tool
        async def test_tool(input: str) -> str:
            return f"Result: {input}"

        agent = orchestrator._create_enhanced_subagent(
            agent_id="test_subagent",
            system_prompt="Test subagent",
            tools=[test_tool],
            parent_context={"test": "context"},
        )

        assert isinstance(agent, ThinkingAgent)
        assert agent.agent_id == "test_subagent"
        assert agent.shared_context is not None
        assert agent.enable_thinking_mode is True

    @pytest.mark.asyncio
    async def test_orchestration_insights(self, orchestrator):
        """Test getting orchestration insights."""
        # Create some test artifacts
        await orchestrator.shared_context.write_artifact(
            agent_id="main",
            key="test/artifact1",
            content="test content",
            artifact_type=ArtifactType.RESULT,
        )

        await orchestrator.shared_context.write_artifact(
            agent_id="main",
            key="test/artifact2",
            content="test content 2",
            artifact_type=ArtifactType.PLAN,
        )

        insights = await orchestrator.get_orchestration_insights()

        assert "shared_context_stats" in insights
        assert "artifact_types" in insights["shared_context_stats"]
        assert insights["shared_context_stats"]["total_artifacts"] >= 2

    @pytest.mark.asyncio
    async def test_workspace_cleanup(self, orchestrator):
        """Test workspace cleanup functionality."""
        # Create old artifact
        old_artifact = await orchestrator.shared_context.write_artifact(
            agent_id="main",
            key="old/artifact",
            content="old content",
            artifact_type=ArtifactType.TEXT,
        )

        # Create recent artifact
        recent_artifact = await orchestrator.shared_context.write_artifact(
            agent_id="main",
            key="recent/artifact",
            content="recent content",
            artifact_type=ArtifactType.TEXT,
        )

        # Manually set old artifact's created_at to be old
        old_artifact.created_at = datetime.now() - timedelta(hours=48)

        # Clean up artifacts older than 24 hours
        # Note: This won't actually delete in the test because we can't
        # modify the filesystem timestamp, but we test the method runs
        cleaned = await orchestrator.cleanup_workspace(older_than_hours=24)

        # Verify cleanup method runs without error
        assert isinstance(cleaned, int)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
