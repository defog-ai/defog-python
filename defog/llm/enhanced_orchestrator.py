"""Enhanced orchestrator with shared context and exploration capabilities."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime

from .orchestrator import AgentOrchestrator, Agent, SubAgentTask, SubAgentResult
from .thinking_agent import ThinkingAgent
from .shared_context import SharedContextStore, ArtifactType
from .enhanced_memory import EnhancedMemoryManager
from .exploration_executor import ExplorationExecutor, ExplorationStrategy
from .providers.base import LLMResponse
from .config import EnhancedOrchestratorConfig

logger = logging.getLogger(__name__)


class EnhancedAgentOrchestrator(AgentOrchestrator):
    """
    Enhanced orchestrator with shared context, thinking agents, and exploration.

    This extends the base orchestrator with:
    - Shared filesystem-based context for cross-agent communication
    - ThinkingAgent support with extended reasoning
    - Alternative path exploration for complex tasks
    - Enhanced memory management with cross-agent sharing
    """

    def __init__(
        self,
        main_agent: Agent,
        available_tools: Optional[List[Callable]] = None,
        config: Optional[EnhancedOrchestratorConfig] = None,
        # Legacy parameters for backward compatibility
        shared_context_path: str = ".agent_workspace",
        max_parallel_tasks: int = 5,
        global_timeout: float = 1200.0,
        **kwargs  # Capture any other legacy parameters
    ):
        """
        Initialize enhanced orchestrator.

        Args:
            main_agent: The main orchestrator agent
            available_tools: Available tools for the orchestrator
            config: Configuration object (recommended)
            shared_context_path: Path for shared context storage (legacy)
            max_parallel_tasks: Maximum parallel tasks (legacy)
            global_timeout: Global timeout (legacy)
            **kwargs: Other legacy parameters (ignored for simplicity)
        """
        # Use provided config or create default
        if config is None:
            config = EnhancedOrchestratorConfig()
            # Only override essential legacy parameters
            config.shared_context.base_path = shared_context_path
            config.max_parallel_tasks = max_parallel_tasks
            config.global_timeout = global_timeout

        self.config = config
        
        super().__init__(
            main_agent=main_agent,
            max_parallel_tasks=config.max_parallel_tasks,
            available_tools=available_tools,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            retry_backoff=config.retry_backoff,
            global_timeout=config.global_timeout,
            **kwargs  # Pass through any other parameters
        )

        # Initialize shared context
        self.shared_context = SharedContextStore(
            base_path=config.shared_context.base_path,
            max_file_size_mb=config.shared_context.max_file_size_mb
        )

        # Initialize exploration executor
        self.exploration_executor = (
            ExplorationExecutor(
                shared_context=self.shared_context,
                max_parallel_explorations=config.exploration.max_parallel_explorations,
                exploration_timeout=config.exploration.exploration_timeout,
                enable_learning=config.exploration.enable_learning,
            )
            if config.enable_exploration
            else None
        )

        # Configuration
        self.enable_thinking_agents = config.enable_thinking_agents
        self.enable_exploration = config.enable_exploration
        self.exploration_strategy = config.exploration.default_strategy
        self.enable_cross_agent_memory = config.enable_cross_agent_memory

        # Enhance main agent if it's not already enhanced
        self._enhance_main_agent()

        logger.info(
            f"EnhancedAgentOrchestrator initialized with shared_context at {config.shared_context.base_path}"
        )

    def _enhance_main_agent(self):
        """Enhance the main agent with shared context and memory."""
        # If main agent is not a ThinkingAgent, we can still enhance its memory
        if self.enable_cross_agent_memory and self.main_agent.memory_manager:
            # Replace with enhanced memory manager
            old_manager = self.main_agent.memory_manager
            self.main_agent.memory_manager = EnhancedMemoryManager(
                token_threshold=old_manager.token_threshold,
                preserve_last_n_messages=old_manager.preserve_last_n_messages,
                summary_max_tokens=old_manager.summary_max_tokens,
                enabled=old_manager.enabled,
                shared_context_store=self.shared_context,
                agent_id=self.main_agent.agent_id,
                cross_agent_sharing=True,
            )

        # If it's already a ThinkingAgent, ensure it has shared context
        if isinstance(self.main_agent, ThinkingAgent):
            self.main_agent.shared_context = self.shared_context

    def _create_enhanced_subagent(
        self,
        agent_id: str,
        system_prompt: str,
        tools: List[Callable],
        parent_context: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        """Create an enhanced subagent with thinking capabilities."""
        if self.enable_thinking_agents:
            # Create ThinkingAgent
            agent = ThinkingAgent(
                agent_id=agent_id,
                provider=self.subagent_provider,
                model=self.subagent_model,
                system_prompt=system_prompt,
                tools=tools,
                reasoning_effort=self.reasoning_effort,
                shared_context_store=self.shared_context,
                enable_thinking_mode=True,
                memory_config={
                    "enabled": self.enable_cross_agent_memory,
                    "token_threshold": 50000,
                    "preserve_last_n_messages": 10,
                    "summary_max_tokens": 4000,
                },
            )

            # If parent context is available, initialize with it
            if parent_context and hasattr(agent, "memory_manager"):
                asyncio.create_task(
                    self._initialize_agent_with_context(agent, parent_context)
                )
        else:
            # Create regular agent but with enhanced memory
            memory_config = None
            if self.enable_cross_agent_memory:
                memory_config = {
                    "enabled": True,
                    "shared_context_store": self.shared_context,
                    "agent_id": agent_id,
                }

            agent = Agent(
                agent_id=agent_id,
                provider=self.subagent_provider,
                model=self.subagent_model,
                system_prompt=system_prompt,
                tools=tools,
                memory_config=memory_config,
                reasoning_effort=self.reasoning_effort,
            )

        return agent

    async def _initialize_agent_with_context(
        self, agent: ThinkingAgent, parent_context: Dict[str, Any]
    ):
        """Initialize a new agent with parent context."""
        if isinstance(agent.memory_manager, EnhancedMemoryManager):
            # Add parent context as initial memory
            context_message = {
                "role": "system",
                "content": f"Context from parent agent:\n{parent_context}",
            }
            await agent.memory_manager.add_messages_with_sharing(
                [context_message],
                tokens=len(str(parent_context)) // 4,
                tags=["parent_context", "initialization"],
                share=True,
            )

    async def _execute_single_task_enhanced(
        self, task: SubAgentTask, agent: Agent
    ) -> SubAgentResult:
        """Enhanced task execution with exploration support."""
        # If exploration is enabled and agent supports it
        if (
            self.enable_exploration
            and self.exploration_executor
            and isinstance(agent, ThinkingAgent)
        ):
            # Use exploration executor
            return await self.exploration_executor.execute_with_exploration(
                agent=agent, task=task, strategy=self.exploration_strategy
            )
        else:
            # Fall back to standard execution
            return await self._execute_single_task(task, agent)

    async def _execute_subagent_tasks(
        self, tasks: List[SubAgentTask]
    ) -> List[SubAgentResult]:
        """Override to use enhanced execution."""
        # Log orchestration start
        orchestration_id = f"orch_{datetime.now().isoformat()}"
        await self.shared_context.write_artifact(
            agent_id=self.main_agent.agent_id,
            key=f"orchestration/{orchestration_id}/start",
            content={
                "tasks": [
                    {
                        "agent_id": t.agent_id,
                        "task_id": getattr(t, "task_id", t.agent_id),
                        "task_description": t.task_description,
                        "execution_mode": t.execution_mode.value,
                        "max_retries": t.max_retries,
                        "retry_delay": t.retry_delay,
                        "retry_backoff": t.retry_backoff,
                        "dependencies": t.dependencies,
                    }
                    for t in tasks
                ],
                "timestamp": datetime.now().isoformat(),
            },
            artifact_type=ArtifactType.PLAN,
        )

        # Execute tasks using parent implementation
        results = await super()._execute_subagent_tasks(tasks)

        # Log orchestration completion
        await self.shared_context.write_artifact(
            agent_id=self.main_agent.agent_id,
            key=f"orchestration/{orchestration_id}/complete",
            content={
                "results": [
                    {
                        "agent_id": r.agent_id,
                        "task_id": r.task_id,
                        "success": r.success,
                        "result": (
                            str(r.result)[:500] if r.result else None
                        ),  # Truncate long results
                        "error": r.error,
                        "error_type": r.error_type,
                        "retry_count": r.retry_count,
                        "execution_time": r.execution_time,
                        "metadata": r.metadata,
                    }
                    for r in results
                ],
                "timestamp": datetime.now().isoformat(),
                "success_count": sum(1 for r in results if r.success),
            },
            artifact_type=ArtifactType.RESULT,
        )

        return results

    async def _plan_and_create_subagents_enhanced(
        self, user_request: str, analysis: str
    ) -> Dict[str, Any]:
        """Enhanced planning that creates thinking agents."""
        # Get parent context if main agent has enhanced memory
        parent_context = None
        if self.enable_cross_agent_memory and isinstance(
            self.main_agent.memory_manager, EnhancedMemoryManager
        ):
            parent_context = (
                await self.main_agent.memory_manager.prepare_context_for_new_agent(
                    focus=user_request, include_cross_agent=True
                )
            )

        # Use parent's planning method
        result = await super().plan_and_create_subagents(
            {"user_request": user_request, "analysis": analysis}
        )

        # If we created agents and have parent context, enhance them
        if parent_context and "created_agents" in result:
            for agent_id in result["created_agents"]:
                if agent_id in self.subagents:
                    agent = self.subagents[agent_id]
                    if isinstance(agent, ThinkingAgent):
                        await self._initialize_agent_with_context(agent, parent_context)

        return result

    async def get_orchestration_insights(self) -> Dict[str, Any]:
        """Get insights about orchestration patterns and performance."""
        insights = {
            "timestamp": datetime.now().isoformat(),
            "shared_context_stats": {},
            "exploration_patterns": {},
            "cross_agent_collaborations": [],
        }

        # Get shared context statistics
        if self.shared_context:
            recent_artifacts = await self.shared_context.get_recent_artifacts(limit=20)
            insights["shared_context_stats"] = {
                "total_artifacts": len(recent_artifacts),
                "artifact_types": {},
            }

            for artifact in recent_artifacts:
                type_name = artifact.artifact_type.value
                if type_name not in insights["shared_context_stats"]["artifact_types"]:
                    insights["shared_context_stats"]["artifact_types"][type_name] = 0
                insights["shared_context_stats"]["artifact_types"][type_name] += 1

        # Get exploration patterns if available
        if self.exploration_executor:
            insights["exploration_patterns"] = {
                "successful_patterns": len(
                    self.exploration_executor.successful_patterns
                )
            }

        # Get cross-agent collaboration info
        for agent_id, agent in self.subagents.items():
            if isinstance(agent, ThinkingAgent):
                # Check for collaboration artifacts
                collab_artifacts = await self.shared_context.list_artifacts(
                    pattern=f"collaboration/*{agent_id}*"
                )
                if collab_artifacts:
                    insights["cross_agent_collaborations"].append(
                        {"agent_id": agent_id, "collaborations": len(collab_artifacts)}
                    )

        return insights

    async def cleanup_workspace(self, older_than_hours: int = 24):
        """Clean up old artifacts from shared workspace."""
        if not self.shared_context:
            return

        cutoff_time = datetime.now().timestamp() - (older_than_hours * 3600)
        artifacts = await self.shared_context.list_artifacts()

        cleaned = 0
        for artifact in artifacts:
            if artifact.created_at.timestamp() < cutoff_time:
                await self.shared_context.delete_artifact(
                    artifact.key, self.main_agent.agent_id
                )
                cleaned += 1

        logger.info(f"Cleaned up {cleaned} old artifacts")

        return cleaned
