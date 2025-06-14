"""Task execution with alternative path exploration capabilities."""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .orchestrator import SubAgentTask, SubAgentResult, ExecutionMode
from .thinking_agent import ThinkingAgent
from .shared_context import SharedContextStore, ArtifactType

logger = logging.getLogger(__name__)


class ExplorationStrategy(Enum):
    """Strategy for exploring alternative paths."""

    SEQUENTIAL = "sequential"  # Try alternatives one by one
    PARALLEL = "parallel"  # Try alternatives in parallel
    ADAPTIVE = "adaptive"  # Decide based on task complexity


@dataclass
class ExplorationPath:
    """Represents an alternative execution path."""

    path_id: str
    description: str
    approach: str
    confidence: float = 0.5
    estimated_complexity: str = "medium"  # low, medium, high
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class ExplorationResult:
    """Result from exploring an alternative path."""

    path_id: str
    success: bool
    result: Any
    execution_time: float
    error: Optional[str] = None
    insights: Optional[Dict[str, Any]] = None
    artifacts_created: List[str] = field(default_factory=list)


class ExplorationExecutor:
    """
    Executor that can explore alternative paths for task completion.

    Features:
    - Generate multiple solution paths for a task
    - Execute paths based on strategy (sequential/parallel/adaptive)
    - Learn from successful and failed attempts
    - Store exploration results for future reference
    """

    def __init__(
        self,
        shared_context: Optional[SharedContextStore] = None,
        max_parallel_explorations: int = 3,
        exploration_timeout: float = 300.0,  # 5 minutes per exploration
        enable_learning: bool = True,
    ):
        """
        Initialize the exploration executor.

        Args:
            shared_context: Shared context store for saving results
            max_parallel_explorations: Maximum parallel exploration paths
            exploration_timeout: Timeout for each exploration path
            enable_learning: Whether to learn from exploration results
        """
        self.shared_context = shared_context
        self.max_parallel_explorations = max_parallel_explorations
        self.exploration_timeout = exploration_timeout
        self.enable_learning = enable_learning

        # Track successful patterns
        self.successful_patterns: Dict[str, List[str]] = {}

        logger.info("ExplorationExecutor initialized")

    async def generate_exploration_paths(
        self, agent: ThinkingAgent, task: SubAgentTask, num_paths: int = 3
    ) -> List[ExplorationPath]:
        """
        Generate alternative execution paths for a task.

        Args:
            agent: The thinking agent to use for generation
            task: The task to generate paths for
            num_paths: Number of paths to generate

        Returns:
            List of exploration paths
        """
        # Use agent's exploration capabilities
        alternatives = await agent.explore_alternatives(
            task=task.task_description, num_alternatives=num_paths
        )

        # Convert to ExplorationPath objects
        paths = []
        for i, alt in enumerate(alternatives):
            path = ExplorationPath(
                path_id=f"{task.agent_id}_path_{i}",
                description=alt.get("description", f"Alternative approach {i+1}"),
                approach=alt.get("approach", ""),
                confidence=0.5,  # Default confidence
                estimated_complexity="medium",
            )
            paths.append(path)

        # Save paths to shared context
        if self.shared_context:
            paths_key = (
                f"exploration_paths/{task.agent_id}/{datetime.now().isoformat()}"
            )
            await self.shared_context.write_artifact(
                agent_id=task.agent_id,
                key=paths_key,
                content=[p.__dict__ for p in paths],
                artifact_type=ArtifactType.PLAN,
                metadata={"task_id": task.task_id, "num_paths": len(paths)},
            )

        return paths

    async def execute_with_exploration(
        self,
        agent: ThinkingAgent,
        task: SubAgentTask,
        strategy: ExplorationStrategy = ExplorationStrategy.ADAPTIVE,
    ) -> SubAgentResult:
        """
        Execute a task with alternative path exploration.

        Args:
            agent: The agent to execute with
            task: The task to execute
            strategy: Exploration strategy to use

        Returns:
            SubAgentResult with the best outcome
        """
        start_time = time.time()

        # First, try the primary approach
        primary_result = await self._execute_primary_approach(agent, task)

        # If successful and confidence is high, return
        if primary_result.success:
            return primary_result

        # Generate alternative paths
        paths = await self.generate_exploration_paths(agent, task, num_paths=3)

        if not paths:
            return primary_result  # No alternatives, return primary result

        # Decide on exploration strategy
        if strategy == ExplorationStrategy.ADAPTIVE:
            strategy = self._decide_strategy(task, paths)

        # Execute exploration based on strategy
        if strategy == ExplorationStrategy.SEQUENTIAL:
            return await self._explore_sequential(agent, task, paths, primary_result)
        else:
            return await self._explore_parallel(agent, task, paths, primary_result)

    async def _execute_primary_approach(
        self, agent: ThinkingAgent, task: SubAgentTask
    ) -> SubAgentResult:
        """Execute the primary approach for a task."""
        try:
            # Let agent think about the task first
            if isinstance(agent, ThinkingAgent):
                messages = [{"role": "user", "content": task.task_description}]
                response = await agent.process_with_thinking(
                    messages=messages, context=task.context, think_first=True
                )
            else:
                messages = [{"role": "user", "content": task.task_description}]
                response = await agent.process(messages, context=task.context)

            return SubAgentResult(
                agent_id=task.agent_id,
                task_id=task.task_id,
                success=True,
                result=response.content,
                execution_time=time.time() - time.time(),
                metadata={
                    "approach": "primary",
                    "tokens": response.input_tokens + response.output_tokens,
                },
            )

        except Exception as e:
            logger.warning(f"Primary approach failed: {e}")
            return SubAgentResult(
                agent_id=task.agent_id,
                task_id=task.task_id,
                success=False,
                result=None,
                error=str(e),
                error_type="PRIMARY_FAILURE",
            )

    async def _explore_sequential(
        self,
        agent: ThinkingAgent,
        task: SubAgentTask,
        paths: List[ExplorationPath],
        primary_result: SubAgentResult,
    ) -> SubAgentResult:
        """Explore paths sequentially until one succeeds."""
        best_result = primary_result
        exploration_results = []

        for path in paths:
            logger.info(f"Exploring path {path.path_id}: {path.description}")

            try:
                # Execute the alternative approach
                result = await self._execute_exploration_path(agent, task, path)
                exploration_results.append(result)

                if result.success:
                    # Found successful path
                    best_result = self._create_final_result(
                        task, result, exploration_results
                    )

                    # Learn from success
                    if self.enable_learning:
                        await self._record_success(task, path, result)

                    break

            except asyncio.TimeoutError:
                logger.warning(f"Path {path.path_id} timed out")
                continue
            except Exception as e:
                logger.error(f"Path {path.path_id} failed: {e}")
                continue

        # Save exploration history
        await self._save_exploration_history(task, exploration_results)

        return best_result

    async def _explore_parallel(
        self,
        agent: ThinkingAgent,
        task: SubAgentTask,
        paths: List[ExplorationPath],
        primary_result: SubAgentResult,
    ) -> SubAgentResult:
        """Explore paths in parallel and return the best result."""
        # Limit parallel explorations
        paths_to_explore = paths[: self.max_parallel_explorations]

        # Create exploration tasks
        exploration_tasks = []
        for path in paths_to_explore:
            task_coro = self._execute_exploration_path(agent, task, path)
            exploration_tasks.append(asyncio.create_task(task_coro))

        # Wait for all to complete or timeout
        try:
            exploration_results = await asyncio.wait_for(
                asyncio.gather(*exploration_tasks, return_exceptions=True),
                timeout=self.exploration_timeout,
            )
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for task in exploration_tasks:
                if not task.done():
                    task.cancel()
            exploration_results = []

        # Process results
        valid_results = [
            r
            for r in exploration_results
            if isinstance(r, ExplorationResult) and not isinstance(r, Exception)
        ]

        # Find best result
        best_exploration = None
        for result in valid_results:
            if result.success:
                best_exploration = result
                break

        # Save exploration history
        await self._save_exploration_history(task, valid_results)

        if best_exploration:
            return self._create_final_result(task, best_exploration, valid_results)
        else:
            return primary_result

    async def _execute_exploration_path(
        self, agent: ThinkingAgent, task: SubAgentTask, path: ExplorationPath
    ) -> ExplorationResult:
        """Execute a single exploration path."""
        start_time = time.time()
        artifacts_created = []

        try:
            # Modify task context with exploration approach
            exploration_context = task.context or {}
            exploration_context["exploration_approach"] = path.approach
            exploration_context["path_description"] = path.description

            # Create modified task
            exploration_task = SubAgentTask(
                agent_id=task.agent_id,
                task_id=f"{task.task_id}_{path.path_id}",
                task_description=f"{task.task_description}\n\nApproach: {path.description}",
                context=exploration_context,
                execution_mode=task.execution_mode,
                dependencies=task.dependencies,
            )

            # Execute with thinking
            messages = [{"role": "user", "content": exploration_task.task_description}]
            response = await agent.process_with_thinking(
                messages=messages,
                context=exploration_context,
                thinking_prompt=f"How to implement: {path.description}",
            )

            # Save result artifact
            if self.shared_context:
                result_key = (
                    f"exploration_result/{path.path_id}/{datetime.now().isoformat()}"
                )
                await self.shared_context.write_artifact(
                    agent_id=agent.agent_id,
                    key=result_key,
                    content=response.content,
                    artifact_type=ArtifactType.RESULT,
                    metadata={"path_id": path.path_id},
                )
                artifacts_created.append(result_key)

            # Reflect on the result
            insights = await agent.reflect_on_results(
                task=path.description, results=response.content, save_reflection=True
            )

            return ExplorationResult(
                path_id=path.path_id,
                success=True,
                result=response.content,
                execution_time=time.time() - start_time,
                insights=insights,
                artifacts_created=artifacts_created,
            )

        except Exception as e:
            logger.error(f"Exploration path {path.path_id} failed: {e}")
            return ExplorationResult(
                path_id=path.path_id,
                success=False,
                result=None,
                execution_time=time.time() - start_time,
                error=str(e),
                artifacts_created=artifacts_created,
            )

    def _decide_strategy(
        self, task: SubAgentTask, paths: List[ExplorationPath]
    ) -> ExplorationStrategy:
        """Decide exploration strategy based on task and paths."""
        # Simple heuristic: use parallel for complex tasks with independent paths
        has_complex_paths = any(p.estimated_complexity == "high" for p in paths)
        has_dependencies = any(p.prerequisites for p in paths)

        if has_dependencies:
            return ExplorationStrategy.SEQUENTIAL
        elif has_complex_paths and len(paths) <= self.max_parallel_explorations:
            return ExplorationStrategy.PARALLEL
        else:
            return ExplorationStrategy.SEQUENTIAL

    def _create_final_result(
        self,
        task: SubAgentTask,
        best_exploration: ExplorationResult,
        all_explorations: List[ExplorationResult],
    ) -> SubAgentResult:
        """Create final result from exploration outcomes."""
        return SubAgentResult(
            agent_id=task.agent_id,
            task_id=task.task_id,
            success=True,
            result=best_exploration.result,
            metadata={
                "exploration_path": best_exploration.path_id,
                "paths_explored": len(all_explorations),
                "successful_path_insights": best_exploration.insights,
                "artifacts": best_exploration.artifacts_created,
            },
        )

    async def _save_exploration_history(
        self, task: SubAgentTask, results: List[ExplorationResult]
    ) -> None:
        """Save exploration history for learning."""
        if not self.shared_context:
            return

        history_key = f"exploration_history/{task.task_id}/{datetime.now().isoformat()}"
        history_data = {
            "task_id": task.task_id,
            "task_description": task.task_description,
            "explorations": [
                {
                    "path_id": r.path_id,
                    "success": r.success,
                    "execution_time": r.execution_time,
                    "error": r.error,
                    "has_insights": r.insights is not None,
                }
                for r in results
            ],
            "timestamp": datetime.now().isoformat(),
        }

        await self.shared_context.write_artifact(
            agent_id=task.agent_id,
            key=history_key,
            content=history_data,
            artifact_type=ArtifactType.RESULT,
            metadata={
                "num_explorations": len(results),
                "had_success": any(r.success for r in results),
            },
        )

    async def _record_success(
        self, task: SubAgentTask, path: ExplorationPath, result: ExplorationResult
    ) -> None:
        """Record successful pattern for future use."""
        # Extract task pattern (simplified)
        task_pattern = task.task_description[:50]  # First 50 chars as pattern

        if task_pattern not in self.successful_patterns:
            self.successful_patterns[task_pattern] = []

        self.successful_patterns[task_pattern].append(path.approach)

        # Save to shared context
        if self.shared_context:
            pattern_key = (
                f"successful_pattern/{task_pattern}/{datetime.now().isoformat()}"
            )
            await self.shared_context.write_artifact(
                agent_id=task.agent_id,
                key=pattern_key,
                content={
                    "pattern": task_pattern,
                    "successful_approach": path.approach,
                    "path_description": path.description,
                    "insights": result.insights,
                },
                artifact_type=ArtifactType.RESULT,
                metadata={"task_id": task.task_id},
            )
