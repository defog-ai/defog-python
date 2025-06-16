"""Configuration classes for enhanced orchestration features."""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

from .enums import ExplorationStrategy


@dataclass
class SharedContextConfig:
    """Configuration for SharedContextStore."""

    base_path: str = ".agent_workspace"
    cache_size_limit: int = 100
    max_file_size_mb: int = 10
    cleanup_older_than_days: int = 7


@dataclass
class ExplorationConfig:
    """Configuration for ExplorationExecutor."""

    max_parallel_explorations: int = 3
    exploration_timeout: float = 300.0  # 5 minutes
    enable_learning: bool = True
    default_strategy: ExplorationStrategy = ExplorationStrategy.ADAPTIVE


@dataclass
class MemoryConfig:
    """Configuration for EnhancedMemoryManager."""

    max_context_length: int = 150000
    summarization_threshold: int = 100000
    max_cross_agent_contexts: int = 10
    summary_model: str = "claude-sonnet-4-20250514"
    reasoning_effort: str = "medium"


@dataclass
class ThinkingConfig:
    """Configuration for ThinkingAgent."""

    enable_thinking_mode: bool = True
    thinking_timeout: float = 120.0  # 2 minutes
    max_thinking_iterations: int = 3
    reflection_enabled: bool = True
    save_thinking_artifacts: bool = True
    thinking_model: str = "claude-sonnet-4-20250514"
    reasoning_effort: str = "medium"


@dataclass
class EnhancedOrchestratorConfig:
    """Main configuration class for EnhancedAgentOrchestrator."""

    # Sub-configurations
    shared_context: SharedContextConfig = field(default_factory=SharedContextConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)

    # Feature flags
    enable_thinking_agents: bool = True
    enable_exploration: bool = True
    enable_cross_agent_memory: bool = True

    # Global settings
    max_parallel_tasks: int = 5
    global_timeout: float = 1200.0  # 20 minutes
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0

    @classmethod
    def from_env(cls) -> "EnhancedOrchestratorConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Override with environment variables if present
        if "ORCHESTRATOR_WORKSPACE_PATH" in os.environ:
            config.shared_context.base_path = os.environ["ORCHESTRATOR_WORKSPACE_PATH"]

        if "ORCHESTRATOR_MAX_PARALLEL" in os.environ:
            config.max_parallel_tasks = int(os.environ["ORCHESTRATOR_MAX_PARALLEL"])

        if "ORCHESTRATOR_TIMEOUT" in os.environ:
            config.global_timeout = float(os.environ["ORCHESTRATOR_TIMEOUT"])

        if "ORCHESTRATOR_EXPLORATION_TIMEOUT" in os.environ:
            config.exploration.exploration_timeout = float(
                os.environ["ORCHESTRATOR_EXPLORATION_TIMEOUT"]
            )

        if "ORCHESTRATOR_DISABLE_THINKING" in os.environ:
            config.enable_thinking_agents = False
            config.thinking.enable_thinking_mode = False

        if "ORCHESTRATOR_DISABLE_EXPLORATION" in os.environ:
            config.enable_exploration = False

        if "ORCHESTRATOR_DISABLE_CROSS_MEMORY" in os.environ:
            config.enable_cross_agent_memory = False

        return config
