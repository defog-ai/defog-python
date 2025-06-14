"""Configuration classes for enhanced orchestration features."""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

from .exploration_executor import ExplorationStrategy


@dataclass
class SharedContextConfig:
    """Configuration for SharedContextStore."""
    
    base_path: str = ".agent_workspace"
    cache_size_limit: int = 100
    max_file_size_mb: int = 10
    allowed_extensions: List[str] = field(default_factory=lambda: ['.json', '.txt', '.md'])
    enable_compression: bool = False
    cleanup_older_than_days: int = 7
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.cache_size_limit < 1:
            raise ValueError("cache_size_limit must be at least 1")
        if self.max_file_size_mb < 1:
            raise ValueError("max_file_size_mb must be at least 1")
        if self.cleanup_older_than_days < 1:
            raise ValueError("cleanup_older_than_days must be at least 1")


@dataclass
class ExplorationConfig:
    """Configuration for ExplorationExecutor."""
    
    max_parallel_explorations: int = 3
    exploration_timeout: float = 300.0  # 5 minutes
    enable_learning: bool = True
    default_strategy: ExplorationStrategy = ExplorationStrategy.ADAPTIVE
    max_paths_per_task: int = 5
    path_confidence_threshold: float = 0.3
    retry_failed_paths: bool = True
    max_retries_per_path: int = 2
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_parallel_explorations < 1:
            raise ValueError("max_parallel_explorations must be at least 1")
        if self.exploration_timeout <= 0:
            raise ValueError("exploration_timeout must be positive")
        if self.max_paths_per_task < 1:
            raise ValueError("max_paths_per_task must be at least 1")
        if not 0 <= self.path_confidence_threshold <= 1:
            raise ValueError("path_confidence_threshold must be between 0 and 1")
        if self.max_retries_per_path < 0:
            raise ValueError("max_retries_per_path must be non-negative")


@dataclass
class MemoryConfig:
    """Configuration for EnhancedMemoryManager."""
    
    max_context_length: int = 150000
    summarization_threshold: int = 100000
    max_cross_agent_contexts: int = 10
    context_cleanup_interval: float = 3600.0  # 1 hour
    enable_context_compression: bool = True
    summary_model: str = "claude-sonnet-4-20250514"
    reasoning_effort: str = "medium"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_context_length < 1000:
            raise ValueError("max_context_length must be at least 1000")
        if self.summarization_threshold < 100:
            raise ValueError("summarization_threshold must be at least 100")
        if self.max_cross_agent_contexts < 1:
            raise ValueError("max_cross_agent_contexts must be at least 1")
        if self.context_cleanup_interval <= 0:
            raise ValueError("context_cleanup_interval must be positive")


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
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.thinking_timeout <= 0:
            raise ValueError("thinking_timeout must be positive")
        if self.max_thinking_iterations < 1:
            raise ValueError("max_thinking_iterations must be at least 1")


@dataclass
class SecurityConfig:
    """Security configuration for filesystem operations."""
    
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10MB
    allowed_file_extensions: List[str] = field(default_factory=lambda: ['.json', '.txt', '.md'])
    forbidden_paths: List[str] = field(default_factory=lambda: ['/etc', '/sys', '/proc', '/dev'])
    enable_path_validation: bool = True
    max_path_length: int = 255
    sanitize_filenames: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_file_size_bytes < 1:
            raise ValueError("max_file_size_bytes must be at least 1")
        if self.max_path_length < 10:
            raise ValueError("max_path_length must be at least 10")


@dataclass
class EnhancedOrchestratorConfig:
    """Main configuration class for EnhancedAgentOrchestrator."""
    
    # Sub-configurations
    shared_context: SharedContextConfig = field(default_factory=SharedContextConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Feature flags
    enable_thinking_agents: bool = True
    enable_exploration: bool = True
    enable_cross_agent_memory: bool = True
    enable_enhanced_error_handling: bool = True
    
    # Global settings
    max_parallel_tasks: int = 5
    global_timeout: float = 1200.0  # 20 minutes
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    
    @classmethod
    def from_env(cls) -> 'EnhancedOrchestratorConfig':
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
            config.exploration.exploration_timeout = float(os.environ["ORCHESTRATOR_EXPLORATION_TIMEOUT"])
        
        if "ORCHESTRATOR_DISABLE_THINKING" in os.environ:
            config.enable_thinking_agents = False
            config.thinking.enable_thinking_mode = False
        
        if "ORCHESTRATOR_DISABLE_EXPLORATION" in os.environ:
            config.enable_exploration = False
        
        if "ORCHESTRATOR_DISABLE_CROSS_MEMORY" in os.environ:
            config.enable_cross_agent_memory = False
        
        return config
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_parallel_tasks < 1:
            raise ValueError("max_parallel_tasks must be at least 1")
        if self.global_timeout <= 0:
            raise ValueError("global_timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if self.retry_backoff < 1:
            raise ValueError("retry_backoff must be at least 1")