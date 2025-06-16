"""Enums for configuration classes."""

from enum import Enum


class ExplorationStrategy(Enum):
    """Strategy for exploring alternative paths."""

    SEQUENTIAL = "sequential"  # Try alternatives one by one
    PARALLEL = "parallel"  # Try alternatives in parallel
    ADAPTIVE = "adaptive"  # Decide based on task complexity


class ArtifactType(Enum):
    """Types of artifacts that can be stored."""

    TEXT = "text"
    JSON = "json"
    SUMMARY = "summary"
    PLAN = "plan"
    RESULT = "result"
    CHECKPOINT = "checkpoint"
    EXPLORATION = "exploration"


class ExecutionMode(Enum):
    """Execution mode for subagent tasks."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
