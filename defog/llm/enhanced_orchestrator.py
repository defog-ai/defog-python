"""
Backward compatibility module for EnhancedAgentOrchestrator.

This module has been merged into orchestrator.py. The functionality is now
available directly in the Orchestrator class (previously AgentOrchestrator).

For backward compatibility, we re-export the class from its new location.
"""

import warnings
from .orchestrator import Orchestrator

# Show deprecation warning
warnings.warn(
    "The enhanced_orchestrator module is deprecated. "
    "EnhancedAgentOrchestrator functionality has been merged into the main Orchestrator class. "
    "Please import from defog.llm.orchestrator instead.",
    DeprecationWarning,
    stacklevel=2,
)

# For backward compatibility
EnhancedAgentOrchestrator = Orchestrator

__all__ = ["EnhancedAgentOrchestrator"]
