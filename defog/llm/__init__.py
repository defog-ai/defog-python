"""LLM module with memory management capabilities."""

from .utils import chat_async, LLMResponse
from .utils_memory import (
    chat_async_with_memory,
    create_memory_manager,
    MemoryConfig,
)
from .memory import (
    MemoryManager,
    ConversationHistory,
    compactify_messages,
    TokenCounter,
)
from .pdf_processor import analyze_pdf, PDFAnalysisInput, ClaudePDFProcessor
from .pdf_data_extractor import PDFDataExtractor, extract_pdf_data

# Enhanced orchestration components
from .shared_context import SharedContextStore, Artifact, ArtifactType
from .enhanced_memory import EnhancedMemoryManager, SharedMemoryEntry
from .thinking_agent import ThinkingAgent
from .exploration_executor import (
    ExplorationExecutor,
    ExplorationStrategy,
    ExplorationPath,
    ExplorationResult,
)
from .enhanced_orchestrator import EnhancedAgentOrchestrator
from .config import (
    EnhancedOrchestratorConfig,
    SharedContextConfig,
    ExplorationConfig,
    MemoryConfig as EnhancedMemoryConfig,
    ThinkingConfig,
)

__all__ = [
    # Core functions
    "chat_async",
    "chat_async_with_memory",
    "LLMResponse",
    # Memory management
    "MemoryManager",
    "ConversationHistory",
    "MemoryConfig",
    "create_memory_manager",
    "compactify_messages",
    "TokenCounter",
    # PDF processing
    "analyze_pdf",
    "PDFAnalysisInput",
    "ClaudePDFProcessor",
    "PDFDataExtractor",
    "extract_pdf_data",
    # Enhanced orchestration
    "SharedContextStore",
    "Artifact",
    "ArtifactType",
    "EnhancedMemoryManager",
    "SharedMemoryEntry",
    "ThinkingAgent",
    "ExplorationExecutor",
    "ExplorationStrategy",
    "ExplorationPath",
    "ExplorationResult",
    "EnhancedAgentOrchestrator",
    # Configuration classes
    "EnhancedOrchestratorConfig",
    "SharedContextConfig",
    "ExplorationConfig",
    "EnhancedMemoryConfig",
    "ThinkingConfig",
]
