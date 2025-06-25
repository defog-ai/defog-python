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
from .pdf_processor import (
    PDFAnalysisInput,
    ClaudePDFProcessor,
    OpenAIPDFProcessor,
)
from .pdf_data_extractor import PDFDataExtractor, extract_pdf_data
from .image_data_extractor import ImageDataExtractor, extract_image_data
from .html_data_extractor import HTMLDataExtractor, extract_html_data
from .text_data_extractor import TextDataExtractor, extract_text_data

# Orchestration components
from .orchestrator import (
    AgentOrchestrator,
    Agent,
    SubAgentTask,
    SubAgentResult,
    ExecutionMode,
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
    "PDFAnalysisInput",
    "ClaudePDFProcessor",
    "OpenAIPDFProcessor",
    "PDFDataExtractor",
    "extract_pdf_data",
    # Image processing
    "ImageDataExtractor",
    "extract_image_data",
    # HTML processing
    "HTMLDataExtractor",
    "extract_html_data",
    # Text processing
    "TextDataExtractor",
    "extract_text_data",
    # Orchestration
    "AgentOrchestrator",
    "Agent",
    "SubAgentTask",
    "SubAgentResult",
    "ExecutionMode",
]
