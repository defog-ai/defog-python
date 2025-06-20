"""Add citations to orchestrator results using tool outputs as sources."""

import json
from typing import Optional
from .citations import citations_tool
from .providers.base import LLMResponse


async def add_citations_to_response(
    response: LLMResponse,
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
    instructions: Optional[str] = None,
) -> str:
    """
    Add citations to an orchestrator response using its tool outputs as source documents.

    Args:
        response: The LLMResponse from orchestrator.process()
        provider: LLM provider for generating citations
        model: Model to use for citations
        instructions: Custom citation instructions

    Returns:
        The response content with citations added
    """
    # Extract tool outputs if available
    if not hasattr(response, "tool_outputs") or not response.tool_outputs:
        return response.content

    # Convert tool outputs to documents
    documents = []
    for i, tool_output in enumerate(response.tool_outputs):
        doc_name = f"{tool_output.get('name', 'tool')}_{i}"
        content = f"Tool: {tool_output.get('name', 'unknown')}\n"
        content += (
            f"Arguments: {json.dumps(tool_output.get('arguments', {}), indent=2)}\n"
        )
        content += f"Result: {tool_output.get('result', '')}"

        documents.append({"document_name": doc_name, "document_content": content})

    # Default instructions
    if not instructions:
        instructions = (
            "Rewrite the response with inline citations from the tool outputs. "
            "Use [Source: tool_name] format. Keep all information but add proper attribution."
        )

    # Generate cited version
    question = f"Add citations to this response:\n\n{response.content}"

    citation_blocks = await citations_tool(
        question=question,
        instructions=instructions,
        documents=documents,
        model=model,
        provider=provider,
        verbose=False,
    )

    # Extract text from citation blocks
    if isinstance(citation_blocks, list):
        cited_text = ""
        for block in citation_blocks:
            if block.get("type") == "text":
                cited_text += block.get("text", "")
        return cited_text if cited_text else response.content

    return response.content
