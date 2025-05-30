from defog.llm.llm_providers import LLMProvider
import os
import asyncio


async def upload_document_to_openai_vector_store(document, store_id):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    file_name = document["document_name"]
    if not file_name.endswith(".txt"):
        file_name = file_name + ".txt"
    file_content = document["document_content"]
    if isinstance(file_content, str):
        # convert to bytes
        file_content = file_content.encode("utf-8")

    # first, upload the file to the vector store
    file = await client.files.create(
        file=(file_name, file_content), purpose="assistants"
    )

    # then add it to the vector store
    await client.vector_stores.files.create(
        vector_store_id=store_id,
        file_id=file.id,
    )


async def citations_tool(
    question: str,
    instructions: str,
    documents: list[dict],
    model: str,
    provider: LLMProvider,
    max_tokens: int = 16000,
):
    """
    Use this tool to get an answer to a well-cited answer to a question,
    given a list of documents.
    """
    if provider in [LLMProvider.OPENAI, LLMProvider.OPENAI.value]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # create an ephemeral vector store
        store = await client.vector_stores.create()
        store_id = store.id

        # Upload all documents in parallel
        await asyncio.gather(
            *[
                upload_document_to_openai_vector_store(document, store_id)
                for document in documents
            ]
        )

        # keep polling until the vector store is ready
        is_ready = False
        while not is_ready:
            store = await client.vector_stores.files.list(vector_store_id=store_id)
            total_completed = sum(
                1 for file in store.data if file.status == "completed"
            )
            is_ready = total_completed == len(documents)
            if not is_ready:
                print(
                    f"Waiting for vector store to be ready before proceeding... {total_completed}/{len(documents)} files completed"
                )
                await asyncio.sleep(1)

        # get the answer
        response = await client.responses.create(
            model=model,
            input=question,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [store_id],
                }
            ],
            tool_choice="required",
            instructions=instructions,
            max_output_tokens=max_tokens,
        )

        # convert the response to a list of blocks
        # similar to a subset of the Anthropic citations API
        blocks = []
        for part in response.output:
            if part.type == "message":
                contents = part.content
                for item in contents:
                    if item.type == "output_text":
                        blocks.append(
                            {
                                "text": item.text,
                                "type": "text",
                                "citations": [
                                    {"document_title": i.filename}
                                    for i in item.annotations
                                ],
                            }
                        )
        return blocks

    elif provider in [LLMProvider.ANTHROPIC, LLMProvider.ANTHROPIC.value]:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        document_contents = []
        for document in documents:
            document_contents.append(
                {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": document["document_content"],
                    },
                    "title": document["document_name"],
                    "citations": {"enabled": True},
                }
            )

        # Create content messages with citations enabled for individual tool calls
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    # Add all individual document contents
                    *document_contents,
                ],
            }
        ]

        response = await client.messages.create(
            model=model,
            messages=messages,
            system=instructions,
            max_tokens=max_tokens,
        )

        response_with_citations = [item.to_dict() for item in response.content]
        return response_with_citations
    else:
        raise ValueError(f"Provider {provider} not supported for citations tool")
