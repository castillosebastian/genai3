# Copyright (c) Microsoft. All rights reserved.
import asyncio
import os
from dotenv import dotenv_values
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    AzureTextCompletion,
    AzureTextEmbedding,
)

# ERROR
# from semantic_kernel.connectors.memory.azure_cognitive_search import (
#     AzureCognitiveSearchMemoryStore,
# )

COLLECTION_NAME = 'finance-bench-small-sk'


async def search_acs_memory_questions(kernel: sk.Kernel) -> None:
    questions = [
        "what's the Revenue of Microsoft",       
    ]

    for question in questions:
        print(f"Question: {question}")
        result = await kernel.memory.search_async(COLLECTION_NAME, question)
        print(f"Answer: {result[0].text}\n")


async def main() -> None:
    kernel = sk.Kernel()

    config = dotenv_values(".env")

    AZURE_COGNITIVE_SEARCH_ENDPOINT = config["AZURE_AISEARCH_ENDPOINT"]
    AZURE_COGNITIVE_SEARCH_ADMIN_KEY = config["AZURE_AISEARCH_API_KEY"]
    AZURE_OPENAI_API_KEY = config["AZURE_OPENAI_API_KEY"]
    AZURE_OPENAI_ENDPOINT = config["AZURE_OPENAI_ENDPOINT"]
    vector_size = 1536

    # Setting up OpenAI services for text completion and text embedding
    kernel.add_text_completion_service(
        "dv",
        AzureTextCompletion(
            deployment_name="text-embedding-ada-002",
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
        ),
    )
    kernel.add_text_embedding_generation_service(
        "ada",
        AzureTextEmbedding(
            deployment_name="text-embedding-ada-002",
            endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
        ),
    )

    connector = AzureCognitiveSearchMemoryStore(
        vector_size, AZURE_COGNITIVE_SEARCH_ENDPOINT, AZURE_COGNITIVE_SEARCH_ADMIN_KEY
    )

    # Register the memory store with the kernel
    kernel.register_memory_store(memory_store=connector)
    
    print("Asking questions... (manually)")
    await search_acs_memory_questions(kernel)

    await connector.close_async()


if __name__ == "__main__":
    asyncio.run(main())
