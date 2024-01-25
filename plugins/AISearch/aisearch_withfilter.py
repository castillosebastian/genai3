import semantic_kernel as sk
import asyncio
import typing as t
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, VectorFilterMode
from langchain_community.embeddings import AzureOpenAIEmbeddings
#from semantic_kernel.sk_pydantic import PydanticField
#from semantic_kernel.orchestration.sk_context import SKContext
from openai import AzureOpenAI
from semantic_kernel.plugin_definition import sk_function, sk_function_context_parameter
from semantic_kernel import SKContext

from semantic_kernel import Kernel, ContextVariables
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.planning import ActionPlanner
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_AISEARCH_ENDPOINT = os.getenv("AZURE_AISEARCH_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDINGS_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME")
AZURE_AISEARCH_INDEX_NAME = os.getenv("AZURE_AISEARCH_INDEX_NAME")
credential = AzureKeyCredential(os.getenv("AZURE_AISEARCH_API_KEY"))

class AISearch:

    @sk_function(
        description="This function search for finance information stored in knowledge data base",
        name="search",
        #input_description="A user query related to factual data or insight from financial documents",
    )
    @sk_function_context_parameter(name="company",description="The company data to look for")    
    async def search(self, ask: str, context: SKContext) -> str:

        company = context.variables.get("company")

        def format_hybrid_search_results(hybrid_search_results):
            formatted_results = [
                f"""ID: {result['Id']}
                    Text: {result['Text']}
                    ExternalSourceName: {result['ExternalSourceName']}
                    Source: {result['Description']}
                    AdditionalMetadata: {result['AdditionalMetadata']}
                    """
                for result in hybrid_search_results
            ]
            formatted_string = ""
            for i, doc in enumerate(formatted_results):
                # formatted_string += f"\n<document {i+1}>\n\n {doc}\n"
                formatted_string += f'\n""" {doc}\n"""\n\n'
            return formatted_string

        def generate_embeddings(text):
            openai_client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
            )
            embeddings = AzureOpenAIEmbeddings(
                azure_deployment="text-embedding-ada-002",
                openai_api_version=AZURE_OPENAI_API_VERSION,
                chunk_size=1000,
            )

            return (
                openai_client.embeddings.create(
                    input=[text], model=AZURE_OPENAI_EMBEDDINGS_MODEL_NAME
                )
                .data[0]
                .embedding
            )

        search_client = SearchClient(
            AZURE_AISEARCH_ENDPOINT, AZURE_AISEARCH_INDEX_NAME, credential=credential
        )

        vquery = generate_embeddings(ask)

        vector_query = VectorizedQuery(
            vector=vquery, k_nearest_neighbors=5, fields="Embedding"
        )

        results = search_client.search(
            search_text=ask,
            vector_queries=[vector_query],
            vector_filter_mode=VectorFilterMode.PRE_FILTER,
            filter=f"Description eq '{company}'",
            select=[
                "Text",
                "Id",
                "ExternalSourceName",
                "Description",
                "AdditionalMetadata",
            ],
            top=4,
        )

        results = format_hybrid_search_results(results)

        return str(results)


class GenerateNumberPlugin:
    """
    Description: Generate a number between a min and a max.
    """

    @sk_function(
        description="Generate a random number between min and max",
        name="GenerateNumber",
    )
    @sk_function_context_parameter(name="min", description="Minimum number of paragraphs.")
    @sk_function_context_parameter(name="max", description="Maximum number of paragraphs.", default_value="10")
    def generate_number(self, context: SKContext) -> str:
        """
        Generate a number between min-max
        Example:
            min="4" max="10" => rand(4,8)
        Args:
            min -- The lower limit for the random number generation
            max -- The upper limit for the random number generation
        Returns:
            int value
        """
        try:
            return str(random.randint(int(context["min"]), int(context["max"])))
        except ValueError as e:
            print(f"Invalid input {context['min']} {context['max']}")
            raise e