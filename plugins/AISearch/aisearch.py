import semantic_kernel as sk
import asyncio
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, VectorFilterMode
from langchain_community.embeddings import AzureOpenAIEmbeddings
from openai import AzureOpenAI
from semantic_kernel.plugin_definition import kernel_function, kernel_function_context_parameter
from semantic_kernel import KernelContext
from semantic_kernel import Kernel, ContextVariables
from semantic_kernel.planning import ActionPlanner
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
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
    @kernel_function(
        description="This function search for finance information stored in knowledge data base",
        name="search",
        input_description="A user query related to factual data or insight from financial documents",
    )
    async def search(self, ask: str) -> str:
        try:
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

            return "No documents found" if results == '' else results
        
        except ValueError as e:
            print(f"Error: {e}")            
            raise e

def build_query_filter(json_object):
    try:
        # Initialize a list to hold filter strings
        filters = []

        # Convert company names to uppercase, or use [None] if company_name is None
        companies = [company.upper() for company in json_object['company_name']] if json_object.get('company_name') is not None else [None]

        # Check and prepare country and dates outside the loop
        country = json_object['country'][0] if json_object.get('country') else None
        year = json_object['dates'][0][:4] if json_object.get('dates') else None

        # Loop through each company
        for company in companies:
            # Initialize components of the filter for this company
            filter_components = []

            # Add company filter if company exists
            if company:
                filter_components.append(f"Description eq '{company}'")

            # Add country filter if country exists
            if country:
                filter_components.append(f"Country eq '{country}'")

            # Add date filter if dates exist
            if year:
                filter_components.append(f"AdditionalMetadata eq '{year}'")

            # Check if any filter component was added for this company/criteria
            if filter_components:
                # Join all components with 'and'
                filter_query = " and ".join(filter_components)
                filters.append(filter_query)

        # Check if any filter was created
        if not filters:
            return None

        return filters

    except Exception as e:
        # Handle any exception that occurs during processing
        return [f"Error building query filter: {e}"]




class AISearchWF:

    @kernel_function(
        description="This function search for finance information stored in knowledge data base",
        name="searchwf",
        #input_description="A user query related to factual data or insight from financial documents",
    )
    @kernel_function_context_parameter(name="ask",description="Ask from the user")
    @kernel_function_context_parameter(name="filter",description="The filter to apply for the search")
    async def searchwf(self, context: KernelContext) -> str:

        try:
            
            metadata_filter = str(context['filter'])
            ask = str(context['ask'])


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

            # if metadata filter is none do not use filter!!!!!!!!!!
            
            results = search_client.search(
                search_text=ask,
                vector_queries=[vector_query],
                vector_filter_mode=VectorFilterMode.PRE_FILTER,
                filter=metadata_filter,                
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
            
            return "No documents found" if results == '' else results

        
        except ValueError as e:
            print(f"Error: {e}")            
            raise e