import json
import semantic_kernel as sk
import asyncio
from typing import List, Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, VectorFilterMode
from langchain_community.embeddings import AzureOpenAIEmbeddings
from openai import AzureOpenAI
from semantic_kernel.plugin_definition import (
    kernel_function,
    kernel_function_context_parameter,
)
from semantic_kernel import KernelContext
from semantic_kernel import Kernel, ContextVariables
from semantic_kernel.planning import ActionPlanner
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)
import os
from dotenv import load_dotenv

load_dotenv()


AZURE_AISEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDINGS_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME")
AZURE_AISEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
embeddings = os.environ["AZURE_OPENAI_EMBEDDINGS_MODEL_NAME"]


class VSearch:
    """
    Python Class that take a user question, preprocess the ask, execute a hybrid search + filter. The latter is executed if a company ticker, location
    or date is extracted from the ask. If more than one ticker is extracted, one query per ticker is executed. 
    Todo: 
        Pos-processing:  the output in caso of ask related to 3 tickers or more
    """
    def __init__(self):
        """
        Initialize the VSearch class with required clients and configurations.
        """
        self.openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",
            openai_api_version=AZURE_OPENAI_API_VERSION,
            chunk_size=1000,
        )
        self.azure_chat_service = AzureChatCompletion(
                deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
            )
        self.azure_text_embedding = AzureTextEmbedding(
                deployment_name=embeddings,
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
        )
    
    # Auxiliary Functions
        
    def build_query_filter(self, json_object) -> str:
        """
        Convert Json object with extracted entities en query filter for AI Search Index.
        """
        if not isinstance(json_object, dict):
            raise TypeError("json_object must be a dictionary")
        
        try:
            # Initialize a list to hold filter strings
            filters = []
            # Convert ticker names to uppercase, or use [None] if ticker_name is None
            tickers = (
                [ticker.upper() for ticker in json_object["ticker"]]
                if json_object.get("ticker") is not None
                else [None]
            )
            # Check and prepare referenced_location and referenced_year outside the loop
            referenced_location = json_object["location"][0] if json_object.get("location") else None
            referenced_year = json_object["dates"][0][:4] if json_object.get("dates") else None
            # Loop through each ticker
            for ticker in tickers:
                # Initialize components of the filter for this ticker
                filter_components = []
                # Add ticker filter if ticker exists
                if ticker:
                    filter_components.append(f"referenced_entity eq '{ticker}'")
                # Add referenced_location filter if referenced_location exists
                if referenced_location:
                    filter_components.append(f"referenced_location eq '{referenced_location}'")
                # Add date filter if referenced_year exist
                if referenced_year:
                    filter_components.append(f"referenced_year eq '{referenced_year}'")
                # Check if any filter component was added for this ticker/criteria
                if filter_components:
                    # Join all components with 'and'
                    filter_query = " and ".join(filter_components)
                    filters.append(filter_query)
            # Check if any filter was created            
            return filters if filters else None
        except Exception as e:
            # Handle any exception that occurs during processing
            return [f"Error building query filter: {e}"]

    def format_search_results(self, documents, metadata_filters=None):
        """
        The AI search return an iterator over the Index, so this function extract the document text and metadata.
        """        
        if not any(doc.get("retrieved_info") for doc in documents):
            return f"No documents found for this question's related search: {metadata_filters}"
        
        # Format and return search results
        try:
            processed_texts = []
            for document in documents:
                joined_text = "\n\n".join(
                    self.result_to_string(result)
                    for result in document["retrieved_info"]
                )
                processed_texts.append(joined_text)
            final_document = "\n\n```\n" + "\n\n```\n\n```\n".join(processed_texts) + "\n```"
            return final_document
        except Exception as e:
            # Handle any exceptions that occur during formatting
            error_message = f"Error occurred while formatting search results: {e}"
            print(error_message)
            return error_message

    
    def string_to_json(self, string):
        try:
            # Convert the string to a JSON object
            json_object = json.loads(string)
            return json_object
        except json.JSONDecodeError as e:
            # Handle the exception if the string is not a valid JSON
            return f"Error converting string to JSON: {e}"

    def result_to_string(self, result):
        return "\n".join(f"{key}: {value}" for key, value in result.items())

    # Auxiliary function: Async

    async def generate_embeddings(self, text: str) -> Optional[List[float]]:
        """
        Generates embeddings for the given text using Azure OpenAI.                
        """
        try:
            response = self.openai_client.embeddings.create(
                input=[text], model=AZURE_OPENAI_EMBEDDINGS_MODEL_NAME
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return None

    async def extract_entities(self, context):
        """
          Extract entities using kernel and plugins.
          Todo: 
            -add ticker reference dinamically to prompt to increasy precision of the ticker extraction. Ticker
             reference come from the unique values of the field 'referenced_entity'.
        """
        try:
            kernel = sk.Kernel()
            kernel.add_chat_service("chat_completion", self.azure_chat_service)
            kernel.add_text_embedding_generation_service("ada", self.azure_text_embedding)
            pluginASKT = kernel.import_semantic_plugin_from_directory("plugins", "ASKProcess")
            extract_entities = pluginASKT["extractEntities"]

            my_context = kernel.create_new_context()
            my_context["ask"] = context["input"]["ask"]

            response = await kernel.run(extract_entities, input_context=my_context)
            return self.string_to_json(response["input"])
        except Exception as e:
            error_message = f"Error occurred while extracting entities: {e}"
            print(error_message)
            return error_message

    # Main native function

    @kernel_function(
        description="This function search finance information from public filings of any ticker.",
        name="retrieve_documents",
        input_description="A user question related to financial information related to a company or companies",
    )
    async def retrieve_documents(self, context: KernelContext) -> str:
        try:           

            entities = await self.extract_entities(context)

            metadata_filters = self.build_query_filter(entities)            

            search_client = SearchClient(
                AZURE_AISEARCH_ENDPOINT,
                AZURE_AISEARCH_INDEX_NAME,
                credential=credential,
            )

            vquery = await self.generate_embeddings(context["input"]["ask"])

            vector_query = VectorizedQuery(
                vector=vquery, k_nearest_neighbors=5, fields="embedding"
            )

            # this list only aply if metada filter
            documents = []

            if metadata_filters:
                for filter in metadata_filters:
                    results = search_client.search(
                        search_text=context["input"]["ask"],
                        vector_queries=[vector_query],
                        vector_filter_mode=VectorFilterMode.PRE_FILTER,
                        filter=filter,
                        select=[
                            "document",
                            "id",
                            "referenced_entity",
                            "referenced_year",
                            "filename",
                        ],
                        top=4,
                    )
                    retrieved_info = [
                        dict(result) for result in results
                    ]  # Convert results to list of dicts
                    documents.append(
                        {"filter": filter, "retrieved_info": retrieved_info}
                    )
            else:
                results = search_client.search(
                    search_text=context["input"]["ask"],
                    vector_queries=[vector_query],
                    select=[
                        "document",
                        "id",
                        "referenced_entity",
                        "referenced_year",
                        "filename",
                    ],
                    top=4,
                )
                retrieved_info = [dict(result) for result in results]                
                documents.append({"filter": None, "retrieved_info": retrieved_info})

            # Process each 'retrieved_info' in the documents
            final_document = self.format_search_results(documents, filter)

            return final_document

        except ValueError as e:
            print(f"Error: {e}")
            raise e
