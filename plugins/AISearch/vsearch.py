import json
import semantic_kernel as sk
import asyncio
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


AZURE_AISEARCH_ENDPOINT = os.getenv("AZURE_AISEARCH_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDINGS_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME")
AZURE_AISEARCH_INDEX_NAME = os.getenv("AZURE_AISEARCH_INDEX_NAME")
credential = AzureKeyCredential(os.getenv("AZURE_AISEARCH_API_KEY"))
embeddings = os.environ["AZURE_OPENAI_EMBEDDINGS_MODEL_NAME"]


class VSearch:
    """
    
    """
    
    def build_query_filter(self, json_object) -> str:
        """
        Convert Json object with extracted entities en query filter for AI Search Index.
        """
        try:
            # Initialize a list to hold filter strings
            filters = []

            # Convert company names to uppercase, or use [None] if company_name is None
            companies = (
                [company.upper() for company in json_object["company_name"]]
                if json_object.get("company_name") is not None
                else [None]
            )

            # Check and prepare country and dates outside the loop
            country = json_object["country"][0] if json_object.get("country") else None
            year = json_object["dates"][0][:4] if json_object.get("dates") else None

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

    async def format_hybrid_search_results(self, hybrid_search_results) -> str:
        """
        Turn multiple document retrieved from AIS Index into single document.
        """
        try:
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
        except Exception as e:
            # Handle any exception that occurs during processing
            return [f"Error formatting retrieved documents: {e}"]

    async def generate_embeddings(self, text):
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

    @kernel_function(
        description="This function search finance information from public filings of any company.",
        name="retrieve_documents",
        # input_description="A user query related to factual data or insight from financial documents",
    )
    async def retrieve_documents(self, context: KernelContext) -> str:
        try:
            azure_chat_service = AzureChatCompletion(
                deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
            )
            azure_text_embedding = AzureTextEmbedding(
                deployment_name=embeddings,
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
            )

            kernel = sk.Kernel()
            kernel.add_chat_service("chat_completion", azure_chat_service)
            kernel.add_text_embedding_generation_service("ada", azure_text_embedding)
            pluginASKT = kernel.import_semantic_plugin_from_directory(
                "plugins", "ASKProcess"
            )
            extract_entities = pluginASKT["extractEntities"]

            my_context = kernel.create_new_context()
            my_context["ask"] = context["input"]["ask"]

            response = await kernel.run(extract_entities, input_context=my_context)
            # response = extract_entities.invoke(context["input"]["ask"])

            ask_entities = self.string_to_json(response["input"])

            metadata_filters = self.build_query_filter(ask_entities)

            # metadata_filter = metadata_filter[0] if metadata_filter else metadata_filter

            search_client = SearchClient(
                AZURE_AISEARCH_ENDPOINT,
                AZURE_AISEARCH_INDEX_NAME,
                credential=credential,
            )

            vquery = await self.generate_embeddings(context["input"]["ask"])

            vector_query = VectorizedQuery(
                vector=vquery, k_nearest_neighbors=5, fields="Embedding"
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
                            "Text",
                            "Id",
                            "ExternalSourceName",
                            "Description",
                            "AdditionalMetadata",
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
                        "Text",
                        "Id",
                        "ExternalSourceName",
                        "Description",
                        "AdditionalMetadata",
                    ],
                    top=4,
                )

            # Process each 'retrieved_info' in the documents
            processed_texts = []
            for document in documents:
                # Join all data in 'retrieved_info' into a single text
                joined_text = "\n\n".join(
                    self.result_to_string(result)
                    for result in document["retrieved_info"]
                )
                processed_texts.append(joined_text)

            # Join all processed texts into one document with the specified format
            final_document = (
                "\n\n```\n" + "\n\n```\n\n```\n".join(processed_texts) + "\n```"
            )

            return final_document

        except ValueError as e:
            print(f"Error: {e}")
            raise e
