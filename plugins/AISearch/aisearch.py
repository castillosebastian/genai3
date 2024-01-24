import asyncio
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, VectorFilterMode
from langchain_community.embeddings import AzureOpenAIEmbeddings
from semantic_kernel.sk_pydantic import PydanticField
from semantic_kernel.orchestration.sk_context import SKContext
from openai import AzureOpenAI
from semantic_kernel.plugin_definition import (
    sk_function,
    sk_function_context_parameter,
)
from semantic_kernel import Kernel,ContextVariables
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

import os
from dotenv import load_dotenv
load_dotenv()

searchazure_ai_search_endpoint = os.getenv("AZURE_AISEARCH_ENDPOINT")
searchazure_ai_search_api_key = os.getenv("AZURE_AISEARCH_API_KEY")
searchazure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
searchazure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
searchazure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_openai_embeddings_model_name = os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME")
index_name=os.getenv("AZURE_AISEARCH_INDEX_NAME")
credential = AzureKeyCredential(searchazure_ai_search_api_key)

class AISearch:             

    @sk_function(
        description="This function search for finance information stored in knowledge data base",
        name="search",
        input_description="A user query related to factual data or insight from financial documents",
    )    
    async def search(self, ask: str) -> str:

        def format_hybrid_search_results(hybrid_search_results):
                formatted_results = [
                    f"""ID: {result['Id']}
                    Text: {result['Text']}
                    ExternalSourceName: {result['ExternalSourceName']}
                    Source: {result['Description']}
                    AdditionalMetadata: {result['AdditionalMetadata']}
                    """ for result in hybrid_search_results
                ]
                formatted_string = ""
                for i, doc in enumerate(formatted_results):
                    #formatted_string += f"\n<document {i+1}>\n\n {doc}\n"
                    formatted_string += f"\n\"\"\" {doc}\n\"\"\"\n\n"
                return formatted_string

        def generate_embeddings(text):        
                openai_client = AzureOpenAI(
                    api_key=searchazure_openai_api_key,
                    api_version="2023-05-15",
                    azure_endpoint=searchazure_openai_endpoint
                )
                embeddings = AzureOpenAIEmbeddings(
                    azure_deployment="text-embedding-ada-002",
                    openai_api_version="2023-05-15",
                    chunk_size=10
                )

                return openai_client.embeddings.create(input=[text], model=azure_openai_embeddings_model_name).data[0].embedding

        search_client = SearchClient(searchazure_ai_search_endpoint, index_name, credential=credential)
                
        vquery = generate_embeddings(ask)
               
        vector_query = VectorizedQuery(vector=vquery, k_nearest_neighbors=5, fields='Embedding')        
       
        results = search_client.search(
            search_text=ask,
            vector_queries=[vector_query],
            select=["Text", "Id","ExternalSourceName","Description","AdditionalMetadata"], 
            top=5
        )

        results =  format_hybrid_search_results(results)
        
        return str(results)
    


