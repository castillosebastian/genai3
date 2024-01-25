import semantic_kernel as sk
import asyncio
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
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
from semantic_kernel.planning import ActionPlanner
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding

import os
from dotenv import load_dotenv
load_dotenv()

AZURE_AISEARCH_ENDPOINT = os.getenv("AZURE_AISEARCH_ENDPOINT")
AZURE_AISEARCH_API_KEY = os.getenv("AZURE_AISEARCH_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDINGS_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME")
AZURE_AISEARCH_INDEX_NAME=os.getenv("AZURE_AISEARCH_INDEX_NAME")
credential = AzureKeyCredential(AZURE_AISEARCH_API_KEY)

# Creating the kernel
# api_key, org_id = sk.azure_aisearch_settings_from_dot_env()
# deployment_name, key, endpoint = sk.azure_openai_settings_from_dot_env()
# embeddings = os.environ["AZURE_OPENAI_EMBEDDINGS_MODEL_NAME"]
# azure_chat_service = AzureChatCompletion(deployment_name=deployment_name, endpoint=endpoint, api_key=key)
# azure_text_embedding = AzureTextEmbedding(deployment_name=embeddings, endpoint=endpoint, api_key=key)
# kernel = sk.Kernel()
# kernel.add_chat_service("chat_completion", azure_chat_service)
# kernel.add_text_embedding_generation_service("ada", azure_text_embedding)
# # Plugins
# pluginFC = kernel.import_semantic_plugin_from_directory("plugin", "AISearch")        
# qrewrite = pluginFC["qrewrite"] 

# def rewrite_ask(ask, qrewrite, history=None):        

#     # Set Context
#     my_context = kernel.create_new_context()
#     my_context['query'] = ask
#     my_context['chat_history'] =  history
    
#     # As SK Context
#     response = await kernel.run_async(qrewrite, input_context=my_context) 
    
#     return response

# def summarize_docs():
#       pass

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
                    api_key=AZURE_OPENAI_API_KEY,
                    api_version="2023-05-15",
                    azure_endpoint=AZURE_OPENAI_ENDPOINT
                )
                embeddings = AzureOpenAIEmbeddings(
                    azure_deployment="text-embedding-ada-002",
                    openai_api_version="2023-05-15",
                    chunk_size=1000
                )

                return openai_client.embeddings.create(input=[text], model=AZURE_OPENAI_EMBEDDINGS_MODEL_NAME).data[0].embedding

        search_client = SearchClient(AZURE_AISEARCH_ENDPOINT, AZURE_AISEARCH_INDEX_NAME, credential=credential)
                
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
    


