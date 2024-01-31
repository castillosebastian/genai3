import os
import semantic_kernel as sk
import asyncio
import inspect
import sys
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, VectorFilterMode
from langchain_community.embeddings import AzureOpenAIEmbeddings
from openai import AzureOpenAI
from semantic_kernel.plugin_definition import kernel_function, kernel_function_context_parameter
from semantic_kernel import KernelContext, Kernel, ContextVariables
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
from dotenv import load_dotenv
load_dotenv()
root_dir = os.getenv("ROOT")
sys.path.insert(0, root_dir)
from plugins.AISearch.aisearch import AISearchWF, build_query_filter
from src.utils import string_to_json

AZURE_AISEARCH_ENDPOINT = os.getenv("AZURE_AISEARCH_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDINGS_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME")
AZURE_AISEARCH_INDEX_NAME = os.getenv("AZURE_AISEARCH_INDEX_NAME")
credential = AzureKeyCredential(os.getenv("AZURE_AISEARCH_API_KEY"))

# Creating the kernel
deployment_name, key, endpoint = sk.azure_openai_settings_from_dot_env()
embeddings = os.environ["AZURE_OPENAI_EMBEDDINGS_MODEL_NAME"]

azure_chat_service = AzureChatCompletion(deployment_name=deployment_name, endpoint=endpoint, api_key=key)
azure_text_embedding = AzureTextEmbedding(deployment_name=embeddings, endpoint=endpoint, api_key=key)

          
async def query(ask=None):
    """
    This function takes an ask from the user related to one or many companies 
    and performs for each company a hybrid-search with filter in Azure AI search Index.
    """
    try:
        kernel = sk.Kernel()
        kernel.add_chat_service("chat_completion", azure_chat_service)
        kernel.add_text_embedding_generation_service("ada", azure_text_embedding)

        pluginASKT = kernel.import_semantic_plugin_from_directory("plugins", "ASKProcess")        
        extract_entities = pluginASKT["extractEntities"]
        pluginAIS = kernel.import_plugin(plugin_instance= AISearchWF(), plugin_name= "AISearchWF")
        searchwf =  pluginAIS["searchwf"]                                 

        my_context = kernel.create_new_context()
        my_context['ask'] = ask

        try:
            response = await kernel.run(extract_entities, input_context=my_context)         
        except Exception as e:
            # Handle exceptions from extract_entities
            print(f"Error in extract_entities: {e}")
            return []

        ask_entities = string_to_json(response['input'])            
        metadata_filter = build_query_filter(ask_entities)    

        documents = []

        for i in metadata_filter:
            context_variables = sk.ContextVariables(variables={"ask":ask,"filter": i})
            try:
                # Retrieve document with Hybrid Search with Filters
                doc = await kernel.run(searchwf, input_vars=context_variables) 
                documents.append(doc)    
            except Exception as e:
                # Handle exceptions from searchwf
                print(f"Error in searchwf with filter {i}: {e}")
                # Optionally continue to the next filter or return/exit

        return documents

    except Exception as e:
        # Main function level error handling
        print(f"Unexpected error in query function: {e}")
        return []
