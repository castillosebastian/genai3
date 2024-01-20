import os
import asyncio
from typing import Tuple
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.connectors.ai.open_ai.request_settings.azure_chat_request_settings import (
    AzureAISearchDataSources,
    AzureChatRequestSettings,
    AzureDataSources,
    ExtraBody,
)
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAITextEmbedding,
    AzureChatCompletion,
    AzureTextEmbedding,
)
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryCaptionResult,
    QueryAnswerResult,
    SemanticErrorMode,
    SemanticErrorReason,
    SemanticSearchResultsType,
    QueryType,
    VectorizedQuery,
    VectorQuery,
    VectorFilterMode,    
)

# Configure environment variables  
service_endpoint = os.getenv("AZURE_AISEARCH_URL") 
index_name = os.getenv("AZURE_AISEARCH_INDEX_NAME")
key = os.getenv("AZURE_AISEARCH_API_KEY") 
model: str = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME")
credential = AzureKeyCredential(key)

# Load Azure AI Search and OpenAI Settings
deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()

azure_ai_search_settings = sk.azure_aisearch_settings_from_dot_env_as_dict()

search_client = SearchClient(service_endpoint, index_name, credential=credential)

client = AzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version = os.getenv("OPENAI_API_VERSION"),
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"))

