import os
import sys
import pandas as pd
import openai
from dotenv import load_dotenv
load_dotenv()
from openai import AzureOpenAI
import json
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient, SearchIndexingBufferedSender  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,    
    QueryType,
    VectorizedQuery,
    VectorQuery,
    VectorFilterMode,    
)
from azure.search.documents.indexes.models import (  
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    SemanticPrioritizedFields,
    SemanticField,  
    SearchField,  
    SemanticSearch,
    VectorSearch,  
    HnswAlgorithmConfiguration,
    HnswParameters,  
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    ExhaustiveKnnParameters,
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    SemanticField,  
    SearchField,  
    VectorSearch,  
    HnswParameters,  
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)  

service_endpoint = os.getenv("AZURE_AISEARCH_ENDPOINT") 
index_name = "finance-bench-small-sk" 
key = os.getenv("AZURE_AISEARCH_API_KEY") 
model: str = os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME") 
credential = AzureKeyCredential(key)
client = AzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version = "2023-05-15",
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)
openai.api_type = os.getenv('AZURE_OPENAI_API_TYPE')
openai.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
openai.api_version = os.getenv('AZURE_OPENAI_API_VERSION')
openai.api_key = os.getenv('AZURE_OPENAI_API_KEY')
model: str = os.getenv('AZURE_OPENAI_EMBEDDINGS_MODEL_NAME') 
azure_search_endpoint: str = os.getenv('AZURE_AISEARCH_ENDPOINT') 
azure_search_key: str = os.getenv('AZURE_AISEARCH_API_KEY')
credential = AzureKeyCredential(azure_search_key)

from langchain_community.embeddings import AzureOpenAIEmbeddings
embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2023-05-15",
        chunk_size=10,
    )

def generate_embeddings(text, model=model):
        return client.embeddings.create(input = [text], model=model).data[0].embedding


vector_search = False

if vector_search: 
     
    # Simple vector Search
    query = "Revenue of Microsoft"  
    
    search_client = SearchClient(service_endpoint, index_name, credential=credential)
    vector_query = VectorizedQuery(vector=generate_embeddings(query), k_nearest_neighbors=3, fields="Embedding")
    
    results = search_client.search(  
        search_text=None,  
        vector_queries= [vector_query],
        select=["Text", "Id", ],
    )  
    
    for result in results:  
        print(f"Id: {result['Id']}")  
        print(f"Text: {result['Text']}")    
        print('-'*100)  

vector_search_wfilter = False

if  vector_search_wfilter:
    # Perform a Pure Vector Search with a filter
    query = "What is the Revenue of 2019"  

    search_client = SearchClient(service_endpoint, index_name, credential=credential)
    
    vector_query = VectorizedQuery(vector=generate_embeddings(query), k_nearest_neighbors=3, fields="Embedding")
    
    results = search_client.search(  
        search_text=None,  
        vector_queries= [vector_query],
        vector_filter_mode=VectorFilterMode.PRE_FILTER,
        filter="Description eq 'BESTBUY'",
        select=["Text", "Id", "Description", 'AdditionalMetadata', 'ExternalSourceName' ],
    )  
    
    for result in results:  
        print(f"Id: {result['Id']}")  
        print(f"Description: {result['Description']}")
        print(f"AdditionalMetadata: {result['AdditionalMetadata']}")
        print(f"ExternalSourceName: {result['ExternalSourceName']}")        
        print('-'*100)  


hybrid_search = True

if hybrid_search:     
    # Perform a Pure Vector Search with a filter
    query = "What is the Revenue of 2019"  
    # ## Perform a Hybrid Search
    # Hybrid Search
    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))  
    vector_query = VectorizedQuery(vector=generate_embeddings(query), k_nearest_neighbors=3, fields="Embedding")

    results = search_client.search(  
        search_text=query,  
        vector_queries=[vector_query],
        select=["Text", "Id", "Description" ],
        top=3
    )  
    
    for result in results:  
        print(f"Id: {result['Id']}")  
        print(f"Description: {result['Description']}")      
        print(f"Text: {result['Text']}")   
        print('-'*100) 