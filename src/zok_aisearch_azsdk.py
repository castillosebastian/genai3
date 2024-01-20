import os
from dotenv import load_dotenv
load_dotenv()

# Azure related imports
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery,VectorFilterMode    

# LangChain Community imports
from langchain_community.embeddings import AzureOpenAIEmbeddings

# OpenAI imports
from openai import AzureOpenAI

# Environment variable definitions
AZURE_AISEARCH_ENDPOINT = os.getenv("AZURE_AISEARCH_ENDPOINT")
AZURE_AISEARCH_API_KEY = os.getenv("AZURE_AISEARCH_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDINGS_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME")
AZURE_OPENAI_API_TYPE = os.getenv("AZURE_OPENAI_API_TYPE")

# Additional constants
INDEX_NAME = "finance-bench-small-sk"

# Azure credentials
credential = AzureKeyCredential(AZURE_AISEARCH_API_KEY)

# AzureOpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",  # Assuming this is a constant and not an environment variable
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# AzureOpenAIEmbeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",  # Assuming this is a constant
    openai_api_version="2023-05-15",  # Assuming this is a constant
    chunk_size=10  # Assuming this is a constant
)

def generate_embeddings(text, model=AZURE_OPENAI_EMBEDDINGS_MODEL_NAME):
        return client.embeddings.create(input = [text], model=model).data[0].embedding


vector_search = False

if vector_search: 
     
    # Simple vector Search
    query = "Revenue of Microsoft"  
    
    search_client = SearchClient(AZURE_AISEARCH_ENDPOINT, INDEX_NAME, credential=credential)
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

    search_client = SearchClient(AZURE_AISEARCH_ENDPOINT, INDEX_NAME, credential=credential)
    
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
    # Hybrid Search
    query = "What is the revenue of Microsoft"      
    
    search_client = SearchClient(AZURE_AISEARCH_ENDPOINT, INDEX_NAME, credential=credential)  
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