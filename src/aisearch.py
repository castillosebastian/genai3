import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, VectorFilterMode
from langchain_community.embeddings import AzureOpenAIEmbeddings
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

class AISearch:
    def __init__(self, index_name="finance-bench-small-sk"):
        # Environment variable definitions
        self.azure_ai_search_endpoint = os.getenv("AZURE_AISEARCH_ENDPOINT")
        self.azure_ai_search_api_key = os.getenv("AZURE_AISEARCH_API_KEY")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.azure_openai_embeddings_model_name = os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME")

        # Azure credentials and clients
        credential = AzureKeyCredential(self.azure_ai_search_api_key)
        self.search_client = SearchClient(self.azure_ai_search_endpoint, index_name, credential=credential)
        self.openai_client = AzureOpenAI(
            api_key=self.azure_openai_api_key,
            api_version="2023-05-15",
            azure_endpoint=self.azure_openai_endpoint
        )
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",
            openai_api_version="2023-05-15",
            chunk_size=10
        )

    def generate_embeddings(self, text):
        return self.openai_client.embeddings.create(input=[text], model=self.azure_openai_embeddings_model_name).data[0].embedding

    def search(self, query="Revenue of Microsoft", k_nearest_neighbors=5, top=5, embed_field='Embedding', select_fields=None, query_type="simple", filter=None):
        vector_query = VectorizedQuery(vector=self.generate_embeddings(query), k_nearest_neighbors=k_nearest_neighbors, fields=embed_field)
        
        if query_type == "simple":
            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=select_fields if select_fields else ["Text", "Id"],
                top=top
            )

        elif query_type == "hybrid":
            results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=select_fields if select_fields else ["Text", "Id"],
                top=top
            )

        elif query_type == "hybrid_wfilter":
            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                vector_filter_mode=VectorFilterMode.PRE_FILTER,
                filter=filter,
                select=select_fields if select_fields else ["Text", "Id"],
                top=top
            )
        else:
            raise ValueError("Invalid query type. Choose from 'simple', 'hybrid', 'hybrid_wfilter'.")

        return results

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Instantiate the AISearch class
    ai_search = AISearch()

    # Example 1: Perform a simple search
    # simple_search_results = ai_search.search(query="Revenue of Microsoft", query_type="simple")
    # print("Simple Search Results:")
    # for result in simple_search_results:
    #     print(f"ID: {result['Id']}, Text: {result['Text']}")

    # Example 2: Perform a hybrid search
    hybrid_search_results = ai_search.search(query="Revenue of Microsoft", query_type="hybrid", top=3,
                                             select_fields=["Text", "Id","ExternalSourceName",
                                                                      "Description","AdditionalMetadata"])
    print("\nHybrid Search Results:")    
    # for result in hybrid_search_results:
    #     print(f"ID: {result['Id']}\n")
    #     print(f"Text: {result['Text']}\n")
    #     print(f"ExternalSourceName: {result['ExternalSourceName']}\n")
    #     print(f"Description: {result['Description']}\n")
    #     print(f"AdditionalMetadata: {result['AdditionalMetadata']}\n")

    # Formated for semantik-kernel pipeline
    formatted_results = [
        f"""ID: {result['Id']}
        Text: {result['Text']}
        ExternalSourceName: {result['ExternalSourceName']}
        Source: {result['Description']}
        AdditionalMetadata: {result['AdditionalMetadata']}
        """ for result in hybrid_search_results
    ]

    print(formatted_results)

    # Example 3: Perform a hybrid search with filters
    # hybrid_wfilter_search_results = ai_search.search(
    #     query="Revenue of Microsoft",
    #     query_type="hybrid_wfilter",
    #     filter="Description eq 'BESTBUY'",
    #     select_fields=["Text", "Id", "Description", 'AdditionalMetadata', 'ExternalSourceName']
    # )
    # print("\nHybrid Search with Filters Results:")
    # for result in hybrid_wfilter_search_results:
    #     print(f"ID: {result['Id']}, Text: {result['Text']}, Description: {result['Description']}")