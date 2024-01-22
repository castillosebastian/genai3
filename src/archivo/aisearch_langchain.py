
import os
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain.embeddings import AzureOpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

model: str = "text-embedding-ada-002"
vector_store_address: str = os.environ["AZURE_AISEARCH_ENDPOINT"]
vector_store_password: str = os.environ["AZURE_AISEARCH_API_KEY"]

embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2023-05-15",
        chunk_size=1000
    )

index_name: str =  os.environ["AZURE_AISEARCH_INDEX_NAME"]
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)

docs = vector_store.similarity_search(
    query="What is the Revenue of Microsoft",
    k=3,
    search_type="hybrid",
)

print(docs[0].page_content)