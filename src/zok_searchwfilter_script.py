import asyncio
import os 
import sys
import json
import inspect
import semantic_kernel as sk
from semantic_kernel.planning import ActionPlanner
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
# Get the root directory of your project (the directory containing 'src' and 'plugins')
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
pluginDirectory = "plugins"
from plugins.AISearch.aisearch import AISearchWF, build_query_filter
from plugins.AISearch.query import query
from src.utils import string_to_json
from dotenv import load_dotenv
load_dotenv()

# Creating the kernel
deployment_name, key, endpoint = sk.azure_openai_settings_from_dot_env()
embeddings = os.environ["AZURE_OPENAI_EMBEDDINGS_MODEL_NAME"]

azure_chat_service = AzureChatCompletion(deployment_name=deployment_name, endpoint=endpoint, api_key=key)
azure_text_embedding = AzureTextEmbedding(deployment_name=embeddings, endpoint=endpoint, api_key=key)


async def main() -> None:

    ask = "What is the Revenue of Microsoft and Pfizer?"

    result = await query(ask)    

    for i in result:
        print("-"*100)
        print(i)
        print("-"*100)
         

if __name__ == "__main__":   
    asyncio.run(main())

