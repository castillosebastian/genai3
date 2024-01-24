import asyncio
import os 
import sys
import inspect
import semantic_kernel as sk
from semantic_kernel.planning import ActionPlanner
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
# Get the root directory of your project (the directory containing 'src' and 'plugins')
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from plugins.AISearch.aisearch import AISearch
pluginDirectory = "plugins"

# Creating the kernel
kernel = sk.Kernel()
api_key, org_id = sk.azure_aisearch_settings_from_dot_env()
deployment_name, key, endpoint = sk.azure_openai_settings_from_dot_env()
embeddings = os.environ["AZURE_OPENAI_EMBEDDINGS_MODEL_NAME"]

azure_chat_service = AzureChatCompletion(deployment_name=deployment_name, endpoint=endpoint, api_key=key)
azure_text_embedding = AzureTextEmbedding(deployment_name=embeddings, endpoint=endpoint, api_key=key)

kernel.add_chat_service("chat_completion", azure_chat_service)
kernel.add_text_embedding_generation_service("ada", azure_text_embedding)

async def main() -> None:
   
    pluginAIS = kernel.import_plugin(plugin_instance= AISearch(), plugin_name= "AISearch")
    search_function = pluginAIS['search']   
    ask = "What is the total Revenue of Microsoft for the years 2023,2022,2021?"
    documents = await kernel.run_async(search_function, input_str=ask)
    print(documents)
    

if __name__ == "__main__":   
    asyncio.run(main())

