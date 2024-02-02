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
from plugins.AISearch.vsearch import VSearch
from dotenv import load_dotenv
load_dotenv()

# Creating the kernel
deployment_name, key, endpoint = sk.azure_openai_settings_from_dot_env()
embeddings = os.environ["AZURE_OPENAI_EMBEDDINGS_MODEL_NAME"]

azure_chat_service = AzureChatCompletion(deployment_name=deployment_name, endpoint=endpoint, api_key=key)
azure_text_embedding = AzureTextEmbedding(deployment_name=embeddings, endpoint=endpoint, api_key=key)


async def main() -> None:

    single_question = True
    question_set = False    


    if single_question:
        # Test single question-------------------------------------------------------
        ask = "What is the revenue of Microsoft?" # Test simple OK
        #ask = "What is the revenue of Pfizer?" # Test simple OK
        ask = "What is the revenue of Microsoft and Pfizer?" # Test simple OK
        #ask = "What is the revenue?" # Test execution with filter == None OK
        #ask = "In agreement with the information outlined in the income statement, what is average net profit margin (as a %) for Best Buy?"

        kernel = sk.Kernel()
        kernel.add_chat_service("chat_completion", azure_chat_service)
        kernel.add_text_embedding_generation_service("ada", azure_text_embedding)

        pluginASKT = kernel.import_semantic_plugin_from_directory("plugins", "ASKProcess")        
        extract_entities = pluginASKT["extractEntities"]
        pluginAIS = kernel.import_plugin(plugin_instance= VSearch(), plugin_name= "VSearch")
        vsearch =  pluginAIS["retrieve_documents"]

        context = kernel.create_new_context()
        context['ask'] = ask
        #response = await kernel.run(extract_entities, input_context=my_context)         
        #print(response['input'])
        #ask_entities = vsearch.retrieve_documents(response['input'])            
               
        doc = vsearch(context)

        print(doc['input'])       
    

if __name__ == "__main__":   
    asyncio.run(main())
