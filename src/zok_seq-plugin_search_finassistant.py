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
from dotenv import load_dotenv
load_dotenv()

# Creating the kernel
deployment_name, key, endpoint = sk.azure_openai_settings_from_dot_env()
embeddings = os.environ["AZURE_OPENAI_EMBEDDINGS_MODEL_NAME"]

azure_chat_service = AzureChatCompletion(deployment_name=deployment_name, endpoint=endpoint, api_key=key)
azure_text_embedding = AzureTextEmbedding(deployment_name=embeddings, endpoint=endpoint, api_key=key)

async def main() -> None:
    
    kernel = sk.Kernel()
    kernel.add_chat_service("chat_completion", azure_chat_service)
    kernel.add_text_embedding_generation_service("ada", azure_text_embedding)

    test_plan = False
    test_chain_plugin = False
    test_qrewrite = True

    ## Test Simple Plan------------------------------------------------------------------     OK  
    if test_plan:    
        planner = ActionPlanner(kernel)   
        ask = "What is the revenue of Microsoft?"
        print(f"Finding the most similar function available to get that done...")
        plan = await planner.create_plan_async(goal=ask)
        print(f"🧲 The best single function to use is `{plan._plugin_name}.{plan._function.name}`")
   
    # Test chain plugin---------------------------------------------------------------      OK    
    if test_chain_plugin:
        # Plugins
        pluginAIS = kernel.import_plugin(plugin_instance= AISearch(), plugin_name= "AISearch")
        search_function = pluginAIS['search']
        pluginFC = kernel.import_semantic_plugin_from_directory(pluginDirectory, "FinanceGenerator")        
        consultant_response = pluginFC["OneCompanyQuestion"]        
        # Context Variables
        ask = "What is the total Revenue of Microsoft for the years 2023,2022,2021?"
        #ask = "Give me a summary of Management's Discussion and Analysis of Best Buy 2019?" # Failed        
        # Retrieve
        #documents = str(await kernel.run_async(search_function, input_str=ask))  # str critically important! No, my mistake, I was not using the class api        
        documents = await kernel.run_async(search_function, input_str=ask)  # str critically important!                

        # As context vars
        # my_context = sk.ContextVariables()
        # my_context["input"] = ask
        # my_context["context"] = documents
        # As Context
        my_context = kernel.create_new_context()
        my_context['input'] = ask
        my_context['context'] =  documents['input']
        # As SK Context
        response = await kernel.run_async(consultant_response, input_context=my_context)           
        # As SK Variables
        #response = await kernel.run_async(consultant_response, input_vars=my_context)       
        print(response)       

    # Test get query intent-------------------------------------------------------------      OK  
    if test_qrewrite:
        #ask = "What is the total Revenue of Microsoft for the years 2023,2022,2021?"
        ask = "Give me a summary of Management's Discussion and Analysis of Best Buy 2019?" # Failed        
        pluginFC = kernel.import_semantic_plugin_from_directory(pluginDirectory, "AISearch")        
        qrewrite = pluginFC["qrewrite"] 
        # Set Context
        my_context = kernel.create_new_context()
        my_context['query'] = ask
        my_context['chat_history'] =  ''
        # As SK Context
        response = await kernel.run_async(qrewrite, input_context=my_context) 
        print(response['input'])




if __name__ == "__main__":   
    asyncio.run(main())

