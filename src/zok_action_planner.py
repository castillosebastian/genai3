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
api_key, org_id = sk.azure_aisearch_settings_from_dot_env()
deployment_name, key, endpoint = sk.azure_openai_settings_from_dot_env()
embeddings = os.environ["AZURE_OPENAI_EMBEDDINGS_MODEL_NAME"]

azure_chat_service = AzureChatCompletion(deployment_name=deployment_name, endpoint=endpoint, api_key=key)
azure_text_embedding = AzureTextEmbedding(deployment_name=embeddings, endpoint=endpoint, api_key=key)




async def main() -> None:
    kernel = sk.Kernel()
    kernel.add_chat_service("chat_completion", azure_chat_service)
    kernel.add_text_embedding_generation_service("ada", azure_text_embedding)

    ## Test Simple Plan------------------------------------------------------------------     OK  
    #planner = ActionPlanner(kernel)   
    #ask = "What is the revenue of Microsoft?"
    #print(f"Finding the most similar function available to get that done...")
    #plan = await planner.create_plan_async(goal=ask)
    #print(f"🧲 The best single function to use is `{plan._plugin_name}.{plan._function.name}`")
   
    # Test chain plugin---------------------------------------------------------------      ERROR
    # Fail because:
    # this code: response = await kernel.run_async(consultant_response, input_context=my_context)    
    # Something went wrong in pipeline step 0. During function invocation: 'FinanceGenerator.OneCompanyQuestion'. 
    # Error description: 'sequence item 3: expected str instance, SKContext found'    
    pluginAIS = kernel.import_plugin(plugin_instance= AISearch(), plugin_name= "AISearch")
    search_function = pluginAIS['search']
    pluginFC = kernel.import_semantic_plugin_from_directory(pluginDirectory, "FinanceGenerator")        
    consultant_response = pluginFC["OneCompanyQuestion"]        

    # # print(type(search_function))
    # # print(type(pluginFC))    
    ask = "What is the total Revenue of Microsoft for the years 2023,2022,2021?"
    #print(type(ask))
    documents = await kernel.run_async(search_function, input_str=ask)    
    text = str(documents)
    #print(type(text))

    # llevar objeto docoment a str   
    
    # As context vars
    # context_vars = sk.ContextVariables()
    # context_vars["input"] = ask
    # context_vars["context"] = documents

    # As context
    my_context = kernel.create_new_context()
    my_context['input'] = ask
    my_context['context'] =  text    

    response = await kernel.run_async(consultant_response, input_context=my_context)   
    print(response)       

    # Test other aproach----------------------------------------------------------------
    # pluginAIS = kernel.import_plugin(plugin_instance= AISearch(), plugin_name= "AISearch")
    # search_function = pluginAIS['search']
    
    # ask = "What is the total Revenue of Microsoft for the years 2023,2022,2021?"
    # documents = await kernel.run_async(search_function, input_str=ask)    

    # sk_prompt = """
    # You are a super-intelligent financial assistant. 
    # You has been asked this question: {{$input}}
    # And the following context has been provided: 

    # ####
    # {{$context}}
    # ####

    # If the context provided is "unknown" then the assistant should not provide any answer and should instead ask for clarification.
    # If the context has been provided, then the assistant proceeds to provide expert answer to resolves the question at hand with available contextual information.
    # The answer is expressed within the proper financial vocabulary. 
    # Add the 'Source' as citation.
    # Assistant:
    # """
    # prompt_config = sk.PromptTemplateConfig(max_tokens=2000, temperature=0.7, top_p=0.4)
    # prompt_template = sk.PromptTemplate(sk_prompt, kernel.prompt_template_engine, prompt_config)
    # function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
    # chat_function = kernel.register_semantic_function("ChatBot", "Chat", function_config)
       
    # context_vars = sk.ContextVariables()
    # context_vars["input"] = ask
    # context_vars["context"] = documents
    # print(type(context_vars))
    # response = await kernel.run_async(chat_function, input_vars=context_vars)  
    # print(response)


if __name__ == "__main__":   
    asyncio.run(main())

