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
    test_rewrite = False
    test_rewrite_retrive = True # Very important
    test_filter_mode = False

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
        #ask = "What is the total Revenue of Microsoft for the years 2023,2022,2021?"
        ask = "Give me a summary of Management's Discussion and Analysis of Best Buy 2019?" # Failed        
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
    if test_rewrite:
        #ask = "What is the total Revenue of Microsoft for the years 2023,2022,2021?"
        ask = "Give me a summary of MD&A of Best Buy 2019?" # Failed        
        pluginFC = kernel.import_semantic_plugin_from_directory(pluginDirectory, "ASKProcess")        
        rewrite = pluginFC["rewrite"] 
        # Set Context
        my_context = kernel.create_new_context()
        my_context['ask'] = ask
        my_context['chat_history'] =  ''
        # As SK Context
        response = await kernel.run_async(rewrite, input_context=my_context) 
        print(response['input'])

    # Test get query intent-------------------------------------------------------------      OK  
    if test_rewrite_retrive:
        #ask = "What is the total Revenue of Microsoft for the years 2023,2022,2021?"
        #ask = "Is Best Buy trying to enrich the lives of consumers through technology?" # Failed        
        #ask = "Give a summary overview of Best Buy challenges"
        #ask = "In agreement with the information outlined in the income statement, what is the FY2015 - FY2017 3 year average net profit margin (as a %) for Best Buy? Answer in units of percents and round to one decimal place."
        # ask= """
        # What is the year end FY2019 total amount of inventories for Best Buy? Answer in USD millions. 
        # Base your judgments on the information provided primarily in the balance sheet.
        # """        
        chat_history = """
        User: Hi Assistant, can you tell me if the growth in Johnson & Johnson's adjusted EPS is expected to accelerate in FY2023? \n
        Assistant: Certainly! Analysts are forecasting a potential acceleration in Johnson & Johnson's adjusted EPS for FY2023. This is based on their strong product pipeline and strategic investments in pharmaceuticals and medical devices.\n
        User: What are the main factors that could drive this acceleration?\n
        Assistant: The main factors include increased sales from their pharmaceutical segment, particularly from new and existing drugs. Additionally, the company's focus on high-margin products and cost-cutting measures are likely to contribute.\n
        User: Are there any recent product launches that could significantly impact their earnings?\n
        Assistant: Yes, Johnson & Johnson has had several successful product launches recently, especially in their pharmaceutical division. These new drugs, treating a range of conditions, are expected to boost revenue growth in FY2023.\n
        User: How does the company's diversification across healthcare sectors affect its earnings stability?\n
        Assistant: Johnson & Johnson's diversification across pharmaceuticals, medical devices, and consumer health products provides a balanced portfolio that can mitigate risks and ensure more stable earnings. This diversity helps them navigate market fluctuations more effectively.\n
        User: Has the company made any recent acquisitions or partnerships that could influence its growth?\n
        Assistant: Indeed, Johnson & Johnson has been actively pursuing strategic acquisitions and partnerships, particularly in the pharmaceutical sector, to expand its product offerings and fuel growth. These moves could positively impact their growth trajectory in FY2023.\n
        User: What are the potential risks that could hinder this expected EPS growth?\n
        Assistant: The risks include potential regulatory challenges, competition from generic and biosimilar products, and global economic uncertainties that could affect healthcare spending. These factors could impact their revenue and, consequently, their EPS growth.\n
        User: Finally, what's the general consensus among analysts regarding JnJ's growth prospects for FY2023?\n
        Assistant: Most analysts hold a positive outlook for Johnson & Johnson's growth in FY2023, citing their strong pipeline, strategic initiatives, and robust market presence. However, it's important to stay updated as market conditions can change.\n
        """
        ask="Is growth in the company adjusted EPS expected to accelerate in that period?" # use indirec reference
        #ask="What can you tell me about it MD&A in that period?" # use indirec reference
        
        pluginAIS = kernel.import_plugin(plugin_instance= AISearch(), plugin_name= "AISearch")
        search = pluginAIS['search']
        pluginFC = kernel.import_semantic_plugin_from_directory(pluginDirectory, "FinanceGenerator")        
        consultant_response = pluginFC["OneCompanyQuestion"]        
        pluginASKT = kernel.import_semantic_plugin_from_directory(pluginDirectory, "ASKProcess")        
        rewrite = pluginASKT["rewrite"] 

        # Set Context
        my_context = kernel.create_new_context()
        my_context['ask'] = ask
        my_context['chat_history'] =  chat_history
        
        # Rewrite
        response = await kernel.run_async(rewrite, input_context=my_context) 
        new_ask = response['input']
        print(f'ask:{ask}')
        print(f'new ask:{new_ask}')

        # Search
        documents = await kernel.run_async(search, input_str=new_ask)       
        print(documents)

        # As Context
        context = kernel.create_new_context()
        context['input'] = new_ask
        context['context'] =  documents['input']
        
        # Generate
        response = await kernel.run_async(consultant_response, input_context=context)           
        
        print(response)       
    

if __name__ == "__main__":   
    asyncio.run(main())

