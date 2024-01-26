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
from plugins.AISearch.aisearch import AISearchWF
pluginDirectory = "plugins"

from dotenv import load_dotenv
load_dotenv()

# Creating the kernel
deployment_name, key, endpoint = sk.azure_openai_settings_from_dot_env()
embeddings = os.environ["AZURE_OPENAI_EMBEDDINGS_MODEL_NAME"]

azure_chat_service = AzureChatCompletion(deployment_name=deployment_name, endpoint=endpoint, api_key=key)
azure_text_embedding = AzureTextEmbedding(deployment_name=embeddings, endpoint=endpoint, api_key=key)


def string_to_json(string):
    try:
        # Convert the string to a JSON object
        json_object = json.loads(string)
        return json_object
    except json.JSONDecodeError as e:
        # Handle the exception if the string is not a valid JSON
        return f"Error converting string to JSON: {e}"
    
def get_json_field(json_string, field_name):
    # Convert the string to a JSON object
    json_object = string_to_json(json_string)

    if not isinstance(json_object, dict):
        return f"Invalid JSON"
    
    if field_name in json_object:
        # Extract the field value
        field_value = json_object[field_name]
        
        if field_value is None:
            return "Field value is null"
        return field_value
    else:
        return f"'{field_name}' key not found"


async def main() -> None:
    
    kernel = sk.Kernel()
    kernel.add_chat_service("chat_completion", azure_chat_service)
    kernel.add_text_embedding_generation_service("ada", azure_text_embedding)

    
    test_filter_mode_basic = True

    # Test get query intent-------------------------------------------------------------      OK  
    if test_filter_mode_basic:
        
        #ask = "Give me a summary of MD&A of Pfizer 2019?" # Failed
        #ask = "Waht elements are mentioned in the MD&A of Pfizer 2019?" # Failed
        ask = "What elements are mentioned in the MD&A of BestBuy 2019?" # Failed
        #ask = "What is the total Revenue of Microsoft for the years 2023,2022,2021?"
        #ask = "Is Best Buy trying to enrich the lives of consumers through technology?" # Failed        
        #ask = "Give a summary overview of Best Buy challenges"
        #ask = "In agreement with the information outlined in the income statement, what is the FY2015 - FY2017 3 year average net profit margin (as a %) for Best Buy? Answer in units of percents and round to one decimal place."
        #ask= """
        #What is the year end FY2019 total amount of inventories for Best Buy? Answer in USD millions. 
        #Base your judgments on the information provided primarily in the balance sheet.
        #"""        
        #ask="Is growth in the company adjusted EPS expected to accelerate in that period?" # use indirec reference
        #chat_history = ""
                
        pluginAIS = kernel.import_plugin(plugin_instance= AISearchWF(), plugin_name= "AISearchWF")
        searchwf =  pluginAIS["searchwf"]    
        pluginFC = kernel.import_semantic_plugin_from_directory(pluginDirectory, "FinanceGenerator")        
        consultant_response = pluginFC["OneCompanyQuestion"]        
        pluginASKT = kernel.import_semantic_plugin_from_directory(pluginDirectory, "ASKProcess")        
        extract_entities = pluginASKT["extractEntities"]                              
        
        # Set Context
        my_context = kernel.create_new_context()
        my_context['ask'] = ask
                
        # Rewrite
        print(ask)
        
        response = await kernel.run_async(extract_entities, input_context=my_context)         
        #ask_entities = string_to_json(response['input'])
        #print(type(ask_entities))
        field_name = "company_name"
        field_value = get_json_field(response['input'], field_name)[0].upper()
        print(field_value)          
               
        context_variables = sk.ContextVariables(variables={"ask":ask,"company": field_value})
        # print(type(ask))
        # print(type(field_value))
        # print(type(context_variables))        
        # context_variables['ask'] = ask
        # context_variables['company'] = field_value

        # Retrieve document with Hybrid Search with Filters
        documents = await kernel.run_async(searchwf, input_vars=context_variables) 
        print(documents)
        
        # As Context
        context = kernel.create_new_context()
        context['input'] = ask
        context['context'] =  documents['input']
        
        # Generate
        response = await kernel.run_async(consultant_response, input_context=context)           
        print(response)           

if __name__ == "__main__":   
    asyncio.run(main())

