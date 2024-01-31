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
from plugins.AISearch.aisearch import AISearchWF, build_query_filter
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
        
        # Single Company Query
        #ask = "What is the revenue of Microsoft?" # OK
        #ask = "What is the revenue of Microsoft in 2019" # OK
        # Multiple Componies Query
        ask = "What elements are mentioned in the MD&A of BestBuy and Microsoft?" # Failed
                
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
        print(f'original_question: {ask}')
        
        
        response = await kernel.run(extract_entities, input_context=my_context)         
        ask_entities = string_to_json(response['input'])        
        print(f'extracted_entities: {ask_entities}')
        metadata_filter = build_query_filter(ask_entities)
        print(f'builded filter: {metadata_filter}')
              

        if len(metadata_filter)==1:
            context_variables = sk.ContextVariables(variables={"ask":ask,"filter": metadata_filter[0]})
            # Retrieve document with Hybrid Search with Filters
            documents = await kernel.run(searchwf, input_vars=context_variables) 
            print(documents)
        else:
            for i in metadata_filter:
                context_variables = sk.ContextVariables(variables={"ask":ask,"filter": i})
                # Retrieve document with Hybrid Search with Filters
                documents = await kernel.run(searchwf, input_vars=context_variables) 
                print('Retrieved document:\n')
                print(documents)
                print('-'*100)        
        
        # # As Context
        # context = kernel.create_new_context()
        # context['input'] = ask
        # context['context'] =  documents['input']
            
        #     # Generate
        # response = await kernel.run(consultant_response, input_context=context)           
        # print(response)           

if __name__ == "__main__":   
    asyncio.run(main())

