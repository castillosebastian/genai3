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

    single_question = True
    question_set = False

    if single_question:
        # Test single question-------------------------------------------------------
        ask = "In agreement with the information outlined in the income statement, what is average net profit margin (as a %) for Best Buy?"
        result = await query(ask)    
        for i in result:
            print("-"*100)
            print(i)
            print("-"*100)
    
    if question_set:
        # Test question-set-----------------------------------------------------------
        # Define test cases
        results_list = []

        test_cases = [
        "What is the Revenue of Microsoft?",
        "In agreement with the information outlined in the income statement, what is average net profit margin (as a %) for Best Buy?",
        "What is the total amount of inventories for Best Buy? Answer in USD millions. Base your judgments on the information provided primarily in the balance sheet.",
        "Is growth in JnJ's adjusted EPS expected to accelerate?",
        "How did JnJ's sales growth compare to international sales growth?",
        "Has Microsoft increased its debt on balance sheet?",
        "How much does Pfizer expect to pay to spin off Upjohn in the future in USD million?",
        "For Pfizer, which geographic region had the biggest drop in year over year revenues?",
        "Is Pfizer spinning off any large business segments?"
        ]

        for ask in test_cases:
            print(f"\nQuery: {ask}")
            documents = await query(ask)
        
            # Create a dictionary for each question and its result
            result_dict = {
                'question': ask,
                'documents': documents[0]['input']
            }

            # Append the dictionary to the results list
            results_list.append(result_dict)

        # Write the results to a JSON file
        with open('query_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_list, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":   
    asyncio.run(main())

