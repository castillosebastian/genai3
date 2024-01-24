# Problem: retriever not working

# source:

# /home/sebacastillo/genai3/semantic-kernel/python/notebooks/04-context-variables-chat.ipynb
# /home/sebacastillo/genai3/semantic-kernel/python/notebooks/06-memory-and-embeddings.ipynb
# /home/sebacastillo/genai3/semantic-kernel/python/semantic_kernel/core_plugins/text_memory_plugin.py
# https://techcommunity.microsoft.com/t5/educator-developer-blog/teach-chatgpt-to-answer-questions-using-azure-ai-search-amp/ba-p/3985395


import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from aisearch_async import AISearch
from dotenv import load_dotenv
load_dotenv()

ai_search = AISearch()

def format_hybrid_search_results(hybrid_search_results):
    formatted_results = [
        f"""ID: {result['Id']}
        Text: {result['Text']}
        ExternalSourceName: {result['ExternalSourceName']}
        Source: {result['Description']}
        AdditionalMetadata: {result['AdditionalMetadata']}
        """ for result in hybrid_search_results
    ]
    formatted_string = ""
    for i, doc in enumerate(formatted_results):
        #formatted_string += f"\n<document {i+1}>\n\n {doc}\n"
        formatted_string += f"\n\"\"\" {doc}\n\"\"\"\n\n"
    return formatted_string
    

sk_prompt = """
{{$chat_history}}

You are a super intetelligent financial assistant. Using exclusively the information provided in the <related_documents> answer the user question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Keep the answer as concise as possible and add the 'Source' as citation.

<related_documents>
{{$docs}}

User:> {{$user_input}}
ChatBot:>
"""

kernel = sk.Kernel()

deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()

kernel.add_chat_service(
        "chat_completion",
        AzureChatCompletion(deployment_name=deployment, endpoint=endpoint, api_key=api_key),
)

prompt_config = sk.PromptTemplateConfig.from_completion_parameters(max_tokens=2000, temperature=0.7, top_p=0.4)
prompt_template = sk.PromptTemplate(sk_prompt, kernel.prompt_template_engine, prompt_config)

function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
chat_function = kernel.register_semantic_function("ChatBot", "Chat", function_config)


async def chat(context_vars: sk.ContextVariables) -> bool:
    try:
        user_input = input("User:> ")

        context_vars["user_input"] = user_input     
        
        hybrid_search_results = await ai_search.search(query=user_input, query_type="hybrid", top=3,
                                                       select_fields=["Text", "Id","ExternalSourceName",
                                                                      "Description","AdditionalMetadata"])       

        docs =  format_hybrid_search_results(hybrid_search_results)
        
        context_vars["docs"]  = docs

    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False
    if user_input == "exit":
        print("\n\nExiting chat...")
        return False

    answer = await kernel.run_async(chat_function, input_vars=context_vars, )
    context_vars["chat_history"] += f"\nUser:> {user_input}\nChatBot:> {answer}\n"

    print(f"Retrieved Documents:> {docs}")
    print("-"*100)
    print(f"ChatBot:> {answer}")
    return True


async def main() -> None:
    context = sk.ContextVariables()
    context["chat_history"] = ""  
    
    chatting = True
    while chatting:        
        chatting = await chat(context)

if __name__ == "__main__":

    #print(documents)
    #ask1 Compare the total Revenue of Microsoft for the years 2023, 2022 and 2021
    #ask2 Explain the Revenue by reportable segment Domestic and International of Best By 2019 report
    asyncio.run(main())
