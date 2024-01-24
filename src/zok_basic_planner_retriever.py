import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.planning import BasicPlanner
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from plugins.AISearch.aisearch import AISearch

deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()

# Creating and tooling the kernel
kernel = sk.Kernel()

kernel.add_chat_service(
        "chat_completion",
        AzureChatCompletion(deployment_name=deployment, endpoint=endpoint, api_key=api_key),
)

aisearch_plugin = kernel.import_plugin(AISearch(),"AISearch")
aisearch_plugin = aisearch_plugin["search"]
results = aisearch_plugin("What is the Revenue of Microsoft")
#print(results)


# Que todo lo haga el planner!



PROMPT = """
You are a planner for the Semantic Kernel.
Your job is to create a properly formatted JSON plan step by step, to satisfy the goal given.
Create a list of subtasks based off the [GOAL] provided.
Each subtask must be from within the [AVAILABLE FUNCTIONS] list. Do not use any functions that are not in the list.
Base your decisions on which functions to use from the description and the name of the function.
Sometimes, a function may take arguments. Provide them if necessary.
The plan should be as short as possible.
For example:

[AVAILABLE FUNCTIONS]
AISearch.search
description: This function search for finance information stored in knowledge data base
args:
- input: A user query related to factual data or insight from financial documents

[GOAL]
"Answer user questions"
[OUTPUT]
    {
        "input": query,
        "subtasks": [
            {"function": "AISearch.search","args": {"query": "What is the Revenue of Microsoft?"}}
        ]
    }

[AVAILABLE FUNCTIONS]
{{$available_functions}}

[GOAL]
{{$goal}}

[OUTPUT]
"""

planner = BasicPlanner()
ask = "What is the Revenue of Best Buy?"

basic_plan = asyncio.run(planner.create_plan_async(goal=ask, kernel=kernel,prompt=PROMPT))
#print("generated plan ",basic_plan.generated_plan)

results = asyncio.run(planner.execute_plan_async(basic_plan, kernel))
#print(results)

sk_prompt = """
You are a super intetelligent financial assistant. Using exclusively the information provided in the <related_documents> answer the user question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Keep the answer as concise as possible and add the 'Source' as citation.

<related_documents>
{{$input}}

User:> "What is the Revenue of Best Buy?"
ChatBot:>
"""

rag = kernel.create_semantic_function(
    prompt_template=sk_prompt,
    function_name="RAG",
    plugin_name="RAGPlugin",
    description="Aswer the user question",
    max_tokens=500,
    temperature=0.5,
    top_p=0.5,
)

async def main() -> None:
   story = await rag.invoke_async(input=results)
   print(story)


if __name__ == "__main__":

    #print(documents)
    #ask1 Compare the total Revenue of Microsoft for the years 2023, 2022 and 2021
    #ask2 Explain the Revenue by reportable segment Domestic and International of Best By 2019 report
    asyncio.run(main())
  
