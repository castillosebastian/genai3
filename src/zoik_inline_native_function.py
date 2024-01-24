
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    OpenAIChatCompletion,
)

kernel = sk.Kernel()

useAzureOpenAI = True

# Configure AI service used by the kernel
if useAzureOpenAI:
    deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
    azure_chat_service = AzureChatCompletion(
        deployment_name="turbo", endpoint=endpoint, api_key=api_key
    )  # set the deployment name to the value of your chat model
    kernel.add_chat_service("chat_completion", azure_chat_service)
else:
    api_key, org_id = sk.openai_settings_from_dot_env()
    oai_chat_service = OpenAIChatCompletion(ai_model_id="gpt-3.5-turbo", api_key=api_key, org_id=org_id)
    kernel.add_chat_service("chat-gpt", oai_chat_service)


import random

from semantic_kernel.skill_definition import sk_function

#from semantic_kernel.plugin_definition import sk_function

class GenerateNumberSkill:
    """
    Description: Generate N random numbers.
    """

    @sk_function(
        description="Generate N random number",
        name="GenerateNRandomNumbers",
    )
    def generate_number_three_or_higher(self, input: str) -> str:
        """
        Generate <input> random numbers.
        Example:
            "8" => (3,8,1,2,4,7,8,9)
        Args:
            input -- the total random number generation
        Returns:
            int value
        """
        try:
            #return str(random.randint(3, int(input)))
            return str([random.randint(0, input-1) for _ in range(input)])
        except ValueError as e:
            print(f"Invalid input {input}")
            raise e

generate_number_plugin = kernel.import_skill(GenerateNumberSkill())
generate_nrandom_numbers = generate_number_plugin["GenerateNRandomNumbers"]
number_result = generate_nrandom_numbers(100)
print(number_result)

