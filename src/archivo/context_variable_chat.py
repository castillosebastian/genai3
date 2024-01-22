
import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    OpenAIChatCompletion,
)

kernel = sk.Kernel()

deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
kernel.add_chat_service(
    "chat_completion",
    AzureChatCompletion(deployment_name=deployment, endpoint=endpoint, api_key=api_key),
)

sk_prompt = """
ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know' if it does not have an answer.

{{$history}}
User: {{$user_input}}
ChatBot: """

chat_function = kernel.create_semantic_function(
    prompt_template=sk_prompt,
    function_name="ChatBot",
    max_tokens=2000,
    temperature=0.7,
    top_p=0.5,
)

context = kernel.create_new_context()
context["history"] = ""
context["user_input"] = "Hi, tell me about AI sector in US Markets"
#bot_answer = await chat_function.invoke_async(context=context)
bot_answer = chat_function.invoke(context=context)
print(bot_answer)
print(context['history'])
context["history"] += f"\nUser: {context['user_input']}\nChatBot: {bot_answer}\n"
print(context["history"])

async def chat(input_text: str) -> None:
    # Save new message in the context variables
    print(f"User: {input_text}")
    context["user_input"] = input_text

    # Process the user message and get an answer
    answer = await chat_function.invoke_async(context=context)

    # Show the response
    print(f"ChatBot: {answer}")

    # Append the new interaction to the chat history
    context["history"] += f"\nUser: {input_text}\nChatBot: {answer}\n"

async def main() -> None:
    chatting = True
    while chatting:
        chatting = await chat()


if __name__ == "__main__":
    asyncio.run(main())
