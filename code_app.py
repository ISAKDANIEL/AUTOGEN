import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load API key from .env
load_dotenv()


async def main():
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ API key not found. Check your .env file.")
        return

    # Create OpenAI model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key,
    )

    # Create Assistant Agent
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful AI assistant."
    )

    print("AutoGen Assistant is Ready!")
    print("Type 'exit' to stop.\n")

    # Chat loop
    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("👋 Goodbye brother!")
            break

        # Send task to assistant
        result = await assistant.run(task=user_input)

        # Print assistant response
        print("\nAssistant:", result.messages[-1].content)
        print("-" * 50)


# Run async program
asyncio.run(main())
