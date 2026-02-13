import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load environment variables
load_dotenv()


async def main():
    # Create model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Create assistant
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful assistant."
    )

    # Run a simple task
    result = await assistant.run(
        task="create a landing page in the name of isak"
    )

    # Print response
    print("\n=== Response ===\n")
    print(result.messages[-1].content)


asyncio.run(main())
