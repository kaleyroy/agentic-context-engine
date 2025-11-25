from browser_use import Agent, Browser, ChatOpenAI
import asyncio
import os
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


async def example():
    browser = Browser(
        # use_cloud=True,  # Uncomment to use a stealth browser on Browser Use Cloud
    )

    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_API_BASE")

    llm = ChatOpenAI(model="qwen-plus", api_key=api_key, base_url=base_url)

    agent = Agent(
        task="Find the number of stars of the browser-use repo",
        llm=llm,
        browser=browser,
    )

    history = await agent.run()
    return history


if __name__ == "__main__":
    history = asyncio.run(example())
