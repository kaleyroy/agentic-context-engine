import os, sys
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import TypedDict
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest


api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("DASHSCOPE_API_BASE")
default_model = "qwen3-max"

llm = ChatOpenAI(temperature=0, model=default_model, api_key=api_key, base_url=base_url)


# --------
# Tools functions
# --------


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# -------
# Agent functions
# --------


class Context(TypedDict):
    system_prompt: str


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    system_prompt = request.runtime.context.get("system_prompt", "")
    if system_prompt:
        return system_prompt
    else:
        return "You are a helpful assistant"


agent = create_agent(
    model=llm,
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=Context,
)


if __name__ == "__main__":
    # Casse 1: Basic
    user_input = "What's the weather like in 上海?"
    agent_state = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        context={"system_prompt": "You are a helpful assistant"},
    )
    print(agent_state["messages"][-1].content)

    # Casse 2: Custom prompt
    agent_state = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        context={"system_prompt": "You are a weather expert, response in JSON format"},
    )
    print(agent_state["messages"][-1].content)
