from pathlib import Path
import os, sys

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Any, List, TypedDict
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from ace import Curator, LiteLLMClient, Playbook, Reflector
from ace.integrations.base import wrap_playbook_context
from ace.roles import GeneratorOutput
from test_langchain_chroma import k2_doc_vectorstore_similarity_search

from dotenv import load_dotenv

load_dotenv()  # load .env file

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


# --------
# ACE Chat Agent Wrapper
# --------


class ACEChatWrapper:

    SYSTEM_PROMPT_TEMPLATE = """
## 角色定义
你是一个专业知识库问答助手，接收用户的问题，并根据用户问题的检索结果，生成最佳的答案。
如果用户问题没有任务检索结果，或者检索结果无法回答用户问题，请根据你的知识进行答案生成。

## 知识检索结果
{search_results}
{playbook_context}
"""

    def __init__(
        self,
        model: str = default_model,
        temperature: float = 0.0,
        ace_model: str = "dashscope/qwen-plus",
        is_learning: bool = True,
        use_memory: bool = False,
    ):

        self.chat_agent = self._chat_agent(
            model=model, temperature=temperature, use_memory=use_memory
        )
        self.is_learning = is_learning
        playbook_path = Path(__file__).parent / "ace_chat_playbook.json"
        self.playbook = (
            Playbook.load_from_file(playbook_path)
            if playbook_path.exists()
            else Playbook()
        )
        _llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(_llm)
        self.curator = Curator(_llm)
        self.conversation_history = []

    def _chat_agent(
        self,
        model: str = default_model,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: str = "You are a helpful assistant",
        use_memory: bool = False,
    ):

        class AgentContext(TypedDict):
            system_prompt: str

        @dynamic_prompt
        def agent_system_prompt(request: ModelRequest) -> str:

            runtime_system_prompt = request.runtime.context.get("system_prompt", "")
            return runtime_system_prompt if runtime_system_prompt else system_prompt

        api_key = os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("DASHSCOPE_API_BASE")

        llm = ChatOpenAI(
            temperature=temperature,
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
        )

        # TODO: Add agent tools

        if use_memory:
            memory_saver = InMemorySaver()
            return create_agent(
                model=llm,
                tools=[],
                middleware=[agent_system_prompt],
                context_schema=AgentContext,
                checkpointer=memory_saver,
            )
        else:
            return create_agent(
                model=llm,
                tools=[],
                middleware=[agent_system_prompt],
                context_schema=AgentContext,
            )

    def chat(self, user_query: str) -> str:
        """Single chat turn with context injection."""

        playbook_context = ""
        results = k2_doc_vectorstore_similarity_search(user_query, k=3)
        search_results = [f"- {res.page_content}" for res, _ in results]

        # Inject playbook on first message
        if self.playbook.bullets():
            playbook_context = wrap_playbook_context(self.playbook)

        params = {
            "search_results": "".join(search_results),
            "playbook_context": playbook_context,
        }
        runtime_system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(**params)
        print(f"SYSTEM_PROMPT:\n{runtime_system_prompt}")

        # Chat
        success = True
        config: RunnableConfig = {"configurable": {"thread_id": "1"}}
        agent_state = self.chat_agent.invoke(
            {"messages": [{"role": "user", "content": user_query}]},
            context={"system_prompt": runtime_system_prompt},
            config=config,
        )

        messages = agent_state["messages"]
        response = messages[-1].content
        print(f"AGENT MESSAGES:\n{messages}")

        # Track conversation
        self.conversation_history.append({"user": user_query, "assistant": response})

        # Learn
        if self.is_learning:
            self.learn_conversation(success=success, feedback="User satisfied")
            # Save playbook to current file dir
            playbook_path = Path(__file__).parent / "ace_chat_playbook.json"
            self.playbook.save_to_file(playbook_path)

        return response

    def learn_conversation(self, success=True, feedback="User satisfied"):
        """Learn from entire conversation at the end."""
        if not self.conversation_history:
            return

        # Build conversation summary
        conversation = "\n".join(
            f"User: {turn['user']}\nAssistant: {turn['assistant']}"
            for turn in self.conversation_history
        )

        # Learn from full conversation
        generator_output = GeneratorOutput(
            reasoning=conversation,
            final_answer=self.conversation_history[-1]["assistant"],
            bullet_ids=[],
            raw={"success": success},
        )

        feedback_text = (
            f"Conversation {'succeeded' if success else 'failed'}, {feedback}"
        )

        user_query = self.conversation_history[0]["user"]
        reflection = self.reflector.reflect(
            question=user_query,
            generator_output=generator_output,
            playbook=self.playbook,
            feedback=feedback_text,
        )

        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=f"User query: {user_query})",
            progress="Conversation completed",
        )

        self.playbook.apply_delta(curator_output.delta)

        # Reset for next conversation
        self.conversation_history = []


# --------

# Test

# --------


if __name__ == "__main__":

    # # Casse 1: Basic
    # user_input = "What's the weather like in 上海?"
    # agent_state = agent.invoke(
    #     {"messages": [{"role": "user", "content": user_input}]},
    #     context={"system_prompt": "You are a helpful assistant"},
    # )
    # print(agent_state["messages"][-1].content)

    # # Casse 2: Custom prompt
    # agent_state = agent.invoke(
    #     {"messages": [{"role": "user", "content": user_input}]},
    #     context={"system_prompt": "You are a weather expert, response in JSON format"},
    # )
    # print(agent_state["messages"][-1].content)

    ace_chat = ACEChatWrapper(use_memory=True)
    while True:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break
        response = ace_chat.chat(user_input)
        print(f"Assistant: {response}")
