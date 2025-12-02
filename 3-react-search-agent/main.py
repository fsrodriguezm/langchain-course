import os
from typing import Any, Iterable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langsmith import Client

load_dotenv()

# define the tools and LLM
tools = [TavilySearch()]
local_llm = ChatOllama(model="llama3.2", temperature=0)

# pull the prompt from LangSmith
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
client = Client(api_key=LANGSMITH_API_KEY)
react_prompt = client.pull_prompt("hwchase17/react", include_model=True)
system_prompt = None

def _extract_system_prompt(prompt: Any) -> str | None:
    """Attempt to pull the system message from a LangSmith prompt."""
    chat_prompt: ChatPromptTemplate | None = None
    if isinstance(prompt, ChatPromptTemplate):
        chat_prompt = prompt
    elif isinstance(prompt, RunnableSequence):
        possible_prompt = prompt.first
        if isinstance(possible_prompt, ChatPromptTemplate):
            chat_prompt = possible_prompt

    if chat_prompt is None:
        return None

    for message_template in chat_prompt.messages:
        if isinstance(message_template, SystemMessagePromptTemplate):
            placeholders = {
                var: f"{{{var}}}" for var in message_template.input_variables
            }
            try:
                formatted_message = message_template.format(**placeholders)
            except Exception:
                continue

            content = formatted_message.content
            if isinstance(content, str):
                return content

            if isinstance(content, Iterable):
                # Content can be a list of typed segments (eg. OpenAI multimodal messages)
                text_chunks = [
                    chunk.get("text", "")
                    for chunk in content
                    if isinstance(chunk, dict) and chunk.get("type") == "text"
                ]
                merged = "\n".join(part for part in text_chunks if part).strip()
                if merged:
                    return merged
    return None

system_prompt = _extract_system_prompt(react_prompt)

agent = create_agent(
    model=local_llm,
    tools=tools,
    system_prompt=system_prompt,
)

def main():
    print("Hello from 3-react-search-agent!")
    query = (
        "Search for 3 AI Platform Engineer job posts using langchain in "
        "Northern Virginia on LinkedIn posted in the last week and list their details."
    )
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    print("Agent state:", result)
    agent_message = result["messages"][-1]
    print("Final response:", agent_message.content)

if __name__ == "__main__":
    main()
