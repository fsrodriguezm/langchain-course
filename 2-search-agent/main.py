from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet.
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching for: {query}")

    # For demonstration purposes, we return a static response.
    # return "El Salvador weather is sunny"

    # In a real implementation, you would call the Tavily search API like this:
    return tavily.search(query=query)

# Create an agent with the search tool
llm = ChatOpenAI(model="gpt-4", temperature=0)
local_llm = ChatOllama(model="llama3.2", temperature=0)
tools = [search]
agent = create_agent(model=local_llm, tools=tools)

def main():
    print("Hello from 2-search-agent!")
    result = agent.invoke({"messages":HumanMessage(content="What is the weather in El Salvador?")})
    print(f"Agent result: {result}")

    agent_message = result["messages"][-1] 
    print (f"## Final Agent Message ##")
    print(f"Agent answer: {agent_message.content}")

if __name__ == "__main__":
    main()