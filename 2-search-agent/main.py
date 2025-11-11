from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field
from functions import tavily_search

load_dotenv()

class Source(BaseModel):
    """Schema for a source used by the agent."""
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources."""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

# # Custom search tool using Tavily
# tavily = TavilyClient()

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over the internet.
#     Args:
#         query: The query to search for
#     Returns:
#         The search results
#     """
#     print(f"Searching for: {query}")

#     # For demonstration purposes, we return a static response.
#     # return "El Salvador weather is sunny"

#     # In a real implementation, you would call the Tavily search API like this:
#     return tavily.search(query=query)

# Create an agent with the search tool
llm = ChatOpenAI(model="gpt-4", temperature=0)
local_llm = ChatOllama(model="llama3.2", temperature=0)
# tools = [search] # Using custom search tool
# tools = [TavilySearch()] # Using Tavily search tool
tools = [tavily_search] # Using Tavily search tool defined in functions.py

# agent = create_agent(model=local_llm, tools=tools) # Using Tavily search tool
agent = create_agent(model=local_llm, tools=tools, response_format=AgentResponse)

def main():
    print("Hello from 2-search-agent!")
    # result = agent.invoke({"messages":HumanMessage(content="What is the weather in El Salvador?")})
    result = agent.invoke({"messages":HumanMessage(content="Search for 3 AI Platform Engineer job posts using langchain in Northern Virginia on LinkedIn and list their details.")})
    print(f"Agent result: {result}")

    agent_message = result["messages"][-1] 
    print (f"## Final Agent Message ##")
    print(f"Agent answer: {agent_message.content}")

if __name__ == "__main__":
    main()