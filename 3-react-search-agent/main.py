import os
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_tavily import TavilySearch
from langsmith import Client
from langchain_ollama import ChatOllama

load_dotenv()

# define the tools and LLM
tools = [TavilySearch()]
local_llm = ChatOllama(model="llama3.2", temperature=0)

# pull the prompt from LangSmith
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
client = Client(api_key=LANGSMITH_API_KEY)
react_prompt = client.pull_prompt("hwchase17/react", include_model=True)

# create the agent
agent = create_react_agent(
    llm=local_llm,
    tools=tools,
    prompt=react_prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def main():
    print("Hello from 3-react-search-agent!")
    result = agent_executor.invoke(
        {
            "input": "Search for 3 AI Platform Engineer job posts using langchain in Northern Virginia on LinkedIn posted in the last week and list their details.",
        }
    )
    print(result)

if __name__ == "__main__":
    main()
