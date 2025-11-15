import os
from dotenv import load_dotenv
# from langchain import hub
# from langchain.agents import AgentExecutor
# from langchain.agents.react.agent import create_react_agent
# from langchain.agents.react.agent import create_react_agent
from langchain_experimental import create_react_agent
# from langchain import create_react_agent
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

def main():
    print("Hello from 3-react-search-agent!")

if __name__ == "__main__":
    main()