from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from langchain-course!")

    # Step 1: Define a prompt template
    information = "Python is a programming language."
    summary_template = """
    given the information {information} I want you to create:
    1. A short summary
    2. Two interesting facts about the topic"""

    summary_prompt_template = PromptTemplate(  
        input_variables=["information"],
        template=summary_template
    )

    # Step 2: Create a chain with an LLM
    # llm = ChatOllama(model="gemma3:270m", temperature=0.9)
    llm = ChatOllama(model="llama3.2", temperature=0.9)
    # llm = ChatOllama(model="dolphin-mistral:7b", temperature=0.9)
    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = summary_prompt_template | llm

    # Step 3: Invoke the chain
    response = chain.invoke({"information": information})
    print("Summary:", response.content)

if __name__ == "__main__":
    main()
