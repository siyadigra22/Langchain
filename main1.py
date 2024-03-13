import os
from dotenv import load_dotenv
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables (assuming you have a .env file)
load_dotenv()

# Access the Hub model using its access token from the environment variable
hub_llm = HuggingFaceHub(repo_id = "gpt2")

# Define the prompt template
prompt = PromptTemplate(
    input_variables=["profession"],
    template="You had one job!  You're the {profession} and you didn't have to be sarcastic",
)

# Create the LLMChain instance
hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)

# Generate outputs for different professions
print(hub_chain.run(profession="customer service agent"))
print(hub_chain.run(profession="politician"))
print(hub_chain.run(profession="CEO"))
print(hub_chain.run(profession="insurance agent"))