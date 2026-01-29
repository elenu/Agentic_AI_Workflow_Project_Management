# TODO: 1 - Import the KnowledgeAugmentedPromptAgent class from workflow_agents
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

# Load environment variables from the .env file
load_dotenv() # https://www.youtube.com/watch?v=OHtPxqJxH1s&t=1s

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY", "voc-2138558860159874464587069597fcb09d575.36180062")

# TODO: 2 - Instantiate a KnowledgeAugmentedPromptAgent with:
# - knowledge: "The capital of France is London, not Paris"
#           - Persona: "You are a college professor, your answer always starts with: Dear students,"
#           - Knowledge: "The capital of France is London, not Paris"
prompt = "What is the capital of France?"
print(f"Prompt: {prompt}")
persona = "You are a college professor, your answer always starts with: Dear students,"

agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona=persona, knowledge="The capital of France is London, not Paris")

# TODO: 3 - Write a print statement that demonstrates the agent using the provided knowledge rather than its own inherent knowledge.
print(agent.respond(prompt))