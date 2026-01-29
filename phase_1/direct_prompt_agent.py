# Test script for DirectPromptAgent class

from workflow_agents.base_agents import DirectPromptAgent  # Import the DirectPromptAgent class from workflow_agents
import os
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv() # https://www.youtube.com/watch?v=OHtPxqJxH1s&t=1s

# Load the OpenAI API key from the environment variables (fallback to placeholder)
openai_api_key = os.getenv("OPENAI_API_KEY", "voc-2138558860159874464587069597fcb09d575.36180062")
# Define the prompt to be sent to the agent
prompt = "What is the Capital of France?"

# TODO: 3 - Instantiate the DirectPromptAgent as direct_agent
# Instantiate the DirectPromptAgent
direct_agent = DirectPromptAgent(openai_api_key)

# Use the agent to send the prompt (the agent implementation exposes `respond`)
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
print(direct_agent_response)

# TODO: 5 - Print an explanatory message describing the knowledge source used by the agent to generate the response
print("The DirectPromptAgent uses direct access to the OpenAI language model to generate responses based on the provided prompt.")