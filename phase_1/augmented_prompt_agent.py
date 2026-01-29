# TODO: 1 - Import the AugmentedPromptAgent class
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import AugmentedPromptAgent # Import the AugmentedPromptAgent class

# Load environment variables from .env file
load_dotenv() # https://www.youtube.com/watch?v=OHtPxqJxH1s&t=1s

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY", "voc-2138558860159874464587069597fcb09d575.36180062")

# Define the prompt and persona for the agent
prompt = "What is the capital of France?"
print(f"Prompt: {prompt}")

persona = "You are a college professor; your answers always start with: 'Dear students,'"

# TODO: 2 - Instantiate an object of AugmentedPromptAgent with the required parameters
augmented_agent = AugmentedPromptAgent(openai_api_key, persona)

# TODO: 3 - Send the 'prompt' to the agent and store the response in a variable named 'augmented_agent_response'
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)

# TODO: 4 - Add a comment explaining:
# - What knowledge the agent likely used to answer the prompt.
# The AugmentedPromptAgent likely used its internal knowledge base, which may include facts about world capitals, to answer the prompt about the capital of France.
# - How the system prompt specifying the persona affected the agent's response.
# The system prompt likely influenced the agent to adopt a more formal and educational tone, consistent with that of a college professor. This would affect not only the language used but also the depth and style of the explanation provided in the response.
# The output includes comments discussing knowledge source and persona impact.
print("It is expected an augmented prompting agent will include a metadata header that ensures traceability of information and explains the persona's influence. In this case, the AugmentedPromptAgent likely used its internal knowledge base, which may include facts about world capitals, to answer the prompt about the capital of France. The system prompt influenced the agent to respond in a formal and educational tone, consistent with that of a college professor.")