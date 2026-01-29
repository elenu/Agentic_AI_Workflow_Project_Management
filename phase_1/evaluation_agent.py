# TODO: 1 - Import EvaluationAgent and KnowledgeAugmentedPromptAgent classes
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY", "voc-2138558860159874464587069597fcb09d575.36180062")
prompt = "What is the capital of France?"
print(f"Prompt: {prompt}")

# Parameters for the Knowledge Agent
kg_persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"
# Instantiate the KnowledgeAugmentedPromptAgent (was a class reference)
knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key, persona=kg_persona, knowledge=knowledge) # TODO: 2 - Instantiate the KnowledgeAugmentedPromptAgent here

# Parameters for the Evaluation Agent
persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."
evaluation_agent = EvaluationAgent(openai_api_key=openai_api_key, persona=persona, evaluation_criteria=evaluation_criteria, worker_agent=knowledge_agent, max_iterations=10, knowledge=knowledge, initial_prompt=prompt) # TODO: 3 - Instantiate the EvaluationAgent with a maximum of 10 interactions here

# TODO: 4 - Evaluate the prompt and print the response from the EvaluationAgent
evaluation_response = evaluation_agent.evaluate(prompt)
print(evaluation_response)
