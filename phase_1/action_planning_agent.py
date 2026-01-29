# TODO: 1 - Import all required libraries, including the ActionPlanningAgent
from workflow_agents.base_agents import ActionPlanningAgent
import os
from dotenv import load_dotenv
load_dotenv()

# TODO: 2 - Load environment variables and define the openai_api_key variable with your OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY", "voc-2138558860159874464587069597fcb09d575.36180062")

knowledge = """
# Fried Egg
1. Heat pan with oil or butter
2. Crack egg into pan
3. Cook until white is set (2-3 minutes)
4. Season with salt and pepper
5. Serve

# Scrambled Eggs
1. Crack eggs into a bowl
2. Beat eggs with a fork until mixed
3. Heat pan with butter or oil over medium heat
4. Pour egg mixture into pan
5. Stir gently as eggs cook
6. Remove from heat when eggs are just set but still moist
7. Season with salt and pepper
8. Serve immediately

# Boiled Eggs
1. Place eggs in a pot
2. Cover with cold water (about 1 inch above eggs)
3. Bring water to a boil
4. Remove from heat and cover pot
5. Let sit: 4-6 minutes for soft-boiled or 10-12 minutes for hard-boiled
6. Transfer eggs to ice water to stop cooking
7. Peel and serve
"""

# TODO: 3 - Instantiate the ActionPlanningAgent, passing the openai_api_key and the knowledge variable
agent = ActionPlanningAgent(openai_api_key, knowledge)

# TODO: 4 - Print the agent's response to the following prompt: "One morning I wanted to have scrambled eggs"
prompt = "One morning I wanted to have scrambled eggs"
print("Prompt:", prompt)

available = [m for m in dir(agent) if callable(getattr(agent, m)) and not m.startswith("__")] # I consulted it here: https://stackoverflow.com/questions/33242357/python-getattr-callable-function
print("Agent callable methods:", available)

if hasattr(agent, "extract_steps_from_prompt"): # I checked it here: https://www.w3schools.com/python/ref_func_hasattr.asp
    steps = agent.extract_steps_from_prompt(prompt)
else:
    raise AttributeError(
        "Agent has no method to handle prompts. Check the printed methods and call the correct one."
    )

# Pretty-print the steps
if isinstance(steps, list): # I learnt this on: https://www.w3schools.com/python/ref_func_isinstance.asp
    for i, step in enumerate(steps, start=1):
        print(f"{i}. {step}")
else:
    print(steps)
