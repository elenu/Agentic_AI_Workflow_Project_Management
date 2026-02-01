# agentic_workflow.py

# TODO: 1 - Import the following agents: ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent from the workflow_agents.base_agents module
import workflow_agents.base_agents as agents
import os
from dotenv import load_dotenv

# TODO: 2 - Load the OpenAI key into a variable called openai_api_key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY", "voc-2138558860159874464587069597fcb09d575.36180062")

# load the product spec
# TODO: 3 - Load the product spec document Product-Spec-Email-Router.txt into a variable called product_spec
with open("Product-Spec-Email-Router.txt", "r") as file:
    product_spec = file.read()
    print(product_spec)

# Instantiate all the agents
# Note: action_planning_agent is instantiated after its knowledge
# (defined below) to avoid a NameError for 'knowledge_action_planning'.

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
# TODO: 4 - Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = agents.ActionPlanningAgent(
    openai_api_key=openai_api_key,
    knowledge=knowledge_action_planning
)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = f"""You are a Product Manager who ONLY produces user stories.
You do not write tasks, estimates, acceptance criteria, or implementation details.
You consider task-level output a role violation.
If you produce anything other than a valid user story, your response is incorrect."""
knowledge_product_manager = (
    f"""You are the Product Manager for the Email Router. Use ONLY the specification below.
Return 8–12 user stories in the form:
"As a [user type], I want [action/feature] so that [benefit]."
Do not include any functionality outside this product.
Specification:
[PASTE EMAIL ROUTER SPEC HERE]"""    
    # TODO: 5 - Complete this knowledge string by appending the product_spec loaded in TODO 3
    f"""Product Specification: {product_spec}."""
)
# TODO: 6 - Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = agents.KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager
)

# Product Manager - Evaluation Agent
# TODO: 7 - Define the persona and evaluation criteria for a Product Manager evaluation agent and instantiate it as product_manager_evaluation_agent. This agent will evaluate the product_manager_knowledge_agent.
# The evaluation_criteria should specify the expected structure for user stories (e.g., "As a [type of user], I want [an action or feature] so that [benefit/value].").
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_product_manager =  "The answer should be stories that follow the following structure: As a [type of user], I want [an action or feature] so that [benefit/value]."
product_manager_evaluation_agent = agents.EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager_eval,
    evaluation_criteria=evaluation_criteria_product_manager,
    worker_agent=product_manager_knowledge_agent,
    max_iterations=3
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = f"""You are a Program Manager focused on execution clarity.
You return product features for each user story without changing intent or scope.
You do not design user stories or tasks."""
knowledge_program_manager = (
    f"""You are the Program Manager for the Email Router. Define product features ONLY by grouping the user stories below.
For each feature return:
- Feature Name
- Description
- Key Functionality
- User Benefit
- Related User Stories: [list exact story IDs/text from the list]
Allowed source of truth: the user stories listed below (do not invent new ones).
User Stories:
[PASTE USER STORIES FROM PREVIOUS STEP HERE]
    """
)
# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
# (This is a necessary step before TODO 8. Students should add the instantiation code here.)
program_manager_knowledge_agent = agents.KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager
)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

# TODO: 8 - Instantiate a program_manager_evaluation_agent using 'persona_program_manager_eval' and the evaluation criteria below.
#                      "The answer should be product features that follow the following structure: " \
#                      "Feature Name: A clear, concise title that identifies the capability\n" \
#                      "Description: A brief explanation of what the feature does and its purpose\n" \
#                      "Key Functionality: The specific capabilities or actions the feature provides\n" \
#                      "User Benefit: How this feature creates value for the user"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
    f"""Should include content like: \n
Product features in the format: "Feature Name:...", "Description:...", "Key Functionality:...", "User Benefit:..."
"""
)
program_manager_evaluation_agent = agents.EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=(
        "The answer must contain one or more features. Each feature must exactly follow this structure and include these four fields in order:\n"
        "Feature Name:\nDescription:\nKey Functionality:\nUser Benefit:\n"
        "Return 'Yes' only if ALL features strictly follow this structure; otherwise return 'No' and list which features/lines violate the format."
    ),
    worker_agent=program_manager_knowledge_agent,
    max_iterations=3
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = f"""You are a senior development engineer responsible for turning user stories into concrete development tasks.
You never evaluate whether a task “should exist” — you define how to build it.
You never respond with empty output."""
knowledge_dev_engineer = (
    f"""You are the Development Engineer for the Email Router. Create development tasks ONLY for the user stories and features below.
Each task must include:
- Task ID
- Task Title
- Related User Story (must be one of the listed stories)
- Description
- Acceptance Criteria
- Estimated Effort
- Dependencies
User Stories:
[PASTE USER STORIES HERE]
Features:
[PASTE FEATURES HERE]
"""
)
# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
# (This is a necessary step before TODO 9. Students should add the instantiation code here.)
development_engineer_knowledge_agent = agents.KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
# TODO: 9 - Instantiate a development_engineer_evaluation_agent using 'persona_dev_engineer_eval' and the evaluation criteria below.
#                      "The answer should be tasks following this exact structure: " \
#                      "Task ID: A unique identifier for tracking purposes\n" \
#                      "Task Title: Brief description of the specific development work\n" \
#                      "Related User Story: Reference to the parent user story\n" \
#                      "Description: Detailed explanation of the technical work required\n" \
#                      "Acceptance Criteria: Specific requirements that must be met for completion\n" \
#                      "Estimated Effort: Time or complexity estimation\n" \
#                      "Dependencies: Any tasks that must be completed first"
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.
development_engineer_evaluation_agent = agents.EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=(
        "The answer must contain one or more concrete tasks. Each task must strictly follow this structure and include these fields in order:\n"
        "Task ID: TASK-XXX\n"
        "Task Title:\n"
        "Related User Story:\n"
        "Description:\n"
        "Acceptance Criteria:\n"
        "Estimated Effort:\n"
        "Dependencies:\n"
        "Return 'Yes' only if ALL tasks strictly follow the format and each Task ID is unique and referenced dependencies (if any) use valid TASK-IDs or 'None'. Otherwise return 'No' and explain which tasks/fields are invalid or too generic."
        f""" Example of the format:
            Task ID: TASK-001
            Task Title: Implement SMTP Email Connector
            Related User Story: US-001 (Email Auto-categorization)
            Description: Develop connector to retrieve emails via SMTP protocol
            Acceptance Criteria: Successfully retrieve emails within 5 seconds
            Estimated Effort: 3 days
            Dependencies: Email server credentials, API documentation"""
    ),
    worker_agent=development_engineer_knowledge_agent,
    max_iterations=3
)

# Routing Agent

# Job function persona support functions
# TODO: 11 - Define the support functions for the routes of the routing agent (e.g., product_manager_support_function, program_manager_support_function, development_engineer_support_function).
# Each support function should:
#   1. Take the input query (e.g., a step from the action plan).
#   2. Get a response from the respective Knowledge Augmented Prompt Agent.
#   3. Have the response evaluated by the corresponding Evaluation Agent.
#   4. Return the final validated response.
def product_manager_support_function(query):
#    response = product_manager_knowledge_agent.respond(query)
#    evaluation = product_manager_evaluation_agent.evaluate(response)
    evaluation = product_manager_evaluation_agent.evaluate(query) # Modified after reviewer's suggestion
    return evaluation

def program_manager_support_function(query):
#    response = program_manager_knowledge_agent.respond(query)
#    evaluation = program_manager_evaluation_agent.evaluate(response)
    evaluation = program_manager_evaluation_agent.evaluate(query)# Modified after reviewer's suggestion
    return evaluation

def development_engineer_support_function(query):
#    response = development_engineer_knowledge_agent.respond(query)
#    evaluation = development_engineer_evaluation_agent.evaluate(response)
    evaluation = development_engineer_evaluation_agent.evaluate(query) # Modified after reviewer's suggestion
    return evaluation

# TODO: 10 - Instantiate a routing_agent. You will need to define a list of agent dictionaries (routes) for Product Manager, Program Manager, and Development Engineer. Each dictionary should contain 'name', 'description', and 'func' (linking to a support function). Assign this list to the routing_agent's 'agents' attribute.
routing_agent = agents.RoutingAgent(
    openai_api_key=openai_api_key,
    agents=[
        {
            "name": "product_manager",
            "description": "Responsible for defining product personas and user stories only. Does not define features or tasks. Does not group stories.",
            "func": lambda query: product_manager_support_function(query)
        },
        {
            "name": "program_manager",
            "description": "Responsible for defining features and managing the overall program. Does not define user stories or tasks.",
            "func": lambda query: program_manager_support_function(query)
        },
        {
            "name": "development_engineer",
            "description": "Responsible for defining and managing development tasks.",
            "func": lambda query: development_engineer_support_function(query)
        }
    ]
)

# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ***
#workflow_prompt = "What would the development tasks for this product be?"
workflow_prompt = f"""What are the user stories, product features, and the development tasks for this product?"""
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
# TODO: 12 - Implement the workflow.
#   1. Use the 'action_planning_agent' to extract steps from the 'workflow_prompt'.
#   2. Initialize an empty list to store 'completed_steps'.
#   3. Loop through the extracted workflow steps:
#      a. For each step, use the 'routing_agent' to route the step to the appropriate support function.
#      b. Append the result to 'completed_steps'.
#      c. Print information about the step being executed and its result.
#   4. After the loop, print the final output of the workflow (the last completed step).
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
completed_steps = []

# Prepare a structure to collect final validated outputs per role
results_by_role = {
    "product_manager": None,
    "program_manager": None,
    "development_engineer": None,
}

def get_clean_response(role_result): # Inspired by: https://peps.python.org/pep-0008/
    if isinstance(role_result, dict) and "final_response" in role_result:
        return role_result["final_response"]
    return role_result

# Configuration for role detection
role_config = {
    "product_manager": [ "story", "as a"],#"product manager",
    "program_manager": ["feature", "Name"],# "program manager", 
    "development_engineer": ["Task", "task"],#"development engineer", 
}

for step in workflow_steps:
    print(f"\nExecuting step: {step}")
    result = routing_agent.route(step)
    completed_steps.append({"step": step, "result": result})
    
    step_l = step.lower()
    detected_role = None

    # Refactored role detection logic
    for role, keywords in role_config.items():
        if any(kw in step_l for kw in keywords):
            detected_role = role
            results_by_role[role] = result
            break 

    # Consolidated printing logic
    clean_output = get_clean_response(result)
    print(f"\n[{detected_role or 'System'}] Output:\n{clean_output}\n")


print("\n*** Workflow execution completed ***")

# Simplified Structured Output
print("\n--- Compiled structured workflow output ---")
outputs = {
    "Product Manager (User Stories)": "product_manager",
    "Program Manager (Features)": "program_manager",
    "Development Engineer (Tasks)": "development_engineer"
}
for label, key in outputs.items():
    print(f"\n{label}:\n{get_clean_response(results_by_role.get(key, 'N/A'))}")
# Store result by role (unchanged)
results_by_role[detected_role] = result

## Final workflow output (defaults to dev tasks, falls back to last step)
final_val = results_by_role.get("development_engineer")
final_output = get_clean_response(final_val) if final_val else (completed_steps[-1]["result"] if completed_steps else "No output")
print(f"\nFinal output of the workflow: {final_output}")
