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
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "For each story, provide the following fields and nothing else, in the exact order and format shown below:\n"
    "User stories: list of stories that starts as 'As a [type of user], I want [an action or feature] so that [benefit/value].'\n"
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    f"""Should include content like: \n
User Stories:
1. As a Customer Support Representative, I want emails to be automatically categorized so that I can focus on complex inquiries.
2. As an IT Administrator, I want to configure routing rules so that emails reach the right experts.
"""
    "Ensure each story is unique and covers different functionalities of the product. "
    "Do NOT include features or tasks—only user stories.\n"
    # TODO: 5 - Complete this knowledge string by appending the product_spec loaded in TODO 3
    f"Product Specification: {product_spec}"
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
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = (
    "You are a Program Manager responsible for organizing user stories into product features. "
    "For each feature, provide the following fields and nothing else, in the exact order and format shown below:\n"
    "Feature Name: <short title>\n"
    "Description: <brief explanation of what the feature does and its purpose>\n"
    "Key Functionality: <bullet or comma-separated list of capabilities or actions>\n"
    "User Benefit: <how this feature creates value for the user>\n"
    "Return each feature separated by a blank line. Do NOT include user stories—only features derived from the provided stories."
    "Use the user stories provided below to define features for the product. "
    f"""Should include content like: \n
    Features:
Feature Name: Email Ingestion System
Description: Real-time email retrieval and preprocessing
Key Functionality: SMTP/IMAP integration, metadata extraction
User Benefit: Seamless integration with existing infrastructure
"""
    "Ensure each feature is unique and covers different functionalities of the product. "
    "Do NOT include user stories or tasks—only features.\n"
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
    Tasks:
Task ID: TASK-001
Task Title: Implement SMTP Email Connector
Related User Story: US-001 (Email Auto-categorization)
Description: Develop connector to retrieve emails via SMTP protocol
Acceptance Criteria: Successfully retrieve emails within 5 seconds
Estimated Effort: 3 days
Dependencies: Email server credentials, API documentation"""
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
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = (
    "You are a Development Engineer responsible for producing concrete, implementable development tasks for the product. "
    "For each user story or feature supplied, create one or more specific tasks that can be assigned to engineers. "
    "Each task MUST follow this exact structure (no extra text):\n"
    "Task ID: TASK-XXX (use incremental IDs starting at TASK-001)\n"
    "Task Title: <brief title>\n"
    "Related User Story: <exact user story sentence or feature reference>\n"
    "Description: <detailed technical work to be done>\n"
    "Acceptance Criteria: <clear, testable criteria>\n"
    "Estimated Effort: <hours or story points>\n"
    "Dependencies: <comma-separated task IDs or 'None'>\n"
    "Return only tasks (one blank line between tasks). DO NOT return general instructions about how to write tasks. Provide concrete tasks tailored to the Email Router product specification and the user stories/features provided."
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
    response = product_manager_knowledge_agent.respond(query)
    evaluation = product_manager_evaluation_agent.evaluate(response)
    return evaluation

def program_manager_support_function(query):
    response = program_manager_knowledge_agent.respond(query)
    evaluation = program_manager_evaluation_agent.evaluate(response)
    return evaluation

def development_engineer_support_function(query):
    response = development_engineer_knowledge_agent.respond(query)
    evaluation = development_engineer_evaluation_agent.evaluate(response)
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
workflow_prompt = "What would the development tasks for this product be?"
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

for step in workflow_steps:
    print(f"\nExecuting step: {step}")
    result = routing_agent.route(step)
    completed_steps.append({"step": step, "result": result})

    # Try to detect which role this step belongs to so we can compile structured output
    step_l = step.lower()
    if "product manager" in step_l or "story" in step_l or "as a" in step_l:
        role = "product_manager"
    elif "program manager" in step_l or "feature" in step_l:
        role = "program_manager"
    elif "development engineer" in step_l or "task" in step_l or "development" in step_l:
        role = "development_engineer"
    else:
        # fallback: let the router pick the agent based on similarity by checking printed router logs
        # If we cannot confidently map, skip structured assignment
        role = None

    if role is not None:
        results_by_role[role] = result

    # Print a concise view of the result for this step
    if isinstance(result, dict) and result.get("final_response"):
        print(f"Result (final_response):\n{result['final_response']}\nEvaluation: {result.get('evaluation')}\nIterations: {result.get('num_iterations')}")
    else:
        print(f"Result:\n{result}\n")

print("\n*** Workflow execution completed ***\n")

# Compile and print structured output
print("\n--- Compiled structured workflow output ---\n")
print("Product Manager (user stories):\n", results_by_role.get("product_manager"))
print("\nProgram Manager (features):\n", results_by_role.get("program_manager"))
print("\nDevelopment Engineer (tasks):\n", results_by_role.get("development_engineer"))

# Provide the most relevant final output (tasks) if available
final_output = results_by_role.get("development_engineer") or (completed_steps[-1] if completed_steps else None)
print(f"\nFinal output of the workflow: {final_output}")