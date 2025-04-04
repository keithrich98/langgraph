# answer_verifier.py - Task for answer verification

import os
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from langgraph.func import task
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage

# Import shared state
from state import ChatState
# Import the logger from logging_config
from logging_config import logger

# Set up the LLM for verification
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# openai_api_key = os.getenv("OPENAI_API_KEY")
# if openai_api_key is None:
#     raise ValueError("OPENAI_API_KEY is not set in your environment.")
# llm = ChatOpenAI(
#     model_name="gpt-4o",
#     openai_api_key=openai_api_key,
#     temperature=0
# )

azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Check that variables are loaded
if None in [azure_api_key, azure_endpoint, azure_deployment, azure_api_version]:
    raise ValueError("Azure environment variables are not fully set.")

# Initialize the AzureChatOpenAI LLM
llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    azure_deployment=azure_deployment,
    api_key=azure_api_key,
    api_version=azure_api_version,
    temperature=0
)

# Verification result model
class VerificationResult(BaseModel):
    is_valid: bool = Field(description="Whether the answer meets all requirements")
    missing_requirements: List[str] = Field(default_factory=list, description="Requirements not met")
    verification_message: str = Field(description="Verification or follow-up message")

def verify_answer(
    question: Dict[str, Any],
    answer: str,
    conversation_history: List[Dict[str, str]],
    config: Optional[RunnableConfig] = None
) -> Tuple[bool, str]:
    """
    Verify if the user's answer meets the requirements of the question.
    
    Args:
        question: The question with requirements
        answer: The user's answer
        conversation_history: History of the conversation
        config: Optional runtime configuration
        
    Returns:
        Tuple of (is_valid, verification_message)
    """
    formatted_requirements = "\n".join([f"- {key}: {value}" for key, value in question['requirements'].items()])
    formatted_history = ""
    for msg in conversation_history:
        role = msg['role']
        content = msg['content']
        formatted_history += f"{role.upper()}: {content}\n\n"
    
    system_message = SystemMessage(content=
        "You are an expert medical validator for a questionnaire about polymicrogyria."
        "Evaluate if the user's answer meets the specified requirements."
    )
    
    user_message = HumanMessage(content=
        f"QUESTION: {question['text']}\n\n"
        f"REQUIREMENTS:\n{formatted_requirements}\n\n"
        f"CONVERSATION HISTORY:\n{formatted_history}\n"
        f"USER'S ANSWER: {answer}\n\n"
        f"Evaluate if this answer meets all the requirements. Explain why if valid or specify missing items."
    )
    
    llm_with_tool = llm.bind_tools(
        tools=[VerificationResult],
        tool_choice={"type": "function", "function": {"name": "VerificationResult"}}
    )
    
    try:
        response = llm_with_tool.invoke([system_message, user_message], config=config)
        if response.tool_calls and len(response.tool_calls) > 0:
            tool_call = response.tool_calls[0]
            result = VerificationResult(**tool_call["args"])
            return result.is_valid, result.verification_message
        if hasattr(response, "content") and response.content:
            is_valid = "valid" in response.content.lower() and "not valid" not in response.content.lower()
            return is_valid, response.content
        return False, "Unable to verify your answer. Please provide more details."
    except Exception as e:
        logger.error(f"Error in verification: {str(e)}")
        return False, f"There was an error verifying your answer: {str(e)}"

@task
def answer_verification_task(
    action: Dict = None,
    state: ChatState = None,
    config: Optional[RunnableConfig] = None
) -> ChatState:
    """
    Task to verify user answers.
    
    Expects an action dict with "action": "answer" with "answer" field.
    Updates the state with the answer and verification results.
    """
    # Log key details of initial state
    logger.debug(f"AV-Initial: conversation_history length: {len(state.conversation_history)}")
    logger.debug(f"AV-Initial: current_question_index: {state.current_question_index}")
    logger.debug(f"AV-Initial: thread_id: {id(state)}")
    
    if action is None or action.get("action") != "answer" or state.is_complete:
        logger.debug("AV: No action to verify or questionnaire is complete")
        return state
    
    # Process user answer
    answer = action.get("answer", "")
    state.conversation_history.append({"role": "human", "content": answer})
    
    # Log key details after human answer
    logger.debug(f"AV-AfterHumanAnswer: conversation_history length: {len(state.conversation_history)}")
    
    # Get the current question and verify the answer
    current_question = state.questions[state.current_question_index]
    is_valid, verification_message = verify_answer(current_question, answer, state.conversation_history, config)
    
    # Add verification message to conversation history
    state.conversation_history.append({"role": "ai", "content": verification_message})
    
    # Log key details after verification
    logger.debug(f"AV-AfterVerification: conversation_history length: {len(state.conversation_history)}")
    logger.debug(f"AV-AfterVerification: is_valid: {is_valid}")
    
    # Include verification result in the state for the question processor
    # We're not modifying the current_question_index or is_complete here - that's for the question processor
    state.verification_result = {
        "action": "verified_answer",
        "question_index": state.current_question_index,
        "answer": answer,
        "verification_message": verification_message,
        "is_valid": is_valid
    }
    
    # Log key details of final state
    logger.debug(f"AV-Final: conversation_history length: {len(state.conversation_history)}")
    
    return state