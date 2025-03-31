# verification_agent.py - Updated for more deterministic behavior

import os  # for environment operations
from typing import Dict, List, Any, Optional, Tuple  # type annotations
from pydantic import BaseModel, Field  # for data validation and schema definitions
from langgraph.func import task  # decorator for defining tasks in the workflow
from langchain_core.runnables import RunnableConfig  # for accessing runtime configuration
from langchain_core.messages import HumanMessage, SystemMessage  # for building messages for the LLM
from langchain_openai import ChatOpenAI  # for using OpenAI's Chat models
from dotenv import load_dotenv  # to load environment variables from a .env file
import uuid  # for generating unique IDs if needed

# Import shared state functions and ChatState model
from state import ChatState, get_state_for_thread, update_state_for_thread
# Import the configured logger for logging
from logging_config import logger

# Log initialization of the verification agent
logger.info("Initializing verification_agent.py")

# Load environment variables
load_dotenv()
# Get the OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    logger.error("OPENAI_API_KEY is not set in your environment.")
    raise ValueError("OPENAI_API_KEY is not set in your environment.")

# Initialize the ChatOpenAI model for verification with GPT-4 and zero temperature (for deterministic output)
logger.info("Initializing LLM for verification agent")
llm = ChatOpenAI(
    model_name="gpt-4",
    openai_api_key=openai_api_key,
    temperature=0
)

# Define a Pydantic model for the verification result returned by the LLM
class VerificationResult(BaseModel):
    is_valid: bool = Field(description="Whether the answer meets all requirements")
    missing_requirements: List[str] = Field(default_factory=list, description="Requirements not met")
    verification_message: str = Field(description="Verification or follow-up message")

# Define the verification task that always uses LLM-based verification
@task
def verify_answer_task(thread_id: str, answer: str, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Task to verify if a user's answer meets the requirements for the current question.
    This function always calls the LLM-based verification (_perform_verification) and passes
    the conversation history (retrieved via thread_id) along with the human's answer.
    """
    # Log the start of the verification task and the answer received
    logger.info(f"verify_answer_task called for thread_id: {thread_id}")
    logger.debug(f"Answer to verify: {answer}")
    
    # If a config is provided, check and correct the thread_id based on the configuration
    if config and "configurable" in config and "thread_id" in config["configurable"]:
        actual_thread_id = config["configurable"]["thread_id"]
        if thread_id != actual_thread_id:
            logger.warning(f"Correcting invalid thread_id: from {thread_id} to {actual_thread_id}")
            thread_id = actual_thread_id
    
    # Retrieve the current conversation state (including history) using the thread_id
    state = get_state_for_thread(thread_id)
    if not state:
        logger.error(f"No state found for thread_id: {thread_id}")
        return {
            "success": True,  # Allow workflow to continue gracefully
            "is_valid": True,  # Assume the answer is valid
            "message": "Answer has been recorded. Let's continue with the questionnaire.",
            "missing_requirements": []
        }
    
    # Log the current question index for debugging
    logger.debug(f"Current question index: {state.current_question_index}")
    
    # Check if there is a current question to verify (i.e. session still active)
    if state.current_question_index >= len(state.questions):
        logger.error(f"No current question to verify, index: {state.current_question_index}")
        return {
            "success": True,  # Workflow should continue
            "is_valid": True,  # Assume valid since no question is left
            "message": "All questions have been completed.",
            "missing_requirements": []
        }
    
    # Retrieve the current question using the state index
    current_question = state.questions[state.current_question_index]
    logger.debug(f"Current question: {current_question['text']}")
    
    # Always use LLM-based verification by passing the human answer along with the full conversation history
    logger.info("Performing LLM-based verification")
    try:
        verification_result = _perform_verification(
            question=current_question,
            answer=answer,
            conversation_history=state.conversation_history,
            config=config
        )
    except Exception as e:
        # If an error occurs during LLM verification, log it and default to accepting the answer
        logger.error(f"Verification error: {str(e)}", exc_info=True)
        verification_result = VerificationResult(
            is_valid=True,  # Force valid to allow progression
            verification_message="Your answer has been recorded. Let's continue with the questionnaire.",
            missing_requirements=[]
        )
    
    logger.debug(f"Verification result - is_valid: {verification_result.is_valid}")
    logger.debug(f"Verification message: {verification_result.verification_message}")
    
    # Append the human answer to the conversation history for future context
    state.conversation_history.append({"role": "human", "content": answer})
    
    # If the answer is verified as valid by the LLM:
    if verification_result.is_valid:
        logger.info("Answer is valid")
        # Save the answer in the state's responses and verified_answers
        state.responses[state.current_question_index] = answer
        state.verified_answers[state.current_question_index] = {
            "question": current_question["text"],
            "answer": answer,
            "verification": verification_result.verification_message
        }
        # Append a verification confirmation message to the conversation history
        state.conversation_history.append({
            "role": "ai", 
            "content": f"VERIFIED: {verification_result.verification_message}"
        })
    else:
        # If the answer does not meet the requirements, append a follow-up message
        logger.info("Answer needs more info")
        state.conversation_history.append({
            "role": "ai", 
            "content": f"NEEDS MORE INFO: {verification_result.verification_message}"
        })
    
    # Update the stored state with the new conversation history and responses
    update_state_for_thread(thread_id, state)
    
    # Return the result of the verification
    return {
        "success": True,
        "is_valid": verification_result.is_valid,
        "message": verification_result.verification_message,
        "missing_requirements": verification_result.missing_requirements
    }

# The helper function _is_simple_valid_answer has been removed because verification now always uses _perform_verification

def _perform_verification(
    question: Dict[str, Any],
    answer: str,
    conversation_history: List[Dict[str, str]],
    config: Optional[RunnableConfig] = None
) -> VerificationResult:
    """
    Helper function to perform the actual verification using the LLM.
    This function constructs a prompt that includes the current question,
    its requirements (as provided in the question's requirements field), and the user's answer,
    along with a relevant portion of the conversation history.
    """
    logger.info("_perform_verification called")
    
    # Format the question's requirements for inclusion in the prompt
    formatted_requirements = "\n".join([f"- {key}: {value}" for key, value in question['requirements'].items()])
    logger.debug(f"Formatted requirements: {formatted_requirements}")
    
    # To reduce token usage, only include the last 4 messages from the conversation history if available
    relevant_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
    formatted_history = ""
    # Loop through each message in the relevant history and format it
    for msg in relevant_history:
        role = msg['role']
        content = msg['content']
        formatted_history += f"{role.upper()}: {content}\n\n"
    
    logger.debug(f"History length: {len(formatted_history)} characters")
    
    # Create a system message to instruct the LLM on its role as a medical validator
    system_message = SystemMessage(content=
        "You are an expert medical validator for a questionnaire about polymicrogyria. "
        "Your only task is to verify if the user's answer meets the specified requirements. "
        "Be lenient - if the answer generally addresses the requirement, consider it met. "
        "Always return a valid VerificationResult with is_valid=true unless requirements are clearly missing."
    )
    # Create a human message that includes the question, the formatted requirements, and the user's answer
    user_message = HumanMessage(content=
        f"QUESTION: {question['text']}\n\n"
        f"REQUIREMENTS:\n{formatted_requirements}\n\n"
        f"USER'S ANSWER: {answer}\n\n"
        f"Does this answer meet all the requirements? If yes, return is_valid=true with a brief verification message. "
        f"If any requirements are missing, return is_valid=false with specific follow-up questions."
    )
    
    # Prepare the LLM with tools so that it can return a structured response matching VerificationResult
    logger.info("Preparing LLM with tools for verification")
    llm_with_tool = llm.bind_tools(
        tools=[VerificationResult],
        tool_choice={"type": "function", "function": {"name": "VerificationResult"}}
    )
    
    try:
        # Call the LLM with the system and human messages, along with any provided configuration
        logger.info("Calling LLM for verification")
        response = llm_with_tool.invoke([system_message, user_message], config=config)
        logger.debug(f"LLM response received: {response}")
        
        # If the LLM response contains tool_calls, process the first one as the structured verification result
        if response.tool_calls and len(response.tool_calls) > 0:
            logger.info("Processing tool call response")
            tool_call = response.tool_calls[0]
            result = VerificationResult(**tool_call["args"])
            logger.debug(f"Tool call result: is_valid={result.is_valid}, message={result.verification_message[:50]}...")
            return result
        
        # As a fallback, if the response has direct content, use it to determine validity heuristically
        if hasattr(response, "content") and response.content:
            logger.info("Processing direct content response")
            is_valid = True  # Default to valid
            if "missing" in response.content.lower() and "requirement" in response.content.lower():
                is_valid = False
            if "not valid" in response.content.lower() or "invalid" in response.content.lower():
                is_valid = False
            
            return VerificationResult(
                is_valid=is_valid,
                verification_message=response.content,
                missing_requirements=[] if is_valid else ["Please provide more complete information"]
            )
            
        # If no valid response format is detected, log a warning and return a default valid result
        logger.warning("No valid response format detected, using default fallback")
        return VerificationResult(
            is_valid=True,
            verification_message="Your response has been recorded.",
            missing_requirements=[]
        )
            
    except Exception as e:
        # If an exception occurs, log the error and return a default verification result that allows progression
        error_msg = f"Error in verification: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return VerificationResult(
            is_valid=True,
            verification_message="Your response has been accepted.",
            missing_requirements=[]
        )
