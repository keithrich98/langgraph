# verification_agent.py
import os
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from langgraph.func import task
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Import shared state
from state import ChatState, get_state_for_thread, update_state_for_thread
from logging_config import logger

logger.info("Initializing verification_agent.py")

# Set up environment and API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    logger.error("OPENAI_API_KEY is not set in your environment.")
    raise ValueError("OPENAI_API_KEY is not set in your environment.")

# Initialize LLM
logger.info("Initializing LLM for verification agent")
llm = ChatOpenAI(
    model_name="gpt-4",
    openai_api_key=openai_api_key,
    temperature=0
)

# Verification result model
class VerificationResult(BaseModel):
    is_valid: bool = Field(description="Whether the answer meets all requirements")
    missing_requirements: List[str] = Field(default_factory=list, description="Requirements not met")
    verification_message: str = Field(description="Verification or follow-up message")

@task
def verify_answer_task(thread_id: str, answer: str, config: Optional[RunnableConfig] = None) -> Dict:
    """
    Task to verify if a user's answer meets the requirements for the current question.
    """
    logger.info(f"verify_answer_task called for thread_id: {thread_id}")
    logger.debug(f"Answer to verify: {answer}")
    
    # Get the current state
    state = get_state_for_thread(thread_id)
    if not state:
        logger.error(f"No state found for thread_id: {thread_id}")
        return {
            "success": False,
            "error": "No active questionnaire session found.",
            "verification_result": None
        }
    
    logger.debug(f"Current question index: {state.current_question_index}")
    
    # Get the current question
    if state.current_question_index >= len(state.questions):
        logger.error(f"No current question to verify, index: {state.current_question_index}")
        return {
            "success": False,
            "error": "No current question to verify.",
            "verification_result": None
        }
    
    current_question = state.questions[state.current_question_index]
    logger.debug(f"Current question: {current_question['text']}")
    
    # Perform verification
    logger.info("Performing verification")
    verification_result = _perform_verification(
        question=current_question,
        answer=answer,
        conversation_history=state.conversation_history,
        config=config
    )
    
    logger.debug(f"Verification result - is_valid: {verification_result.is_valid}")
    logger.debug(f"Verification message: {verification_result.verification_message}")
    
    # Update the state based on verification results
    state.conversation_history.append({"role": "human", "content": answer})
    
    if verification_result.is_valid:
        logger.info("Answer is valid")
        # Record the verified answer
        state.responses[state.current_question_index] = answer
        state.verified_answers[state.current_question_index] = {
            "question": current_question["text"],
            "answer": answer,
            "verification": verification_result.verification_message
        }
        
        # Add verification message to conversation history
        state.conversation_history.append({
            "role": "ai", 
            "content": f"VERIFIED: {verification_result.verification_message}"
        })
    else:
        logger.info("Answer needs more info")
        # Add follow-up questions to conversation history
        state.conversation_history.append({
            "role": "ai", 
            "content": f"NEEDS MORE INFO: {verification_result.verification_message}"
        })
    
    # Update the state
    update_state_for_thread(thread_id, state)
    
    # Return verification results
    return {
        "success": True,
        "is_valid": verification_result.is_valid,
        "message": verification_result.verification_message,
        "missing_requirements": verification_result.missing_requirements
    }

def _perform_verification(
    question: Dict[str, Any],
    answer: str,
    conversation_history: List[Dict[str, str]],
    config: Optional[RunnableConfig] = None
) -> VerificationResult:
    """
    Helper function to perform the actual verification using the LLM.
    """
    logger.info("_perform_verification called")
    
    # Format requirements and conversation history for the LLM
    formatted_requirements = "\n".join([f"- {key}: {value}" for key, value in question['requirements'].items()])
    logger.debug(f"Formatted requirements: {formatted_requirements}")
    
    formatted_history = ""
    for msg in conversation_history:
        role = msg['role']
        content = msg['content']
        formatted_history += f"{role.upper()}: {content}\n\n"
    
    logger.debug(f"History length: {len(formatted_history)} characters")
    
    # Create messages for the LLM
    system_message = SystemMessage(content=
        "You are an expert medical validator for a questionnaire about polymicrogyria. "
        "Evaluate if the user's answer meets the specified requirements."
    )
    user_message = HumanMessage(content=
        f"QUESTION: {question['text']}\n\n"
        f"REQUIREMENTS:\n{formatted_requirements}\n\n"
        f"CONVERSATION HISTORY:\n{formatted_history}\n"
        f"USER'S ANSWER: {answer}\n\n"
        f"Evaluate if this answer meets all the requirements. If all requirements are met, "
        f"respond with a verification message. If any requirements are missing, ask specific "
        f"follow-up questions to get the missing information."
    )
    
    # Prepare the LLM with tools
    logger.info("Preparing LLM with tools for verification")
    llm_with_tool = llm.bind_tools(
        tools=[VerificationResult],
        tool_choice={"type": "function", "function": {"name": "VerificationResult"}}
    )
    
    try:
        # Get verification response from LLM
        logger.info("Calling LLM for verification")
        response = llm_with_tool.invoke([system_message, user_message], config=config)
        logger.debug(f"LLM response received: {response}")
        
        # Parse the verification result
        if response.tool_calls and len(response.tool_calls) > 0:
            logger.info("Processing tool call response")
            tool_call = response.tool_calls[0]
            result = VerificationResult(**tool_call["args"])
            logger.debug(f"Tool call result: is_valid={result.is_valid}, message={result.verification_message[:50]}...")
            return result
        
        # Handle direct response format as fallback
        if hasattr(response, "content") and response.content:
            logger.info("Processing direct content response")
            # Simple heuristic to determine validity from text
            is_valid = "valid" in response.content.lower() and "not valid" not in response.content.lower()
            return VerificationResult(
                is_valid=is_valid,
                verification_message=response.content,
                missing_requirements=[] if is_valid else ["Unable to determine specific missing requirements"]
            )
            
        # Default fallback
        logger.warning("No valid response format detected, using default fallback")
        return VerificationResult(
            is_valid=False,
            verification_message="Unable to verify your answer. Please provide more details.",
            missing_requirements=["Unable to determine specific missing requirements"]
        )
            
    except Exception as e:
        error_msg = f"Error in verification: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return VerificationResult(
            is_valid=False,
            verification_message=error_msg,
            missing_requirements=["Verification process encountered an error"]
        )