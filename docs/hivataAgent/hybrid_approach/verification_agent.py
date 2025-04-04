# verification_agent.py
from langgraph.func import task
from typing import Dict, Any, List, Tuple
from state import SessionState, get_current_question, add_to_extraction_queue
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Import logger
from logging_config import logger

# Load environment variables from .env file
load_dotenv()

# Setup Azure OpenAI
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Check that variables are loaded
if None in [azure_api_key, azure_endpoint, azure_deployment, azure_api_version]:
    logger.warning("Azure environment variables are not fully set. Will use fallback verification.")
    use_llm = False
else:
    # Initialize the AzureChatOpenAI LLM
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        api_key=azure_api_key,
        api_version=azure_api_version,
        temperature=0
    )
    use_llm = True
    logger.info("Successfully initialized Azure OpenAI for verification")

@task
def verify_answer(state: SessionState) -> SessionState:
    """
    Verify if the user's answer meets the requirements of the current question.
    
    Uses an LLM to check if the answer meets all specified requirements when available.
    Falls back to simple verification if LLM is not configured.
    
    Important: This considers the ENTIRE conversation context, not just the most recent answer.
    """
    current_index = state.current_index
    logger.info(f"Starting verification for answer to question {current_index}")
    
    question = get_current_question(state)
    if not question:
        logger.warning(f"No question found for index {current_index}, verification skipped")
        return state
    
    if current_index not in state.responses:
        logger.warning(f"No answer found for question {current_index}, verification skipped")
        return state
    
    # Instead of verifying only the most recent answer, we'll consider the entire
    # conversation history for this question
    current_answer = state.responses[current_index]
    logger.debug(f"Verifying answer for question {current_index}: {current_answer[:50]}...")
    
    # Get the requirements for the current question
    requirements = question.get("requirements", {})
    logger.debug(f"Question has {len(requirements)} requirements: {list(requirements.keys())}")
    
    # Use LLM for verification if available, otherwise use simple verification
    if use_llm:
        logger.info("Using LLM for answer verification with conversation context")
        is_valid, verification_message = llm_verification_with_context(
            current_answer, 
            requirements, 
            question["text"],
            state.conversation_history
        )
    else:
        logger.info("Using simple verification (LLM not available)")
        is_valid, verification_message = simple_verification(current_answer, requirements)
    
    # Log the verification result
    if is_valid:
        logger.info(f"Answer to question {current_index} is VALID")
    else:
        logger.info(f"Answer to question {current_index} is INVALID")
        logger.debug(f"Verification message: {verification_message}")
    
    # Store verification results
    state.verified_responses[current_index] = is_valid
    state.verification_messages[current_index] = verification_message
    
    # Add verification message to conversation history
    state.conversation_history.append({"role": "system", "content": verification_message})
    logger.debug("Added verification message to conversation history")
    
    # If valid, add to extraction queue for future term extraction
    if is_valid:
        logger.debug(f"Adding question {current_index} to term extraction queue")
        state = add_to_extraction_queue(state, current_index)
    
    # Store the verification result for the parent workflow
    state.verification_result = {
        "action": "verified_answer",
        "question_index": current_index,
        "answer": current_answer,
        "verification_message": verification_message,
        "is_valid": is_valid
    }
    
    logger.debug("Updated state with verification result")
    return state

def llm_verification_with_context(
    current_answer: str, 
    requirements: Dict[str, str], 
    question_text: str,
    conversation_history: List[Dict[str, str]]
) -> Tuple[bool, str]:
    """
    Use LLM to verify if the requirements are met, considering the full conversation context.
    
    Args:
        current_answer: The user's most recent answer text
        requirements: Dictionary of requirements (key: name, value: description)
        question_text: The original question text
        conversation_history: Full conversation history for context
    
    Returns:
        Tuple of (is_valid, verification_message)
    """
    try:
        # Format conversation history for context
        context = []
        for i, msg in enumerate(conversation_history):
            if msg["role"] == "system" and "Requirements:" in msg["content"] and i == 0:
                # This is the original question
                continue
            elif msg["role"] == "system" and "Analysis of Patient's Answer" in msg["content"]:
                # This is a verification message, skip it
                continue
            else:
                # Add to context
                context.append(f"{msg['role'].upper()}: {msg['content']}")
        
        # Build context string
        context_str = "\n".join(context)
        
        # Construct prompt for the LLM
        requirements_text = "\n".join([f"- {key}: {desc}" for key, desc in requirements.items()])
        
        system_message = f"""
        You are a medical questionnaire verification assistant. Your task is to verify that a patient's answers
        collectively meet all the requirements for a specific question.
        
        You should consider the ENTIRE conversation history, not just the most recent answer.
        Previous answers in the conversation may contain information that satisfies some requirements.
        """
        
        user_message = f"""
        Question: {question_text}
        
        Requirements:
        {requirements_text}
        
        Conversation History (includes question and all answers):
        {context_str}
        
        Most recent answer: "{current_answer}"
        
        Analyze ALL information provided in the conversation history and determine if ALL requirements have been met.
        For each requirement, check if the information has been provided anywhere in the conversation.
        
        First, list each requirement and whether it was met or not met.
        Then, provide your overall assessment (valid or invalid).
        If any requirements are not met, explain what specific information is still missing.
        If all requirements are met, provide a positive acknowledgment.
        """
        
        # Call the LLM with system and user messages
        logger.debug("Sending verification request to LLM with conversation context")
        
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]
        
        response = llm.invoke(messages)
        verification_message = response.content.strip()
        logger.debug(f"Received LLM verification response: {verification_message[:100]}...")
        
        # Check if the response indicates the answer is valid
        # Look for keywords indicating validity
        is_valid = any(phrase in verification_message.lower() for phrase in [
            "meets all requirement", 
            "all requirements are met",
            "all requirements have been met",
            "valid", 
            "all criteria are satisfied"
        ]) and not any(phrase in verification_message.lower() for phrase in [
            "invalid", 
            "not met", 
            "missing", 
            "incomplete",
            "does not provide"
        ])
        
        return is_valid, verification_message
    
    except Exception as e:
        logger.error(f"Error during LLM verification: {str(e)}", exc_info=True)
        # Fall back to simple verification
        logger.warning("Falling back to simple verification due to LLM error")
        return simple_verification(current_answer, requirements)

def simple_verification(answer: str, requirements: Dict[str, str]) -> Tuple[bool, str]:
    """
    Simple verification logic as a backup when LLM is unavailable.
    
    This is a placeholder that performs very basic checks.
    """
    logger.debug("Running simple verification checks")
    
    # Check if the answer is too short to be meaningful
    if len(answer.strip()) < 20:
        logger.debug("Answer is too brief (< 20 chars)")
        return False, "Your answer is too brief. Please provide more detailed information that meets all requirements."
    
    missing_requirements = []
    for req_key, req_description in requirements.items():
        # Very simplistic check - in reality, an LLM would do this much better
        if req_key.lower() not in answer.lower():
            logger.debug(f"Requirement '{req_key}' not found in answer")
            missing_requirements.append(f"{req_key}: {req_description}")
    
    if missing_requirements:
        logger.debug(f"Found {len(missing_requirements)} missing requirements")
        missing_text = "\n".join([f"- {item}" for item in missing_requirements])
        return False, f"Your answer appears to be missing the following required information:\n{missing_text}\n\nPlease provide a more complete answer."
    
    logger.debug("All requirements appear to be met")
    return True, "Thank you. Your answer meets all the requirements."