# answer_verifier.py - Refactored for StateGraph API

import os
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

# Import the logger from logging_config
from logging_config import logger

# Set up the LLM for verification
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in your environment.")
llm = ChatOpenAI(
    model_name="gpt-4o",
    openai_api_key=openai_api_key,
    temperature=0
)

# Verification result model
class VerificationResult(BaseModel):
    is_valid: bool = Field(description="Whether the answer meets all requirements")
    missing_requirements: List[str] = Field(default_factory=list, description="Requirements not met")
    verification_message: str = Field(description="Verification or follow-up message")

def extract_consolidated_answer(conversation_history: List[Dict[str, str]]) -> str:
    """
    Extract and consolidate all human responses from the conversation history.
    Combines multiple human messages into a single comprehensive answer.
    
    Args:
        conversation_history: List of message dictionaries
        
    Returns:
        str: Consolidated answer from all human messages
    """
    # Find the first AI message (which should be the question)
    question_index = -1
    for i, msg in enumerate(conversation_history):
        if msg.get('role') == 'ai':
            question_index = i
            break
    
    # Extract all human messages that come after the question
    human_responses = []
    for msg in conversation_history[question_index+1:]:
        if msg.get('role') == 'human':
            human_responses.append(msg.get('content', ''))
    
    # Combine all human responses
    consolidated_answer = " ".join(human_responses)
    logger.debug(f"Consolidated answer from {len(human_responses)} human message(s): {consolidated_answer[:50]}...")
    
    return consolidated_answer

def verify_answer(
    question: Dict[str, Any],
    answer: str,
    conversation_history: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Verify if the user's answer meets the requirements of the question.
    
    Args:
        question: The question with requirements
        answer: The most recent user answer (may be used for logging, but not for verification)
        conversation_history: Complete history of the conversation
        
    Returns:
        Dict with is_valid and verification_message
    """
    # Format the requirements as a bullet list
    formatted_requirements = "\n".join([f"- {key}: {value}" for key, value in question['requirements'].items()])
    
    # Format the conversation history for context
    formatted_history = ""
    for msg in conversation_history:
        role = msg['role']
        content = msg['content']
        formatted_history += f"{role.upper()}: {content}\n\n"
    
    # Extract a consolidated answer from all human responses
    consolidated_answer = extract_consolidated_answer(conversation_history)
    
    # Create an improved system message with clearer instructions
    system_message = SystemMessage(content=
        "You are an expert medical validator for a questionnaire about polymicrogyria. "
        "Your task is to evaluate if the user's answers collectively meet all specified requirements. "
        "Important: Consider ALL information provided across multiple user messages when evaluating. "
        "Users often provide answers in a conversational style across multiple messages. "
        "Be somewhat lenient but ensure all key information is present."
    )
    
    # Create an improved user message that emphasizes considering the complete context
    user_message = HumanMessage(content=
        f"QUESTION: {question['text']}\n\n"
        f"REQUIREMENTS:\n{formatted_requirements}\n\n"
        f"CONVERSATION HISTORY:\n{formatted_history}\n"
        f"USER'S CONSOLIDATED ANSWER: {consolidated_answer}\n\n"
        f"Evaluate if the user's responses COLLECTIVELY meet all the requirements. "
        f"Look for required information across ALL user messages in the conversation history. "
        f"If the information is spread across multiple messages, that's perfectly acceptable. "
        f"Explain why the answer is valid or specify exactly which requirements are still missing."
    )
    
    logger.debug(f"Sending verification request with consolidated answer: {consolidated_answer[:50]}...")
    
    # Bind the VerificationResult tool to the LLM
    llm_with_tool = llm.bind_tools(
        tools=[VerificationResult],
        tool_choice={"type": "function", "function": {"name": "VerificationResult"}}
    )
    
    try:
        response = llm_with_tool.invoke([system_message, user_message])
        if response.tool_calls and len(response.tool_calls) > 0:
            tool_call = response.tool_calls[0]
            result = VerificationResult(**tool_call["args"])
            return {"is_valid": result.is_valid, "verification_message": result.verification_message}
        if hasattr(response, "content") and response.content:
            is_valid = "valid" in response.content.lower() and "not valid" not in response.content.lower()
            return {"is_valid": is_valid, "verification_message": response.content}
        return {"is_valid": False, "verification_message": "Unable to verify your answer. Please provide more details."}
    except Exception as e:
        logger.error(f"Error in verification: {str(e)}")
        return {"is_valid": False, "verification_message": f"There was an error verifying your answer: {str(e)}"}

def verification_node(state: Dict) -> Dict:
    """
    Graph node for verifying user answers.
    Updates the state with the answer and verification results.
    
    Note: This function is not used in the current implementation. The version in
    parent_workflow.py is used instead. This is kept for reference.
    """
    # Log key details of initial state
    idx = state["current_question_index"]
    logger.debug(f"Verification node for question index {idx}")
    
    # Get the latest response (should be from the answer_node)
    answer = state["responses"].get(idx, "")
    
    # Log key details after human answer
    logger.debug(f"Verifying answer: {answer[:30]}...")
    
    # Get the current question and verify the answer
    current_question = state["questions"][idx]
    
    # When used with a state schema that has add_messages reducer, this would be correct:
    from langchain_core.messages import AIMessage
    verification_result = verify_answer(current_question, answer, state["conversation_history"])
    verification_message = verification_result["verification_message"]
    is_valid = verification_result["is_valid"]
    
    logger.debug(f"Verification result: valid={is_valid}")
    
    # Add the verification message to the conversation history
    # Using AIMessage format to work with the add_messages reducer
    state_update = {
        "conversation_history": [AIMessage(content=verification_message)],
        "verification_result": {
            "action": "verified_answer",
            "question_index": idx,
            "answer": answer,
            "verification_message": verification_message,
            "is_valid": is_valid
        }
    }
    
    return state_update