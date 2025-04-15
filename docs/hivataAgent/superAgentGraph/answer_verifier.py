import os
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from logging_config import logger
from langchain_openai import ChatOpenAI
from state import ChatState
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in your environment.")

llm = ChatOpenAI(
    model_name="gpt-4o",
    openai_api_key=openai_api_key,
    temperature=0
)

# Verification result model with a clear JSON schema.
class VerificationResult(BaseModel):
    is_valid: bool = Field(description="Whether the answer meets all requirements")
    missing_requirements: List[str] = Field(default_factory=list, description="Requirements not met")
    verification_message: str = Field(description="Verification or follow-up message")

# --- Prompt Template Helpers ---
def get_system_message() -> SystemMessage:
    """
    Returns the system message for answer verification.
    This message instructs the LLM to act as a medical validator.
    """
    content = (
        "You are an expert medical validator for a questionnaire about polymicrogyria. "
        "Your task is to evaluate if the user's answers collectively meet all specified requirements. "
        "Important: Consider ALL information provided across multiple user messages when evaluating. "
        "Users often provide answers in a conversational style across multiple messages. "
        "Be somewhat lenient but ensure all key information is present."
    )
    return SystemMessage(content=content)

def get_user_message(question: Dict[str, Any],
                     conversation_history: List[Dict[str, str]],
                     consolidated_answer: str) -> HumanMessage:
    """
    Returns the user message for answer verification.
    
    Combines the question text, formatted requirements, conversation history,
    and the consolidated answer into a single message for the LLM.
    
    Args:
        question: The question object containing 'text' and 'requirements'.
        conversation_history: A list of previous messages (as dicts).
        consolidated_answer: The consolidated answer from all human responses.
    
    Returns:
        A HumanMessage with the constructed content.
    """
    formatted_requirements = "\n".join(
        [f"- {key}: {value}" for key, value in question.get("requirements", {}).items()]
    )
    # Format the conversation history.
    formatted_history = ""
    for msg in conversation_history:
        role = msg.get("role", "").upper()
        content = msg.get("content", "")
        formatted_history += f"{role}: {content}\n\n"
        
    content = (
        f"QUESTION: {question.get('text', '')}\n\n"
        f"REQUIREMENTS:\n{formatted_requirements}\n\n"
        f"CONVERSATION HISTORY:\n{formatted_history}\n"
        f"USER'S CONSOLIDATED ANSWER: {consolidated_answer}\n\n"
        "Evaluate if the user's responses COLLECTIVELY meet all the requirements. "
        "Look for required information across ALL user messages in the conversation history. "
        "If the information is spread across multiple messages, that's acceptable. "
        "Explain why the answer is valid or specify exactly which requirements are still missing."
    )
    return HumanMessage(content=content)

def extract_consolidated_answer(conversation_history: List[Dict[str, str]]) -> str:
    """
    Extract and consolidate all human responses from the conversation history.
    
    Args:
        conversation_history: List of message dictionaries.
        
    Returns:
        A consolidated answer string.
    """
    question_index = -1
    for i, msg in enumerate(conversation_history):
        if msg.get('role') == 'ai':
            question_index = i
            break

    human_responses = [msg.get('content', '') for msg in conversation_history[question_index + 1:] if msg.get('role') == 'human']
    consolidated_answer = " ".join(human_responses)
    logger.debug(f"Consolidated answer from {len(human_responses)} human messages: {consolidated_answer[:50]}...")
    return consolidated_answer

# --- Main Verification Logic ---
def verify_answer(
    question: Dict[str, Any],
    answer: str,
    conversation_history: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Verifies if the user's answer meets the requirements.
    
    Args:
        question: The question object with requirements.
        answer: The most recent answer provided (for context).
        conversation_history: The full conversation history.
        
    Returns:
        A dictionary with keys 'is_valid' and 'verification_message'.
    """
    # Format requirements as bullet list.
    formatted_requirements = "\n".join([f"- {key}: {value}" for key, value in question.get('requirements', {}).items()])

    # Consolidate human responses.
    consolidated_answer = extract_consolidated_answer(conversation_history)

    # Build system and user messages using helper functions.
    system_message = get_system_message()
    user_message = get_user_message(question, conversation_history, consolidated_answer)

    logger.debug(f"Sending verification request with consolidated answer: {consolidated_answer[:50]}...")

    # Bind the VerificationResult tool to the LLM.
    llm_with_tool = llm.bind_tools(
        tools=[VerificationResult],
        tool_choice={"type": "function", "function": {"name": "VerificationResult"}}
    )

    try:
        response = llm_with_tool.invoke([system_message, user_message])
        # If the response includes a tool call with arguments, validate with our JSON schema.
        if response.tool_calls and len(response.tool_calls) > 0:
            tool_call = response.tool_calls[0]
            result = VerificationResult(**tool_call["args"])
            return {"is_valid": result.is_valid, "verification_message": result.verification_message}
        # Fallback: if a response content exists, use it with simple heuristics.
        if hasattr(response, "content") and response.content:
            is_valid = "valid" in response.content.lower() and "not valid" not in response.content.lower()
            return {"is_valid": is_valid, "verification_message": response.content}
        return {"is_valid": False, "verification_message": "Unable to verify your answer. Please provide more details."}
    except Exception as e:
        logger.error(f"Error in verification: {str(e)}")
        return {"is_valid": False, "verification_message": f"There was an error verifying your answer: {str(e)}"}

def verification_node(state: ChatState) -> Dict:
    """
    Graph node for verifying user answers.
    Updates state with verification results.
    (This node is kept for reference.)
    """
    idx = state.current_question_index
    logger.debug(f"Verification node for question index {idx}")
    
    answer = state.responses.get(idx, "")
    logger.debug(f"Verifying answer: {answer[:30]}...")
    
    current_question = state.questions[idx]
    verification_result = verify_answer(current_question, answer, state.conversation_history)
    verification_message = verification_result["verification_message"]
    is_valid = verification_result["is_valid"]
    
    logger.debug(f"Verification result: valid={is_valid}")
    
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