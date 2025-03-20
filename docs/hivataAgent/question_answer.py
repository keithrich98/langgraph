# question_answer.py - Modified from graph.py to work in the multi-agent system

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Import the shared state
from state import ChatState, add_to_extraction_queue

# Set up the LLM for verification
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in your environment.")
llm = ChatOpenAI(
    model_name="gpt-4",
    openai_api_key=openai_api_key,
    temperature=0  # Lower temperature for more consistent responses
)

# Verification result model for function calling
class VerificationResult(BaseModel):
    """Tool for verifying if an answer meets medical questionnaire requirements."""
    is_valid: bool = Field(
        description="Whether the answer meets all requirements"
    )
    missing_requirements: List[str] = Field(
        description="List of requirements that weren't met", 
        default_factory=list
    )
    verification_message: str = Field(
        description="Explanation or follow-up questions for the user"
    )

def verify_answer(question: Dict[str, Any], answer: str, conversation_history: List[Dict[str, str]], config: Optional[RunnableConfig] = None) -> (bool, str):
    """
    Uses LangChain function calling with Pydantic models to verify if the answer meets requirements.
    Returns a tuple of (is_valid, verification_message).
    """
    # Format the requirements as bullet points for clarity
    formatted_requirements = "\n".join([f"- {key}: {value}" for key, value in question['requirements'].items()])
    
    # Format the conversation history
    formatted_history = ""
    for msg in conversation_history:
        role = msg['role']
        content = msg['content']
        formatted_history += f"{role.upper()}: {content}\n\n"
    
    # Create system message
    system_message = SystemMessage(content=
        "You are an expert medical validator for a questionnaire about polymicrogyria. "
        "Your task is to evaluate if a user's answer meets the specified requirements. "
        "Be thorough but fair in your assessment."
    )
    
    # Create user message with all context
    user_message = HumanMessage(content=
        f"QUESTION: {question['text']}\n\n"
        f"REQUIREMENTS:\n{formatted_requirements}\n\n"
        f"CONVERSATION HISTORY:\n{formatted_history}\n"
        f"USER'S ANSWER: {answer}\n\n"
        f"Evaluate if this answer meets all the requirements. If it's valid, explain why. "
        f"If it's invalid, specify which requirements weren't met and provide helpful follow-up questions."
    )
    
    # Bind the tool (function) to the LLM 
    llm_with_verify_tool = llm.bind_tools(
        tools=[VerificationResult],
        tool_choice={"type": "function", "function": {"name": "VerificationResult"}}
    )
    
    try:
        # Invoke the model with function calling
        response = llm_with_verify_tool.invoke([system_message, user_message], config=config)
        
        # Check if we got tool calls
        if response.tool_calls and len(response.tool_calls) > 0:
            # Extract the verification result
            tool_call = response.tool_calls[0]
            
            # Use Pydantic to validate and create a proper object
            result = VerificationResult(**tool_call["args"])
            
            # Return the verification result
            return result.is_valid, result.verification_message
        
        # Fallback if no tool calls were made (rare case)
        if hasattr(response, "content") and response.content:
            # Try to infer validity from content
            is_valid = "valid" in response.content.lower() and "not valid" not in response.content.lower()
            return is_valid, response.content
            
        # Ultimate fallback
        return False, "Unable to verify your answer. Please provide more details."
        
    except Exception as e:
        print(f"Error in verification: {str(e)}")
        return False, f"There was an error verifying your answer. Please try again with more details."

# Questions list
def get_questions():
    return [
        {
            "text": "At what age were you diagnosed with polymicrogyria, and what were the primary signs or symptoms?",
            "requirements": {
                "age": "Provide age (based on birthdate)",
                "diagnosis_date": "Provide the date of diagnosis",
                "symptoms": "Describe the key signs and symptoms"
            }
        },
        {
            "text": "What symptoms or neurological issues do you experience, and how would you rate their severity?",
            "requirements": {
                "symptoms": "List each symptom or neurological issue experienced",
                "severity": "Include a severity rating for each (mild, moderate, severe)",
                "context": "Provide any additional context about how symptoms impact daily life"
            }
        },
        {
            "text": "Can you describe the key findings from your brain imaging studies (MRI/CT)?",
            "requirements": {
                "imaging_modality": "Specify the imaging modality used (MRI, CT, etc.)",
                "findings": "Detail the main imaging findings",
                "remarks": "Include any remarks from radiology reports"
            }
        }
    ]

# Define a separate checkpointer for the question_answer workflow
question_answer_memory = MemorySaver()

# The workflow with modifications to work in the multi-agent system
@entrypoint(checkpointer=question_answer_memory)
def question_answer_workflow(action: Dict = None, *, previous: ChatState = None, config: Optional[RunnableConfig] = None) -> ChatState:
    """
    A questionnaire workflow with a verification step using LLM function calling.
    
    Actions:
      {"action": "start"} - Start a new questionnaire session.
      {"action": "answer", "answer": "user answer"} - Submit an answer.
    
    Modified to add verified answers to the extraction queue for the term extractor.
    """
    # Use previous state or initialize a new one
    state = previous if previous is not None else ChatState(questions=get_questions())
    
    if action is None:
        # No action provided, just return current state
        return state
    
    if action.get("action") == "start":
        # Initialize the session
        state.current_question_index = 0
        state.is_complete = False
        state.responses = {}
        state.verified_answers = {}
        state.term_extraction_queue = []
        state.extracted_terms = {}
        
        question = state.questions[0]
        # Format requirements as bullet points for better readability
        formatted_requirements = "\n".join([f"- {key}: {value}" for key, value in question['requirements'].items()])
        prompt = f"{question['text']}\n\nRequirements:\n{formatted_requirements}"
        state.conversation_history = [{"role": "ai", "content": prompt}]
    
    elif action.get("action") == "answer" and not state.is_complete:
        answer = action.get("answer", "")
        # Append the user's answer to the conversation
        state.conversation_history.append({"role": "human", "content": answer})
        
        # Perform verification using the LLM with function calling
        current_question = state.questions[state.current_question_index]
        is_valid, verification_message = verify_answer(current_question, answer, state.conversation_history, config)
        
        # Append the verification message from the LLM to the conversation
        state.conversation_history.append({"role": "ai", "content": verification_message})
        
        if is_valid:
            # Valid answer: store and move to the next question
            state.responses[state.current_question_index] = answer
            
            # Add to verified answers for term extraction
            state.verified_answers[state.current_question_index] = {
                "question": current_question["text"],
                "answer": answer,
                "verification": verification_message
            }
            
            # Add to the extraction queue for the term extractor
            add_to_extraction_queue(state, state.current_question_index)
            
            state.current_question_index += 1
            if state.current_question_index >= len(state.questions):
                state.is_complete = True
                state.conversation_history.append({
                    "role": "ai", 
                    "content": "Thank you for completing all the questions. Your responses have been recorded."
                })
            else:
                # Present the next question
                next_question = state.questions[state.current_question_index]
                # Format requirements as bullet points for better readability
                formatted_requirements = "\n".join([f"- {key}: {value}" for key, value in next_question['requirements'].items()])
                prompt = f"{next_question['text']}\n\nRequirements:\n{formatted_requirements}"
                state.conversation_history.append({"role": "ai", "content": prompt})
        else:
            # If the answer is invalid, the follow-up questions remain visible.
            # The same question remains active so the user can try answering again.
            pass
    
    return state