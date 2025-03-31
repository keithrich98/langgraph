# parent_supervisor.py (revised version)
import os
from typing import List, Dict, Any, Optional
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from logging_config import logger

# Import State and Questions
from state import ChatState, get_state_for_thread, update_state_for_thread
from question_agent import get_questions, ask_question_tool, advance_to_next_question
from verification_agent import verify_answer_task
from dotenv import load_dotenv
load_dotenv()

# Setup API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in your environment.")

# Setup model for supervisor and agents
llm = ChatOpenAI(
    model_name="gpt-4",
    openai_api_key=openai_api_key,
    temperature=0
)

# Setup memory
memory_saver = MemorySaver()

# Question Agent
question_system_prompt = """
You are a helpful medical questionnaire assistant. Your job is to ask questions to a patient
about their medical condition, specifically focusing on polymicrogyria. 
Ask one question at a time and don't move to the next question until the current one is complete.
"""

question_agent = create_react_agent(
    model=llm,
    tools=[ask_question_tool, advance_to_next_question],
    name="question_agent",
    prompt=question_system_prompt
)

# Verification Agent
verification_system_prompt = """
You are an expert medical verification agent that evaluates patient responses to ensure they've
provided all the required information. Carefully check if their answers meet all 
requirements for the current question. If they don't, ask follow-up questions to get
the missing information. Only mark answers as verified when all requirements are met.
"""

verification_agent = create_react_agent(
    model=llm,
    tools=[verify_answer_task],
    name="verification_agent",
    prompt=verification_system_prompt
)

# Supervisor Agent
supervisor_system_prompt = """
You are a medical questionnaire supervisor that coordinates the questionnaire process.
You have two agents at your disposal:

1. question_agent: Use this agent to ask the patient a new question.
2. verification_agent: Use this agent to verify if the patient's answer meets all requirements.

Your workflow:
1. Start with the question_agent to ask the first question.
2. When the patient answers, determine if verification is needed:
   - If the answer indicates the question is not applicable (e.g., "no" to "Do you have headaches?"), 
     skip verification and use question_agent to move to the next question.
   - Otherwise, use verification_agent to confirm the answer meets all requirements.
3. If verification_agent indicates more information is needed, it will handle follow-up questions.
4. Once verification_agent confirms an answer is complete, use question_agent to move to the next question.
5. Repeat this process until all questions are answered.

Be conversational and supportive, but focus on efficiently completing the questionnaire.
"""

# Create the supervisor workflow
supervisor = create_supervisor(
    [question_agent, verification_agent],
    model=llm,
    prompt=supervisor_system_prompt,
    output_mode="last_message",
    supervisor_name="medical_supervisor",
)

# Compile the workflow with the memory saver
medical_questionnaire = supervisor.compile(checkpointer=memory_saver, name="medical_questionnaire")

def start_questionnaire(thread_id: str) -> Dict:
    """
    Start a new questionnaire session.
    
    This function takes a deterministic approach for the first question
    to avoid unnecessary LLM calls.
    """
    logger.info(f"Starting new questionnaire session with thread_id: {thread_id}")
    
    # Directly initialize a new state with questions
    questions = get_questions()
    state = ChatState(
        questions=questions,
        current_question_index=0,
        is_complete=False,
        conversation_history=[]
    )
    
    # Format the first question with requirements
    first_question = questions[0]
    formatted_requirements = "\n".join([f"- {k}: {v}" for k, v in first_question["requirements"].items()])
    question_text = f"{first_question['text']}\n\nRequirements:\n{formatted_requirements}"
    
    # Add the question directly to conversation history
    state.conversation_history.append({
        "role": "ai", 
        "content": question_text
    })
    
    # Save the state
    update_state_for_thread(thread_id, state)
    
    logger.info(f"State initialized with first question for thread_id: {thread_id}")
    logger.debug(f"First question: {question_text[:50]}...")
    
    return {
        "conversation_history": state.conversation_history,
        "current_question_index": state.current_question_index,
        "is_complete": state.is_complete
    }

def process_answer(thread_id: str, answer: str) -> Dict:
    """
    Process a user answer within the questionnaire.
    
    This function invokes the multi-agent workflow (medical_questionnaire) which includes the
    verification_agent. If the verification_agent confirms that the answer meets all requirements,
    the current question index is advanced and the next question is appended to the conversation history.
    Otherwise, if verification is unsuccessful or incomplete, the function retains the current question
    and only adds the follow-up verification message to prompt the human for more information.
    """
    logger.info(f"Processing answer for thread_id: {thread_id}")
    logger.debug(f"Answer: {answer}")
    
    # Retrieve the existing state for the conversation
    state = get_state_for_thread(thread_id)
    if not state:
        logger.error(f"No active session found with thread_id: {thread_id}")
        raise ValueError(f"No active session found with thread_id: {thread_id}")
    
    # Create a user message from the human's answer
    user_message = {
        "role": "user",
        "content": answer
    }
    
    # Append the user's answer to the current conversation history and update state
    existing_messages = state.conversation_history + [user_message]
    state.conversation_history = existing_messages
    update_state_for_thread(thread_id, state)
    
    # Invoke the multi-agent workflow with the full conversation history
    logger.info("Invoking medical_questionnaire workflow with user answer for verification")
    config = {"configurable": {"thread_id": thread_id}}
    result = medical_questionnaire.invoke({"messages": existing_messages}, config)
    
    # Initialize a new conversation history to build our updated history
    conversation_history = []
    # Variables to track the verification outcome
    verification_complete = False
    verification_successful = False
    
    # Process each message returned from the workflow
    for msg in result.get("messages", []):
        # Handle human messages
        if isinstance(msg, HumanMessage):
            conversation_history.append({"role": "human", "content": msg.content})
        # Handle AI messages
        elif isinstance(msg, AIMessage):
            content = msg.content if msg.content else ""
            conversation_history.append({"role": "ai", "content": content})
            # Check for cues that indicate verification status
            if "could you also provide" in content.lower() or "need more information" in content.lower():
                verification_complete = True
                verification_successful = False
                logger.info("Verification incomplete - more information needed")
            elif "verified" in content.lower() or "all requirements are met" in content.lower():
                verification_complete = True
                verification_successful = True
                logger.info("Verification completed successfully")
        # Handle tool messages
        elif isinstance(msg, ToolMessage):
            conversation_history.append({"role": "system", "content": msg.content})
        # Handle messages that are dictionaries
        elif isinstance(msg, dict):
            conversation_history.append(msg)
            content = msg.get("content", "")
            if "could you also provide" in content.lower() or "need more information" in content.lower():
                verification_complete = True
                verification_successful = False
                logger.info("Verification incomplete - more information needed")
            elif "verified" in content.lower() or "all requirements are met" in content.lower():
                verification_complete = True
                verification_successful = True
                logger.info("Verification completed successfully")
    
    # Only if verification is successful, advance to the next question.
    if verification_successful:
        logger.info("Verification successful, advancing to next question")
        # Record the verified answer in state
        state.responses[state.current_question_index] = answer
        # Advance to the next question index
        state.current_question_index += 1
        logger.info(f"Advanced to question index: {state.current_question_index}")
        
        # If there are no more questions, mark the session complete and add a completion message.
        if state.current_question_index >= len(state.questions):
            logger.info("Reached end of questions, marking as complete")
            state.is_complete = True
            update_state_for_thread(thread_id, state)
            completion_message = {
                "role": "ai", 
                "content": "All questions have been completed. Thank you for participating in this questionnaire."
            }
            conversation_history.append(completion_message)
        else:
            # If more questions remain, format and append the next question
            next_question = state.questions[state.current_question_index]
            formatted_requirements = "\n".join([f"- {k}: {v}" for k, v in next_question["requirements"].items()])
            next_question_text = f"{next_question['text']}\n\nRequirements:\n{formatted_requirements}"
            next_question_message = {
                "role": "ai", 
                "content": next_question_text
            }
            conversation_history.append(next_question_message)
    else:
        # If verification is not successful, do not advance the question index.
        # Retain the current question and the verification follow-up messages.
        logger.info("Verification unsuccessful or incomplete; retaining the current question")
        # (Optionally, you may choose to filter out any accidental advancement messages here)
    
    # Update state with the new conversation history
    state.conversation_history = conversation_history
    update_state_for_thread(thread_id, state)
    
    # Retrieve the updated state for consistency and return the necessary information
    updated_state = get_state_for_thread(thread_id)
    
    return {
        "conversation_history": conversation_history,
        "current_question_index": updated_state.current_question_index if updated_state else 0,
        "is_complete": updated_state.is_complete if updated_state else False
    }


def get_questionnaire_state(thread_id: str) -> Dict:
    """
    Get the current state of a questionnaire session.
    """
    logger.info(f"Getting questionnaire state for thread_id: {thread_id}")
    
    state = get_state_for_thread(thread_id)
    if not state:
        logger.warning(f"No state found for thread_id: {thread_id}")
        return {
            "error": "No session found",
            "conversation_history": [],
            "current_question_index": 0,
            "is_complete": False
        }
    
    # No need to convert message objects since we're storing them directly as dictionaries
    return {
        "conversation_history": state.conversation_history,
        "current_question_index": state.current_question_index,
        "is_complete": state.is_complete,
        "responses": state.responses
    }