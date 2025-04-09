# api.py with HIL API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from typing import Dict, Optional, List, Any
import uvicorn

# Import our graph and ChatState from parent_workflow.py
from parent_workflow import graph, ChatState, init_node
from langgraph.types import Command  # Import Command for resuming execution
from logging_config import logger

app = FastAPI()

# Configure CORS for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StartResponse(BaseModel):
    session_id: str
    conversation_history: List[Dict[str, str]]
    current_question: str
    waiting_for_input: bool

class NextRequest(BaseModel):
    session_id: str
    answer: str

class NextResponse(BaseModel):
    conversation_history: List[Dict[str, str]]
    current_question: Optional[str] = None
    finished: bool
    waiting_for_input: bool

def convert_messages_to_dict(messages):
    """
    Convert LangChain message objects to dictionaries.
    
    Ensures consistent conversion from various message formats to a standardized dict format.
    Handles both objects with attributes and dictionary-style messages.
    
    Args:
        messages: A list of message objects (AIMessage, HumanMessage, dict, etc.)
        
    Returns:
        list: A list of dicts with 'role' and 'content' keys
    """
    result = []
    if not messages:
        return result
        
    for msg in messages:
        # Handle dict-style messages
        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
            # Already in the right format
            result.append(msg.copy())  # Use copy to avoid mutating original
            continue
            
        # Handle message objects with attributes
        if hasattr(msg, 'content'):
            content = msg.content
            
            # Determine the role with fallbacks
            if hasattr(msg, 'type'):
                role = msg.type.lower()
            elif hasattr(msg, '_message_type'):
                msg_type = msg._message_type.lower() if hasattr(msg, '_message_type') else ''
                if 'ai' in msg_type or 'assistant' in msg_type:
                    role = 'ai'
                elif 'human' in msg_type or 'user' in msg_type:
                    role = 'human'
                else:
                    role = 'ai'  # Default fallback
            else:
                # Try to infer from class name
                class_name = msg.__class__.__name__.lower()
                if 'ai' in class_name or 'assistant' in class_name:
                    role = 'ai'
                elif 'human' in class_name or 'user' in class_name:
                    role = 'human'
                else:
                    role = 'ai'  # Default fallback
                    
            result.append({"role": role, "content": content})
            continue
            
        # Log any unhandled message types
        logger.warning(f"Unhandled message type: {type(msg)}")
        
    logger.debug(f"Converted {len(messages)} messages to {len(result)} dict messages")
    return result

def get_current_question(conversation_history: List) -> str:
    """Extract the latest AI message from the conversation history."""
    # First convert any message objects to dictionaries
    dict_history = convert_messages_to_dict(conversation_history)
    
    # Then extract the last AI message
    for msg in reversed(dict_history):
        if msg.get("role") == "ai":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""

@app.post("/start", response_model=StartResponse)
def start_session():
    """Start a new questionnaire session."""
    # Generate a unique thread ID
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        logger.debug(f"Starting new session with thread_id: {thread_id}")
        
        # Initialize with minimal required state
        initial_state = {
            "current_question_index": 0,
            "questions": [],
            "conversation_history": [],
            "responses": {},
            "is_complete": False,
            "verified_answers": {},
            "term_extraction_queue": [],
            "extracted_terms": {},
            "last_extracted_index": -1,
            "verification_result": {}
        }
        logger.debug(f"Created initial state: {initial_state}")
        
        # Run the graph until it hits the first interrupt
        logger.debug("About to invoke graph")
        result = graph.invoke(initial_state, config)
        logger.debug(f"Graph invoke result type: {type(result)}")
        logger.debug(f"Graph invoke result: {result}")
        
        # Get the current state
        logger.debug("Getting current state")
        current_state = graph.get_state(config)
        logger.debug(f"Current state type: {type(current_state)}")
        logger.debug(f"Current state attributes: {dir(current_state)}")
        logger.debug(f"Current state values type: {type(current_state.values)}")
        logger.debug(f"Current state values keys: {current_state.values.keys() if hasattr(current_state.values, 'keys') else 'No keys method'}")
        
        # Check if waiting for input
        waiting_for_input = False
        if current_state.tasks and len(current_state.tasks) > 0:
            task = current_state.tasks[0]
            logger.debug(f"Task: {task}")
            if hasattr(task, 'interrupts') and task.interrupts:
                waiting_for_input = True
        
        # Get conversation history
        state_values = current_state.values
        logger.debug(f"State values: {state_values}")
        
        # Try different ways to access conversation history
        raw_conversation_history = []
        if isinstance(state_values, dict):
            logger.debug("State values is a dict, using get method")
            raw_conversation_history = state_values.get('conversation_history', [])
        elif hasattr(state_values, 'conversation_history'):
            logger.debug("State values has conversation_history attribute")
            raw_conversation_history = state_values.conversation_history
        else:
            logger.debug("Couldn't find conversation_history, using empty list")
        
        # Convert message objects to dictionaries
        conversation_history = convert_messages_to_dict(raw_conversation_history)
        logger.debug(f"Converted conversation history: {conversation_history}")
        
        # Get the current question
        current_question = get_current_question(raw_conversation_history)
        
        logger.debug(f"Session started successfully, conversation history length: {len(conversation_history)}")
        
        return StartResponse(
            session_id=thread_id,
            conversation_history=conversation_history,
            current_question=current_question,
            waiting_for_input=waiting_for_input
        )
    except Exception as e:
        import traceback
        logger.error(f"Error starting session: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")

@app.post("/next", response_model=NextResponse)
def next_step(req: NextRequest):
    """Process the user's answer and return the next question."""
    logger.debug(f"Processing next step for session: {req.session_id}")
    config = {"configurable": {"thread_id": req.session_id}}
    
    try:
        # Resume execution with the user's answer
        graph.invoke(Command(resume=req.answer), config)
        
        # Get the updated state
        updated_state = graph.get_state(config)
        
        # Check if waiting for input
        waiting_for_input = False
        if updated_state.tasks and len(updated_state.tasks) > 0:
            task = updated_state.tasks[0]
            if hasattr(task, 'interrupts') and task.interrupts:
                waiting_for_input = True
        
        # Check if finished
        finished = not waiting_for_input and not updated_state.next
        
        # Get conversation history and current question
        state_values = updated_state.values
        # Use get() method since state_values is a dictionary
        raw_conversation_history = state_values.get('conversation_history', [])
        
        # Convert message objects to dictionaries
        conversation_history = convert_messages_to_dict(raw_conversation_history)
        
        current_question = get_current_question(raw_conversation_history)
        
        logger.debug(f"Next step processed, waiting_for_input: {waiting_for_input}, finished: {finished}")
        
        return NextResponse(
            conversation_history=conversation_history,
            current_question=current_question,
            finished=finished,
            waiting_for_input=waiting_for_input
        )
    except Exception as e:
        import traceback
        logger.error(f"Error processing answer: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing answer: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)