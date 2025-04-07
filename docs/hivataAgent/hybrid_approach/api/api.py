# api.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
import uvicorn
import time

# Import the parent workflow functions
from hivataAgent.hybrid_approach.core.parent_workflow import start_session, process_user_answer, get_session_state
from hivataAgent.hybrid_approach.services.thread_manager import thread_manager

# Import logger
from hivataAgent.hybrid_approach.config.logging_config import logger

app = FastAPI(title="Medical Questionnaire API", 
              description="API for a questionnaire with verification")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request and response models
class StartSessionResponse(BaseModel):
    """Response model for starting a new session."""
    session_id: str
    current_question: Optional[str] = None
    conversation_history: List[Dict[str, str]]
    is_complete: bool

class AnswerRequest(BaseModel):
    """Request model for submitting an answer."""
    session_id: str
    answer: str

class AnswerResponse(BaseModel):
    """Response model for answer processing."""
    session_id: str
    current_question: Optional[str] = None
    conversation_history: List[Dict[str, str]]
    is_complete: bool
    verified_responses: Dict[str, bool] = {}  # Use string keys
    verification_messages: Dict[str, str] = {}  # Use string keys
    verified_answers: Dict[str, Dict[str, str]] = {}  # Use string keys
    extracted_terms: Dict[str, List[str]] = {}  # Use string keys
    term_extraction_queue: List[int] = []

class SessionStateResponse(BaseModel):
    """Response model for session state retrieval."""
    session_id: str
    current_index: int
    conversation_history: List[Dict[str, str]]
    is_complete: bool
    responses: Dict[str, str]  # Use string keys
    verified_responses: Dict[str, bool] = {}  # Use string keys
    verification_messages: Dict[str, str] = {}  # Use string keys
    verified_answers: Dict[str, Dict[str, str]] = {}  # Use string keys
    extracted_terms: Dict[str, List[str]] = {}  # Use string keys
    term_extraction_queue: List[int] = []

# Helper function to convert numeric dictionary keys to strings
def convert_numeric_keys_to_strings(data: dict) -> dict:
    """Convert numeric dictionary keys to strings for Pydantic model compatibility."""
    if not data:
        return data
        
    result = {}
    for key, value in data.items():
        # Convert key to string if it's numeric
        str_key = str(key)
        
        # If value is a dict, recursively convert its keys too
        if isinstance(value, dict):
            result[str_key] = convert_numeric_keys_to_strings(value)
        elif isinstance(value, list):
            # If value is a list of dicts, convert each dict's keys
            if value and isinstance(value[0], dict):
                result[str_key] = [convert_numeric_keys_to_strings(item) if isinstance(item, dict) else item for item in value]
            else:
                result[str_key] = value
        else:
            result[str_key] = value
            
    return result

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log requests and responses."""
    request_id = f"{time.time():.0f}"
    
    # Get client IP and requested path
    client_host = request.client.host if request.client else "unknown"
    path = request.url.path
    method = request.method
    
    # Log the request
    logger.info(f"REQ [{request_id}] {method} {path} from {client_host}")
    
    # Track request timing
    start_time = time.time()
    
    # Process the request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log successful response
        logger.info(
            f"RES [{request_id}] {method} {path} completed with status {response.status_code} "
            f"in {process_time:.3f}s"
        )
        return response
    except Exception as e:
        # Log exception
        process_time = time.time() - start_time
        logger.error(
            f"ERR [{request_id}] {method} {path} failed after {process_time:.3f}s: {str(e)}",
            exc_info=True
        )
        raise

# API Endpoints
@app.post("/start", response_model=StartSessionResponse)
def api_start_session():
    """Start a new questionnaire session."""
    logger.info("API: Starting new questionnaire session")
    
    try:
        session = start_session()
        session_id = session["session_id"]
        logger.info(f"API: Session started successfully with ID: {session_id}")
        logger.debug(f"API: Initial session contains {len(session.get('conversation_history', []))} messages")
        return session
    except Exception as e:
        logger.error(f"API: Error starting session: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")

@app.post("/answer", response_model=AnswerResponse)
def api_process_answer(request: AnswerRequest):
    """Process a user's answer, verify it, and get the next question if valid."""
    session_id = request.session_id
    answer_preview = request.answer[:50] + "..." if len(request.answer) > 50 else request.answer
    
    logger.info(f"API: Processing answer for session {session_id}")
    logger.debug(f"API: Answer content preview: {answer_preview}")
    
    try:
        # Get the current session state to verify it exists
        session_state = get_session_state(session_id)
        if not session_state:
            logger.warning(f"API: Session not found: {session_id}")
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Process the answer
        start_time = time.time()
        result = process_user_answer(session_id, request.answer)
        processing_time = time.time() - start_time
        
        # Convert numeric keys to string keys for Pydantic compatibility
        result = convert_numeric_keys_to_strings(result)
        
        # Log the verification result - need to handle string keys
        current_index = str(session_state.get("current_index", 0))
        is_valid = result.get("verified_responses", {}).get(current_index, False)
        logger.info(
            f"API: Answer processed in {processing_time:.3f}s - "
            f"valid: {is_valid}, is_complete: {result.get('is_complete', False)}"
        )
        
        return result
    except HTTPException as e:
        # Re-raise HTTP exceptions
        logger.warning(f"API: HTTP exception in answer processing: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"API: Error processing answer: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing answer: {str(e)}")

@app.get("/session/{session_id}", response_model=SessionStateResponse)
def api_get_session(session_id: str):
    """Get the current state of a session."""
    logger.info(f"API: Retrieving session state for {session_id}")
    
    try:
        session_state = get_session_state(session_id)
        if not session_state:
            logger.warning(f"API: Session not found: {session_id}")
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Convert numeric keys to string keys for Pydantic compatibility
        session_state = convert_numeric_keys_to_strings(session_state)
        
        logger.debug(
            f"API: Retrieved session {session_id} - "
            f"current_index: {session_state.get('current_index', 'unknown')}, "
            f"is_complete: {session_state.get('is_complete', False)}, "
            f"responses: {len(session_state.get('responses', {}))}"
        )
        
        return session_state
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"API: Error retrieving session: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving session: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint."""
    logger.debug("API: Health check request received")
    return {"status": "healthy"}

# Thread management and monitoring endpoints
@app.get("/threads")
def get_thread_status():
    """Get current thread manager status."""
    logger.debug("API: Thread status request received")
    return thread_manager.get_status_summary()

@app.get("/threads/{task_id}")
def get_task_status(task_id: str):
    """Get status of a specific task by task ID."""
    logger.debug(f"API: Task status request for task_id: {task_id}")
    return thread_manager.get_task_status(task_id)

@app.get("/threads/active")
def get_active_tasks():
    """Get all currently active tasks."""
    logger.debug("API: Active tasks request received")
    return thread_manager.get_active_tasks()

@app.get("/threads/history")
def get_recent_tasks(limit: int = 10, task_type: Optional[str] = None):
    """Get recently completed tasks with optional filtering by type."""
    logger.debug(f"API: Task history request received (limit={limit}, type={task_type})")
    return thread_manager.get_recent_tasks(limit=limit, task_type=task_type)

# Run the server when the script is executed directly
if __name__ == "__main__":
    logger.info("Starting Medical Questionnaire API server")
    uvicorn.run(app, host="0.0.0.0", port=8000)