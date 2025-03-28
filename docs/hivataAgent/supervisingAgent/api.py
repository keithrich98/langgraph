# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from typing import Dict, Optional, List, AsyncIterator, Any
from langchain_core.messages import AIMessage, HumanMessage
import uvicorn
import json
import asyncio

from fastapi.responses import StreamingResponse

# Import the new supervisor workflow
from parent_supervisor import (
    start_questionnaire, 
    process_answer, 
    get_questionnaire_state,
    medical_questionnaire
)

# Import logging configuration
from logging_config import logger

logger.info("Initializing API")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS middleware configured")

# Response models
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

def get_current_question(conversation_history: List[Any]) -> str:
    """Extract the latest AI message from the conversation history."""
    logger.info(f"get_current_question called with {len(conversation_history)} messages")
    
    for idx, msg in enumerate(reversed(conversation_history)):
        logger.debug(f"Examining message {len(conversation_history) - idx} of type {type(msg)}")
        
        # Handle both dictionary format and LangChain message objects
        if isinstance(msg, dict) and msg.get("role") == "ai":
            content = msg.get("content", "")
            logger.debug(f"Found AI message (dict): {content[:50]}...")
            return content if isinstance(content, str) else str(content)
        elif isinstance(msg, AIMessage):
            content = msg.content
            logger.debug(f"Found AI message (AIMessage): {content[:50]}...")
            return content if isinstance(content, str) else str(content)
    
    logger.warning("No AI message found in conversation history")
    return ""

@app.post("/start", response_model=StartResponse)
def start_session():
    """Start a new questionnaire session."""
    thread_id = str(uuid4())
    logger.info(f"Starting new session with thread_id: {thread_id}")
    
    try:
        # Start the session
        logger.info("Calling start_questionnaire")
        result = start_questionnaire(thread_id)
        logger.debug(f"start_questionnaire returned {len(result['conversation_history'])} messages")
        
        conversation_history = result["conversation_history"]
        current_question = get_current_question(conversation_history)
        is_complete = result["is_complete"]
        
        logger.info(f"Session started successfully, current_question: {current_question[:50]}...")
        
        return StartResponse(
            session_id=thread_id,
            conversation_history=conversation_history,
            current_question=current_question,
            waiting_for_input=not is_complete
        )
    except Exception as e:
        import traceback
        logger.error(f"Error starting session: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")

@app.post("/next", response_model=NextResponse)
def next_step(req: NextRequest):
    """Process the user's answer and return the next question or follow-up verification."""
    logger.info(f"Processing next step for thread_id: {req.session_id}")
    logger.debug(f"Answer: {req.answer}")
    
    try:
        logger.info("Calling process_answer")
        result = process_answer(req.session_id, req.answer)
        logger.debug(f"process_answer returned {len(result['conversation_history'])} messages")
        
        conversation_history = result["conversation_history"]
        current_question = get_current_question(conversation_history)
        is_complete = result["is_complete"]
        
        logger.info(f"Next step processed successfully, current_question: {current_question[:50]}...")
        
        return NextResponse(
            conversation_history=conversation_history,
            current_question=current_question,
            finished=is_complete,
            waiting_for_input=not is_complete
        )
    except Exception as e:
        import traceback
        logger.error(f"Error processing answer: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing answer: {str(e)}")

@app.post("/stream-next")
async def stream_next_step(req: NextRequest) -> StreamingResponse:
    """Stream the response while processing the answer."""
    logger.info(f"Streaming next step for thread_id: {req.session_id}")
    config = {"configurable": {"thread_id": req.session_id}}
    
    try:
        # Get the current state
        logger.info("Getting current state")
        current_state = get_questionnaire_state(req.session_id)
        current_conversation_history = current_state["conversation_history"]
        logger.debug(f"Current conversation history has {len(current_conversation_history)} messages")
        
        # Create user message
        user_message = {
            "role": "user",
            "content": req.answer
        }
        
        # Combine with existing messages
        messages = current_conversation_history + [user_message]
        logger.info(f"Prepared {len(messages)} messages for streaming")
        
        async def response_stream() -> AsyncIterator[bytes]:
            token_count = 0
            logger.info(f"Streaming started for session {req.session_id}")
            
            # Stream the processing
            logger.info("Starting medical_questionnaire.astream")
            try:
                async for chunk in medical_questionnaire.astream(
                    {"messages": messages},
                    config,
                    stream_mode="updates"
                ):
                    # Extract tokens from the chunk based on your graph output format
                    if chunk:
                        token_count += 1
                        logger.debug(f"Streaming token #{token_count}: {str(chunk)[:50]}...")
                        await asyncio.sleep(0.05)  # slight delay for visibility
                        yield f"data: {json.dumps({'text': str(chunk), 'token_num': token_count})}\n\n".encode('utf-8')
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n".encode('utf-8')
            
            # Send final state after streaming is complete
            logger.info("Streaming complete, retrieving final state")
            final_state = get_questionnaire_state(req.session_id)
            final_conversation = final_state["conversation_history"]
            
            final_data = {
                "final": True,
                "conversation_history": final_conversation,
                "current_question": get_current_question(final_conversation),
                "finished": final_state["is_complete"],
                "waiting_for_input": not final_state["is_complete"],
                "total_tokens": token_count
            }
            logger.info(f"Sending final state with {len(final_conversation)} messages")
            yield f"data: {json.dumps(final_data)}\n\n".encode('utf-8')
        
        return StreamingResponse(response_stream(), media_type="text/event-stream")
    except Exception as e:
        import traceback
        logger.error(f"Error in streaming: {str(e)}")
        logger.error(traceback.format_exc())
        async def error_stream():
            error_data = {"final": True, "error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')
        return StreamingResponse(error_stream(), media_type="text/event-stream")

@app.get("/state/{session_id}")
def get_session_state(session_id: str):
    """Get the current state of a session."""
    logger.info(f"Getting session state for thread_id: {session_id}")
    
    try:
        state = get_questionnaire_state(session_id)
        logger.info(f"Retrieved state with {len(state.get('conversation_history', []))} messages")
        return state
    except Exception as e:
        import traceback
        logger.error(f"Error getting session state: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error getting session state: {str(e)}")

@app.get("/log-levels/{level}")
def set_log_level(level: str):
    """Dynamically change the log level."""
    level = level.upper()
    if level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        numeric_level = getattr(logging, level)
        logger.setLevel(numeric_level)
        logger.info(f"Log level set to {level}")
        return {"message": f"Log level set to {level}"}
    else:
        return {"error": f"Invalid log level: {level}"}

logger.info("API routes configured")

if __name__ == "__main__":
    logger.info("Starting uvicorn server")
    uvicorn.run(app, host="0.0.0.0", port=8000)