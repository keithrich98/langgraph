# api.py - Updated to support both question_answer and term extraction agents

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from typing import Dict, Optional, List, AsyncIterator, Any
import uvicorn
import json
import asyncio

from fastapi.responses import StreamingResponse

# Import the parent workflow that coordinates both agents
from parent_workflow import parent_workflow, get_full_state
from term_extractor import term_extraction_workflow 

app = FastAPI()

# Configure CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class TermsResponse(BaseModel):
    extracted_terms: Dict[str, List[str]]
    current_index: int
    is_complete: bool
    extraction_queue_length: int

# Helper function: Extract the latest AI message as the current question
def get_current_question(conversation_history: List[Dict[str, str]]) -> str:
    for msg in reversed(conversation_history):
        if msg.get("role") == "ai":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""

@app.post("/start", response_model=StartResponse)
def start_session():
    """Start a new questionnaire session."""
    # Generate a unique thread ID for checkpointing
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Start the parent workflow which initializes both agents
        result = parent_workflow.invoke({"action": "start"}, config)
        conversation_history = result.conversation_history
        current_question = get_current_question(conversation_history)
        is_completed = result.is_complete
        
        return StartResponse(
            session_id=thread_id,
            conversation_history=conversation_history,
            current_question=current_question,
            waiting_for_input=not is_completed
        )
    except Exception as e:
        import traceback
        print(f"[DEBUG] Error starting session: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")

@app.post("/next", response_model=NextResponse)
def next_step(req: NextRequest):
    """Process the user's answer and return the next question (or follow-up verification)."""
    config = {"configurable": {"thread_id": req.session_id}}
    
    try:
        result = parent_workflow.invoke({"action": "answer", "answer": req.answer}, config)
        conversation_history = result.conversation_history
        current_question = get_current_question(conversation_history)
        is_completed = result.is_complete
        
        return NextResponse(
            conversation_history=conversation_history,
            current_question=current_question,
            finished=is_completed,
            waiting_for_input=not is_completed
        )
    except Exception as e:
        import traceback
        print(f"[DEBUG] Error processing answer: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing answer: {str(e)}")

@app.post("/stream-next")
async def stream_next_step(req: NextRequest) -> StreamingResponse:
    """Process the user's answer and stream the LLM verification response."""
    config = {"configurable": {"thread_id": req.session_id}}
    
    try:
        current_state = parent_workflow.get_state(config)
        current_conversation_length = len(current_state.values.conversation_history)
        
        async def response_stream() -> AsyncIterator[bytes]:
            token_count = 0
            print(f"[DEBUG] Streaming started for session {req.session_id}")
            async for event, chunk in parent_workflow.astream(
                {"action": "answer", "answer": req.answer},
                config,
                stream_mode="messages"
            ):
                if hasattr(chunk, "content") and chunk.content:
                    token_count += 1
                    await asyncio.sleep(0.05)  # 50ms delay for visibility
                    print(f"[DEBUG] Token #{token_count}: '{chunk.content}'")
                    yield f"data: {json.dumps({'text': chunk.content, 'token_num': token_count})}\n\n".encode('utf-8')
            
            print(f"[DEBUG] Finished streaming {token_count} tokens")
            final_state = parent_workflow.get_state(config)
            final_conversation = final_state.values.conversation_history
            
            if len(final_conversation) >= current_conversation_length + 2:
                final_data = {
                    "final": True,
                    "conversation_history": final_conversation,
                    "current_question": get_current_question(final_conversation),
                    "finished": final_state.values.is_complete,
                    "waiting_for_input": not final_state.values.is_complete,
                    "total_tokens": token_count
                }
                yield f"data: {json.dumps(final_data)}\n\n".encode('utf-8')
            else:
                error_data = {
                    "final": True,
                    "error": "Failed to process answer correctly",
                    "total_tokens": token_count
                }
                yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')
        
        return StreamingResponse(response_stream(), media_type="text/event-stream")
    except Exception as e:
        import traceback
        print(f"[DEBUG] Error in streaming: {str(e)}")
        print(traceback.format_exc())
        async def error_stream():
            error_data = {"final": True, "error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')
        return StreamingResponse(error_stream(), media_type="text/event-stream")

@app.post("/debug-extraction-queue/{session_id}")
def debug_extraction_queue(session_id: str):
    """Debug endpoint to inspect the extraction queue and related state."""
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        state_snapshot = parent_workflow.get_state(config)
        state = state_snapshot.values if hasattr(state_snapshot, "values") else None
        if not state:
            return {"error": "No state found for this session"}
        
        # Debug the extraction queue before and after updating it.
        from state import add_to_extraction_queue
        before_queue = getattr(state, "term_extraction_queue", []).copy()
        updated_state = add_to_extraction_queue(state, 0)
        print(f"[DEBUG] Extraction queue before: {before_queue}")
        print(f"[DEBUG] Extraction queue after update: {updated_state.term_extraction_queue}")
        parent_workflow.update_state(config, updated_state)
        updated_snapshot = parent_workflow.get_state(config)
        updated_state = updated_snapshot.values if hasattr(updated_snapshot, "values") else None
        
        extraction_result = parent_workflow.invoke({"action": "extract_terms"}, config=config)
        final_snapshot = parent_workflow.get_state(config)
        final_state = final_snapshot.values if hasattr(final_snapshot, "values") else None
        
        extract_snapshot = term_extraction_workflow.get_state(config)
        extract_state = extract_snapshot.values if hasattr(extract_snapshot, "values") else None
        
        return {
            "before": {"extraction_queue": before_queue},
            "after_update": {"extraction_queue": getattr(updated_state, "term_extraction_queue", [])},
            "after_extraction": {
                "extraction_queue": getattr(final_state, "term_extraction_queue", []),
                "extracted_terms": getattr(extract_state, "extracted_terms", {}),
                "last_extracted_index": getattr(extract_state, "last_extracted_index", -1)
            }
        }
    except Exception as e:
        import traceback
        print(f"[DEBUG] Error debugging extraction queue: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}


@app.get("/extracted-terms/{session_id}")
def get_extracted_terms(session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # Get parent state snapshot
        parent_state_snapshot = parent_workflow.get_state(config)
        parent_state = parent_state_snapshot.values if hasattr(parent_state_snapshot, "values") else None
        
        # Also get term extractor state to make sure we have the most recent terms
        extract_state_snapshot = term_extraction_workflow.get_state(config)
        extract_state = extract_state_snapshot.values if hasattr(extract_state_snapshot, "values") else None
        
        # Log states for debugging
        print(f"[DEBUG API] Parent state snapshot: {parent_state}")
        print(f"[DEBUG API] Extract state snapshot: {extract_state}")
        
        # Get values from parent state
        if parent_state is not None:
            current_index = getattr(parent_state, "current_question_index", 0)
            is_complete = getattr(parent_state, "is_complete", False)
            queue_length = len(getattr(parent_state, "term_extraction_queue", []))
            
            # Get extracted terms, preferring the extraction workflow's state if available
            if extract_state is not None and hasattr(extract_state, "extracted_terms"):
                extracted_terms = extract_state.extracted_terms
            else:
                extracted_terms = getattr(parent_state, "extracted_terms", {})
        else:
            extracted_terms = {}
            current_index = 0
            is_complete = False
            queue_length = 0
        
        print(f"[DEBUG API] Returning extracted_terms: {extracted_terms}")
        
        return {
            "extracted_terms": extracted_terms,
            "current_index": current_index,
            "is_complete": is_complete,
            "extraction_queue_length": queue_length
        }
    except Exception as e:
        # Exception handling remains the same
        import traceback
        print(f"[DEBUG API] Error retrieving extracted terms: {str(e)}")
        print(traceback.format_exc())
        return {
            "extracted_terms": {},
            "current_index": 0,
            "is_complete": False,
            "extraction_queue_length": 0,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)