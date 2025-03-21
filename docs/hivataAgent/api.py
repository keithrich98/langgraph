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

@app.post("/force-extraction/{session_id}")
def force_extraction(session_id: str):
    """Force term extraction for verified answers."""
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        parent_snapshot = parent_workflow.get_state(config)
        parent_state = parent_snapshot.values if hasattr(parent_snapshot, "values") else None
        if not parent_state:
            return {"error": "No parent state found"}
        
        verified_answers = getattr(parent_state, "verified_answers", {}) 
        if not verified_answers:
            return {"error": "No verified answers found"}
        
        from term_extractor import extract_terms
        results = {}
        for idx, answer_data in verified_answers.items():
            question = answer_data.get("question", "")
            answer = answer_data.get("answer", "")
            print(f"[DEBUG] Force extraction for question index {idx}: {question[:50]}...")
            terms = extract_terms(question, answer, config=config).result()
            results[idx] = terms
        
        extract_snapshot = term_extraction_workflow.get_state(config)
        extract_state = extract_snapshot.values if hasattr(extract_snapshot, "values") else None
        if extract_state:
            extract_state.extracted_terms = results
            extract_state.term_extraction_queue = []
            extract_state.last_extracted_index = max(verified_answers.keys()) if verified_answers else -1
            term_extraction_workflow.update_state(config, extract_state)
        
        return {
            "success": True,
            "extracted_terms": results,
            "message": "Forced extraction completed"
        }
    except Exception as e:
        import traceback
        print(f"[DEBUG] Error forcing extraction: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}

@app.post("/direct-extract/{session_id}")
def direct_extract(session_id: str):
    """Directly extract terms without using the task abstraction."""
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        parent_snapshot = parent_workflow.get_state(config)
        parent_state = parent_snapshot.values if hasattr(parent_snapshot, "values") else None
        if not parent_state:
            return {"error": "No parent state found"}
        
        verified_answers = getattr(parent_state, "verified_answers", {}) 
        if not verified_answers:
            return {"error": "No verified answers found"}
        
        extract_terms_results = {}
        for idx, answer_data in verified_answers.items():
            question = answer_data.get("question", "")
            answer = answer_data.get("answer", "")
            print(f"[DEBUG] Direct extraction for Q{idx}: {question[:50]}...")
            print(f"[DEBUG] Answer: {answer[:50]}...")
            from term_extractor import extraction_prompt
            import json
            formatted_prompt = extraction_prompt.format(question=question, answer=answer)
            from term_extractor import llm
            response = llm.invoke(formatted_prompt)
            content = response.content
            try:
                content = content.strip()
                if not content.startswith('['):
                    start_idx = content.find('[')
                    if start_idx >= 0:
                        content = content[start_idx:]
                        bracket_count = 0
                        for i, char in enumerate(content):
                            if char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    content = content[:i+1]
                                    break
                terms = json.loads(content)
                if isinstance(terms, list):
                    extract_terms_results[idx] = terms
                else:
                    extract_terms_results[idx] = ["ERROR: Expected array of terms"]
            except Exception as e:
                print(f"[DEBUG] Error parsing terms: {str(e)}")
                lines = content.split('\n')
                terms = []
                for line in lines:
                    line = line.strip()
                    if line and line.startswith('-'):
                        terms.append(line[1:].strip())
                    elif line and line.startswith('"') and line.endswith('"'):
                        terms.append(line.strip('"'))
                extract_terms_results[idx] = terms if terms else ["ERROR: Failed to extract terms"]
        
        extract_snapshot = term_extraction_workflow.get_state(config)
        extract_state = extract_snapshot.values if hasattr(extract_snapshot, "values") else None
        if extract_state:
            if hasattr(extract_state, "extracted_terms"):
                extract_state.extracted_terms = extract_terms_results
            else:
                term_extraction_workflow.update_state(config, {"extracted_terms": extract_terms_results})
        
        return {
            "success": True,
            "extracted_terms": extract_terms_results,
            "message": "Direct extraction completed"
        }
    except Exception as e:
        import traceback
        print(f"[DEBUG] Error in direct extraction: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}

@app.get("/extracted-terms/{session_id}")
def get_extracted_terms(session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # Get parent state snapshot
        parent_state_snapshot = parent_workflow.get_state(config)
        parent_state = parent_state_snapshot.values if hasattr(parent_state_snapshot, "values") else None
        
        # Log parent's state for debugging
        print(f"[DEBUG API] Parent state snapshot: {parent_state}")
        
        if parent_state is not None:
            # Read extracted_terms from the parent's merged state
            extracted_terms = getattr(parent_state, "extracted_terms", {})
            current_index = getattr(parent_state, "current_question_index", 0)
            is_complete = getattr(parent_state, "is_complete", False)
            queue_length = len(getattr(parent_state, "term_extraction_queue", []))
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


@app.post("/extract-terms/{session_id}")
def trigger_term_extraction(session_id: str):
    """Manually trigger term extraction."""
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        result = parent_workflow.invoke({"action": "extract_terms"}, config=config)
        extract_snapshot = term_extraction_workflow.get_state(config)
        extract_state = extract_snapshot.values if hasattr(extract_snapshot, "values") else None
        if extract_state is not None:
            if hasattr(extract_state, "term_extraction_queue"):
                queue_length = len(extract_state.term_extraction_queue)
                extracted_terms = extract_state.extracted_terms
            else:
                queue_length = len(extract_state.get("term_extraction_queue", []))
                extracted_terms = extract_state.get("extracted_terms", {})
        else:
            queue_length = 0
            extracted_terms = {}
        
        return {
            "success": True,
            "extraction_queue_length": queue_length,
            "extracted_terms": extracted_terms,
            "message": "Term extraction triggered successfully"
        }
    except Exception as e:
        import traceback
        print(f"[DEBUG] Error triggering term extraction: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error triggering term extraction: {str(e)}")

@app.get("/debug/{session_id}")
def debug_session(session_id: str):
    """Retrieve the current session state for overall debugging."""
    config = {"configurable": {"thread_id": session_id}}
    try:
        # Get combined state from the parent workflow
        combined_state = get_full_state(session_id)
        
        # Get detailed state information from parent workflow
        current_state_snapshot = parent_workflow.get_state(config)
        if hasattr(current_state_snapshot, "values"):
            state = current_state_snapshot.values
            detailed_state = {
                "current_index": state.current_question_index,
                "is_complete": state.is_complete,
                "conversation_history": state.conversation_history,
                "responses": state.responses,
                "verified_answers": state.verified_answers,
                "extraction_queue": state.term_extraction_queue,
                "last_extracted_index": state.last_extracted_index
            }
        else:
            detailed_state = {}
        
        # Get term extraction details
        extraction_state_snapshot = term_extraction_workflow.get_state(config)
        if hasattr(extraction_state_snapshot, "values"):
            extraction_state = extraction_state_snapshot.values
            extraction_details = {
                "extraction_queue": extraction_state.term_extraction_queue,
                "extracted_terms": extraction_state.extracted_terms,
                "last_extracted_index": extraction_state.last_extracted_index
            }
        else:
            extraction_details = {}
        
        debug_info = {
            "parent_state": detailed_state,
            "extraction_state": extraction_details,
            "combined_state": combined_state
        }
        print(f"[DEBUG] Debug session info for {session_id}: {json.dumps(debug_info, indent=2)}")
        return debug_info
    except Exception as e:
        import traceback
        print(f"[DEBUG] Error in debug_session: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
