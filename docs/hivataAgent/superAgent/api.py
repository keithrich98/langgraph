from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from typing import Dict, Optional, List, AsyncIterator, Any
import uvicorn
import json
import asyncio

from fastapi.responses import StreamingResponse

# Import the parent workflow (which now orchestrates the QA and term extraction tasks)
from parent_workflow import parent_workflow, get_full_state

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

def get_current_question(conversation_history: List[Dict[str, str]]) -> str:
    """Extract the latest AI message from the conversation history."""
    for msg in reversed(conversation_history):
        if msg.get("role") == "ai":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""

@app.post("/start", response_model=StartResponse)
def start_session():
    """Start a new questionnaire session."""
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Start the session by invoking the parent workflow with "start" action.
        result = parent_workflow.invoke({"action": "start"}, config)
        conversation_history = result.conversation_history
        current_question = get_current_question(conversation_history)
        is_complete = result.is_complete
        
        return StartResponse(
            session_id=thread_id,
            conversation_history=conversation_history,
            current_question=current_question,
            waiting_for_input=not is_complete
        )
    except Exception as e:
        import traceback
        print(f"[DEBUG] Error starting session: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")

@app.post("/next", response_model=NextResponse)
def next_step(req: NextRequest):
    """Process the user's answer and return the next question (or follow-up verification)."""
    print(f"[DEBUG API] Using thread_id: {req.session_id}")
    config = {"configurable": {"thread_id": req.session_id}}
    
    try:
        result = parent_workflow.invoke({"action": "answer", "answer": req.answer}, config)
        conversation_history = result.conversation_history
        current_question = get_current_question(conversation_history)
        is_complete = result.is_complete
        
        return NextResponse(
            conversation_history=conversation_history,
            current_question=current_question,
            finished=is_complete,
            waiting_for_input=not is_complete
        )
    except Exception as e:
        import traceback
        print(f"[DEBUG] Error processing answer: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing answer: {str(e)}")

@app.post("/stream-next")
async def stream_next_step(req: NextRequest) -> StreamingResponse:
    """Stream the LLM verification response while processing the answer."""
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
                    await asyncio.sleep(0.05)  # slight delay for visibility
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
    """Endpoint to debug the extraction queue."""
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        state_snapshot = parent_workflow.get_state(config)
        state = state_snapshot.values if hasattr(state_snapshot, "values") else None
        if not state:
            return {"error": "No state found for this session"}
        
        # For debugging, display the extraction queue before and after triggering extraction.
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
        
        return {
            "before": {"extraction_queue": before_queue},
            "after_update": {"extraction_queue": getattr(updated_state, "term_extraction_queue", [])},
            "after_extraction": {
                "extraction_queue": getattr(final_state, "term_extraction_queue", []),
                "extracted_terms": getattr(final_state, "extracted_terms", {}),
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
        state_snapshot = parent_workflow.get_state(config)
        state = state_snapshot.values if hasattr(state_snapshot, "values") else None
        if state is None:
            return {"extracted_terms": {}, "current_index": 0, "is_complete": False, "extraction_queue_length": 0}
        
        current_index = getattr(state, "current_question_index", 0)
        is_complete = getattr(state, "is_complete", False)
        queue_length = len(getattr(state, "term_extraction_queue", []))
        extracted_terms = getattr(state, "extracted_terms", {})
        
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
        return {"extracted_terms": {}, "current_index": 0, "is_complete": False, "extraction_queue_length": 0, "error": str(e)}

@app.get("/inspect-model-context/{session_id}")
def inspect_model_context(session_id: str, question_index: int = None):
    """
    Debug endpoint to inspect the context sent to the LLM.
    """
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        parent_state_snapshot = parent_workflow.get_state(config)
        if not hasattr(parent_state_snapshot, "values"):
            return {"error": "No state found for this session"}
        parent_state = parent_state_snapshot.values
        
        # Assume conversation and questions are stored in the parent's state.
        questions = parent_state.get("questions", [])
        conversation_history = parent_state.get("conversation_history", [])
        current_index = parent_state.get("current_question_index", 0)
        
        if not questions:
            from question_processor import get_questions
            questions = get_questions()
        
        if question_index is None:
            question_index = current_index
        
        if question_index < 0 or question_index >= len(questions):
            return {"error": f"Invalid question index: {question_index}"}
        
        current_question = questions[question_index]
        formatted_requirements = "\n".join([f"- {k}: {v}" for k, v in current_question.get("requirements", {}).items()])
        question_text = current_question.get("text", "")
        
        formatted_history = ""
        for msg in conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            formatted_history += f"{role.upper()}: {content}\n\n"
        
        system_prompt = (
            "You are an expert medical validator for a questionnaire about polymicrogyria. "
            "Evaluate if a user's answer meets the specified requirements."
        )
        
        user_prompt = (
            f"QUESTION: {question_text}\n\n"
            f"REQUIREMENTS:\n{formatted_requirements}\n\n"
            f"CONVERSATION HISTORY:\n{formatted_history}\n"
            f"USER'S ANSWER: [Most recent answer here]\n\n"
            f"Evaluate if the answer meets all requirements."
        )
        
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4")
            system_tokens = len(encoding.encode(system_prompt))
            user_tokens = len(encoding.encode(user_prompt))
            total_tokens = system_tokens + user_tokens
        except Exception as e:
            total_tokens = "N/A"
            system_tokens = "N/A"
            user_tokens = "N/A"
        
        return {
            "session_id": session_id,
            "current_question_index": question_index,
            "question": question_text,
            "total_conversation_history_length": len(conversation_history),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "token_estimates": {
                "system_tokens": system_tokens,
                "user_tokens": user_tokens,
                "total_tokens": total_tokens,
                "note": "Estimates based on GPT-4 tokenization"
            },
            "state_info": {
                "questions_count": len(questions),
                "is_complete": parent_state.get("is_complete", False)
            }
        }
    except Exception as e:
        import traceback
        print(f"[DEBUG] Error inspecting model context: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
