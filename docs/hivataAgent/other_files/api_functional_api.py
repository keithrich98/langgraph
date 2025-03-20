# api.py - Updated with streaming verification

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from typing import Dict, Optional, List, AsyncIterator
import uvicorn
import json
import asyncio
from fastapi.responses import StreamingResponse

# Import our updated workflow and ChatState from graph.py
from graph import questionnaire_workflow, ChatState

app = FastAPI()

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

def get_current_question(conversation_history: List[Dict[str, str]]) -> str:
    """Extract the most recent AI message from conversation history as a string."""
    for msg in reversed(conversation_history):
        if msg.get("role") == "ai":
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            return content
    return ""


@app.post("/start", response_model=StartResponse)
def start_session():
    """Start a new questionnaire session."""
    # Generate a unique thread ID for checkpointing
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        result = questionnaire_workflow.invoke({"action": "start"}, config)
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
        print(f"Error starting session: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")

@app.post("/next", response_model=NextResponse)
def next_step(req: NextRequest):
    """Process the user's answer and return the next question (or follow-up verification)."""
    config = {"configurable": {"thread_id": req.session_id}}
    
    try:
        result = questionnaire_workflow.invoke(
            {"action": "answer", "answer": req.answer}, 
            config
        )
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
        print(f"Error processing answer: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing answer: {str(e)}")

@app.post("/stream-next")
async def stream_next_step(req: NextRequest) -> StreamingResponse:
    """Process the user's answer and stream the LLM verification response."""
    config = {"configurable": {"thread_id": req.session_id}}
    
    try:
        # First get the current state to properly compare later
        current_state = questionnaire_workflow.get_state(config)
        current_conversation_length = len(current_state.values.conversation_history)
        
        # Use astream to stream message tokens
        async def response_stream() -> AsyncIterator[bytes]:
            # Collect verification message
            verification_text = ""
            token_count = 0
            
            print(f"Streaming started for session {req.session_id}")
            
            async for event, chunk in questionnaire_workflow.astream(
                {"action": "answer", "answer": req.answer},
                config,
                stream_mode="messages"
            ):
                # Only process message chunks
                if hasattr(chunk, "content") and chunk.content:
                    token_count += 1
                    # Add a small delay to make streaming more visible
                    await asyncio.sleep(0.1)  # 100ms delay between tokens
                    
                    # Debug output for each token
                    print(f"Token #{token_count}: '{chunk.content}'")
                    
                    # Send token with its count for debugging
                    yield f"data: {json.dumps({'text': chunk.content, 'token_num': token_count})}\n\n".encode('utf-8')
                    verification_text += chunk.content
            
            print(f"Finished streaming {token_count} tokens")
            
            # Get the final state after streaming is complete
            final_state = questionnaire_workflow.get_state(config)
            final_conversation = final_state.values.conversation_history
            
            # Calculate which is the verification message (should be the new message after user input)
            # This should always be the conversation history after user input (at minimum len +2)
            if len(final_conversation) >= current_conversation_length + 2:
                # Get the full state to return to client
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
                # Something went wrong, return error
                error_data = {
                    "final": True,
                    "error": "Failed to process answer correctly",
                    "total_tokens": token_count
                }
                yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')
        
        return StreamingResponse(
            response_stream(),
            media_type="text/event-stream"
        )
    except Exception as e:
        import traceback
        print(f"Error in streaming: {str(e)}")
        print(traceback.format_exc())
        # Return error response
        async def error_stream():
            error_data = {
                "final": True,
                "error": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n".encode('utf-8')
        
        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream"
        )

@app.get("/debug/{session_id}")
def debug_session(session_id: str):
    """Retrieve the current session state for debugging."""
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        current_state = questionnaire_workflow.get_state(config)
        if hasattr(current_state, "values"):
            state = current_state.values
            return {
                "current_index": state.current_question_index,
                "is_complete": state.is_complete,
                "conversation_history": state.conversation_history,
                "responses": state.responses
            }
        return {"error": "No state values found"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)