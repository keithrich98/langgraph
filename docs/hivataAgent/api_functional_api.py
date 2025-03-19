# api.py - Simplified approach without HIL API

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from typing import Dict, Optional, List
import uvicorn

# Import our workflow from graph.py
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
    """Extract the most recent AI message from conversation history."""
    for msg in reversed(conversation_history):
        if msg.get("role") == "ai":
            return msg.get("content", "")
    return ""

@app.post("/start", response_model=StartResponse)
def start_session():
    """Start a new questionnaire session."""
    # Generate a unique thread ID
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Start the workflow
        result = questionnaire_workflow.invoke({"action": "start"}, config)
        
        # Extract state information
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
    """Process user answer and get the next question."""
    config = {"configurable": {"thread_id": req.session_id}}
    
    try:
        # Submit the answer
        result = questionnaire_workflow.invoke(
            {"action": "answer", "answer": req.answer}, 
            config
        )
        
        # Extract state information
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

# Debug endpoint
@app.get("/debug/{session_id}")
def debug_session(session_id: str):
    """Get details about the current session state."""
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # Get current state directly
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