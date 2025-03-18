# api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from typing import Dict, Optional, List
import uvicorn

# Import our graph and ChatState from graph.py.
# Note: We now import ask_node_api instead of ask_node.
from graph import graph, ChatState, init_node, ask_node_api, increment_node, END

app = FastAPI()

# In-memory session store.
sessions: Dict[str, ChatState] = {}

class StartResponse(BaseModel):
    session_id: str
    conversation_history: List[Dict[str, str]]
    current_question: str

class NextRequest(BaseModel):
    session_id: str
    answer: str

class NextResponse(BaseModel):
    conversation_history: List[Dict[str, str]]
    current_question: Optional[str] = None
    finished: bool

@app.post("/start", response_model=StartResponse)
def start_session():
    session_id = str(uuid4())
    state = ChatState()  # Create a new ChatState instance.
    state = init_node(state)  # Initialize the state.
    # Use the non-interactive ask node (ask_node_api) to generate the first question.
    command = ask_node_api(state)
    sessions[session_id] = state
    # The current question is in the conversation_history as the last message.
    current_question = state.conversation_history[-1]["content"] if state.conversation_history else ""
    return StartResponse(
        session_id=session_id, 
        conversation_history=state.conversation_history, 
        current_question=current_question
    )

@app.post("/next", response_model=NextResponse)
def next_step(req: NextRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    state = sessions[req.session_id]
    
    # Record the user's answer.
    idx = state.current_question_index
    state.conversation_history.append({"role": "human", "content": req.answer})
    state.responses[idx] = req.answer

    # Advance the conversation using increment_node.
    command = increment_node(state)
    if command.goto == END:
        sessions.pop(req.session_id)
        return NextResponse(conversation_history=state.conversation_history, finished=True)
    else:
        # Generate the next question using the non-interactive ask node.
        command = ask_node_api(state)
        sessions[req.session_id] = state
        current_question = state.conversation_history[-1]["content"]
        return NextResponse(
            conversation_history=state.conversation_history, 
            current_question=current_question, 
            finished=False
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
