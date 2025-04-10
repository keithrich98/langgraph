# api.py with HIL API

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from typing import Dict, Optional, List
import uvicorn

# Import our graph and ChatState from graph.py
from parent_workflow import graph, ChatState, init_node
from langgraph.types import Command  # Import Command for resuming execution

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

@app.post("/start", response_model=StartResponse)
def start_session():
    # Generate a unique thread ID
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Initialize the state and run until interrupt
    state = ChatState()
    state = init_node(state)

    # Run the graph through initialization and the first ask_node
    # This will generate the first question and then stop at answer_node
    result = graph.invoke(state, config)

    # Get the current state
    current_state = graph.get_state(config)

    # Check if waiting for input
    waiting_for_input = False
    if current_state.tasks and len(current_state.tasks) > 0:
        task = current_state.tasks[0]
        if hasattr(task, 'interrupts') and task.interrupts:
            waiting_for_input = True

    # Get conversation history
    state_values = current_state.values
    conversation_history = state_values.get('conversation_history', [])

    # Get the current question
    current_question = ""
    for msg in reversed(conversation_history):
        if msg["role"] == "ai":
            current_question = msg["content"]
            break

    return StartResponse(
        session_id=thread_id,
        conversation_history=conversation_history,
        current_question=current_question,
        waiting_for_input=waiting_for_input
    )

@app.post("/next", response_model=NextResponse)
def next_step(req: NextRequest):
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
        finished = not updated_state.next and not waiting_for_input

        # Get conversation history and current question
        state_values = updated_state.values
        conversation_history = state_values.get('conversation_history', [])

        current_question = ""
        for msg in reversed(conversation_history):
            if msg["role"] == "ai":
                current_question = msg["content"]
                break

        return NextResponse(
            conversation_history=conversation_history,
            current_question=current_question,
            finished=finished,
            waiting_for_input=waiting_for_input
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing answer: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)