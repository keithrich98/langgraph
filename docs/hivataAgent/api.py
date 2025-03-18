# api.py

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from uuid import uuid4
from typing import Dict, Optional, List, Union, Any
import uvicorn

from graph import (
    graph,
    ChatState,
    init_node,
    ask_node_api,
    increment_node,
    validation_response,
    # New imports for functional API
    functional_graph,
    Command
)

app = FastAPI()
sessions: Dict[str, Dict[str, Any]] = {}

class StartResponse(BaseModel):
    session_id: str
    conversation_history: List[Dict[str, str]]
    current_question: str
    using_functional_api: bool = False

class NextRequest(BaseModel):
    session_id: str
    answer: str

class NextResponse(BaseModel):
    conversation_history: List[Dict[str, str]]
    current_question: Optional[str] = None
    finished: bool

# Original API implementation
@app.post("/start", response_model=StartResponse)
def start_session(use_functional: bool = Query(False, description="Use the functional API implementation")):
    session_id = str(uuid4())
    
    if use_functional:
        # Use functional API implementation
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        # Start the workflow and run until the first interrupt
        for step in functional_graph.stream(None, config):
            pass  # Process steps if needed
        
        # Get the current state at the interrupt point
        state = functional_graph.get_state(config).values
        
        sessions[session_id] = {
            "thread_id": thread_id,
            "using_functional": True
        }
        
        conversation_history = state.get("conversation_history", [])
        current_question = ""
        if "current_question" in state:
            question_obj = state["current_question"]
            current_question = f"{question_obj['text']}\nRequirements: {question_obj['requirements']}"
        
        return StartResponse(
            session_id=session_id,
            conversation_history=conversation_history,
            current_question=current_question,
            using_functional_api=True
        )
    else:
        # Use original StateGraph implementation
        state = ChatState()
        state = init_node(state)  # build default questions, etc.
        # Send the first question
        command = ask_node_api(state)  # attaches question to conversation_history
        # We do not invoke the graph here, we just do the node function directly
        sessions[session_id] = {
            "state": state,
            "using_functional": False
        }
        current_question = state.conversation_history[-1]["content"] if state.conversation_history else ""
        return StartResponse(
            session_id=session_id,
            conversation_history=state.conversation_history,
            current_question=current_question,
            using_functional_api=False
        )

@app.post("/next", response_model=NextResponse)
def next_step(req: NextRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = sessions[req.session_id]
    
    if session_data.get("using_functional", False):
        # Handle with functional API
        thread_id = session_data["thread_id"]
        config = {"configurable": {"thread_id": thread_id}}
        
        # Resume execution with the user's answer
        for step in functional_graph.stream(Command(resume=req.answer), config):
            pass  # Process steps if needed
        
        # Get the updated state
        state = functional_graph.get_state(config).values
        
        # Check if finished
        if state.get("finished", False):
            sessions.pop(req.session_id)
            return NextResponse(
                conversation_history=state.get("conversation_history", []),
                finished=True
            )
        
        # Get the current question text
        current_question = ""
        if "current_question" in state:
            question_obj = state["current_question"]
            current_question = f"{question_obj['text']}\nRequirements: {question_obj['requirements']}"
        
        # Return the next response
        return NextResponse(
            conversation_history=state.get("conversation_history", []),
            current_question=current_question,
            finished=False
        )
    else:
        # Handle with original implementation
        state = session_data["state"]

        # 1) Record user's answer in the conversation
        idx = state.current_question_index
        user_answer = req.answer
        state.conversation_history.append({"role": "human", "content": user_answer})
        state.responses[idx] = user_answer

        # 2) Validate with LLM
        question_obj = state.questions[idx]
        validation_result = validation_response(
            question_text=question_obj["text"],
            requirements=question_obj["requirements"],
            user_answer=user_answer,
            conversation_so_far=state.conversation_history
        )

        if validation_result["valid"] == "true":
            # The LLM says the answer meets requirements → we can proceed to the next question
            # (or finish if this was the last).
            # Optionally, we can store the LLM's "confirmation" message in conversation:
            state.conversation_history.append(
                {"role": "ai", "content": validation_result["message"]}
            )

            # Move to next question
            command = increment_node(state)
            if command.goto == "END":
                # All questions answered
                sessions.pop(req.session_id)
                return NextResponse(
                    conversation_history=state.conversation_history,
                    finished=True
                )
            else:
                # Not finished, so ask the next question
                command = ask_node_api(state)
                current_question = state.conversation_history[-1]["content"]
                sessions[req.session_id]["state"] = state
                return NextResponse(
                    conversation_history=state.conversation_history,
                    current_question=current_question,
                    finished=False
                )
        else:
            # The LLM says the answer is invalid
            # We'll store the clarifying question or prompt in conversation_history
            clarifying_msg = validation_result["message"]
            state.conversation_history.append({"role": "ai", "content": clarifying_msg})

            # We do NOT increment the question index. The user must call /next again, presumably
            # with a new or corrected answer for the same question.
            # Return the clarifying prompt as the "current_question".
            sessions[req.session_id]["state"] = state
            return NextResponse(
                conversation_history=state.conversation_history,
                current_question=clarifying_msg,
                finished=False
            )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)