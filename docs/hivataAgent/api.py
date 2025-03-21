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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        # Start the parent workflow which will initialize both agents
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
        print(f"Error starting session: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error starting session: {str(e)}")

@app.post("/next", response_model=NextResponse)
def next_step(req: NextRequest):
    """Process the user's answer and return the next question (or follow-up verification)."""
    config = {"configurable": {"thread_id": req.session_id}}
    
    try:
        # Use the parent workflow to process the answer
        # This will automatically trigger term extraction after a valid answer
        result = parent_workflow.invoke(
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
        current_state = parent_workflow.get_state(config)
        current_conversation_length = len(current_state.values.conversation_history)
        
        # Use astream to stream message tokens
        async def response_stream() -> AsyncIterator[bytes]:
            # Collect verification message
            verification_text = ""
            token_count = 0
            
            print(f"Streaming started for session {req.session_id}")
            
            # We use the parent workflow to ensure term extraction happens automatically
            async for event, chunk in parent_workflow.astream(
                {"action": "answer", "answer": req.answer},
                config,
                stream_mode="messages"
            ):
                # Only process message chunks
                if hasattr(chunk, "content") and chunk.content:
                    token_count += 1
                    # Add a small delay to make streaming more visible
                    await asyncio.sleep(0.05)  # 50ms delay between tokens
                    
                    # Debug output for each token
                    print(f"Token #{token_count}: '{chunk.content}'")
                    
                    # Send token with its count for debugging
                    yield f"data: {json.dumps({'text': chunk.content, 'token_num': token_count})}\n\n".encode('utf-8')
                    verification_text += chunk.content
            
            print(f"Finished streaming {token_count} tokens")
            
            # Get the final state after streaming is complete
            final_state = parent_workflow.get_state(config)
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
    
@app.post("/debug-extraction-queue/{session_id}")
def debug_extraction_queue(session_id: str):
    """Debug endpoint to test the extraction queue functionality."""
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # Get the current state
        state_snapshot = parent_workflow.get_state(config)
        state = state_snapshot.values if hasattr(state_snapshot, "values") else None
        
        if not state:
            return {"error": "No state found for this session"}
        
        # Check verified answers
        verified_answers = getattr(state, "verified_answers", {})
        
        # Check if the first answer is verified
        if 0 in verified_answers:
            # Manually add to extraction queue
            from state import add_to_extraction_queue
            
            # Create a before/after comparison
            before_queue = getattr(state, "term_extraction_queue", []).copy()
            
            # Manually add to queue
            updated_state = add_to_extraction_queue(state, 0)
            
            # Update the state
            parent_workflow.update_state(config, updated_state)
            
            # Get updated state
            updated_snapshot = parent_workflow.get_state(config)
            updated_state = updated_snapshot.values if hasattr(updated_snapshot, "values") else None
            
            # Trigger term extraction
            extraction_result = parent_workflow.invoke({"action": "extract_terms"}, config=config)
            
            # Get final state
            final_snapshot = parent_workflow.get_state(config)
            final_state = final_snapshot.values if hasattr(final_snapshot, "values") else None
            
            # Get extraction state
            extract_snapshot = term_extraction_workflow.get_state(config)
            extract_state = extract_snapshot.values if hasattr(extract_snapshot, "values") else None
            
            return {
                "before": {
                    "extraction_queue": before_queue,
                },
                "after_update": {
                    "extraction_queue": getattr(updated_state, "term_extraction_queue", []),
                },
                "after_extraction": {
                    "extraction_queue": getattr(final_state, "term_extraction_queue", []),
                    "extracted_terms": getattr(extract_state, "extracted_terms", {}),
                    "last_extracted_index": getattr(extract_state, "last_extracted_index", -1)
                }
            }
        else:
            return {"error": "No verified answers found"}
    
    except Exception as e:
        import traceback
        print(f"Error debugging extraction queue: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}

@app.post("/force-extraction/{session_id}")
def force_extraction(session_id: str):
    """Force term extraction for verified answers."""
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # Get the current states
        parent_snapshot = parent_workflow.get_state(config)
        parent_state = parent_snapshot.values if hasattr(parent_snapshot, "values") else None
        
        if not parent_state:
            return {"error": "No parent state found"}
        
        # Get verified answers
        verified_answers = getattr(parent_state, "verified_answers", {}) 
        
        if not verified_answers:
            return {"error": "No verified answers found"}
        
        # Manually create input for term extraction
        from term_extractor import extract_terms
        
        results = {}
        for idx, answer_data in verified_answers.items():
            question = answer_data.get("question", "")
            answer = answer_data.get("answer", "")
            
            # Directly call the extraction function
            terms = extract_terms(question, answer, config=config).result()
            
            # Store results
            results[idx] = terms
        
        # Update the term extraction state
        extract_snapshot = term_extraction_workflow.get_state(config)
        extract_state = extract_snapshot.values if hasattr(extract_snapshot, "values") else None
        
        if extract_state:
            # Update extracted terms
            extract_state.extracted_terms = results
            # Clear queue
            extract_state.term_extraction_queue = []
            # Update last extracted index
            extract_state.last_extracted_index = max(verified_answers.keys()) if verified_answers else -1
            
            # Update state
            term_extraction_workflow.update_state(config, extract_state)
        
        return {
            "success": True,
            "extracted_terms": results,
            "message": "Forced extraction completed"
        }
    
    except Exception as e:
        import traceback
        print(f"Error forcing extraction: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}

@app.post("/direct-extract/{session_id}")
def direct_extract(session_id: str):
    """Directly extract terms without using the task abstraction."""
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # Get the current states
        parent_snapshot = parent_workflow.get_state(config)
        parent_state = parent_snapshot.values if hasattr(parent_snapshot, "values") else None
        
        if not parent_state:
            return {"error": "No parent state found"}
        
        # Get verified answers
        verified_answers = getattr(parent_state, "verified_answers", {}) 
        
        if not verified_answers:
            return {"error": "No verified answers found"}
        
        # Directly use the term extraction workflow
        extract_terms_results = {}
        
        # For each verified answer
        for idx, answer_data in verified_answers.items():
            question = answer_data.get("question", "")
            answer = answer_data.get("answer", "")
            
            # Create a request to process this answer
            request_data = {
                "question": question,
                "answer": answer
            }
            
            # Print for debugging
            print(f"Processing extraction for Q{idx}: {question[:50]}...")
            print(f"Answer: {answer[:50]}...")
            
            # Call the LLM directly - this avoids the task abstraction
            from term_extractor import extraction_prompt
            import json
            
            # Format the prompt
            formatted_prompt = extraction_prompt.format(
                question=question,
                answer=answer
            )
            
            # Use the LLM directly
            from term_extractor import llm
            response = llm.invoke(formatted_prompt)
            content = response.content
            
            # Try to parse the response as JSON
            try:
                # Find JSON array in the response if it's not clean JSON
                content = content.strip()
                if not content.startswith('['):
                    # Try to find the start of a JSON array
                    start_idx = content.find('[')
                    if start_idx >= 0:
                        content = content[start_idx:]
                        # Find the matching end bracket
                        bracket_count = 0
                        for i, char in enumerate(content):
                            if char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    content = content[:i+1]
                                    break
                
                # Parse the JSON array
                terms = json.loads(content)
                if isinstance(terms, list):
                    extract_terms_results[idx] = terms
                else:
                    extract_terms_results[idx] = ["ERROR: Expected array of terms"]
            except Exception as e:
                print(f"Error parsing terms: {str(e)}")
                # Try a simple fallback approach
                lines = content.split('\n')
                terms = []
                for line in lines:
                    line = line.strip()
                    if line and line.startswith('-'):
                        terms.append(line[1:].strip())
                    elif line and line.startswith('"') and line.endswith('"'):
                        terms.append(line.strip('"'))
                
                extract_terms_results[idx] = terms if terms else ["ERROR: Failed to extract terms"]
        
        # Update the term extraction state if there are results
        if extract_terms_results:
            extract_snapshot = term_extraction_workflow.get_state(config)
            extract_state = extract_snapshot.values if hasattr(extract_snapshot, "values") else None
            
            if extract_state:
                # Update extracted terms
                if hasattr(extract_state, "extracted_terms"):
                    extract_state.extracted_terms = extract_terms_results
                else:
                    # Fall back to dictionary approach if needed
                    term_extraction_workflow.update_state(
                        config, 
                        {"extracted_terms": extract_terms_results}
                    )
        
        return {
            "success": True,
            "extracted_terms": extract_terms_results,
            "message": "Direct extraction completed"
        }
    
    except Exception as e:
        import traceback
        print(f"Error in direct extraction: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}

@app.get("/extracted-terms/{session_id}")
def get_extracted_terms(session_id: str):
    """Retrieve the terms extracted by the term_extractor agent."""
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # Get the debug information which we know works
        debug_info = debug_session(session_id)
        
        # Extract the terms directly from the debug info
        extracted_terms = debug_info.get("extracted_terms", {})
        current_index = debug_info.get("current_index", 0)
        is_complete = debug_info.get("is_complete", False)
        extraction_queue = debug_info.get("extraction_queue", [])
        
        return {
            "extracted_terms": extracted_terms,
            "current_index": current_index,
            "is_complete": is_complete,
            "extraction_queue_length": len(extraction_queue)
        }
    except Exception as e:
        import traceback
        print(f"Error retrieving extracted terms: {str(e)}")
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
        # Use parent_workflow with extract_terms action
        # Make sure the config is passed as a keyword argument
        result = parent_workflow.invoke(
            {"action": "extract_terms"}, 
            config=config  # Pass as keyword argument
        )
        
        # Get the updated extraction status
        extract_state_snapshot = term_extraction_workflow.get_state(config)
        extract_state = extract_state_snapshot.values if hasattr(extract_state_snapshot, "values") else None
        
        # Check if we got a valid state object
        if extract_state is not None:
            # Access attributes directly if it's a ChatState object, or as dict if it's a dict
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
        print(f"Error triggering term extraction: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error triggering term extraction: {str(e)}")

@app.get("/debug/{session_id}")
def debug_session(session_id: str):
    """Retrieve the current session state for debugging."""
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # Get data from both agents using the parent workflow
        combined_state = get_full_state(session_id)
        
        # Get more detailed state information
        current_state = parent_workflow.get_state(config)
        if hasattr(current_state, "values"):
            state = current_state.values
            detailed_state = {
                "current_index": state.current_question_index,
                "is_complete": state.is_complete,
                "conversation_history": state.conversation_history,
                "responses": state.responses,
                "extraction_queue": state.term_extraction_queue,
                "verified_answers": state.verified_answers,
                "last_extracted_index": state.last_extracted_index
            }
            
            # Combine with the full state
            return {
                **detailed_state,
                "extracted_terms": combined_state.get("extracted_terms", {})
            }
        
        # Fallback to just returning the combined state
        return combined_state
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)