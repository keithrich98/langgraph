# Medical Questionnaire System

A scalable medical questionnaire system with question flow and answer verification, built using LangGraph.

## Architecture

This system uses a modular, agent-based architecture for scalability:

1. **State Management** (`state.py`): Defines the central state structure shared across all agents
2. **Memory Management** (`shared_memory.py`): Provides persistence across sessions using LangGraph's checkpointing
3. **Question Agent** (`question_agent.py`): Manages the question flow
4. **Verification Agent** (`verification_agent.py`): Verifies user answers against requirements
5. **Parent Workflow** (`parent_workflow.py`): Orchestrates the communication between agents
6. **API** (`api.py`): Provides a REST interface for client applications

## Getting Started

### Prerequisites

- Python 3.8+
- FastAPI
- LangGraph

### Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install langgraph fastapi uvicorn pydantic
```

### Running the Application

1. Start the API server:

```bash
python api.py
```

2. In a separate terminal, run the test script to see the system in action:

```bash
python test_questionnaire.py
```

## API Endpoints

### Start a Session

```
POST /start
```

Starts a new questionnaire session.

**Response:**
- `session_id`: Unique identifier for the session
- `current_question`: The first question text
- `conversation_history`: Full conversation up to this point
- `is_complete`: Boolean indicating if the questionnaire is complete

### Answer a Question

```
POST /answer
```

Submit an answer to the current question.

**Request Body:**
```json
{
  "session_id": "your-session-id",
  "answer": "Your answer text"
}
```

**Response:**
- `session_id`: Session identifier
- `current_question`: Next question or follow-up verification
- `conversation_history`: Updated conversation history
- `is_complete`: Boolean indicating if the questionnaire is complete
- `verified_responses`: Map of which answers have been verified
- `verification_messages`: Feedback messages from verification

### Get Session State

```
GET /session/{session_id}
```

Retrieve the full state of a session.

**Response:** Complete session data including all responses and verification results.

## Extending the System

This architecture is designed to be scalable. To add new agents:

1. Create a new agent file (e.g., `term_extractor.py`)
2. Define tasks for that agent
3. Update the parent workflow to incorporate the new agent

For example, to add a term extraction agent:

```python
# In parent_workflow.py
from term_extractor import extract_terms

# Then add to the workflow:
if state.verification_result.get("is_valid", False):
    # Run term extraction on valid answers
    state = extract_terms(state).result()
```

## Architecture Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Question Agent │     │ Verification    │     │  Future Agents  │
│                 │     │  Agent          │     │  (Term Extract) │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────┬───────┴───────────────┬───────┘
                         │                       │
                ┌────────▼───────────┐  ┌────────▼───────────┐
                │  Parent Workflow   │  │    Shared State    │
                │   Orchestration    │  │                    │
                └────────┬───────────┘  └────────────────────┘
                         │
                ┌────────▼───────────┐
                │    REST API        │
                │                    │
                └────────┬───────────┘
                         │
                         ▼
                  Client Applications
```