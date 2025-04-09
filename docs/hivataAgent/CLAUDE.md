# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run Commands
- Run API: `python -m hivataAgent.hybrid_approach.api.api`
- Run automated tests: `python -m hivataAgent.hybrid_approach.tests.test_questionnaire_automated`
- Run manual tests: `python -m hivataAgent.hybrid_approach.tests.test_questionnaire_manual`

## Project Structure
- `agents/` - Agent implementation files (question, verification, term extraction)
- `api/` - REST API endpoints and request handling
- `config/` - Configuration files including logging setup
- `core/` - Core workflow and state management
- `services/` - Utility services like thread management
- `tests/` - Test files for automated and manual testing

## Code Style Guidelines
- **Imports**: Standard library first, third-party libraries next, local modules last
- **Formatting**: 4-space indentation, 80-120 character line length
- **Types**: Use type annotations for all function parameters and return values
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Error Handling**: Use try/except with specific exceptions, log errors with context
- **Logging**: Use the configured logger from logging_config.py with appropriate levels

## Architecture
- State management in `core/state.py` using Pydantic models
- REST API endpoints in `api/api.py`
- Business logic in `core/parent_workflow.py` and agent files
- Memory/persistence in `core/shared_memory.py`
- Asynchronous processing in `services/thread_manager.py`

## LangGraph Integration
The project uses LangGraph's functional API for workflow orchestration. See `code_review_task_list.txt` for a detailed analysis of our LangGraph integration approach and improvements that have been made, including:

1. Pydantic-based state management
2. Graph-based routing
3. Improved task composition for term extraction
4. Thread management for asynchronous operations

## Known Challenges
- Serialization of Future objects from LangGraph's @task decorator
- Balance between LangGraph idioms and practical implementation
- Asynchronous term extraction requiring careful thread management

## Documentation
- The code_review_task_list.txt file contains detailed information about architectural decisions, implementation details, and future improvements
- Each module has docstrings explaining its purpose and functionality
- API endpoints are documented with FastAPI annotations