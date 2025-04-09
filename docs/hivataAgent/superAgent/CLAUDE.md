# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run Commands
- Run API: `python -m api`
- Run StreamlitQA: `streamlit run StreamlitQA.py`
- Run tests: `python -m unittest discover -p "test_*.py"`

## Project Structure
- Core workflow orchestration: `parent_workflow.py`
- State management: `state.py` 
- Agent tasks: `question_processor.py`, `answer_verifier.py`, `term_extractor.py`
- API endpoints: `api.py`
- Shared memory: `shared_memory.py`
- Logging: `logging_config.py`

## Code Style Guidelines
- **Imports**: Standard library first, third-party libraries next, local modules last
- **Formatting**: 4-space indentation, 80-120 character line length
- **Types**: Use type annotations for all function parameters and return values
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Error Handling**: Use try/except with specific exceptions, log errors with context
- **Logging**: Use the configured logger from logging_config.py with appropriate levels
- **Documentation**: Docstrings for classes and functions describing purpose and parameters

## Architecture
This project implements a multi-agent workflow using LangGraph's functional API:
- Pydantic-based state management with ChatState class
- Task composition with @task decorator from langgraph.func
- Asynchronous term extraction using threading
- Shared memory persistence with checkpointing
- FastAPI endpoints for RESTful API access