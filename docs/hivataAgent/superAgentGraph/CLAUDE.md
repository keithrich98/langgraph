# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run Commands
- Run API: `python -m api`
- Run StreamlitQA: `streamlit run StreamlitQA.py`
- Run all tests: `python -m unittest discover -p "test_*.py"`
- Run a single test: `python -m unittest test_questionnaire_manual.py`

## Project Structure
- Core workflow orchestration: `parent_workflow.py` with LangGraph StateGraph
- State management: `state.py` using Pydantic models
- Agent tasks: `question_processor.py`, `answer_verifier.py`, `term_extractor.py`
- API endpoints: `api.py` using FastAPI
- Shared memory: `shared_memory.py` for state persistence
- Logging: `logging_config.py` for centralized log configuration

## Code Style Guidelines
- **Imports**: Standard library first, third-party next, local modules last
- **Formatting**: 4-space indentation, 80-120 character line length
- **Types**: Type annotations for function parameters and return values
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Error Handling**: Try/except with specific exceptions, detailed logging
- **Logging**: Use configured logger with appropriate levels (debug, info, error)
- **Documentation**: Docstrings for all functions and classes

## Architecture
This project implements a questionnaire system using LangGraph:
- Graph-based workflow with nodes for question processing, verification, and term extraction
- Human-in-the-loop functionality using LangGraph's interrupt mechanism
- Pydantic-based state management with immutable state updates
- Asynchronous term extraction with proper thread management
- FastAPI endpoints for RESTful API access
- Comprehensive logging with context and state information