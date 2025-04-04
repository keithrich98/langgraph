# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run Commands
- Run API: `python hybrid_approach/api.py`
- Run automated tests: `python hybrid_approach/test_questionnaire_automated.py`
- Run manual tests: `python hybrid_approach/test_questionnaire_manual.py`

## Code Style Guidelines
- **Imports**: Standard library first, third-party libraries next, local modules last
- **Formatting**: 4-space indentation, 80-120 character line length
- **Types**: Use type annotations for all function parameters and return values
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Error Handling**: Use try/except with specific exceptions, log errors with context
- **Logging**: Use the configured logger from logging_config.py with appropriate levels

## Architecture
- State management in state.py
- REST API endpoints in api.py
- Business logic in parent_workflow.py and agent files
- Memory/persistence in shared_memory.py