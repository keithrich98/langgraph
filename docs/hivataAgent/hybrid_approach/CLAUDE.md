# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run Commands
- Run API (recommended): `python hybrid_approach/run_api.py`
- Run automated tests: `python hybrid_approach/run_tests.py --mode=automated`
- Run manual tests: `python hybrid_approach/run_tests.py --mode=manual`

Alternatively, you can use the Python module approach, but you need to be in the parent directory:
```bash
# From the docs directory
cd /path/to/docs
python -m hivataAgent.hybrid_approach.api.api
python -m hivataAgent.hybrid_approach.tests.test_questionnaire_automated
python -m hivataAgent.hybrid_approach.tests.test_questionnaire_manual
```

## Code Style Guidelines
- **Imports**: Standard library first, third-party libraries next, local modules last
- **Formatting**: 4-space indentation, 80-120 character line length
- **Types**: Use type annotations for all function parameters and return values
- **Naming**: snake_case for variables/functions, CamelCase for classes
- **Error Handling**: Use try/except with specific exceptions, log errors with context
- **Logging**: Use the configured logger from logging_config.py with appropriate levels

## Architecture
- State management in core/state.py
- REST API endpoints in api/api.py
- Business logic in core/parent_workflow.py and agents/ files
- Memory/persistence in core/shared_memory.py
- Logging configuration in config/logging_config.py

## Package Structure
The project follows a modular package structure:
```
hivataAgent/
└── hybrid_approach/
    ├── agents/               # Agent implementations
    │   ├── question_agent.py
    │   ├── term_extractor_agent.py
    │   └── verification_agent.py
    ├── api/                  # API endpoints and server
    │   └── api.py
    ├── config/               # Configuration files
    │   ├── logging_config.py
    │   └── logs/             # Log files directory
    ├── core/                 # Core business logic
    │   ├── parent_workflow.py
    │   ├── shared_memory.py
    │   └── state.py
    └── tests/                # Test files
        ├── test_questionnaire_automated.py
        └── test_questionnaire_manual.py
```

When importing modules, use the full package path:
```python
from hivataAgent.hybrid_approach.core.state import SessionState
from hivataAgent.hybrid_approach.config.logging_config import logger
```