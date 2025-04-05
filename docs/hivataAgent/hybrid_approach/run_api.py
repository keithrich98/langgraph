#!/usr/bin/env python3
"""
API Runner script for the Hybrid Approach questionnaire

This script adds the necessary path configurations to run the API
without needing to adjust the PYTHONPATH environment variable.
"""
import sys
import os
import uvicorn

# Add the parent directory (containing hivataAgent) to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

# Import API app to verify imports work
from hivataAgent.hybrid_approach.api.api import app

if __name__ == "__main__":
    print(f"Starting API server...")
    print(f"Python path includes: {parent_dir}")
    
    # Run the Uvicorn server
    uvicorn.run(
        "hivataAgent.hybrid_approach.api.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload for development
    )