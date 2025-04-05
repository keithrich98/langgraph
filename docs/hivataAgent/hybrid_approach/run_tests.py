#!/usr/bin/env python3
"""
Test Runner script for the Hybrid Approach questionnaire

This script adds the necessary path configurations to run the tests
without needing to adjust the PYTHONPATH environment variable.
"""
import sys
import os
import importlib
import argparse

# Add the parent directory (containing hivataAgent) to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, parent_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the questionnaire tests")
    parser.add_argument(
        "--mode", 
        choices=["automated", "manual"], 
        default="manual",
        help="Test mode: automated or manual"
    )
    
    args = parser.parse_args()
    
    # Import and run the specified test module
    if args.mode == "automated":
        print("Running automated tests...")
        test_module = importlib.import_module("hivataAgent.hybrid_approach.tests.test_questionnaire_automated")
    else:
        print("Running manual tests...")
        test_module = importlib.import_module("hivataAgent.hybrid_approach.tests.test_questionnaire_manual")
        
        # If the module has a main function, call it
        if hasattr(test_module, "main"):
            test_module.main()