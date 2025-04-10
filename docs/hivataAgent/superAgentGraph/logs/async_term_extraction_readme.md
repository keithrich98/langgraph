# Async Term Extraction Implementation Guide

This document provides instructions on how to test and verify the asynchronous term extraction implementation.

## Background

The term extraction functionality has been redesigned to run as a true background process outside of the LangGraph flow. This allows the questionnaire flow to continue without being blocked by term extraction, while terms are processed in parallel.

## Key Files Modified

1. **parent_workflow.py**
   - Added `process_extraction_in_background` function to process extraction in a separate thread
   - Modified `process_answer_node` to start background threads for extraction
   - Simplified the router logic since extraction no longer needs special routing

2. **api.py**
   - Simplified the `/terms/{session_id}` endpoint to just return data without triggering extraction
   - The extraction is now handled automatically in the background via threading

3. **test_questionnaire_manual.py**
   - Enhanced with better term extraction testing
   - Added functionality to check terms from previous questions
   - Provides visual feedback on extraction status

## How the Async Extraction Works

1. When a valid answer is verified, a background thread is immediately spawned to process the extraction
2. The term extraction happens completely outside the graph flow
3. The state is updated atomically once extraction is complete
4. The API endpoints simply report the current state without trying to trigger extraction

## Testing the Implementation

### Step 1: Start the API Server

```bash
cd /Users/keithrichards/Projects/langgraph/docs/hivataAgent/superAgentGraph
python -m api
```

The API server should start and listen on port 8000.

### Step 2: Run the Test Script

In a different terminal window:

```bash
cd /Users/keithrichards/Projects/langgraph/docs/hivataAgent/superAgentGraph
python -m test_questionnaire_manual
```

The test script will:
1. Start a new session
2. Automatically answer questions using sample answers
3. Check for extracted terms after each question
4. Provide a final verification of all extracted terms

### What to Look For

- The test script should show the conversation progress with automated answers
- After answering the first question, it will start checking for terms from previous questions
- You should see the "EXTRACTED TERMS" section populated as terms are extracted
- The extraction happens in the background while you continue answering questions
- The final check should show terms for all answered questions

### Manual Testing

If you want to run a manual test (answering questions yourself), modify the script call:

```python
if __name__ == "__main__":
    main(use_automated_answers=False)
```

## Verification Results

If the implementation is working correctly, you should see:

1. Successful extraction of terms in the background
2. Uninterrupted question flow (no delays between questions)
3. Complete extraction of all terms by the end of the questionnaire
4. No errors or exceptions during the process

## Troubleshooting

If terms aren't being extracted:

1. Check the logs for any error messages from the background thread
2. Verify that the thread_id is being properly passed to process_extraction_in_background
3. Make sure the extraction thread has access to the required libraries and models
4. Increase the delay or retry count in check_terms_for_previous_questions for more time