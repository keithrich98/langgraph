#!/usr/bin/env python3
# automated_test_questionnaire.py
import requests
import json
import time
import sys

# Base URL of the API
BASE_URL = "http://localhost:8000"

# Predefined answers for testing
TEST_ANSWERS = {
    # Question 1 - First attempt (incomplete)
    "q1_attempt1": "3 months old",
    
    # Question 1 - Second attempt (completing the answer)
    "q1_attempt2": "november 20 2014 date of diagnosis and microcephaly was only symptom that led to the diagnosis",
    
    # Question 2 - Complete answer
    "q2": ("I experience seizures (severe), delayed speech development (moderate), and weakness "
           "on my right side (mild to moderate). The seizures significantly impact my ability to "
           "drive and participate in some activities. The speech delays make communication in "
           "group settings challenging, and the weakness affects my ability to perform certain "
           "physical tasks requiring fine motor control."),
    
    # Question 3 - Complete answer
    "q3": ("I had an MRI performed in 2014 that showed extensive polymicrogyria affecting the "
           "left frontoparietal region. The radiologist noted abnormal cortical development with "
           "excessive number of small gyri and shallow sulci. The report mentioned that this pattern "
           "is consistent with perisylvian polymicrogyria and likely explains the neurological "
           "symptoms I'm experiencing.")
}

def print_with_color(text, color):
    """Print text with color."""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'end': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def print_separator():
    """Print a separator line."""
    print_with_color("=" * 80, 'cyan')

def print_conversation(conversation):
    """Print the conversation history in a readable format."""
    for message in conversation:
        role = message["role"].upper()
        content = message["content"]
        
        if role == "SYSTEM":
            if "Requirements:" in content:
                # This is a question
                print_with_color(f"{role}: {content}", 'yellow')
            elif "Analysis of Patient's Answer" in content:
                # This is verification feedback
                print_with_color(f"{role}: {content}", 'purple')
            else:
                # Other system messages
                print_with_color(f"{role}: {content}", 'blue')
        else:
            # User messages
            print_with_color(f"{role}: {content}", 'green')
        print()

def print_latest_system_message(conversation_history):
    """Print only the latest system message."""
    system_messages = [msg for msg in conversation_history if msg["role"] == "system"]
    if system_messages:
        latest_message = system_messages[-1]
        
        if "Requirements:" in latest_message["content"]:
            # This is a question
            print_with_color(f"SYSTEM: {latest_message['content']}", 'yellow')
        elif "Analysis of Patient's Answer" in latest_message["content"]:
            # This is verification feedback
            print_with_color(f"SYSTEM: {latest_message['content']}", 'purple')
        else:
            # Other system messages
            print_with_color(f"SYSTEM: {latest_message['content']}", 'blue')
        print()

def analyze_system_message(conversation_history):
    """
    Analyze the latest system message to determine:
    1. Whether it's a verification message
    2. Whether verification failed or passed
    3. Whether it's a new question
    
    Returns a tuple: (is_verification, passed_verification, is_new_question)
    """
    system_messages = [msg for msg in conversation_history if msg["role"] == "system"]
    if not system_messages:
        return (False, False, False)
    
    latest_message = system_messages[-1]
    content = latest_message["content"].lower()
    
    # Check if it's a verification message
    is_verification = any(phrase in content for phrase in [
        "analysis", "assessment", "requirement", "valid", "invalid"
    ]) and not "requirements:" in content
    
    # Check if verification passed
    passed_verification = is_verification and any(phrase in content for phrase in [
        "all requirements have been met",
        "all requirements are met",
        "valid",
        "meets all requirement"
    ]) and not "invalid" in content
    
    # Check if it's a new question (contains Requirements: but not verification terms)
    is_new_question = "requirements:" in content and not is_verification
    
    return (is_verification, passed_verification, is_new_question)

def run_automated_test():
    """Run the automated test for the questionnaire."""
    print_with_color("Starting automated questionnaire testing...", 'cyan')
    print_separator()
    
    # Start a new session
    try:
        response = requests.post(f"{BASE_URL}/start")
        response.raise_for_status()  # Raise exception for non-200 responses
    except requests.RequestException as e:
        print_with_color(f"Error starting session: {str(e)}", 'red')
        return
    
    session_data = response.json()
    session_id = session_data["session_id"]
    print_with_color(f"Session ID: {session_id}", 'cyan')
    print_separator()
    
    # Print the first question
    print_with_color("INITIAL QUESTION:", 'cyan')
    print_conversation(session_data["conversation_history"])
    print_separator()
    
    # Track the current question index
    current_question_index = 0
    
    # Track completed questions and whether we're on a retry attempt
    attempted_questions = set()
    is_retry_attempt = False
    current_question_text = ""
    
    # Extract the current question text from initial question
    if session_data["conversation_history"]:
        current_question_text = session_data["conversation_history"][0]["content"].split("\n\n")[0]
    
    # Loop until the questionnaire is complete
    while not session_data.get("is_complete", False):
        # Analyze the latest system message to determine our state
        is_verification, passed_verification, is_new_question = analyze_system_message(session_data["conversation_history"])
        
        # If there's a new question, update our tracking
        if is_new_question:
            system_messages = [msg for msg in session_data["conversation_history"] if msg["role"] == "system"]
            current_question_text = system_messages[-1]["content"].split("\n\n")[0]
            is_retry_attempt = False
            
        # Determine which answer to submit based on the question and verification status
        if "at what age were you diagnosed" in current_question_text.lower():
            if is_verification and not passed_verification and not is_retry_attempt:
                # If verification failed on first attempt, use the second attempt answer
                answer = TEST_ANSWERS["q1_attempt2"]
                print_with_color("SUBMITTING SECOND ATTEMPT FOR QUESTION 1:", 'cyan')
                is_retry_attempt = True
                attempted_questions.add(0)
            elif not attempted_questions or 0 not in attempted_questions:
                # First attempt for question 1
                answer = TEST_ANSWERS["q1_attempt1"]
                print_with_color("SUBMITTING FIRST ATTEMPT FOR QUESTION 1:", 'cyan')
                attempted_questions.add(0)
        elif "what symptoms or neurological issues" in current_question_text.lower():
            answer = TEST_ANSWERS["q2"]
            print_with_color("SUBMITTING ANSWER FOR QUESTION 2:", 'cyan')
            attempted_questions.add(1)
        elif "describe the key findings" in current_question_text.lower():
            answer = TEST_ANSWERS["q3"]
            print_with_color("SUBMITTING ANSWER FOR QUESTION 3:", 'cyan')
            attempted_questions.add(2)
        else:
            print_with_color(f"Unexpected question: {current_question_text[:50]}...", 'red')
            break
        
        print_with_color(f"USER: {answer}", 'green')
        print()
        
        # Submit the answer with a retry mechanism
        max_retries = 3
        retry_delay = 1  # seconds
        for retry in range(max_retries):
            try:
                response = requests.post(
                    f"{BASE_URL}/answer",
                    json={"session_id": session_id, "answer": answer}
                )
                response.raise_for_status()
                break
            except requests.RequestException as e:
                if retry < max_retries - 1:
                    print_with_color(f"Error submitting answer, retrying in {retry_delay}s: {str(e)}", 'yellow')
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print_with_color(f"Error submitting answer after {max_retries} retries: {str(e)}", 'red')
                    return
        
        session_data = response.json()
        
        # Print the system's response
        print_with_color("SYSTEM RESPONSE:", 'cyan')
        print_latest_system_message(session_data["conversation_history"])
        print_separator()
        
        # If all questions have been attempted and the last verification passed, we're likely done
        if len(attempted_questions) >= 3 and (passed_verification or is_new_question):
            print_with_color("All questions have been attempted. Checking completion status...", 'cyan')
            # Double-check with server to confirm completion status
            status_response = requests.get(f"{BASE_URL}/session/{session_id}")
            session_data = status_response.json()
            if not session_data.get("is_complete", False):
                print_with_color("Server reports questionnaire not yet complete. Continuing...", 'yellow')
            else:
                print_with_color("Server confirms questionnaire completion.", 'green')
                break
    
    # After completion, retrieve the full session state
    print_with_color("\n=== QUESTIONNAIRE COMPLETED SUCCESSFULLY ===", 'cyan')
    
    try:
        response = requests.get(f"{BASE_URL}/session/{session_id}")
        response.raise_for_status()
    except requests.RequestException as e:
        print_with_color(f"Error getting session summary: {str(e)}", 'red')
        return
    
    summary = response.json()
    
    print_with_color(f"\nTotal questions answered: {len(summary.get('responses', {}))}", 'cyan')
    
    # Print a summary of all verified responses
    print_with_color("\n=== VERIFIED ANSWERS ===", 'cyan')
    verified_responses = summary.get("verified_responses", {})
    
    # Get all questions from conversation history
    all_questions = []
    system_messages = summary.get("conversation_history", [])
    for msg in system_messages:
        if msg["role"] == "system" and "Requirements:" in msg["content"]:
            all_questions.append(msg["content"].split("\n\n")[0])
    
    # Identify each question by its content
    question_mapping = {
        0: next((q for q in all_questions if "at what age were you diagnosed" in q.lower()), ""),
        1: next((q for q in all_questions if "what symptoms or neurological issues" in q.lower()), ""),
        2: next((q for q in all_questions if "describe the key findings" in q.lower()), "")
    }
    
    # Print each question and its response
    for q_idx_str, is_valid in sorted(verified_responses.items()):
        q_idx = int(q_idx_str)
        
        # Get the appropriate question text using the mapping
        if q_idx in question_mapping and question_mapping[q_idx]:
            question_text = question_mapping[q_idx]
            print_with_color(f"Question {q_idx + 1}: {question_text}", 'yellow')
            print_with_color(f"Valid: {is_valid}", 'green' if is_valid else 'red')
            
            # Print the full response
            if q_idx_str in summary.get("responses", {}):
                print_with_color(f"Your answer: {summary['responses'][q_idx_str]}", 'green')
            print()

if __name__ == "__main__":
    try:
        run_automated_test()
    except KeyboardInterrupt:
        print_with_color("\nTest interrupted by user.", 'yellow')
        sys.exit(0)
    except Exception as e:
        print_with_color(f"\nUnexpected error: {str(e)}", 'red')
        sys.exit(1)