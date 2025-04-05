# test_questionnaire.py
import requests
import json
from time import sleep

# Base URL of the API (change if deployed elsewhere)
BASE_URL = "http://localhost:8000"

def print_conversation(conversation):
    """Print the conversation history in a readable format."""
    for message in conversation:
        role = message["role"].upper()
        content = message["content"]
        print(f"{role}: {content}\n")

def main():
    """Interactive questionnaire flow with verification."""
    print("Starting a new questionnaire session...\n")
    
    # Start a new session
    response = requests.post(f"{BASE_URL}/start")
    if response.status_code != 200:
        print(f"Error starting session: {response.text}")
        return
    
    session_data = response.json()
    session_id = session_data["session_id"]
    print(f"Session ID: {session_id}\n")
    
    # Print the first question
    print_conversation(session_data["conversation_history"])
    
    # Loop until the questionnaire is complete
    while not session_data.get("is_complete", False):
        # Get user input
        answer = input("Your answer: ")
        print()  # Empty line for readability
        
        # Submit the answer
        response = requests.post(
            f"{BASE_URL}/answer",
            json={"session_id": session_id, "answer": answer}
        )
        
        if response.status_code != 200:
            print(f"Error submitting answer: {response.text}")
            return
        
        session_data = response.json()
        
        # Print the AI's response (verification or next question)
        # Get the last message (the AI's response)
        ai_messages = [msg for msg in session_data["conversation_history"] if msg["role"] == "system"]
        if ai_messages:
            latest_ai_message = ai_messages[-1]
            print(f"AI: {latest_ai_message['content']}\n")
    
    # After completion, retrieve the full session state
    print("\n=== Questionnaire Completed Successfully ===")
    
    response = requests.get(f"{BASE_URL}/session/{session_id}")
    if response.status_code != 200:
        print(f"Error getting session summary: {response.text}")
        return
    
    summary = response.json()
    
    print(f"\nTotal questions answered: {len(summary.get('responses', {}))}")
    
    # Print a summary of all verified responses
    print("\n=== Verified Answers ===")
    verified_responses = summary.get("verified_responses", {})
    for q_idx_str, is_valid in verified_responses.items():
        q_idx = int(q_idx_str)
        if q_idx < len(summary.get("conversation_history", [])):
            # Find the question text (it's at even indices, starting from 0)
            question_entry = summary["conversation_history"][q_idx * 2]
            question_text = question_entry["content"].split("\n\n")[0]  # Get just the question without requirements
            print(f"Question {q_idx + 1}: {question_text}")
            print(f"Valid: {is_valid}")
            
            # Print the response if it exists
            if q_idx_str in summary.get("responses", {}):
                print(f"Your answer: {summary['responses'][q_idx_str]}")
            print()

if __name__ == "__main__":
    main()