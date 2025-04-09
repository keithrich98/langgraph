# test_api.py
import requests
from time import sleep

# Base URL of the API (modify if needed)
BASE_URL = "http://localhost:8000"

def print_conversation(conversation):
    """Print the conversation history in a readable format."""
    for msg in conversation:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        print(f"{role}: {content}\n")

def main():
    """Interactive flow to test the /start and /next endpoints."""
    print("Starting a new questionnaire session...\n")
    
    # Start a new session
    response = requests.post(f"{BASE_URL}/start")
    if response.status_code != 200:
        print("Error starting session:", response.text)
        return

    session_data = response.json()
    session_id = session_data["session_id"]
    print(f"Session ID: {session_id}\n")
    
    # Print the conversation history (e.g. the first question)
    print("Initial conversation:")
    print_conversation(session_data["conversation_history"])
    
    # Loop until the session indicates completion
    finished = session_data.get("finished", False)
    waiting = session_data.get("waiting_for_input", False)
    while waiting and not finished:
        # Prompt the user for an answer
        answer = input("Your answer: ")
        print()  # For readability
        
        # Submit the answer to the /next endpoint
        response = requests.post(
            f"{BASE_URL}/next",
            json={"session_id": session_id, "answer": answer}
        )
        if response.status_code != 200:
            print("Error submitting answer:", response.text)
            return
        session_data = response.json()
        
        # Print the updated conversation history
        print("Updated conversation:")
        print_conversation(session_data["conversation_history"])
        
        # Optionally show the current question
        current_question = session_data.get("current_question", "")
        if current_question:
            print("Current Question:")
            print(current_question, "\n")
        
        waiting = session_data.get("waiting_for_input", False)
        finished = session_data.get("finished", False)
    
    if finished:
        print("Questionnaire session completed successfully.")
    else:
        print("No further input is required.")

if __name__ == "__main__":
    main()
