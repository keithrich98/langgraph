# test_questionnaire_manual.py
import requests
import time
import json
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored terminal text
init()

# Base URL of the API (modify if needed)
BASE_URL = "http://localhost:8000"

def print_colored(text, color=Fore.WHITE, bold=False):
    """Print text with specified color."""
    style = Style.BRIGHT if bold else ""
    print(f"{style}{color}{text}{Style.RESET_ALL}")

def print_conversation(conversation):
    """Print the conversation history in a readable format."""
    print_colored("\n=== CONVERSATION HISTORY ===", Fore.CYAN, bold=True)
    for i, msg in enumerate(conversation):
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        
        if role == "AI":
            print_colored(f"[{i}] {role}: {content}\n", Fore.GREEN)
        elif role == "HUMAN":
            print_colored(f"[{i}] {role}: {content}\n", Fore.YELLOW)
        else:
            print_colored(f"[{i}] {role}: {content}\n", Fore.WHITE)

def check_terms_for_previous_questions(session_id, question_indices, max_retries=5):  # default increased to 5
    """
    Check extracted terms for specific question indices.
    Retries a few times with delays to allow background extraction to complete.
    """
    if not question_indices:
        return
        
    print_colored(f"\nChecking for extracted terms from questions {question_indices}...", Fore.CYAN)
    
    for attempt in range(max_retries):
        response = requests.get(f"{BASE_URL}/terms/{session_id}")
        if response.status_code != 200:
            print_colored(f"Error retrieving terms: {response.text}", Fore.RED)
            return
            
        data = response.json()
        terms = data.get("terms", {})
        pending = data.get("pending_extraction", [])
        
        extracted_indices = [int(idx) for idx in terms.keys()]
        pending_indices = [idx for idx in question_indices if idx in pending]
        missing_indices = [idx for idx in question_indices if idx not in extracted_indices and idx not in pending]
        
        if any(str(idx) in terms for idx in question_indices):
            print_colored("\n=== EXTRACTED TERMS ===", Fore.MAGENTA, bold=True)
            for idx in question_indices:
                str_idx = str(idx)
                if str_idx in terms:
                    print_colored(f"Question {idx}:", Fore.MAGENTA, bold=True)
                    for term in terms[str_idx]:
                        print_colored(f"  • {term}", Fore.MAGENTA)
        
        if pending_indices:
            print_colored(f"Terms still being extracted for questions: {pending_indices}", Fore.YELLOW)
        if missing_indices:
            print_colored(f"No terms found or pending for questions: {missing_indices}", Fore.RED)
        
        if all(str(idx) in terms for idx in question_indices) or attempt == max_retries - 1:
            break
            
        wait_time = 2 * (attempt + 1)
        print_colored(f"Waiting for extraction to complete (attempt {attempt+1}/{max_retries}). Sleeping for {wait_time} seconds...", Fore.YELLOW)
        time.sleep(wait_time)

def main(use_automated_answers=True):
    """
    Interactive test flow with improved term extraction testing.
    
    Args:
        use_automated_answers: If True, use predefined answers instead of prompting for input
    """
    print_colored("=== QUESTIONNAIRE TEST WITH TERM EXTRACTION ===", Fore.CYAN, bold=True)
    
    # Sample predefined answers for automated testing
    sample_answers = [
        "I was diagnosed with polymicrogyria at 3 months old in June 2018.", #My primary symptoms included speech delays, difficulty with fine motor skills, and occasional seizures that started when I was around 4 years old.
        "My neurologist recommended physical therapy 3 times a week, speech therapy twice weekly, and prescribed Keppra (levetiracetam) at 500mg twice daily to control the seizures.",
        "Yes, I have family members with neurological conditions. My father has epilepsy and my maternal aunt was diagnosed with cerebral palsy as a child."
    ]
    
    try:
        # Start a new session
        print_colored("\nStarting a new questionnaire session...", Fore.CYAN)
        response = requests.post(f"{BASE_URL}/start")
        if response.status_code != 200:
            print_colored(f"Error starting session: {response.text}", Fore.RED)
            return

        # Extract and display session information
        session_data = response.json()
        session_id = session_data["session_id"]
        print_colored(f"Session ID: {session_id}", Fore.GREEN, bold=True)
        
        # Track answered questions for term extraction checking
        answered_questions = []
        answer_index = 0
        
        # Loop until the session indicates completion
        finished = session_data.get("finished", False)
        waiting = session_data.get("waiting_for_input", True)
        
        while waiting and not finished:
            # Show current question
            print_conversation(session_data["conversation_history"])
            current_question = session_data.get("current_question", "")
            if current_question:
                print_colored("\nCurrent Question:", Fore.CYAN)
                print_colored(current_question, Fore.WHITE)
            
            # Get answer - either from predefined list or from user input
            if use_automated_answers and answer_index < len(sample_answers):
                answer = sample_answers[answer_index]
                answer_index += 1
                print_colored(f"\nAutomated answer: {answer}", Fore.YELLOW)
            else:
                answer = input("\nYour answer: ")
            
            # Submit the answer to the /next endpoint
            print_colored("\nSubmitting answer...", Fore.CYAN)
            response = requests.post(
                f"{BASE_URL}/next",
                json={"session_id": session_id, "answer": answer}
            )
            
            if response.status_code != 200:
                print_colored(f"Error submitting answer: {response.text}", Fore.RED)
                return
                
            session_data = response.json()
            
            # After each answer, check if we have a new question
            if not session_data.get("waiting_for_input", True) or session_data.get("finished", False):
                # Session ended
                break
            
            # Record that we just answered a question successfully
            current_idx = len(answered_questions)
            answered_questions.append(current_idx)
            
            # Check for terms from previous questions (not the one just answered)
            if len(answered_questions) > 1:
                check_terms_for_previous_questions(session_id, answered_questions[:-1])
            
            # Update loop control variables
            waiting = session_data.get("waiting_for_input", False)
            finished = session_data.get("finished", False)
        
        if finished:
            print_colored("\nQuestionnaire session completed successfully.", Fore.GREEN, bold=True)
            
            # Final check for all extracted terms
            print_colored("\nPerforming final check for extracted terms...", Fore.CYAN)
            time.sleep(3)  # Give a bit more time for extraction to complete
            check_terms_for_previous_questions(session_id, answered_questions)
            
            # Additional checks with longer delay to ensure all extractions have completed
            time.sleep(5)
            print_colored("\nFinal verification with extended waiting time...", Fore.CYAN)
            check_terms_for_previous_questions(session_id, answered_questions, max_retries=5)
            
        else:
            print_colored("\nNo further input is required.", Fore.YELLOW)
            
    except KeyboardInterrupt:
        print_colored("\nTest interrupted by user.", Fore.YELLOW)
    except Exception as e:
        print_colored(f"\nError during test: {str(e)}", Fore.RED)
        import traceback
        print_colored(f"\nStacktrace: {traceback.format_exc()}", Fore.RED)

if __name__ == "__main__":
    main()