# term_extractor.py - Medical Term Extraction Agent

import os
import time
from typing import Dict, List, Any, Optional
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate

# In term_extractor.py
# Replace the existing term_extraction_memory definition
from shared_memory import shared_memory

# The entrypoint decorator should remain the same as it will now use the shared memory

# Import the shared state
from state import ChatState, get_next_extraction_task, mark_extraction_complete

# Set up the LLM for term extraction
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in your environment.")
llm = ChatOpenAI(
    model_name="gpt-4", 
    openai_api_key=openai_api_key,
    temperature=0
)

# Define the prompt template for term extraction
extraction_prompt = ChatPromptTemplate.from_template("""
You are an agent who is an expert in generating API search terms that are going to be used to hit the National Library of Medicine's Unified Medical Language System (UMLS) API.

<api_description>
The Search API is designed to help users query the UMLS Metathesaurus to find relevant biomedical and health-related concepts. It acts as a search engine for the UMLS, enabling users to look up terms, concepts, and their associated metadata. For example, if you search for a medical term like "myocardial infarction," the API will return information about the concept, including its unique identifier (CUI), definitions, synonyms, and relationships to other concepts in the UMLS.
</api_description>

<task>
- Use the context given to you to generate one or more human readable term, such as 'gestational diabetes', 'heart attack' to be used as the `query` parameter for the Search API.
- Use your vast medical knowledge to look for potential abbreviations or de-abbreviations for a search term thus generating variations for the same search term if possible.
- Extract medical, social, and other entities from context, like "paternal cousin", "great grand mother", "sugar", "cigarettes", etc.
</task>

<context>
Summary:
Question: {question}
Answer: {answer}
Purpose or reason for the UMLS code:
To extract relevant medical terminology for better understanding and categorization of patient data.
</context>

Based on the context, provide a list of search terms for the UMLS API. Format your response as a JSON array of strings.
""")

@task
def extract_terms(question: str, answer: str, config: Optional[RunnableConfig] = None) -> List[str]:
    """
    Extract medical terminology from a question-answer pair.
    
    Args:
        question: The medical question
        answer: The user's answer
        config: Optional runtime configuration
    
    Returns:
        List of extracted medical terms
    """
    try:
        # Prepare the prompt with the question and answer
        formatted_prompt = extraction_prompt.format(
            question=question,
            answer=answer
        )
        
        # Extract terms using the LLM
        response = llm.invoke(formatted_prompt, config=config)
        content = response.content if hasattr(response, "content") else str(response)
        
        # Parse the response - expected format is a JSON array of strings
        # Handle potential formatting issues
        try:
            import json
            content = content.strip()
            if not content.startswith('['):
                start_idx = content.find('[')
                if start_idx >= 0:
                    content = content[start_idx:]
                    bracket_count = 0
                    for i, char in enumerate(content):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                content = content[:i+1]
                                break
            terms = json.loads(content)
            if isinstance(terms, list):
                return terms
            return ["ERROR: Expected array of terms"]
        except Exception as e:
            print(f"Error parsing extracted terms: {str(e)}")
            lines = content.split('\n')
            terms = []
            for line in lines:
                line = line.strip()
                if line and line.startswith('-'):
                    terms.append(line[1:].strip())
                elif line and line.startswith('"') and line.endswith('"'):
                    terms.append(line.strip('"'))
            return terms if terms else ["ERROR: Failed to extract terms"]
    
    except Exception as e:
        print(f"Error in term extraction: {str(e)}")
        return ["ERROR: Term extraction failed"]

def debug_state(prefix: str, state: ChatState):
    """Log the key details of the state for debugging."""
    print(f"[DEBUG {prefix}] conversation_history length: {len(state.conversation_history)}")
    print(f"[DEBUG {prefix}] current_question_index: {state.current_question_index}")
    print(f"[DEBUG {prefix}] responses count: {len(state.responses)}")
    print(f"[DEBUG {prefix}] term_extraction_queue: {state.term_extraction_queue}")
    print(f"[DEBUG {prefix}] extracted_terms count: {len(state.extracted_terms)}")
    print(f"[DEBUG {prefix}] thread_id: {id(state)}")  # Help identify if state objects are the same

@entrypoint(checkpointer=shared_memory)
def term_extraction_workflow(action: Dict = None, *, previous: ChatState = None, config: Optional[RunnableConfig] = None) -> ChatState:
    # Get state from action or previous, or create new if neither exists
    state = action.get("state", None) or (previous if previous is not None else ChatState())
    
    # Debugging track the thread_id if available
    thread_id = config.get("configurable", {}).get("thread_id") if config else "unknown"
    print(f"[DEBUG TE] Running with thread_id: {thread_id}")
    debug_state("TE-Initial", state)
    
    if action is None:
        print("[DEBUG TE] No action provided; returning state.")
        return state
    
    if action.get("action") == "process":
        print("[DEBUG TE] Processing extraction action.")
        next_index = get_next_extraction_task(state)
        
        if next_index is not None and next_index in state.verified_answers:
            verified_item = state.verified_answers[next_index]
            question = verified_item.get("question", "")
            answer = verified_item.get("answer", "")
            
            # Log the question and answer we're processing (truncated for readability)
            print(f"[DEBUG TE] Processing extraction for index {next_index}")
            print(f"[DEBUG TE] Question: {question[:50]}{'...' if len(question) > 50 else ''}")
            print(f"[DEBUG TE] Answer: {answer[:50]}{'...' if len(answer) > 50 else ''}")
            
            # Extract terms
            terms = extract_terms(question, answer).result()
            print(f"[DEBUG TE] Extracted terms: {terms}")
            
            # Update state with extracted terms
            state.extracted_terms[next_index] = terms
            state = mark_extraction_complete(state, next_index)
            debug_state("TE-AfterExtraction", state)
        else:
            print("[DEBUG TE] No valid items in extraction queue.")
    
    elif action.get("action") == "status":
        extraction_status = {
            "queue_length": len(state.term_extraction_queue),
            "processed_count": len(state.extracted_terms),
            "next_in_queue": get_next_extraction_task(state)
        }
        print(f"[DEBUG TE] Extraction status: {extraction_status}")
    
    debug_state("TE-Final", state)
    return state
