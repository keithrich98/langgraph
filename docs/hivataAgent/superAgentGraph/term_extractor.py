# term_extractor.py - Implementation using langgraph.graph API

import os
import threading
import time
from typing import Dict, List, Any, Optional, Union
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# Import the logger from logging_config
from logging_config import logger

# Set up the LLM for term extraction
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in your environment.")
llm = ChatOpenAI(
    model_name="gpt-4o",
    openai_api_key=openai_api_key,
    temperature=0
)

# Thread-safe lock for asynchronous processing
extraction_lock = threading.Lock()

# Extract terms from verified answers
extraction_prompt = ChatPromptTemplate.from_template("""
You are an agent who is an expert in generating API search terms for the UMLS API.

<api_description>
The Search API helps users query the UMLS Metathesaurus for biomedical and health-related concepts.
</api_description>

<task>
- Use the provided context to generate one or more human-readable terms (e.g. "heart attack") to be used as the query.
- Consider variations, abbreviations, and related entities.
</task>

<context>
Question: {question}
Answer: {answer}
Purpose:
Extract relevant medical terminology.
</context>

Provide your answer as a JSON array of strings.
""")

def extract_terms(question: str, answer: str) -> List[str]:
    """
    Extract medical terms from the question and answer using LLM.
    
    Args:
        question: The question text
        answer: The user's verified answer
        
    Returns:
        List of extracted medical terms
    """
    try:
        logger.debug(f"Extracting terms for Q: {question[:30]}... A: {answer[:30]}...")
        formatted_prompt = extraction_prompt.format(question=question, answer=answer)
        
        # Invoke the LLM with the formatted prompt
        response = llm.invoke(formatted_prompt)
        content = response.content if hasattr(response, "content") else str(response)
        
        # Try to parse the JSON array
        try:
            content = content.strip()
            # If the response doesn't start with '[', find the first '[' and extract the JSON array
            if not content.startswith('['):
                start_idx = content.find('[')
                if start_idx >= 0:
                    content = content[start_idx:]
                    # Find the matching closing bracket
                    bracket_count = 0
                    for i, char in enumerate(content):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                content = content[:i+1]
                                break
            
            # Parse the JSON array
            terms = json.loads(content)
            if isinstance(terms, list):
                logger.info(f"Successfully extracted {len(terms)} terms")
                return terms
            else:
                logger.warning("Expected array of terms but got different format")
                return ["ERROR: Expected array of terms"]
                
        except Exception as e:
            # Fallback: try to extract terms from bullet points or quoted strings
            logger.warning(f"Error parsing terms as JSON: {str(e)}")
            lines = content.split('\n')
            terms = []
            for line in lines:
                line = line.strip()
                if line and line.startswith('-'):
                    terms.append(line[1:].strip())
                elif line and line.startswith('"') and line.endswith('"'):
                    terms.append(line.strip('"'))
            
            if terms:
                logger.info(f"Extracted {len(terms)} terms using fallback method")
                return terms
            else:
                logger.error("Failed to extract terms with fallback method")
                return ["ERROR: Failed to extract terms"]
                
    except Exception as e:
        logger.error(f"Error in term extraction: {str(e)}")
        return ["ERROR: Term extraction failed"]

def extract_terms_async(state: Dict, idx: int) -> Dict:
    """
    Asynchronously extract terms for a specific answer.
    
    Args:
        state: The current state dictionary
        idx: The index of the question/answer to process
        
    Returns:
        Dict with updates to the state
    """
    with extraction_lock:
        try:
            logger.info(f"Starting extract_terms_async for index {idx}")
            
            # Make sure the index is in verified_answers
            if idx not in state["verified_answers"]:
                logger.error(f"Index {idx} not found in verified_answers - state keys: {list(state.keys())}")
                # Log the available verified_answers
                logger.debug(f"Available verified_answers keys: {list(state.get('verified_answers', {}).keys())}")
                return {}
                
            # Get the question and answer
            verified_item = state["verified_answers"][idx]
            question = verified_item.get("question", "")
            answer = verified_item.get("answer", "")
            
            logger.info(f"Extract_terms_async for index {idx} - Q: {question[:30]}... A: {answer[:30]}...")
            
            # Extract terms
            terms = extract_terms(question, answer)
            
            if terms and len(terms) > 0:
                logger.info(f"Successfully extracted {len(terms)} terms for index {idx}: {terms[:3]}...")
                # Update the state with extracted terms and remove from queue
                return {
                    "extracted_terms": {idx: terms},
                    "last_extracted_index": idx
                }
            else:
                logger.warning(f"No terms were extracted for index {idx}")
                return {
                    "extracted_terms": {idx: ["No medical terms found"]},
                    "last_extracted_index": idx
                }
            
        except Exception as e:
            logger.error(f"Error in async term extraction: {str(e)}")
            import traceback
            logger.error(f"Async extraction error traceback: {traceback.format_exc()}")
            return {}

# Function to be executed in a separate thread
def execute_extraction_thread(state: Dict, idx: int, thread_id: str, memory_saver):
    """
    Execute term extraction in a separate thread and update the state.
    
    Args:
        state: The current state dictionary
        idx: The index of the question/answer to process
        thread_id: The thread ID for state retrieval
        memory_saver: The memory saver instance
    """
    try:
        logger.info(f"Executing extraction thread for index {idx} with thread_id {thread_id}")
        
        # Wait a bit to allow the main thread to continue
        time.sleep(1)
        
        # First, check if we can load the current state
        try:
            # Load the current state to verify memory access
            test_load = memory_saver.load(thread_id)
            logger.info(f"Successfully loaded state for thread_id {thread_id}")
        except Exception as load_error:
            logger.error(f"Error loading state in extraction thread: {str(load_error)}")
            import traceback
            logger.error(f"State load error traceback: {traceback.format_exc()}")
            return
        
        # Extract terms
        logger.info(f"Starting term extraction for verified answer at index {idx}")
        state_updates = extract_terms_async(state, idx)
        
        if state_updates and state_updates.get("extracted_terms"):
            logger.info(f"Got extraction results: {state_updates.get('extracted_terms')}")
            
            try:
                # Load the current state again to ensure we have the latest
                current_state = memory_saver.load(thread_id)
                
                # Update extracted_terms - ensure we're using string keys
                extracted_terms = {**current_state.get("extracted_terms", {})}
                for k, v in state_updates.get("extracted_terms", {}).items():
                    # Convert key to string to ensure proper JSON storage
                    str_k = str(k)
                    extracted_terms[str_k] = v
                    logger.info(f"Added terms for index {k}: {v[:3] if len(v) > 3 else v}... (total: {len(v)})")
                
                # Create updated state - make a complete copy to avoid reference issues
                updated_state = {}
                for key, value in current_state.items():
                    updated_state[key] = value
                
                # Update the specific fields we want to change
                updated_state["extracted_terms"] = extracted_terms
                updated_state["last_extracted_index"] = idx
                
                # Remove the processed item from the queue
                if "term_extraction_queue" in updated_state:
                    updated_state["term_extraction_queue"] = [
                        i for i in updated_state["term_extraction_queue"] if i != idx
                    ]
                
                # Save the updated state
                logger.info(f"Saving state with {len(extracted_terms)} extracted term sets")
                memory_saver.save(thread_id, updated_state)
                
                # Verify the save was successful
                verification = memory_saver.load(thread_id)
                if verification and str(idx) in verification.get("extracted_terms", {}):
                    logger.info(f"Successfully verified state save with terms for index {idx}")
                else:
                    logger.warning(f"State verification failed - terms for index {idx} not found after save")
                
                logger.info(f"Term extraction thread completed for index {idx}, state updated with {len(state_updates.get('extracted_terms', {}).get(idx, []))} terms")
                
            except Exception as save_error:
                logger.error(f"Error saving updated state: {str(save_error)}")
                import traceback
                logger.error(f"State save error traceback: {traceback.format_exc()}")
        else:
            logger.warning(f"No terms extracted for index {idx} or empty state updates")
            
    except Exception as e:
        logger.error(f"Error in extraction thread: {str(e)}")
        import traceback
        logger.error(f"Extraction thread error traceback: {traceback.format_exc()}")

def start_extraction_thread(state: Dict, idx: int, thread_id: str, memory_saver) -> bool:
    """
    Start a new thread for term extraction.
    """
    try:
        import copy
        state_copy = copy.deepcopy(state)  # changed to use deep copy
        
        logger.info(f"Starting extraction thread for index {idx} with thread_id: {thread_id}")
        logger.info(f"Extraction queue before starting thread: {state_copy.get('term_extraction_queue', [])}")
        
        import threading
        extraction_thread = threading.Thread(
            target=execute_extraction_thread,
            args=(state_copy, idx, thread_id, memory_saver),
            daemon=True
        )
        extraction_thread.start()
        logger.info(f"Successfully started extraction thread for index {idx}")
        return True
    except Exception as e:
        logger.error(f"Failed to start extraction thread: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False