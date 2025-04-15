# term_extractor.py - Implementation using langgraph.graph API
import os
import threading
import time
from typing import Dict, List, Any, Optional, Union
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from logging_config import logger
from state import ChatState
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

# Thread-safe lock for asynchronous processing.
extraction_lock = threading.Lock()

# Prompt Template for Extraction
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
    Extract medical terms from the provided question and verified answer using the LLM.
    
    The LLM is expected to return a valid JSON array of strings. For example:
    
        ["heart attack", "myocardial infarction"]
    
    If the response is not a valid JSON array, a fallback extraction will attempt to
    parse bullet points or quoted strings.
    
    Args:
        question: The question text.
        answer: The user's verified answer.
    
    Returns:
        A list of extracted medical terms. In case of failure, the returned list will contain
        an error string.
    """
    try:
        logger.debug(f"Extracting terms for Q: {question[:30]}... A: {answer[:30]}...")
        formatted_prompt = extraction_prompt.format(question=question, answer=answer)
        
        # Invoke the LLM with the formatted prompt.
        response = llm.invoke(formatted_prompt)
        content = response.content if hasattr(response, "content") else str(response)
        
        # Primary attempt: use standard JSON parsing.
        try:
            content = content.strip()
            # If the response doesn't start with '[', attempt to isolate the JSON array.
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
                logger.info(f"Successfully extracted {len(terms)} terms")
                return terms
            else:
                logger.warning("Primary JSON parse: Expected a list but got a different type")
                return ["ERROR: Expected array of terms"]
        except Exception as parse_error:
            logger.warning(f"Primary JSON parse error: {parse_error}; attempting fallback extraction.")
            # Fallback extraction: look for bullet points or quoted strings.
            lines = content.split('\n')
            terms = []
            for line in lines:
                line = line.strip()
                if line.startswith('-'):
                    terms.append(line[1:].strip())
                elif line.startswith('"') and line.endswith('"'):
                    terms.append(line.strip('"'))
            if terms:
                logger.info(f"Extracted {len(terms)} terms using fallback method")
                return terms
            else:
                logger.error("Fallback extraction failed")
                return ["ERROR: Failed to extract terms"]
    except Exception as e:
        logger.error(f"Error in term extraction: {str(e)}")
        return ["ERROR: Term extraction failed"]

def extract_terms_async(state: ChatState, idx: int) -> Dict:
    """
    Asynchronously extract medical terms for a given question index.
    
    This function should be called while holding the extraction_lock.
    
    Args:
        state: The current state dictionary.
        idx: The index of the question/answer pair to process.
    
    Returns:
        A dictionary update with the new extracted terms and updated last_extracted_index.
        If extraction fails, the extracted_terms value may contain an error string.
    """
    with extraction_lock:
        try:
            logger.info(f"Starting extract_terms_async for index {idx}")
            if idx not in state.get("verified_answers", {}):
                logger.error(f"Index {idx} not found in verified_answers - available keys: {list(state.get('verified_answers', {}).keys())}")
                return {}
                
            verified_item = state["verified_answers"][idx]
            question = verified_item.get("question", "")
            answer = verified_item.get("answer", "")
            logger.info(f"Extract_terms_async for index {idx} - Q: {question[:30]}... A: {answer[:30]}...")
            
            terms = extract_terms(question, answer)
            if terms and len(terms) > 0:
                logger.info(f"Successfully extracted {len(terms)} terms for index {idx}: {terms[:3]}...")
                return {
                    "extracted_terms": {idx: terms},
                    "last_extracted_index": idx
                }
            else:
                logger.warning(f"No terms extracted for index {idx}")
                return {
                    "extracted_terms": {idx: ["No medical terms found"]},
                    "last_extracted_index": idx
                }
        except Exception as e:
            logger.error(f"Error in async term extraction: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

def execute_extraction_thread(state: Dict, idx: int, thread_id: str, memory_saver):
    """
    Execute term extraction in a background thread and update shared memory.
    
    Args:
        state: The current state dictionary.
        idx: The index of the question/answer pair to process.
        thread_id: A unique thread identifier for state retrieval.
        memory_saver: An instance of a shared memory saver.
    """
    context = {"thread_id": thread_id, "question_index": idx, "function": "execute_extraction_thread"}
    try:
        logger.info(f"Executing extraction thread for index {idx} with thread_id {thread_id}", extra=context)
        time.sleep(1)  # Brief pause to let the main thread continue
        
        try:
            test_load = memory_saver.load(thread_id)
            logger.info(f"Successfully loaded state for thread_id {thread_id}", extra=context)
        except Exception as load_error:
            logger.error(f"Error loading state in extraction thread: {str(load_error)}", extra=context)
            import traceback
            logger.error(traceback.format_exc(), extra=context)
            return
        
        logger.info(f"Starting term extraction for verified answer at index {idx}", extra=context)
        state_updates = extract_terms_async(state, idx)
        
        if state_updates and state_updates.get("extracted_terms"):
            logger.info(f"Got extraction results: {state_updates.get('extracted_terms')}", extra=context)
            try:
                current_state = memory_saver.load(thread_id)
                extracted_terms = {**current_state.get("extracted_terms", {})}
                for k, v in state_updates.get("extracted_terms", {}).items():
                    str_k = str(k)
                    extracted_terms[str_k] = v
                    logger.info(f"Added terms for index {k}: {v[:3] if len(v) > 3 else v} (total: {len(v)})", extra=context)
                
                updated_state = current_state.copy()
                updated_state["extracted_terms"] = extracted_terms
                updated_state["last_extracted_index"] = idx
                if "term_extraction_queue" in updated_state:
                    updated_state["term_extraction_queue"] = [i for i in updated_state["term_extraction_queue"] if i != idx]
                
                logger.info(f"Saving state with {len(extracted_terms)} extracted term sets", extra=context)
                memory_saver.save(thread_id, updated_state)
                
                verification_state = memory_saver.load(thread_id)
                if verification_state and "extracted_terms" in verification_state:
                    if str(idx) in verification_state["extracted_terms"]:
                        logger.info(f"Successfully verified state save for index {idx} with {len(verification_state['extracted_terms'][str(idx)])} terms", extra=context)
                    else:
                        logger.warning(f"Terms for index {idx} not found after save", extra=context)
                else:
                    logger.error("Verification failed: Could not load updated state", extra=context)
                
                logger.info(f"Extraction thread completed for index {idx}, found {len(state_updates.get('extracted_terms', {}).get(idx, []))} terms", extra=context)
                
            except Exception as save_error:
                logger.error(f"Error saving updated state: {str(save_error)}", extra=context)
                import traceback
                logger.error(traceback.format_exc(), extra=context)
        else:
            logger.warning(f"No terms extracted or empty update for index {idx}", extra=context)
    except Exception as e:
        logger.error(f"Extraction thread error for index {idx}: {str(e)}", extra=context)
        import traceback
        logger.error(traceback.format_exc(), extra=context)

def start_extraction_thread(state: Dict, idx: int, thread_id: str, memory_saver) -> bool:
    """
    Start a new thread for term extraction.
    
    Abstracts background thread creation so that additional background processing can be integrated easily.
    
    Args:
        state: The current state dictionary.
        idx: The question/answer index to process.
        thread_id: The unique thread identifier.
        memory_saver: The memory saver instance to update state.
    
    Returns:
        True if the thread was started successfully, False otherwise.
    """
    context = {"thread_id": thread_id, "question_index": idx, "function": "start_extraction_thread"}
    try:
        import copy
        state_copy = copy.deepcopy(state)  # Use deep copy to decouple thread state.
        logger.info(f"Starting extraction thread for index {idx} with thread_id: {thread_id}", extra=context)
        logger.info(f"Extraction queue before starting thread: {state_copy.get('term_extraction_queue', [])}", extra=context)
        
        import threading
        extraction_thread = threading.Thread(
            target=execute_extraction_thread,
            args=(state_copy, idx, thread_id, memory_saver),
            daemon=True
        )
        extraction_thread.start()
        logger.info(f"Successfully started extraction thread for index {idx}", extra=context)
        return True
    except Exception as e:
        logger.error(f"Failed to start extraction thread: {str(e)}", extra=context)
        import traceback
        logger.error(traceback.format_exc(), extra=context)
        return False