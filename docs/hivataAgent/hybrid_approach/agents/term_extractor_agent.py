# term_extractor_agent.py
import os
from typing import Dict, List, Any, Optional
# No need to import task decorator
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate

from hivataAgent.hybrid_approach.core.state import (
    SessionState, 
    get_next_extraction_task, 
    mark_extraction_complete, 
    create_updated_state
)
from hivataAgent.hybrid_approach.config.logging_config import logger

# Import the LLM
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# Initialize LLM
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    logger.error("OPENAI_API_KEY is not set in your environment.")
    raise ValueError("OPENAI_API_KEY is not set in your environment.")

llm = ChatOpenAI(
    model_name="gpt-4o",
    openai_api_key=openai_api_key,
    temperature=0
)

# Define the extraction prompt
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

# Simple cache for extracted terms to avoid redundant extractions
_term_extraction_cache = {}
_MAX_CACHE_SIZE = 100

def extract_terms(question: str, answer: str, config: Optional[RunnableConfig] = None) -> List[str]:
    """
    Extract medical terms from a verified answer.
    
    Features:
    - Caching: Avoids re-extracting terms for identical inputs
    - Robust parsing: Handles various LLM response formats
    - Error recovery: Uses fallback parsing for non-JSON responses
    
    Args:
        question: The question text
        answer: The verified answer
        config: Optional runnable config
        
    Returns:
        List of extracted medical terms
    """
    logger.debug("Starting medical term extraction")
    
    # Generate a cache key - use both question and answer to ensure context is preserved
    cache_key = f"{question}|||{answer}"
    
    # Check cache first
    global _term_extraction_cache
    if cache_key in _term_extraction_cache:
        cached_result = _term_extraction_cache[cache_key]
        logger.info(f"Using cached extraction result with {len(cached_result)} terms")
        return cached_result.copy()  # Return a copy to prevent mutation
    
    try:
        # Format the prompt
        formatted_prompt = extraction_prompt.format(question=question, answer=answer)
        logger.debug("Prompt prepared for term extraction")
        
        # Call the LLM with retry logic
        max_retries = 2
        retry_count = 0
        while retry_count <= max_retries:
            try:
                logger.debug(f"Calling LLM for term extraction (attempt {retry_count + 1})")
                response = llm.invoke(formatted_prompt, config=config)
                content = response.content if hasattr(response, "content") else str(response)
                logger.debug(f"LLM response received: {content[:100]}...")
                break  # Success, exit retry loop
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    raise  # Re-raise if we've exhausted retries
                logger.warning(f"LLM call failed, retrying ({retry_count}/{max_retries}): {str(e)}")
                time.sleep(1)  # Wait between retries
        
        # Parse the response
        terms = parse_extraction_response(content)
        
        # Cache the result (with cache size management)
        if len(_term_extraction_cache) >= _MAX_CACHE_SIZE:
            # Remove a random entry if cache is full
            remove_key = next(iter(_term_extraction_cache))
            del _term_extraction_cache[remove_key]
        _term_extraction_cache[cache_key] = terms.copy()
        
        logger.info(f"Successfully extracted and cached {len(terms)} terms")
        return terms
        
    except Exception as e:
        logger.error(f"Error in term extraction: {str(e)}", exc_info=True)
        return ["ERROR: Term extraction failed"]


def parse_extraction_response(content: str) -> List[str]:
    """
    Parse the LLM response to extract terms, with multiple fallback strategies.
    
    Args:
        content: The LLM response text
        
    Returns:
        List of extracted terms
    """
    try:
        # First attempt: Parse as JSON
        import json
        content = content.strip()
        
        # Handle cases where the LLM might include explanatory text before/after JSON
        if not content.startswith('['):
            start_idx = content.find('[')
            if start_idx >= 0:
                content = content[start_idx:]
                # Find matching closing bracket
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
            logger.info(f"Successfully parsed {len(terms)} terms as JSON")
            
            # Clean up terms - remove empty strings, normalize whitespace
            terms = [term.strip() for term in terms if term and term.strip()]
            return terms
            
        logger.warning("Extracted content was JSON but not a list")
        
    except Exception as e:
        # JSON parsing failed, try other formats
        logger.warning(f"JSON parsing failed: {str(e)}")
    
    # Fallback parsing for non-JSON responses
    lines = content.split('\n')
    terms = []
    
    # Try various patterns
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Extract terms from bullet points
        if line.startswith('-') or line.startswith('*'):
            term = line[1:].strip()
            if term:
                terms.append(term)
        # Extract quoted text
        elif (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
            term = line[1:-1].strip()
            if term:
                terms.append(term)
        # Extract numbered list items
        elif line[0].isdigit() and line[1:3] in ['. ', ') ']:
            term = line[3:].strip()
            if term:
                terms.append(term)
    
    if terms:
        logger.info(f"Extracted {len(terms)} terms using fallback parsing")
        return terms
    
    # Last resort: split by commas if we found nothing else
    if ',' in content:
        terms = [term.strip() for term in content.split(',') if term.strip()]
        if terms:
            logger.info(f"Extracted {len(terms)} terms by comma splitting")
            return terms
    
    logger.warning("Could not extract any terms using any parsing method")
    return ["ERROR: Failed to extract terms"]

def term_extraction_task(
    state: SessionState,
    action: Dict = None,
    config: Optional[RunnableConfig] = None
) -> SessionState:
    """
    Process term extraction for items in the extraction queue.
    
    Features:
    - Batch processing: Can handle multiple extractions at once
    - Prioritization: Processes newest questions first
    - Robust error handling: Individual extraction failures don't affect others
    - Caching: Uses extraction cache to avoid redundant work
    
    Args:
        state: Current application state
        action: Action dictionary with {"action": "extract_terms"}
        config: Optional runnable config
        
    Returns:
        Updated state with extraction results
    """
    logger.debug(f"Term extraction task started, queue size: {len(state.term_extraction_queue)}")
    
    # Check for the correct action
    if action is None or action.get("action") != "extract_terms":
        logger.warning("No 'extract_terms' action provided; returning state unchanged.")
        return state

    # Get the next question index that needs extraction
    next_index = get_next_extraction_task(state)
    if next_index is None:
        logger.debug("No items in extraction queue")
        return state
    
    updated_state = state
    max_batch_size = 1  # Currently process one at a time, but could be increased
    
    # Process up to max_batch_size items in the queue
    for _ in range(min(max_batch_size, len(state.term_extraction_queue))):
        next_index = get_next_extraction_task(updated_state)
        if next_index is None:
            break  # No more items in queue
            
        # Skip if we don't have verified answer data
        if next_index not in updated_state.verified_answers:
            logger.warning(f"No verified answer found for question {next_index}, skipping extraction")
            # Remove from queue and continue
            updated_state = mark_extraction_complete(updated_state, next_index)
            continue
            
        # Get question and answer data
        verified_item = updated_state.verified_answers[next_index]
        question = verified_item.get("question", "")
        answer = verified_item.get("answer", "")
        
        logger.info(f"Processing term extraction for question index {next_index}")
        logger.debug(f"Question: {question[:50]}...")
        logger.debug(f"Answer: {answer[:50]}...")
        
        try:
            # Call the extraction function with improved error handling
            terms = extract_terms(question, answer, config=config)
            
            if terms and not (len(terms) == 1 and terms[0].startswith("ERROR:")):
                logger.info(f"Successfully extracted {len(terms)} terms for question {next_index}")
                logger.debug(f"Terms: {terms}")
                
                # Create new extracted_terms dictionary and update
                new_extracted_terms = updated_state.extracted_terms.copy()
                new_extracted_terms[next_index] = terms
                
                # Update state with new terms
                updated_state = updated_state.update(extracted_terms=new_extracted_terms)
            else:
                # Log extraction failure but continue with other items
                logger.warning(f"Term extraction failed for question {next_index}: {terms}")
        except Exception as e:
            # Log error but continue processing other items
            logger.error(f"Error extracting terms for question {next_index}: {str(e)}")
        
        # Mark as complete in the queue regardless of success/failure
        # to avoid getting stuck on problematic items
        updated_state = mark_extraction_complete(updated_state, next_index)
        logger.debug(f"Term extraction processed for question {next_index}, queue size now: {len(updated_state.term_extraction_queue)}")
    
    return updated_state