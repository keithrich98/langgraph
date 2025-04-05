# term_extractor_agent.py
import os
from typing import Dict, List, Any, Optional
from langgraph.func import task
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

@task
def extract_terms(question: str, answer: str, config: Optional[RunnableConfig] = None) -> List[str]:
    """
    Task to extract medical terms from a verified answer.
    
    Args:
        question: The question text
        answer: The verified answer
        config: Optional runnable config
        
    Returns:
        List of extracted medical terms
    """
    logger.debug("Starting medical term extraction")
    
    try:
        # Format the prompt
        formatted_prompt = extraction_prompt.format(question=question, answer=answer)
        logger.debug(f"Prompt prepared for term extraction")
        
        # Call the LLM
        logger.debug("Calling LLM for term extraction")
        response = llm.invoke(formatted_prompt, config=config)
        content = response.content if hasattr(response, "content") else str(response)
        logger.debug(f"LLM response received: {content[:100]}...")
        
        try:
            # Parse the JSON response
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
                logger.info(f"Successfully extracted {len(terms)} terms")
                return terms
                
            logger.warning("Extracted content was not a list")
            return ["ERROR: Expected array of terms"]
        except Exception as e:
            # Fallback parsing for non-JSON responses
            logger.warning(f"Error parsing terms as JSON: {str(e)}")
            lines = content.split('\n')
            terms = []
            for line in lines:
                line = line.strip()
                # Extract terms from bullet points or quoted text
                if line and line.startswith('-'):
                    terms.append(line[1:].strip())
                elif line and line.startswith('"') and line.endswith('"'):
                    terms.append(line.strip('"'))
            
            logger.info(f"Extracted {len(terms)} terms using fallback parsing")
            return terms if terms else ["ERROR: Failed to extract terms"]
    except Exception as e:
        logger.error(f"Error in term extraction: {str(e)}", exc_info=True)
        return ["ERROR: Term extraction failed"]

# Removed @task decorator to prevent Future objects
def term_extraction_task(
    state: SessionState,
    action: Dict = None,
    config: Optional[RunnableConfig] = None
) -> SessionState:
    """
    Task to process term extraction.
    Expects an action {"action": "extract_terms"}.
    Updates state with extracted terms and removes the processed index from the queue.
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
    
    # Process if there's a valid verified answer
    if next_index in state.verified_answers:
        verified_item = state.verified_answers[next_index]
        question = verified_item.get("question", "")
        answer = verified_item.get("answer", "")
        
        logger.info(f"Processing term extraction for question index {next_index}")
        logger.debug(f"Question: {question[:50]}...")
        logger.debug(f"Answer: {answer[:50]}...")
        
        # Call the extraction task and resolve the future immediately to prevent serialization issues
        terms_future = extract_terms(question, answer, config=config)
        terms = terms_future.result() if hasattr(terms_future, 'result') else terms_future
        logger.info(f"Extracted {len(terms)} terms for question {next_index}")
        logger.debug(f"Terms: {terms}")
        
        # Create new extracted_terms dictionary and update
        new_extracted_terms = state.extracted_terms.copy()
        new_extracted_terms[next_index] = terms
        
        # Create updated state with new terms
        new_state = create_updated_state(state, extracted_terms=new_extracted_terms)
        
        # Mark as complete in the queue
        new_state = mark_extraction_complete(new_state, next_index)
        logger.debug(f"Term extraction completed for question {next_index}, queue size now: {len(new_state.term_extraction_queue)}")
        
        return new_state
    else:
        logger.warning(f"No verified answer found for question {next_index}")
    
    return state