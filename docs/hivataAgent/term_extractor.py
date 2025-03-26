# term_extractor.py - Medical Term Extraction Agent

import os
import time
from typing import Dict, List, Any, Optional
from langgraph.func import task
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate

# Import the shared state helpers
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
        question: The medical question.
        answer: The user's answer.
        config: Optional runtime configuration.
    
    Returns:
        A list of extracted medical terms.
    """
    try:
        formatted_prompt = extraction_prompt.format(
            question=question,
            answer=answer
        )
        response = llm.invoke(formatted_prompt, config=config)
        content = response.content if hasattr(response, "content") else str(response)
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
    """Log key details of the state for debugging."""
    print(f"[DEBUG {prefix}] conversation_history length: {len(state.conversation_history)}")
    print(f"[DEBUG {prefix}] current_question_index: {state.current_question_index}")
    print(f"[DEBUG {prefix}] responses count: {len(state.responses)}")
    print(f"[DEBUG {prefix}] term_extraction_queue: {state.term_extraction_queue}")
    print(f"[DEBUG {prefix}] extracted_terms count: {len(state.extracted_terms)}")
    print(f"[DEBUG {prefix}] thread_id: {id(state)}")

@task
def process_term_extraction(state: ChatState, config: Optional[RunnableConfig] = None) -> ChatState:
    """
    Task that processes term extraction from the state.
    
    It looks at the extraction queue, extracts medical terms from the corresponding verified answer,
    updates the state with the extracted terms, and marks the extraction as complete.
    
    Returns:
        The updated state.
    """
    thread_id = config.get("configurable", {}).get("thread_id") if config else "unknown"
    print(f"[DEBUG TE] Running with thread_id: {thread_id}")
    debug_state("TE-Initial", state)
    
    next_index = get_next_extraction_task(state)
    if next_index is not None and next_index in state.verified_answers:
        verified_item = state.verified_answers[next_index]
        question = verified_item.get("question", "")
        answer = verified_item.get("answer", "")
        print(f"[DEBUG TE] Processing extraction for index {next_index}")
        print(f"[DEBUG TE] Question: {question[:50]}{'...' if len(question) > 50 else ''}")
        print(f"[DEBUG TE] Answer: {answer[:50]}{'...' if len(answer) > 50 else ''}")
        
        # Call the extract_terms task and wait for the result.
        terms = extract_terms(question, answer, config=config).result()
        print(f"[DEBUG TE] Extracted terms: {terms}")
        
        state.extracted_terms[next_index] = terms
        state = mark_extraction_complete(state, next_index)
        debug_state("TE-AfterExtraction", state)
    else:
        print("[DEBUG TE] No valid items in extraction queue.")
    
    debug_state("TE-Final", state)
    return state
