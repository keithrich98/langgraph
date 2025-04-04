# term_extractor.py - Refactored as a task

import os
import time
from typing import Dict, List, Any, Optional
from langgraph.func import task
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate

from state import ChatState, get_next_extraction_task, mark_extraction_complete
from shared_memory import shared_memory  # not used directly here

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
    try:
        formatted_prompt = extraction_prompt.format(question=question, answer=answer)
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
            print(f"Error parsing terms: {str(e)}")
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
    print(f"[DEBUG {prefix}] conversation_history length: {len(state.conversation_history)}")
    print(f"[DEBUG {prefix}] current_question_index: {state.current_question_index}")
    print(f"[DEBUG {prefix}] responses count: {len(state.responses)}")
    print(f"[DEBUG {prefix}] term_extraction_queue: {state.term_extraction_queue}")
    print(f"[DEBUG {prefix}] extracted_terms count: {len(state.extracted_terms)}")
    print(f"[DEBUG {prefix}] thread_id: {id(state)}")

@task
def term_extraction_task(
    state: ChatState,
    action: Dict = None,
    config: Optional[RunnableConfig] = None
) -> ChatState:
    """
    Task to process term extraction.
    Expects an action {"action": "extract_terms"}.
    Updates state with extracted terms and removes the processed index from the queue.
    """
    debug_state("TE-Initial", state)
    
    # Check for the correct action key; update from "process" to "extract_terms"
    if action is None or action.get("action") != "extract_terms":
        print("[DEBUG TE] No 'extract_terms' action provided; returning state unchanged.")
        return state

    next_index = get_next_extraction_task(state)
    if next_index is not None and next_index in state.verified_answers:
        verified_item = state.verified_answers[next_index]
        question = verified_item.get("question", "")
        answer = verified_item.get("answer", "")
        print(f"[DEBUG TE] Processing extraction for index {next_index}")
        print(f"[DEBUG TE] Verified question: {question}")
        print(f"[DEBUG TE] Verified answer: {answer}")
        # Call the extraction task and capture the response
        terms = extract_terms(question, answer, config=config).result()
        print(f"[DEBUG TE] LLM returned raw extracted terms: {terms}")
        state.extracted_terms[next_index] = terms
        state = mark_extraction_complete(state, next_index)
        debug_state("TE-AfterExtraction", state)
    else:
        print("[DEBUG TE] No valid verified answer found in the extraction queue.")
    
    debug_state("TE-Final", state)
    return state


