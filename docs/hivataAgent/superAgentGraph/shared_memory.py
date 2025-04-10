# shared_memory.py
from langgraph.checkpoint.memory import MemorySaver
import logging
logger = logging.getLogger("questionnaire")
import copy  

class LoggingMemorySaver(MemorySaver):
    def __init__(self):
        super().__init__()
        self._states = {}

    def save(self, thread_id, data):
        """
        Save state for a thread_id with improved handling for complex state objects.
        """
        logger.debug(f"[MemorySaver] Saving checkpoint for thread_id {thread_id}. Data keys: {list(data.keys())}")
        
        # Use deep copy to avoid reference issues
        state_copy = copy.deepcopy(data)
        
        # Ensure extracted_terms is properly formatted with string keys
        if "extracted_terms" in state_copy and isinstance(state_copy["extracted_terms"], dict):
            extracted_terms = {str(k): v for k, v in state_copy["extracted_terms"].items()}
            state_copy["extracted_terms"] = extracted_terms
            logger.debug(f"[MemorySaver] Normalized extracted_terms keys to strings: {list(extracted_terms.keys())}")
        
        self._states[thread_id] = state_copy
        
        if thread_id in self._states:
            extracted_terms = self._states[thread_id].get("extracted_terms", {})
            if extracted_terms:
                logger.debug(f"[MemorySaver] Save verified - extracted_terms keys: {list(extracted_terms.keys())}")
        else:
            logger.warning(f"[MemorySaver] Failed to save state for thread_id {thread_id}")
        
        return state_copy

    def load(self, thread_id):
        """
        Load state for a thread_id with improved error handling.
        """
        logger.debug(f"[MemorySaver] Loading checkpoint for thread_id {thread_id}.")
        
        if thread_id not in self._states:
            logger.warning(f"[MemorySaver] No data found for thread_id {thread_id}")
            logger.debug(f"[MemorySaver] Available thread_ids: {list(self._states.keys())}")
            return {}
        
        # Return a deep copy of the state to avoid reference issues
        state_data = self._states[thread_id]
        state_copy = copy.deepcopy(state_data)
        
        if "extracted_terms" in state_copy:
            logger.debug(f"[MemorySaver] Loaded extracted_terms with keys: {list(state_copy['extracted_terms'].keys())}")
        
        return state_copy

shared_memory = LoggingMemorySaver()
