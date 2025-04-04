# shared_memory.py
from langgraph.checkpoint.memory import MemorySaver
import logging

logger = logging.getLogger("questionnaire")

class LoggingMemorySaver(MemorySaver):
    """A MemorySaver that logs operations to help debug."""
    
    def save(self, thread_id, data):
        logger.debug(f"Saving checkpoint for thread_id: {thread_id}")
        super().save(thread_id, data)
    
    def load(self, thread_id):
        logger.debug(f"Loading checkpoint for thread_id: {thread_id}")
        return super().load(thread_id)

# Create a single shared checkpointer to be used across all workflows
shared_memory = LoggingMemorySaver()