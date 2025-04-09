# shared_memory.py
from langgraph.checkpoint.memory import MemorySaver
import logging
logger = logging.getLogger("questionnaire")

class LoggingMemorySaver(MemorySaver):
    def save(self, thread_id, data):
        logger.debug(f"[MemorySaver] Saving checkpoint for thread_id {thread_id}. Data keys: {list(data.keys())}")
        return super().save(thread_id, data)
    
    def load(self, thread_id):
        logger.debug(f"[MemorySaver] Loading checkpoint for thread_id {thread_id}.")
        return super().load(thread_id)

shared_memory = LoggingMemorySaver()
