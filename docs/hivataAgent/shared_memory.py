# shared_memory.py
from langgraph.checkpoint.memory import MemorySaver

# Create a single shared checkpointer to be used across all workflows
shared_memory = MemorySaver()