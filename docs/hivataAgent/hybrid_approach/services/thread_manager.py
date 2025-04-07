"""
Thread Manager Service.

Provides a centralized thread management system to handle asynchronous tasks
like term extraction in a controlled and observable way.
"""
import threading
import time
import uuid
from typing import Dict, Optional, Callable, Any, List
from hivataAgent.hybrid_approach.config.logging_config import logger


class ThreadManager:
    """
    Manages background threads for asynchronous operations.
    
    Provides thread tracking, rate limiting, and error handling
    for long-running background tasks like term extraction.
    """
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = ThreadManager()
        return cls._instance
    
    def __init__(self):
        """Initialize the thread manager."""
        self.active_threads: Dict[str, threading.Thread] = {}
        self.thread_status: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.max_concurrent_threads = 5
        self.task_history: List[Dict[str, Any]] = []  # Store completed task info
        self.max_history = 100  # Keep track of last 100 tasks
        
    def schedule_task(self, task_type: str, task_fn: Callable, *args, **kwargs) -> Optional[str]:
        """
        Schedule a task to run in a background thread.
        
        Args:
            task_type: Type of task (e.g., "extraction", "processing")
            task_fn: The function to call
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            task_id: A unique ID for tracking this task or None if scheduling failed
        """
        task_id = str(uuid.uuid4())
        
        # Check if we're already at maximum capacity
        with self.lock:
            active_count = len(self.active_threads)
            if active_count >= self.max_concurrent_threads:
                logger.warning(f"Maximum concurrent threads reached ({self.max_concurrent_threads})")
                # Could implement queue here instead of returning None
                return None
                
            # Setup the thread
            def wrapped_task():
                self._execute_task(task_id, task_type, task_fn, *args, **kwargs)
            
            # Create the thread
            thread = threading.Thread(target=wrapped_task)
            thread.daemon = True
            
            # Track the thread
            self.active_threads[task_id] = thread
            self.thread_status[task_id] = {
                "task_id": task_id,
                "status": "scheduled",
                "type": task_type,
                "started_at": time.time(),
                "completed_at": None,
                "success": None,
                "error": None,
                "args_summary": f"{len(args)} args, {len(kwargs)} kwargs"
            }
            
            # Start the thread
            thread.start()
            logger.debug(f"Thread manager: Started {task_type} task {task_id}")
            
            return task_id
            
    def _execute_task(self, task_id: str, task_type: str, task_fn: Callable, *args, **kwargs):
        """Execute the task and track its status."""
        try:
            # Update status
            with self.lock:
                self.thread_status[task_id]["status"] = "running"
            
            # Execute task
            result = task_fn(*args, **kwargs)
            
            # Mark successful completion
            with self.lock:
                self.thread_status[task_id]["status"] = "completed"
                self.thread_status[task_id]["completed_at"] = time.time()
                self.thread_status[task_id]["success"] = True
                
                # Add to history and clean up
                self._add_to_history(self.thread_status[task_id])
                
            logger.debug(f"Thread manager: {task_type} task {task_id} completed successfully")
            return result
            
        except Exception as e:
            # Log the error
            logger.error(f"Thread manager: Error in {task_type} task {task_id}: {str(e)}")
            
            # Update status
            with self.lock:
                self.thread_status[task_id]["status"] = "failed"
                self.thread_status[task_id]["completed_at"] = time.time()
                self.thread_status[task_id]["success"] = False
                self.thread_status[task_id]["error"] = str(e)
                
                # Add to history and clean up
                self._add_to_history(self.thread_status[task_id])
            
            # Could implement retry logic here
            return None
        finally:
            # Clean up the thread reference
            with self.lock:
                if task_id in self.active_threads:
                    del self.active_threads[task_id]
    
    def _add_to_history(self, task_info: Dict[str, Any]):
        """Add completed task to history and maintain maximum history size."""
        self.task_history.append(task_info.copy())
        # Trim history if needed
        if len(self.task_history) > self.max_history:
            self.task_history = self.task_history[-self.max_history:]
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task."""
        with self.lock:
            # Check active tasks first
            if task_id in self.thread_status:
                return self.thread_status[task_id].copy()
            
            # Check historical tasks
            for task in reversed(self.task_history):
                if task.get("task_id") == task_id:
                    return task.copy()
            
            # Not found
            return {"status": "unknown", "task_id": task_id}
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of all tasks."""
        with self.lock:
            # Count tasks by status
            active_count = len(self.active_threads)
            status_counts = {"scheduled": 0, "running": 0, "completed": 0, "failed": 0}
            
            for status in self.thread_status.values():
                status_key = status.get("status", "unknown")
                if status_key in status_counts:
                    status_counts[status_key] += 1
            
            # Historical stats
            completed_count = sum(1 for task in self.task_history if task.get("status") == "completed")
            failed_count = sum(1 for task in self.task_history if task.get("status") == "failed")
            
            return {
                "active_count": active_count,
                "max_concurrent": self.max_concurrent_threads,
                "status_counts": status_counts,
                "history": {
                    "total": len(self.task_history),
                    "completed": completed_count,
                    "failed": failed_count
                },
                "active_tasks": [task_id for task_id in self.active_threads.keys()]
            }
    
    def get_active_tasks(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all currently active tasks, optionally filtered by type."""
        with self.lock:
            if task_type:
                return [
                    status.copy() for status in self.thread_status.values()
                    if status.get("type") == task_type
                ]
            else:
                return [status.copy() for status in self.thread_status.values()]
    
    def get_recent_tasks(self, limit: int = 10, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recently completed tasks."""
        with self.lock:
            if task_type:
                filtered = [task for task in self.task_history if task.get("type") == task_type]
            else:
                filtered = self.task_history.copy()
            
            # Return most recent first
            return sorted(filtered, key=lambda x: x.get("completed_at", 0), reverse=True)[:limit]
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Shutdown the thread manager, optionally waiting for tasks to complete.
        
        Args:
            wait: Whether to wait for threads to complete
            timeout: Maximum time to wait in seconds
            
        Returns:
            success: Whether shutdown completed successfully
        """
        logger.info(f"Thread manager: Shutting down with {len(self.active_threads)} active threads")
        
        if not wait:
            # Just clear everything without waiting
            with self.lock:
                self.active_threads.clear()
                self.thread_status.clear()
            return True
        
        # Copy active threads to avoid modification during iteration
        with self.lock:
            active_threads = list(self.active_threads.values())
        
        # Wait for threads to complete
        start_time = time.time()
        for thread in active_threads:
            remaining_time = None
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    logger.warning("Thread manager: Shutdown timeout reached")
                    break
                remaining_time = timeout - elapsed
            
            thread.join(remaining_time)
        
        # Check if any threads are still active
        with self.lock:
            active_count = len(self.active_threads)
            self.active_threads.clear()
            self.thread_status.clear()
            
            if active_count > 0:
                logger.warning(f"Thread manager: {active_count} threads still active during shutdown")
                return False
            
            return True


# Create a singleton instance
thread_manager = ThreadManager.get_instance()