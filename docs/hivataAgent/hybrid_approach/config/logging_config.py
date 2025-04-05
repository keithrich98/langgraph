# logging_config.py
import logging
import sys
from datetime import datetime
import os

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
log_filename = f"logs/questionnaire_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a logger instance that can be imported by other modules
logger = logging.getLogger("questionnaire")