# question_processor.py - Simplified version
from typing import Dict, List
from logging_config import logger

def get_questions() -> List[Dict]:
    """
    Returns the list of questions with their requirements for the polymicrogyria questionnaire.
    """
    return [
        {
            "text": "At what age were you diagnosed with polymicrogyria, and what were the primary signs or symptoms?",
            "requirements": {
                "age": "Provide age (based on birthdate)",
                "diagnosis_date": "Provide the date of diagnosis",
                "symptoms": "Describe the key signs and symptoms"
            }
        },
        {
            "text": "What symptoms or neurological issues do you experience, and how would you rate their severity?",
            "requirements": {
                "symptoms": "List each symptom experienced",
                "severity": "Include a severity rating (mild, moderate, severe)",
                "context": "Provide additional context about how symptoms impact daily life"
            }
        },
        {
            "text": "Can you describe the key findings from your brain imaging studies (MRI/CT)?",
            "requirements": {
                "imaging_modality": "Specify the imaging modality used (MRI, CT, etc.)",
                "findings": "Detail the main imaging findings",
                "remarks": "Include any remarks from radiology reports"
            }
        }
    ]

def format_question(question_obj: Dict) -> str:
    """
    Formats a question object into a presentable string.
    """
    formatted_requirements = "\n".join([f"- {k}: {v}" for k, v in question_obj["requirements"].items()])
    return f"{question_obj['text']}\n\nRequirements:\n{formatted_requirements}"

def get_next_question_index(current_index: int, total_questions: int) -> int:
    """
    Calculates the next question index, or returns -1 if we're at the end.
    """
    next_index = current_index + 1
    return next_index if next_index < total_questions else -1