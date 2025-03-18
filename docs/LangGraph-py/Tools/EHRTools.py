#!/usr/bin/env python3
"""
This module defines tools for the LangGraph-based agent to work with Electronic Health Records (EHR).
It includes:
  - medical_term_extraction_tool: Extracts potential medical terms from a report and purpose using a language model,
    then retrieves corresponding UMLS codes.
  - umls_api_tool: Retrieves UMLS codes and associated semantic type details for a given list of medical terms.
"""

import os
import json
import urllib.parse
import requests
from typing import List, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langfuse import Langfuse
from pydantic import BaseModel, Field

class TermsSchema(BaseModel):
    terms: List[str] = Field(..., description="A list of terms or statements that might map to a medical UMLS CUI.")

# -----------------------------
# Helper Function: Retrieve UMLS codes from terms
# -----------------------------
def get_umls_codes_from_terms(terms: List[str]) -> str:
    api_key = os.environ.get("NLM_API_KEY")
    if not api_key:
        raise ValueError("NLM_API_KEY environment variable not set.")

    def fetch_umls_code_details(uri: str) -> Optional[List[str]]:
        url = f"{uri}?apiKey={api_key}"
        response = requests.get(url)
        if not response.ok:
            return None
        data = response.json()
        semantic_types = data.get("result", {}).get("semanticTypes", [])
        return [st.get("name") for st in semantic_types]

    def fetch_umls_code(term: str) -> List[Dict[str, Optional[str]]]:
        params = {
            "apiKey": api_key,
            "string": term,
            "partialSearch": "true",
            "pageSize": "10",
            "searchType": "normalizedString"
        }
        search_params = urllib.parse.urlencode(params)
        base = os.environ.get("NLM_API_BASE", "")
        endpoint = os.environ.get("NLM_API_ENDPOINT", "")
        api_url = f"{base}{endpoint}?{search_params}"
        response = requests.get(api_url)
        data = response.json()
        probable_codes = []
        results = data.get("result", {}).get("results", [])
        if results:
            for result in results:
                probable_codes.append({
                    "umls_code": result.get("ui"),
                    "uri": result.get("uri"),
                    "name": result.get("name")
                })
        codes_with_semantic = []
        for code in probable_codes:
            semantic = fetch_umls_code_details(code["uri"]) if code.get("uri") else None
            codes_with_semantic.append({
                "umls_code": code.get("umls_code"),
                "name": code.get("name"),
                "semanticTypes": semantic
            })
        return codes_with_semantic

    results = []
    for term in terms:
        codes = fetch_umls_code(term)
        results.append({
            "term": term,
            "possible_codes": codes
        })
    return json.dumps(results)

# -----------------------------
# Tool 1: Medical Term Extraction Tool
# -----------------------------

@tool
def medical_term_extraction_tool(report: str, purpose: str) -> str:
    """
    Extract potential medical terms from a report and purpose, then retrieve corresponding UMLS codes.

    Parameters:
      - report: A comprehensive summary containing medical, social, and other relevant information.
      - purpose: Explanation of why these medical terms are needed and how the UMLS codes will be applied.

    Returns:
      A JSON string containing UMLS codes and details associated with the extracted terms.
    """
    langfuse = Langfuse()
    term_extraction_prompt = langfuse.get_prompt("umls-term-extraction-agent", None, {"label": "latest"})

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", term_extraction_prompt.get_langchain_prompt()[[0]](https://api.python.langchain.com/en/latest/ollama_api_reference.html).content)
    ])

    result = prompt_template.invoke({
        "summary": report,
        "purpose": purpose
    })

    terms = result.get("terms")
    if not terms:
        raise ValueError("No terms were extracted.")
    return get_umls_codes_from_terms(terms)

# -----------------------------
# Tool 2: UMLS API Tool
# -----------------------------
@tool
def umls_api_tool(terms: List[str]) -> str:
    """
    Retrieve UMLS codes and associated information for a list of medical terms.

    Parameters:
      - terms: A list of medical terms obtained from the medical_term_extraction_tool.

    Returns:
      A JSON string containing UMLS codes along with additional details for each term.
    """
    return get_umls_codes_from_terms(terms)

# Expose the tools for external import
__all__ = ["medical_term_extraction_tool", "umls_api_tool"]
