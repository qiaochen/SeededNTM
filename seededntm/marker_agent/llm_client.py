"""Azure OpenAI GPT-4o client wrapper for the Marker Search Agent."""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _load_env():
    """Load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def get_llm_client():
    """Create and return an AzureOpenAI client instance."""
    _load_env()
    from openai import AzureOpenAI

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    base_url = endpoint.split("/openai")[0]

    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=base_url,
        api_version="2025-04-01-preview",
    )


def llm_complete(
    prompt: str,
    system: str = "",
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> str:
    """Send a chat completion request to Azure OpenAI.

    Returns the content of the first choice's message.
    """
    client = get_llm_client()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-2")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("LLM completion failed: %s", e)
        raise


def llm_extract_genes(text: str, cell_type: str) -> list[str]:
    """Use LLM to extract gene symbols from free text for a given cell type."""
    system = (
        "You are a molecular biology expert. Extract gene symbols from the "
        "provided text that are markers for the specified cell type. "
        "Return ONLY a JSON list of uppercase gene symbols, nothing else."
    )
    prompt = (
        f"Cell type: {cell_type}\n\n"
        f"Text:\n{text}\n\n"
        "Return a JSON list of gene symbols."
    )
    import json

    try:
        result = llm_complete(prompt, system=system, temperature=0.0)
        result = result.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(result)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to extract genes via LLM: %s", e)
        return []


def llm_resolve_cell_type(cell_type: str, context: str = "") -> list[str]:
    """Use LLM to resolve cell type aliases for database queries."""
    system = (
        "You are a cell biology nomenclature expert. Given a cell type name, "
        "provide alternative names/synonyms that databases might use. "
        "Return ONLY a JSON list of strings, including the original name."
    )
    prompt = (
        f"Cell type: {cell_type}\n"
        f"Context: {context}\n\n"
        "Return a JSON list of synonyms/alternative names for this cell type."
    )
    import json

    try:
        result = llm_complete(prompt, system=system, temperature=0.0)
        result = result.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[1].rsplit("```", 1)[0]
        names = json.loads(result)
        if cell_type not in names:
            names.insert(0, cell_type)
        return names
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to resolve cell type via LLM: %s", e)
        return [cell_type]
