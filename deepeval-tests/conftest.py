"""
Shared fixtures and configuration for deepeval LLM evaluation tests.
"""

import os
from typing import Optional, Tuple

import requests
import pytest
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCaseParams


# ---------------------------------------------------------------------------
# Custom evaluation model backed by the project's Ollama Cloud API
# ---------------------------------------------------------------------------

class OllamaEvaluationModel(DeepEvalBaseLLM):
    """DeepEval-compatible model that calls the Ollama Cloud chat API.

    Uses the same OLLAMA_BASE_URL / OLLAMA_API_KEY / OLLAMA_MODEL_EVAL
    environment variables that the backend container uses, so no extra
    secrets are required in CI.
    """

    def __init__(self):
        self._base_url = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com").rstrip("/")
        self._api_key = os.environ.get("OLLAMA_API_KEY", "")
        model_name = os.environ.get(
            "OLLAMA_MODEL_EVAL",
            os.environ.get("OLLAMA_MODEL_CLASSIFY", "gemma4:31b"),
        )
        super().__init__(model=model_name)

    def get_model_name(self) -> str:
        return self.name

    def load_model(self):
        return self

    def _chat(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"******"
        response = requests.post(
            f"{self._base_url}/api/chat",
            headers=headers,
            json={
                "model": self.name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def generate(self, prompt: str, schema=None) -> Tuple[str, float]:
        content = self._chat(prompt)
        if schema is not None:
            return schema.model_validate_json(content), 0
        return content, 0

    async def a_generate(self, prompt: str, schema=None) -> Tuple[str, float]:
        return self.generate(prompt, schema)


eval_model = OllamaEvaluationModel()


# ---------------------------------------------------------------------------
# Reusable GEval metric factories
# ---------------------------------------------------------------------------

def json_schema_metric(schema_description: str):
    """Creates a GEval metric that checks JSON schema compliance."""
    return GEval(
        name="JSON Schema Compliance",
        criteria=(
            "Evaluate whether the actual output is valid JSON that conforms to "
            "the required schema. Only check structure, key names, and data "
            "types — do NOT penalize for specific values. "
            + schema_description
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=eval_model,
    )


def output_correctness_metric():
    """Creates a GEval metric that checks factual/logical correctness."""
    return GEval(
        name="Output Correctness",
        criteria=(
            "Determine whether the actual output is logically correct and "
            "reasonable given the input text. The analysis should make sense "
            "for the provided input."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=eval_model,
    )


def answer_relevancy_metric():
    """Creates a GEval metric that checks whether the output is topically
    relevant to the input.  Unlike AnswerRelevancyMetric (which assumes a
    Q&A format), this works for classification and analysis endpoints where
    the output is structured metadata about the input text."""
    return GEval(
        name="Answer Relevancy",
        criteria=(
            "Evaluate whether the actual output is topically relevant to the "
            "input text. The labels, categories, or analysis in the output "
            "should directly relate to the subject matter of the input. "
            "Structured metadata (labels, categories, confidence scores) that "
            "accurately describes the input text should be considered relevant."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=eval_model,
    )
