"""Optional OpenAI-compatible LLM query planner.

The LLM only selects one of the application's existing specialist workflows.
It does not get to execute code or invent a new task, and the deterministic
parser remains the fallback when the service is not configured or fails.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict
from urllib import request


ALLOWED_TASKS = {
    "single": {"vqa", "caption", "grounding"},
    "cross_modal": {"fusion_analysis"},
    "bi_temporal": {"change_vqa", "change_description"},
}


class LLMQueryPlanner:
    def __init__(self, endpoint: str, api_key: str, model: str, timeout: float = 20.0):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def __call__(self, query: str, scenario: str) -> Dict[str, Any]:
        allowed = sorted(ALLOWED_TASKS.get(scenario, set()))
        if not allowed:
            raise ValueError(f"Unsupported scenario for LLM planning: {scenario}")
        system = (
            "You are a remote-sensing query router. Return JSON only, with keys "
            "task and params. Choose exactly one task from the allowed list. "
            "For target land cover use one of water, vegetation, built_up, bare_soil, "
            "or null. Never answer the question; only route it."
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({
                    "scenario": scenario, "allowed_tasks": allowed, "query": query
                })},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.endpoint}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
        content = raw["choices"][0]["message"]["content"]
        result = json.loads(content) if isinstance(content, str) else content
        task = result.get("task")
        if task not in allowed:
            raise ValueError(f"LLM returned unsupported task: {task!r}")
        params = result.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("LLM params must be an object")
        target = params.get("target_class")
        if target is not None and target not in {"water", "vegetation", "built_up", "bare_soil"}:
            params.pop("target_class", None)
        return {"task": task, "params": params}


def configured_planner() -> LLMQueryPlanner | None:
    endpoint = os.environ.get("SATQUERY_LLM_ENDPOINT", "").strip()
    api_key = os.environ.get("SATQUERY_LLM_API_KEY", "").strip()
    model = os.environ.get("SATQUERY_LLM_MODEL", "").strip()
    if not endpoint and not api_key and not model:
        return None
    if not endpoint or not api_key or not model:
        raise ValueError(
            "SATQUERY_LLM_ENDPOINT, SATQUERY_LLM_API_KEY, and SATQUERY_LLM_MODEL "
            "must all be set to enable LLM query understanding."
        )
    return LLMQueryPlanner(endpoint, api_key, model)
