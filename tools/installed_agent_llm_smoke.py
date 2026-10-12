"""Smoke-test an installed wheel with the optional M4 LLM transport extra."""

from __future__ import annotations

import os
import sys
from importlib import metadata
from pathlib import Path

import NEAT
from NEAT.domain.agent_model import AgentModelMode, AgentModelSettings
from NEAT.services import agent_model

checkout = Path(os.environ["NEAT_CHECKOUT_ROOT"]).resolve()
for entry in sys.path:
    if entry and Path(entry).resolve().is_relative_to(checkout):
        raise AssertionError(f"Checkout leaked onto sys.path: {entry}")
if Path(NEAT.__file__).resolve().is_relative_to(checkout):
    raise AssertionError("NEAT imported from checkout instead of the installed wheel")

for distribution_name in ("langgraph", "httpx", "keyring"):
    metadata.distribution(distribution_name)

settings = AgentModelSettings()
if settings.mode is not AgentModelMode.OFFLINE:
    raise AssertionError("Installed Agent model settings did not default to offline")
if "httpx" in sys.modules or "keyring" in sys.modules:
    raise AssertionError("Importing the installed M4 service eagerly loaded optional clients")
if any(
    name.startswith(("langchain_openai", "langchain_anthropic", "langchain_google"))
    for name in sys.modules
):
    raise AssertionError("M4 service imported an Assistant provider")
if agent_model.OpenAICompatibleStructuredModel is None:
    raise AssertionError("Installed M4 model adapter did not resolve")

print(f"Installed NEAT={NEAT.__file__}; offline Agent M4 imports are ready")
