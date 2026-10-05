"""Run from a temporary directory to verify an installed NEAT artifact."""

from __future__ import annotations

import os
import sys
from importlib import metadata
from pathlib import Path

import NEAT
import tools
from NEAT.domain import IndividualEdgeFitConfig, WavelengthRegion
from NEAT.package_resources import assistant_knowledge_root, launch_splash_resource
from NEAT.services.fitting_engine import FittingEngine
from tools.assistant_retrieval import (
    BM25Retriever,
    KNOWLEDGE_FILENAMES,
    load_knowledge_base,
)


checkout = Path(os.environ["NEAT_CHECKOUT_ROOT"]).resolve()
for entry in sys.path:
    if entry and Path(entry).resolve().is_relative_to(checkout):
        raise AssertionError(f"Checkout leaked onto sys.path: {entry}")
for label, module in (("NEAT", NEAT), ("tools", tools)):
    origin = Path(module.__file__).resolve()
    if origin.is_relative_to(checkout):
        raise AssertionError(f"{label} imported from checkout: {origin}")
    print(f"{label}.__file__={origin}")

distribution = metadata.distribution("NEAT")
if distribution.version != NEAT.__version__:
    raise AssertionError(
        f"Installed metadata version {distribution.version} != NEAT.__version__ {NEAT.__version__}"
    )
print(f"NEAT distribution version={distribution.version}")

scripts = {
    entry_point.name
    for entry_point in distribution.entry_points
    if entry_point.group == "console_scripts"
}
required_scripts = {
    "neat",
    "neat-assistant-server",
    "neat-assistant-laptop-setup",
}
if not required_scripts <= scripts:
    raise AssertionError(f"Missing console-script metadata: {required_scripts - scripts}")

if "PyQt5" in sys.modules:
    raise AssertionError("Headless domain/service imports loaded PyQt5")
if FittingEngine is None or IndividualEdgeFitConfig is None or WavelengthRegion is None:
    raise AssertionError("Scientific domain/service imports did not resolve")

sections = load_knowledge_base(assistant_knowledge_root())
loaded_filenames = {section.filename for section in sections}
if loaded_filenames != set(KNOWLEDGE_FILENAMES):
    raise AssertionError(f"Installed knowledge files differ from allow-list: {loaded_filenames}")
if not sections:
    raise AssertionError("Installed knowledge base is empty")
results = BM25Retriever(sections).search("How do I fit a Bragg edge?", limit=3)
if not results:
    raise AssertionError("Installed BM25 retrieval returned no result")

splash = launch_splash_resource()
if not splash.is_file() or not splash.read_bytes():
    raise AssertionError("Installed launch splash resource is missing or empty")
print(f"Knowledge sections={len(sections)}; BM25 results={len(results)}")
print(f"Splash resource={splash}")
