"""Filesystem-only discovery rules for preprocessing folder layouts.

This module deliberately classifies folders by their immediate directory
structure only. It does not inspect image files or scientific metadata.
"""

import os
from typing import Literal, NamedTuple


class ClassicBatchDiscovery(NamedTuple):
    """Immediate datasets selected for a classic batch workflow."""

    folders: list[str]
    has_child_folders: bool


class StandaloneSummationLayout(NamedTuple):
    """Directory-depth classification for standalone Summation."""

    kind: Literal["too_few", "mixed", "two_level", "three_level"]
    sample_folders: list[str]
    run_folders_by_sample: dict[str, list[str]]


class FullProcessSummationDiscovery(NamedTuple):
    """Immediate Full Process runs and whether their presence triggers summation."""

    folders: list[str]
    should_sum: bool


def immediate_child_directories(folder: str) -> list[str]:
    """Return immediate child directories in the order supplied by os.listdir."""
    return [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, name))
    ]


def discover_full_process_summation(folder: str) -> FullProcessSummationDiscovery:
    """Discover Full Process runs; any immediate child means summation is attempted."""
    folders = immediate_child_directories(folder)
    return FullProcessSummationDiscovery(folders, bool(folders))


def discover_classic_batch(folder: str) -> ClassicBatchDiscovery:
    """Use immediate children as datasets, or the selected folder if none exist."""
    children = immediate_child_directories(folder)
    return ClassicBatchDiscovery(children or [folder], bool(children))


def classify_standalone_summation(folder: str) -> StandaloneSummationLayout:
    """Classify the existing two-level or three-level Summation layouts.

    The minimum top-level dataset check happens before nested folders are
    inspected, matching the GUI's existing validation sequence. Empty and
    unrelated directories count because classification is structure-only.
    """
    samples = immediate_child_directories(folder)
    if len(samples) < 2:
        return StandaloneSummationLayout("too_few", samples, {})

    has_runs = [
        any(
            os.path.isdir(os.path.join(sample, name))
            for name in os.listdir(sample)
        )
        for sample in samples
    ]
    if all(has_runs):
        runs_by_sample = {
            sample: immediate_child_directories(sample) for sample in samples
        }
        return StandaloneSummationLayout("three_level", samples, runs_by_sample)
    if any(has_runs):
        return StandaloneSummationLayout("mixed", samples, {})
    return StandaloneSummationLayout("two_level", samples, {})
