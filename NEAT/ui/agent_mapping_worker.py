"""Tracked NEAT mapping worker with a compact LangGraph trace wrapper."""

from __future__ import annotations

import os
from typing import TypedDict

import numpy as np
from PyQt5.QtCore import Qt

from ..domain.agent_model import AgentModelSettings
from ..services.agent_model import AgentCredentialResolver, AgentModelError
from ..services.agent_tracing import agent_langsmith_context
from ..workers.batch import BatchFitEdgesWorker


class _MappingTraceState(TypedDict, total=False):
    run_id: str
    dataset_id: str
    revision: int
    macro_pixel_size: int
    step_x: int
    step_y: int
    mapping_area: tuple[int, int, int, int]
    hkls: tuple[tuple[int, int, int], ...]
    shape_parameters: tuple[tuple[tuple[int, int, int], float, float], ...]
    fix_t: bool
    fix_eta: bool
    interpolation_enabled: bool
    status: str
    output_paths: tuple[str, ...]
    successful_centers: tuple[tuple[tuple[int, int, int], int, int], ...]


class AgentMappingWorker(BatchFitEdgesWorker):
    """Run NEAT's existing mapping loop and expose only compact trace metadata."""

    def __init__(
        self,
        *args,
        run_id: str,
        dataset_id: str,
        revision: int,
        agent_settings: AgentModelSettings,
        langsmith_credential_resolver: AgentCredentialResolver | None,
        mapping_area: tuple[int, int, int, int],
        parent=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.setParent(parent)
        self.run_id = run_id
        self.dataset_id = dataset_id
        self.revision = revision
        self.mapping_area = mapping_area
        self.agent_settings = agent_settings
        self.langsmith_credential_resolver = langsmith_credential_resolver
        self.output_payload = ""
        self.successful_center_mask: np.ndarray | None = None
        self.finished.connect(self._capture_output, Qt.DirectConnection)

    def _capture_output(self, payload: str) -> None:
        self.output_payload = str(payload or "")

    def _langsmith_api_key(self) -> str | None:
        if (
            not self.agent_settings.langsmith_tracing_enabled
            or self.langsmith_credential_resolver is None
        ):
            return None
        try:
            return self.langsmith_credential_resolver.get()
        except AgentModelError:
            return None

    def interpolate_results(self) -> None:
        # Preserve the real fit coverage before NEAT fills skipped locations.
        self.successful_center_mask = np.isfinite(self.a_array).copy()
        super().interpolate_results()

    def run(self) -> None:
        try:
            from langgraph.graph import END, START, StateGraph
        except (ImportError, OSError):  # Keep NEAT's mapping usable without tracing extras.
            super().run()
            return

        def run_mapping(_state: _MappingTraceState) -> _MappingTraceState:
            super(AgentMappingWorker, self).run()
            paths = tuple(
                path
                for path in (
                    line.strip() for line in self.output_payload.splitlines()
                )
                if path and os.path.isfile(path)
            )
            center_mask = self.successful_center_mask
            if center_mask is None:
                center_mask = np.isfinite(self.a_array)
            center_rows = np.arange(
                self.min_y + self.box_height // 2,
                self.max_y - self.box_height + 1,
                self.step_y,
            )
            center_columns = np.arange(
                self.min_x + self.box_width // 2,
                self.max_x - self.box_width + 1,
                self.step_x,
            )
            successful_centers = tuple(
                (
                    config.hkl,
                    int(
                        np.isfinite(
                            center_mask[
                                np.ix_(center_rows, center_columns, [edge_index])
                            ]
                        ).sum()
                    ),
                    int(len(center_rows) * len(center_columns)),
                )
                for edge_index, config in enumerate(self.edge_configs)
                if config.hkl is not None
            )
            return {
                "status": (
                    "completed" if len(paths) >= 2
                    else "partial" if paths
                    else "stopped_or_no_fits"
                ),
                "output_paths": paths,
                "successful_centers": successful_centers,
            }

        graph = StateGraph(_MappingTraceState)
        graph.add_node("fit_mapping_grid", run_mapping)
        graph.add_edge(START, "fit_mapping_grid")
        graph.add_edge("fit_mapping_grid", END)
        compiled = graph.compile()
        initial_state: _MappingTraceState = {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "revision": self.revision,
            "macro_pixel_size": self.box_width,
            "step_x": self.step_x,
            "step_y": self.step_y,
            "mapping_area": self.mapping_area,
            "hkls": tuple(
                config.hkl for config in self.edge_configs if config.hkl is not None
            ),
            "shape_parameters": tuple(
                (config.hkl, float(config.t), float(config.eta))
                for config in self.edge_configs
                if config.hkl is not None
            ),
            "fix_t": bool(self.fix_t),
            "fix_eta": bool(self.fix_eta),
            "interpolation_enabled": bool(self.interpolation_enabled),
        }
        try:
            with agent_langsmith_context(
                enabled=self.agent_settings.langsmith_tracing_enabled,
                api_key=self._langsmith_api_key(),
                agent_id=self.agent_settings.langsmith_agent_id,
                environment=self.agent_settings.langsmith_environment,
                region=self.agent_settings.langsmith_region,
                workspace_id=self.agent_settings.langsmith_workspace_id,
                workflow="neat_macro_pixel_mapping",
                thread_id=self.run_id,
            ):
                compiled.invoke(
                    initial_state,
                    config={
                        "run_name": "NEAT: Macro-pixel mapping",
                        "configurable": {"thread_id": self.run_id},
                        "recursion_limit": 4,
                    },
                )
        except Exception as exc:  # noqa: BLE001 - contain errors at the Qt worker boundary.
            self.message.emit(f"Mapping failed safely ({type(exc).__name__}).")
            if not self.output_payload:
                self.finished.emit("")


__all__ = ["AgentMappingWorker"]
