"""Observation rendering: camera frames -> image parts (+ overlays) for the model."""

from controlr.observation.renderers import ObservationRenderer, RenderedObservation, project_points

__all__ = ["ObservationRenderer", "RenderedObservation", "project_points"]
