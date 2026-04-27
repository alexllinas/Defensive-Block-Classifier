"""Abstract base class for tracking data loaders."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import TrackingInput


class BaseLoader(ABC):
    """Interface that all tracking data loaders must implement.

    A loader reads raw data from a specific format (CSV, HDF5, JSON, …)
    and returns a TrackingInput — the common internal representation used
    by the rest of the pipeline.

    Example implementation::

        class MyLoader(BaseLoader):
            def load(self, filepath, analyzed_team_idx, **kwargs) -> TrackingInput:
                ...
    """

    @abstractmethod
    def load(self, *args, **kwargs) -> TrackingInput:
        """Load tracking data and return a TrackingInput.

        Concrete signatures vary by format. See MetricaLoader for a reference
        implementation.
        """
