from typing import Protocol


class MetricsHook(Protocol):
    def record_latency(self, name: str, value_ms: float) -> None:
        """Record a latency metric.

        Args:
            name: Name of the metric.
            value_ms: Latency in milliseconds.
        """
        ...


class NoOpMetricsHook:
    def record_latency(self, name: str, value_ms: float) -> None:
        pass
