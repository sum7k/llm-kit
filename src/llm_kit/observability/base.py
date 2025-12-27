from typing import Protocol


class MetricsHook(Protocol):
    def record_latency(
        self,
        name: str,
        value_ms: float,
        labels: dict[str, str] | None = None,
    ) -> None: ...

    def increment(
        self,
        name: str,
        value: int = 1,
        labels: dict[str, str] | None = None,
    ) -> None: ...

    def record_gauge(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None: ...


class NoOpMetricsHook:
    def record_latency(
        self, name: str, value_ms: float, labels: dict[str, str] | None = None
    ) -> None:
        pass

    def increment(
        self, name: str, value: int = 1, labels: dict[str, str] | None = None
    ) -> None:
        pass

    def record_gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        pass
