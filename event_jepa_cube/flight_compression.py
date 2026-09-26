"""Generated standalone Arrow Flight IPC policy. Do not edit by hand.

Regenerate with ``python3 scripts/generate_flight_ipc_policy.py``.  This leaf
module intentionally imports no Mycelia API internals so its package remains
independently installable.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pyarrow as pa

PACKAGE_ID = "jcube"
IPC_COMPRESSION_ENV = "MYCELIA_FLIGHT_IPC_COMPRESSION"
POST_DECODE_PROFILE = "python-client32-postdecode"
POST_DECODE_MAX_FRAMES = 1024
POST_DECODE_MAX_BATCHES = 256
POST_DECODE_MAX_BATCH_BYTES = 32 * 1024 * 1024
POST_DECODE_MAX_STREAM_BYTES = 256 * 1024 * 1024
BOUNDED_DECODE_READY = False
MAX_EXACT_OBSERVATION_INTEGER = (1 << 53) - 1
ROUTE_KEYS = ("jcube.client.dynamic",)
ROUTE_POLICIES = {
    "jcube.client.dynamic": {
        "allowed_codecs": ("none",),
        "default_codec": "none",
        "eligible_after_proof": False,
        "receiver_profile": "unnegotiated",
    },
}
REQUIRED_READER_CODECS = ("none",)
_MISSING = object()
_LOGGER = logging.getLogger(__name__)


class FlightIpcCompression(StrEnum):
    NONE = "none"
    LZ4 = "lz4"
    ZSTD = "zstd"


class FlightIpcDirection(StrEnum):
    READ = "read"
    WRITE = "write"


class FlightRpcAction(StrEnum):
    DO_GET = "do_get"
    DO_PUT = "do_put"
    DO_EXCHANGE = "do_exchange"
    DO_ACTION = "do_action"


class FlightIpcOutcome(StrEnum):
    OK = "ok"
    ENCODE_ERROR = "encode_error"
    DECODE_ERROR = "decode_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    PEER_NOT_READY = "peer_not_ready"
    CANCELLED = "cancelled"


class FlightIpcCompressionError(ValueError):
    """Invalid standalone Flight IPC codec without echoing its value."""


class FlightCodecUnavailableError(FlightIpcCompressionError):
    """A required PyArrow IPC codec is unavailable."""


class FlightPolicyConfigError(FlightIpcCompressionError):
    """A none-only standalone route policy is invalid."""


class FlightPostDecodeBudgetError(ValueError):
    """Stable payload-free post-decode budget failure."""


class FlightObservationError(ValueError):
    """A finite observation value is invalid."""


def parse_flight_ipc_compression(raw: str | None) -> FlightIpcCompression:
    if raw is None:
        return FlightIpcCompression.NONE
    if not isinstance(raw, str):
        raise FlightIpcCompressionError("invalid Flight IPC codec")
    try:
        return FlightIpcCompression(raw)
    except ValueError as exc:
        raise FlightIpcCompressionError("invalid Flight IPC codec") from exc


def codec_availability() -> dict[str, bool]:
    available = {FlightIpcCompression.NONE.value: True}
    for codec in (FlightIpcCompression.LZ4, FlightIpcCompression.ZSTD):
        try:
            available[codec.value] = bool(pa.Codec.is_available(codec.value))
        except Exception:
            available[codec.value] = False
    return available


def assert_required_reader_codecs(
    codecs: Iterable[str] = REQUIRED_READER_CODECS,
) -> tuple[FlightIpcCompression, ...]:
    availability = codec_availability()
    checked: list[FlightIpcCompression] = []
    for raw in codecs:
        codec = parse_flight_ipc_compression(raw)
        if not availability[codec.value]:
            raise FlightCodecUnavailableError(
                "required PyArrow Flight codec is unavailable",
            )
        if codec not in checked:
            checked.append(codec)
    return tuple(checked)


assert_pyarrow_ipc_codecs = assert_required_reader_codecs


def ipc_write_options(codec: FlightIpcCompression | str) -> pa.ipc.IpcWriteOptions:
    if isinstance(codec, FlightIpcCompression):
        selected = codec
    else:
        selected = parse_flight_ipc_compression(codec)
    compression = None if selected is FlightIpcCompression.NONE else selected.value
    return pa.ipc.IpcWriteOptions(compression=compression)


@dataclass(frozen=True, slots=True)
class FlightCompressionConfig:
    routes: tuple[str, ...]
    codec: FlightIpcCompression

    @classmethod
    def from_value(
        cls,
        raw: str | None,
        *,
        routes: Iterable[str] = ROUTE_KEYS,
    ) -> FlightCompressionConfig:
        selected_routes = tuple(routes)
        if not selected_routes or len(set(selected_routes)) != len(selected_routes):
            raise FlightPolicyConfigError("invalid standalone Flight route inventory")
        if any(route not in ROUTE_POLICIES for route in selected_routes):
            raise FlightPolicyConfigError("unknown standalone Flight route")
        selected = parse_flight_ipc_compression(raw)
        for route in selected_routes:
            if selected.value not in ROUTE_POLICIES[route]["allowed_codecs"]:
                raise FlightPolicyConfigError(
                    "compression forbidden for standalone Flight route",
                )
        assert_required_reader_codecs()
        return cls(selected_routes, selected)

    @classmethod
    def from_values(
        cls,
        default: str | None,
        overrides: str | None = None,
        *,
        routes: Iterable[str] = ROUTE_KEYS,
    ) -> FlightCompressionConfig:
        if overrides not in (None, "", "{}"):
            raise FlightPolicyConfigError(
                "standalone Flight route overrides are unsupported",
            )
        return cls.from_value(default, routes=routes)

    @classmethod
    def from_env(cls, *, routes: Iterable[str] = ROUTE_KEYS) -> FlightCompressionConfig:
        return cls.from_value(os.getenv(IPC_COMPRESSION_ENV), routes=routes)

    def codec_for(self, route: str) -> FlightIpcCompression:
        if route not in self.routes:
            raise FlightPolicyConfigError("standalone Flight route is not configured")
        return self.codec

    selected_codec = codec_for

    def selected_codec_name(self, route: str) -> str:
        return self.codec_for(route).value

    def options_for(self, route: str) -> pa.ipc.IpcWriteOptions:
        return ipc_write_options(self.codec_for(route))


@dataclass(frozen=True, slots=True)
class FlightPostDecodeBudget:
    max_frames: int = POST_DECODE_MAX_FRAMES
    max_batches: int = POST_DECODE_MAX_BATCHES
    max_batch_bytes: int = POST_DECODE_MAX_BATCH_BYTES
    max_stream_bytes: int = POST_DECODE_MAX_STREAM_BYTES

    def __post_init__(self) -> None:
        values = (
            self.max_frames,
            self.max_batches,
            self.max_batch_bytes,
            self.max_stream_bytes,
        )
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise FlightPolicyConfigError("invalid Flight post-decode budget")


PYTHON_CLIENT32_POSTDECODE = FlightPostDecodeBudget()


def logical_nbytes(value: Any) -> int:
    size = getattr(value, "nbytes", None)
    if size is None and isinstance(value, (bytes, bytearray, memoryview)):
        size = len(value)
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise FlightPostDecodeBudgetError("flight post-decode budget exceeded")
    return size


def iter_flight_batches(
    reader: Iterable[Any],
    budget: FlightPostDecodeBudget = PYTHON_CLIENT32_POSTDECODE,
) -> Iterator[Any]:
    frames = 0
    batches = 0
    stream_bytes = 0
    for chunk in reader:
        frames += 1
        if frames > budget.max_frames:
            raise FlightPostDecodeBudgetError("flight post-decode budget exceeded")
        batch = getattr(chunk, "data", _MISSING)
        if batch is None:
            continue
        if batch is _MISSING:
            batch = chunk
        batches += 1
        if batches > budget.max_batches:
            raise FlightPostDecodeBudgetError("flight post-decode budget exceeded")
        batch_bytes = logical_nbytes(batch)
        if batch_bytes > budget.max_batch_bytes:
            raise FlightPostDecodeBudgetError("flight post-decode budget exceeded")
        stream_bytes += batch_bytes
        if stream_bytes > budget.max_stream_bytes:
            raise FlightPostDecodeBudgetError("flight post-decode budget exceeded")
        yield batch


def collect_flight_table(
    reader: Iterable[Any],
    budget: FlightPostDecodeBudget = PYTHON_CLIENT32_POSTDECODE,
) -> pa.Table:
    schema = getattr(reader, "schema", None)
    schema = schema() if callable(schema) else schema
    batches = list(iter_flight_batches(reader, budget))
    if batches:
        return pa.Table.from_batches(batches)
    if isinstance(schema, pa.Schema):
        return pa.Table.from_batches([], schema=schema)
    return pa.table({})


@dataclass(frozen=True, slots=True)
class FlightIpcObservation:
    route: str
    codec: FlightIpcCompression
    direction: FlightIpcDirection
    action: FlightRpcAction
    outcome: FlightIpcOutcome
    logical_bytes: int
    duration_seconds: float

    def __post_init__(self) -> None:
        if self.route not in ROUTE_KEYS:
            raise FlightObservationError("invalid Flight IPC observation")
        enum_values = (self.codec, self.direction, self.action, self.outcome)
        enum_types = (
            FlightIpcCompression,
            FlightIpcDirection,
            FlightRpcAction,
            FlightIpcOutcome,
        )
        for value, expected in zip(enum_values, enum_types):
            if not isinstance(value, expected):
                raise FlightObservationError("invalid Flight IPC observation")
        if (
            isinstance(self.logical_bytes, bool)
            or not isinstance(self.logical_bytes, int)
            or self.logical_bytes < 0
            or self.logical_bytes > MAX_EXACT_OBSERVATION_INTEGER
            or isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds < 0
        ):
            raise FlightObservationError("invalid Flight IPC observation")

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "route": self.route,
            "codec": self.codec.value,
            "direction": self.direction.value,
            "action": self.action.value,
            "outcome": self.outcome.value,
            "logical_bytes": self.logical_bytes,
            "duration_seconds": float(self.duration_seconds),
        }


ObservationHook = Callable[[FlightIpcObservation], None]


def emit_flight_observation(
    observation: FlightIpcObservation,
    hook: ObservationHook | None = None,
) -> None:
    if hook is not None:
        hook(observation)
    else:
        _LOGGER.info("flight_ipc_observation %s", observation.as_dict())


class _FlightOperationObservation:
    def __init__(
        self,
        *,
        route: str,
        codec: FlightIpcCompression,
        direction: FlightIpcDirection,
        action: FlightRpcAction,
        hook: ObservationHook | None,
    ) -> None:
        if route not in ROUTE_KEYS or not isinstance(codec, FlightIpcCompression):
            raise FlightObservationError("invalid Flight IPC observation")
        valid_direction = isinstance(direction, FlightIpcDirection)
        valid_action = isinstance(action, FlightRpcAction)
        if not valid_direction or not valid_action:
            raise FlightObservationError("invalid Flight IPC observation")
        self._route = route
        self._codec = codec
        self._direction = direction
        self._action = action
        self._hook = hook
        self._logical_bytes = 0
        self._started = 0.0
        self._emitted = False

    def __enter__(self) -> _FlightOperationObservation:
        if self._started or self._emitted:
            raise FlightObservationError("Flight IPC observation already started")
        self._started = time.monotonic()
        return self

    def add_logical_bytes(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FlightObservationError("invalid Flight IPC observation")
        total = self._logical_bytes + value
        if total > MAX_EXACT_OBSERVATION_INTEGER:
            raise FlightObservationError("invalid Flight IPC observation")
        self._logical_bytes = total

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._emitted:
            raise FlightObservationError("Flight IPC observation already emitted")
        self._emitted = True
        if exc_type is None:
            outcome = FlightIpcOutcome.OK
        elif issubclass(exc_type, FlightPostDecodeBudgetError):
            outcome = FlightIpcOutcome.RESOURCE_EXHAUSTED
        elif issubclass(exc_type, (GeneratorExit, KeyboardInterrupt, SystemExit)):
            outcome = FlightIpcOutcome.CANCELLED
        elif self._direction is FlightIpcDirection.READ:
            outcome = FlightIpcOutcome.DECODE_ERROR
        else:
            outcome = FlightIpcOutcome.ENCODE_ERROR
        emit_flight_observation(
            FlightIpcObservation(
                route=self._route,
                codec=self._codec,
                direction=self._direction,
                action=self._action,
                outcome=outcome,
                logical_bytes=self._logical_bytes,
                duration_seconds=max(time.monotonic() - self._started, 0.0),
            ),
            self._hook,
        )
        return False


def observe_flight_operation(
    *,
    route: str,
    codec: FlightIpcCompression,
    direction: FlightIpcDirection,
    action: FlightRpcAction,
    hook: ObservationHook | None = None,
) -> _FlightOperationObservation:
    return _FlightOperationObservation(
        route=route,
        codec=codec,
        direction=direction,
        action=action,
        hook=hook,
    )


def codec_capability_payload() -> dict[str, Any]:
    available = codec_availability()
    read_codecs = []
    for codec in FlightIpcCompression:
        if available[codec.value]:
            read_codecs.append(codec.value)
    write_routes = {}
    for route in ROUTE_KEYS:
        write_routes[route] = ROUTE_POLICIES[route]["default_codec"]
    return {
        "receiver_identity": PACKAGE_ID,
        "receiver_profile": POST_DECODE_PROFILE,
        "read_codecs": read_codecs,
        "bounded_decode_ready": BOUNDED_DECODE_READY,
        "post_decode_profile": POST_DECODE_PROFILE,
        "write_routes": write_routes,
    }
