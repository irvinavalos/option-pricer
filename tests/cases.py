from dataclasses import dataclass


@dataclass(frozen=True)
class MarketState:
    S: float
    K: float
    T: float
    r: float
    vol: float


@dataclass(frozen=True)
class TestMetadata:
    name: str
    source: str
    notes: str | None = None


@dataclass(frozen=True)
class OptionTestCase:
    market: MarketState
    metadata: TestMetadata
    expected_call: float
    expected_put: float


@dataclass(frozen=True)
class DeltaTestCase:
    market: MarketState
    metadata: TestMetadata
    expected_call: float
    expected_put: float


@dataclass(frozen=True)
class ThetaTestCase:
    market: MarketState
    metadata: TestMetadata
    expected_call_calendar: float
    expected_put_calendar: float
    expected_call_trading: float
    expected_put_trading: float


@dataclass(frozen=True)
class GammaTestCase:
    market: MarketState
    metadata: TestMetadata
    expected_gamma: float


@dataclass(frozen=True)
class VegaTestCase:
    market: MarketState
    metadata: TestMetadata
    expected_vega: float


@dataclass(frozen=True)
class RhoTestCase:
    market: MarketState
    metadata: TestMetadata
    expected_call: float
    expected_put: float
