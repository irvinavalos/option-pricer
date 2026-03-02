from bspx.greeks.analytical import AnalyticalBackend
from bspx.greeks.numerical import NumericalBackend
from bspx.instruments import Greeks
from bspx.types import _F64, DayCount, GreeksBackend, OptionType

__all__ = [
    "AnalyticalBackend",
    "NumericalBackend",
    "delta",
    "gamma",
    "rho",
    "theta",
    "vega",
    "calculate_greeks",
]


def delta(backend: GreeksBackend, option_type: OptionType = "call") -> _F64:
    return backend.delta(option_type)


def theta(
    backend: GreeksBackend,
    option_type: OptionType = "call",
    day_count: DayCount = DayCount.CALENDAR,
) -> _F64:
    return backend.theta(option_type, day_count)


def gamma(backend: GreeksBackend) -> _F64:
    return backend.gamma()


def vega(backend: GreeksBackend) -> _F64:
    return backend.vega()


def rho(backend: GreeksBackend, option_type: OptionType = "call") -> _F64:
    return backend.rho(option_type)


def calculate_greeks(
    backend: GreeksBackend,
    option_type: OptionType = "call",
    day_count: DayCount = DayCount.CALENDAR,
) -> Greeks:
    return Greeks(
        delta=delta(backend, option_type),
        theta=theta(backend, option_type, day_count),
        gamma=gamma(backend),
        vega=vega(backend),
        rho=rho(backend, option_type),
    )
