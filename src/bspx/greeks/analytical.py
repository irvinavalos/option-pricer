from bspx.greeks.formulas import analytical
from bspx.pricing import BlackScholesState
from bspx.types import _F64, DayCount, DiffMethod, OptionType


class AnalyticalBackend:
    method = DiffMethod.ANALYTICAL

    def __init__(self, state: BlackScholesState) -> None:
        self._state = state

    def delta(self, option_type: OptionType = "call") -> _F64:
        return analytical.delta(self._state, option_type)

    def theta(
        self, option_type: OptionType = "call", day_count: DayCount = DayCount.CALENDAR
    ) -> _F64:
        return analytical.theta(self._state, option_type, day_count)

    def gamma(self) -> _F64:
        return analytical.gamma(self._state)

    def vega(self) -> _F64:
        return analytical.vega(self._state)

    def rho(self, option_type: OptionType = "call") -> _F64:
        return analytical.rho(self._state, option_type)
