from numpy.typing import ArrayLike

from bspx.greeks.formulas import numerical
from bspx.types import _F64, DayCount, DiffMethod, OptionType, PricingFunction


class NumericalBackend:
    method = DiffMethod.NUMERICAL

    def __init__(
        self,
        pricing_func: PricingFunction,
        S: ArrayLike,
        K: ArrayLike,
        T: ArrayLike,
        r: ArrayLike,
        vol: ArrayLike,
    ) -> None:
        self._pricing_func = pricing_func
        self._S = S
        self._K = K
        self._T = T
        self._r = r
        self._vol = vol

    def delta(self, option_type: OptionType = "call") -> _F64:
        return numerical.delta_fd(
            self._pricing_func,
            self._S,
            self._K,
            self._T,
            self._r,
            self._vol,
            option_type,
        )

    def theta(
        self, option_type: OptionType = "call", day_count: DayCount = DayCount.CALENDAR
    ) -> _F64:
        return numerical.theta_fd(
            self._pricing_func,
            self._S,
            self._K,
            self._T,
            self._r,
            self._vol,
            option_type,
            day_count,
        )

    def gamma(self) -> _F64:
        return numerical.gamma_fd(
            self._pricing_func,
            self._S,
            self._K,
            self._T,
            self._r,
            self._vol,
        )

    def vega(self) -> _F64:
        return numerical.vega_fd(
            self._pricing_func,
            self._S,
            self._K,
            self._T,
            self._r,
            self._vol,
        )

    def rho(self, option_type: OptionType = "call") -> _F64:
        return numerical.rho_fd(
            self._pricing_func,
            self._S,
            self._K,
            self._T,
            self._r,
            self._vol,
            option_type,
        )
