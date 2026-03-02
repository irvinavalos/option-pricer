import pytest
from tests.cases import OptionTestCase

from bspx.pricing import black_scholes_price
from bspx.volatility import implied_vol


def test_implied_vol_hull(hull_15: OptionTestCase):
    market = hull_15.market
    call_price = hull_15.expected_call
    iv = implied_vol(
        market_price=call_price,
        S=market.S,
        K=market.K,
        T=market.T,
        r=market.r,
        option_type="call",
    )
    bs_price = black_scholes_price(market.S, market.K, market.T, market.r, iv, "call")
    assert bs_price == pytest.approx(call_price, abs=0.01)
