import pytest
from hypothesis import given
from tests.cases import OptionTestCase
from tests.constants import (
    IMPLIED_VOL_PUT_CALL_REL,
    IMPLIED_VOL_RECOVERY_REL,
    IMPLIED_VOL_REL,
)
from tests.hypothesis_strategies import gen_black_scholes_parameters

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


@pytest.mark.slow
@given(bs_params=gen_black_scholes_parameters())
def test_implied_vol_round_trip(bs_params):
    S, K, T, r, vol = bs_params
    for option_type in ("call", "put"):
        market_price = float(black_scholes_price(S, K, T, r, vol, option_type))
        iv = implied_vol(market_price, S, K, T, r, option_type)
        recovered = float(black_scholes_price(S, K, T, r, iv, option_type))
        assert recovered == pytest.approx(market_price, rel=IMPLIED_VOL_REL)


@pytest.mark.slow
@given(bs_params=gen_black_scholes_parameters())
def test_implied_vol_recovers_input_vol(bs_params):
    """IV should recover the original vol used to generate the price.

    Note: Deep OTM options have flat price surfaces so vol recovery
    is numerically ill-conditioned and requires looser tolerance.
    """
    S, K, T, r, vol = bs_params
    market_price = float(black_scholes_price(S, K, T, r, vol, "call"))
    iv = implied_vol(market_price, S, K, T, r, "call")
    assert iv == pytest.approx(vol, rel=IMPLIED_VOL_RECOVERY_REL)


def test_implied_vol_call_put_consistency(hull_15: OptionTestCase):
    """IV from call and put should be equal for same strike and expiry.

    Note: Slight discrepancy expected due to rounding in Hull textbook prices.
    """
    market = hull_15.market
    iv_call = implied_vol(
        hull_15.expected_call, market.S, market.K, market.T, market.r, "call"
    )
    iv_put = implied_vol(
        hull_15.expected_put, market.S, market.K, market.T, market.r, "put"
    )
    assert iv_call == pytest.approx(iv_put, rel=IMPLIED_VOL_PUT_CALL_REL)


def test_implied_vol_deep_itm(hull_15: OptionTestCase):
    """IV solver should handle deep in-the-money options."""
    market = hull_15.market
    deep_itm_price = float(
        black_scholes_price(market.S, market.K * 0.5, market.T, market.r, 0.2, "call")
    )
    iv = implied_vol(
        deep_itm_price, market.S, market.K * 0.5, market.T, market.r, "call"
    )
    recovered = float(
        black_scholes_price(market.S, market.K * 0.5, market.T, market.r, iv, "call")
    )
    assert recovered == pytest.approx(deep_itm_price, rel=IMPLIED_VOL_REL)
