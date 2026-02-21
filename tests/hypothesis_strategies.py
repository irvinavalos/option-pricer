from hypothesis import strategies as st

from bspx.constants import MIN_T_CALENDAR, MIN_T_TRADING

BS_S_BOUND = (0.01, 1_000.0)
BS_K_BOUND = (0.01, 1_000.0)

# Note: for below could use 10.0 instead of 3.0, but results
# from using 3.0 are better representations to real world market
BS_T_TRADING_BOUND = (MIN_T_TRADING, 3.0)
BS_T_CALENDAR_BOUND = (MIN_T_CALENDAR, 3.0)

BS_R_BOUND = (-0.05, 0.20)
BS_VOL_BOUND = (0.01, 2.0)


@st.composite
def gen_black_scholes_parameters(draw):
    return (
        draw(st.floats(*BS_S_BOUND, allow_nan=False, allow_infinity=False)),
        draw(st.floats(*BS_K_BOUND, allow_nan=False, allow_infinity=False)),
        draw(st.floats(*BS_T_TRADING_BOUND, allow_nan=False, allow_infinity=False)),
        draw(st.floats(*BS_R_BOUND, allow_nan=False, allow_infinity=False)),
        draw(st.floats(*BS_VOL_BOUND, allow_nan=False, allow_infinity=False)),
    )
