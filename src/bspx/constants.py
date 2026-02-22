# Note: 365.25 is technically more accurate but Hull opts for 366
CALENDAR_DAYS_PER_YEAR = 365

TRADING_DAYS_PER_YEAR = 252

# Minimum time to expiration based on number of days considered
MIN_T_CALENDAR = 1 / CALENDAR_DAYS_PER_YEAR
MIN_T_TRADING = 1 / TRADING_DAYS_PER_YEAR
