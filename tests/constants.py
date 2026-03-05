from typing import Final

# fmt: off
HULL_ABS: Final[float] = 0.01                   # Absolute tolerance for examples taken from Hull's textbook (~2 decimal places)
PUT_CALL_PARITY_REL: Final[float] = 1e-6        # For put-call parity in model pricing

GREEK_ACC_REL: Final[float] = 1e-3              # Central Diff. is O(h^2), using a BUMP=1e-4 gives ~1e-3 practical precision
GREEK_IDENT_ATOL: Final[float] = 1e-8           # For Greek mathematical properties (put-call parity, symmetry, etc.)
THETA_IDENT_ATOL: Final[float] = 1e-6           # For Theta only as it gives more slack in the case of numerical instability due to time decay

IMPLIED_VOL_REL: Final[float] = 1e-4            #IV solver convergence tolerance
IMPLIED_VOL_RECOVERY_REL: Final[float] = 1e-2   # Looser tolerance for vol recovery -> Deep OTM prices have flat price surfaces
IMPLIED_VOL_PUT_CALL_REL: Final[float] = 1e-2   # Call and put IV consistency since Hull uses prices with rounding error
# fmt: on
