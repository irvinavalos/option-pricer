APP = bspx
DATA_DIR = data

SRC = src
PACKAGE = bspx

TESTS = tests

PRICING = pricing
GREEKS = greeks

sync:
	@uv sync

run: sync
	@uv run python -m $(APP)

clean:
	@rm -rf  $(DATA_DIR)
	@rm -rf __pycache__ .pytest_cache .hypothesis

test:
	@uv run pytest

test-pricing:
	@uv run pytest $(TESTS)/$(PRICING)

test-greeks:
	@uv run pytest $(TESTS)/$(GREEKS)

test-fast:
	@uv run pytest -m "not slow"

test-cov:
	@uv run pytest --cov/src/bspx --cov-report=term-missing

.PHONY: sync run clean test test-pricing test-fast test-cov
