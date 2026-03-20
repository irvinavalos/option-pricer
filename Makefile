PACKAGE = quantoptions

.PHONY: sync clean lint lint-fix format type-check

sync:
	@uv sync

sync-dev:
	@uv sync --extra dev

clean:
	@rm -rf __pycache__

lint:
	@uv run ruff check $(PACKAGE)

lint-fix:
	@uv run ruff check --fix $(PACKAGE)

format:
	@uv run ruff format $(PACKAGE)

type-check:
	@uv run pyright $(PACKAGE)
