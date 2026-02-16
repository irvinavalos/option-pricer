APP = bspx
DATA_DIR = data

sync:
	@uv sync

run: sync
	@uv run python -m $(APP)

clean:
	@rm -rf  $(DATA_DIR)
	@rm -rf __pycache__ .pytest_cache

.PHONY: sync run clean
