#!/bin/bash
./run_autoformat.sh
uv run mypy .
uv run pylint --rcfile=.pylintrc src/uncertain_feedback evaluation tests
uv run pytest tests/
