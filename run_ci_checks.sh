#!/bin/bash
./run_autoformat.sh
uv run mypy . --exclude src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model --exclude src/uncertain_feedback/data_collection/sam-3d-body --exclude src/uncertain_feedback/data_collection/MHR
uv run pylint --rcfile=.pylintrc src/uncertain_feedback tests
uv run pytest tests/
