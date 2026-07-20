#!/bin/bash
./run_autoformat.sh
mypy . --exclude src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model --exclude src/uncertain_feedback/data_collection/sam-3d-body --exclude src/uncertain_feedback/data_collection/MHR
pylint --rcfile=.pylintrc src/uncertain_feedback tests
pytest tests/
