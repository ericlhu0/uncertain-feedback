#!/bin/bash
./run_autoformat.sh
mypy . --exclude src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model
pytest . --ignore=src/uncertain_feedback/motion_generators/mdm/motion-diffusion-model --pylint -m pylint --pylint-rcfile=.pylintrc
pytest tests/
