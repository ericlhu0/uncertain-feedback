#!/bin/bash
python -m black .
docformatter -i -r . --exclude venv --exclude motion-diffusion-model
isort .
