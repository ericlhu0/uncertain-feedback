#!/bin/bash
python -m black .
docformatter -i -r . --exclude venv --exclude motion-diffusion-model --exclude sam-3d-body --exclude MHR --exclude sam-3d-body --exclude MHR
isort .
