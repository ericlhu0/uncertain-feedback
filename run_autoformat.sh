#!/bin/bash
python -m black .
# docformatter is intentionally not run here: a repo-wide pass rewraps every
# existing docstring and its summary wrapping breaks hyphenated words. Run it
# manually if you want it.
isort .
