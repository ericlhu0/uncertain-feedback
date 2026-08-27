"""Where each data-collection method keeps the data it generates.

``data/`` mirrors the method packages one level up, so a method's code and its
generated data sit at matching paths. The whole tree is gitignored.

Absolute, not relative to the launch directory: the MDM loader ``os.chdir()``s
into its submodule and never restores, so a relative default would move
underneath a run that loads the generator.
"""

from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
VIDEO_DATA_DIR = DATA_ROOT / "dataset_video"
AUTO_CORRECTION_DATA_DIR = DATA_ROOT / "dataset_auto_correction"

# The clip set `generate.py` writes and `label.py` captions when neither is
# given a directory.
DEFAULT_CLIP_SET = AUTO_CORRECTION_DATA_DIR / "clips"
