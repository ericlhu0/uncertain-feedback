"""Cost generation: what an approach distills from corrections into planning."""

from evaluation.approaches.cost_gen.base import COST_GEN_SOURCES, CostGen
from evaluation.approaches.cost_gen.consolidate import ConsolidateCostGen
from evaluation.approaches.cost_gen.immediate import ImmediateCostGen
from evaluation.approaches.cost_gen.none import NoCostGen

# The conf/approach/cost_gen/ group names run_comparison.py may force per arm.
COST_GEN_MODES = ("none", "immediate", "consolidate")

__all__ = [
    "COST_GEN_MODES",
    "COST_GEN_SOURCES",
    "ConsolidateCostGen",
    "CostGen",
    "ImmediateCostGen",
    "NoCostGen",
]
