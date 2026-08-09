"""Approaches: the system variants under evaluation."""

from evaluation.approaches.base import Approach
from evaluation.approaches.edit_baseline import ParameterizedEditApproach
from evaluation.approaches.system import SystemApproach

__all__ = ["Approach", "ParameterizedEditApproach", "SystemApproach"]
