"""Steering a diffusion sampler toward a user cost model.

Two mechanisms, selected by :attr:`SteeringConfig.mode`:

``cg``
    Classifier guidance: nudge the reverse-diffusion mean by
    ``-guidance_weight · ∇ₓ cost(x̂0(x))``. This is the default steering
    method: it also covers the regime where the prompt and the cost genuinely
    conflict, at the price of a per-cost ``guidance_weight`` calibration
    (~1e4–1e5), a cost that must be differentiable w.r.t. ``x̂0``, and roughly
    2x sampling time.

``resample``
    Alternative: at chosen denoising steps, score every chain's ``x̂0``
    prediction with the cost and systematically resample the population with
    weights ``softmax(-z(cost) / temperature)``. Cost-agnostic (the cost need
    not be differentiable), scale-free in the temperature knob, and free at
    production batch sizes, but it can only reweight the mass the model already
    puts on satisfying motions.

This module is imported while parsing YAML configs, so it must stay torch-free
at module scope: ``torch.Tensor`` appears only in string annotations under
``TYPE_CHECKING`` and ``torch`` itself is imported lazily inside
:func:`make_cond_fn`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor

    from uncertain_feedback.motion_generators.base import MotionGenerator
    from uncertain_feedback.planners.mpc.costs.generated import GeneratedPythonCost
    from uncertain_feedback.simulated_users.base import Bound

STEERING_MODES = ("off", "cg", "resample")


@dataclass(frozen=True)
class SteeringConfig:
    """Which steering mechanism to run, and with what knobs.

    ``resample_steps`` and ``guide_from`` are indices into the denoising loop
    (0 = the first, noisiest step), not diffusion timesteps. Entries at or past
    the sampler's step count simply never fire.

    ``bounds`` is the only cost steering compiles: a learned cost's bounds, or
    ones written in the config. The simulated user's hidden bounds are never a
    fallback, so with no ``bounds`` sampling is unsteered whatever the persona.

    Learned limits steer through a smooth cost of fixed scale, the squared
    excess averaged over time: ``guidance_weight`` multiplies path limits
    (rad²) and ``speed_guidance_weight`` speed limits ((rad/s)² or (m/s)²).
    """

    mode: str = "cg"
    resample_steps: tuple[int, ...] = (15, 25, 35, 45)
    temperature: float = 0.5
    guide_from: int = 10
    guidance_weight: float = 1e5
    speed_guidance_weight: float = 1e4
    bounds: tuple["Bound", ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in STEERING_MODES:
            raise ValueError(
                f"Unknown steering mode {self.mode!r}; choose from "
                f"{list(STEERING_MODES)}."
            )
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}.")
        steps = tuple(self.resample_steps)
        if any(step < 0 for step in steps):
            raise ValueError(f"resample_steps must be nonnegative, got {steps}.")
        if any(later <= earlier for earlier, later in zip(steps, steps[1:])):
            raise ValueError(f"resample_steps must be ascending, got {steps}.")


@dataclass(frozen=True)
class SteeringEvent:
    """Diagnostics recorded at one scoring step of the denoising loop."""

    step: int
    cost_mean: float
    frac_violating: float
    ess: float
    unique_ancestors: int | None = None


@dataclass(frozen=True)
class SteeringSpec:
    """A compiled steering cost plus the configuration to apply it with.

    ``cost`` maps a normalized ``x̂0`` batch ``(N, 263, 1, T)`` to per-sample
    costs ``(N,)``, and must be differentiable w.r.t. its input for ``cg``.
    """

    cost: Callable[[Tensor], Tensor]
    config: SteeringConfig
    seed: int = 0


def build_steering_spec(
    gen: MotionGenerator,
    config: SteeringConfig,
    seed: int,
) -> SteeringSpec | None:
    """Compile ``config.bounds`` into a steering spec, or ``None`` if unsteerable.

    Takes no persona: the simulated user's hidden bounds are evaluation ground
    truth, so the one run that steers with them (the hidden-limit ceiling)
    passes them in through ``config.bounds`` itself. Skips (with a printed
    reason) when there are no bounds, the backend does not implement steering,
    or none of the bounds is supported by the torch feature path.
    """
    if config.mode == "off":
        return None
    if not config.bounds:
        print("steering: no steering bounds given, sampling unsteered")
        return None
    # pylint: disable=import-outside-toplevel
    from uncertain_feedback.motion_generators.mdm.mdm_api import MdmMotionGenerator
    from uncertain_feedback.simulated_users.base import SimulatedUser

    if not isinstance(gen, MdmMotionGenerator):
        print("steering: unsupported backend, sampling unsteered")
        return None
    target = SimulatedUser(
        name="steering.bounds",
        description="",
        feedback_text="",
        bounds=tuple(config.bounds),
    )
    cost = gen.build_user_steering_cost(target)
    if cost is None:
        print("steering: no supported steering bounds, sampling unsteered")
        return None
    return SteeringSpec(cost=cost, config=config, seed=seed)


def build_learned_steering_spec(
    gen: MotionGenerator,
    costs: Sequence[GeneratedPythonCost],
    config: SteeringConfig,
    seed: int,
) -> SteeringSpec | None:
    """Compile the learned planner costs themselves, or ``None`` if unsteerable."""
    if config.mode == "off" or not costs:
        return None
    # pylint: disable-next=import-outside-toplevel
    from uncertain_feedback.motion_generators.mdm.mdm_api import MdmMotionGenerator

    if not isinstance(gen, MdmMotionGenerator):
        print("steering: unsupported backend, sampling unsteered")
        return None
    cost = gen.build_generated_steering_cost(
        costs, config.speed_guidance_weight / config.guidance_weight
    )
    if cost is None:
        print("steering: no steerable learned cost, sampling unsteered")
        return None
    return SteeringSpec(cost=cost, config=config, seed=seed)


def resample_indices(
    costs: np.ndarray, temperature: float, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Return systematic-resampling ancestor indices and the effective sample size.

    Costs are z-scored before the softmax so ``temperature`` transfers unchanged
    across costs of different scales. A degenerate (zero-spread) population
    resamples to the identity.
    """
    costs = np.asarray(costs, dtype=np.float64)
    n = costs.shape[0]
    std = costs.std()
    if std < 1e-12:
        return np.arange(n), float(n)
    logits = -(costs - costs.mean()) / (std * temperature)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    ess = 1.0 / float((weights**2).sum())
    positions = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(weights), positions), ess


def make_cond_fn(
    cost: Callable[[Tensor], Tensor], guidance_weight: float
) -> Callable[..., Tensor]:
    """Return a classifier-guidance ``cond_fn`` for ``p_sample_with_grad``.

    The signature is dictated by ``condition_mean_with_grad``, which calls
    ``cond_fn(x, t, p_mean_var, **model_kwargs)`` and scales the returned
    gradient by the posterior variance — hence the large ``guidance_weight``.
    """
    import torch  # pylint: disable=import-outside-toplevel

    def cond_fn(x: Tensor, t: Tensor, p_mean_var: dict, y=None, **kwargs) -> Tensor:
        del t, y, kwargs
        (grad,) = torch.autograd.grad(cost(p_mean_var["pred_xstart"]).sum(), x)
        return -guidance_weight * grad

    return cond_fn


def conflict_warning(event: SteeringEvent, n_samples: int) -> str | None:
    """Flag a prompt/cost conflict, or ``None`` if the population is healthy.

    Every chain violating the cost *and* the weights collapsing onto a handful
    of ancestors means the model puts almost no mass on motions that satisfy the
    cost, so reweighting the population cannot help.
    """
    if event.frac_violating >= 0.99 and event.ess < 0.05 * n_samples:
        return (
            f"steering conflict at step {event.step}: {event.frac_violating:.0%} of "
            f"samples violate the user cost and ESS collapsed to {event.ess:.1f}/"
            f"{n_samples}. The prompt and the cost likely conflict: the model puts "
            "almost no mass on satisfying motions, so mode 'resample' cannot fix "
            "it — only guidance ('cg', the default) or a reworded prompt can."
        )
    return None
