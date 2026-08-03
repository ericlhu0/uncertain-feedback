# Method-level evaluation

This directory is the home for **evaluating the whole pipeline end to end** — the
scripts and experiments a researcher runs to measure how well the method works, as
opposed to anything the method itself executes at runtime.

It is currently empty. The previous runners under
`src/uncertain_feedback/experiments/` (persona/backend/cluster/transfer/multi-round
comparisons and the automated episode loop) were deleted in the stage-package
refactor and will be rewritten here on top of the per-stage façades:

| Stage | Façade |
| --- | --- |
| text → candidate motions | `uncertain_feedback.motion_generators` |
| cluster + select | `uncertain_feedback.uncertainty` |
| correction → cost code | `uncertain_feedback.cost_generation` |
| score a generated cost | `uncertain_feedback.evaluation_mechanism` |
| execution | `uncertain_feedback.planners.mpc.rollout` |
| simulated care recipients | `uncertain_feedback.simulated_users` |

Living outside `src/` is deliberate: this is researcher tooling, not part of the
shipped method. The in-`src` package `evaluation_mechanism/` is a different thing —
it is how the method scores *its own* generated cost functions, and the codex
sandbox imports it at runtime.
