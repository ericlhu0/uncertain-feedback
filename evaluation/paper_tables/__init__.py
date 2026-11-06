"""The scripts that turn run outputs into the paper's tables.

They were written for the 2026-10-20 audit of the paper's experiments and lived
in that audit's output folder, which git ignores; they are here so the numbers
in the paper can be regenerated from the runs, and so the corrections the audit
made to them (pairing episodes by their folder, not by persona, which repeats;
one index per probe menu) stay made.

- ``episode_table``: success within k user turns per arm, paired tests
  against the first arm (the learning table and the grounder comparison).
- ``steering_table``: candidate-menu violation per arm of a bound-transfer
  run, paired over probe menus (the steering table).
- ``caption_oracle_agreement``: do a clip set's utterances name the sign the
  oracle's wrist moves along (a benchmark sanity check).
- ``steering_targets``: for every learned-steering run, which bounds the
  sampler was actually steered with, recovered by re-executing the steering
  path from the saved learner (the run log now prints this as well).

The grounding comparison over seeds and the prior under both T2M evaluators
need the ``direction`` metric and ``evaluation/metrics/prior.py`` of the
prior-preservation branch and travel with it.
"""
