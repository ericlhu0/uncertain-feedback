"""Generate and render informative fixed or sampled-preference correction scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.benchmarks.informative_scenarios import (
    ScenarioCriteria,
    generate_scenarios,
    render_scenarios,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-dir", type=Path)
    parser.add_argument(
        "--mpc-config", type=Path, default=Path("evaluation/conf/mpc_demo_low1.yaml")
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--personas",
        nargs="+",
        default=[
            "adhesive_capsulitis",
            "stroke_flexor_synergy",
            "triceps_long_head_contracture",
        ],
    )
    parser.add_argument(
        "--sampled-bounds",
        type=int,
        default=0,
        help="Number of synthetic cases, alternating constant/coupled bounds; replaces --personas",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-attempts", type=int, default=300)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if not args.render_only:
        if args.geometry_dir is None:
            parser.error("--geometry-dir is required for generation")
        generate_scenarios(
            args.geometry_dir,
            args.mpc_config,
            args.out_dir,
            args.personas,
            args.seed,
            args.max_attempts,
            ScenarioCriteria(),
            sampled_bounds=args.sampled_bounds,
        )
    if not args.no_render:
        render_scenarios(args.out_dir)


if __name__ == "__main__":
    main()
