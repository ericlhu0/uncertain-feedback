"""Zero the real Kinova Gen3's joint positions (upright reference pose).

Connects to a running emprise-gen3-controller ZMQ server and moves every
joint to zero, waiting until the arm has settled::

    uv run python src/uncertain_feedback/envs/zero_kinova.py
    uv run python src/uncertain_feedback/envs/zero_kinova.py --host 192.168.1.10 --yes
"""

from __future__ import annotations

import argparse

from uncertain_feedback.envs.real_mirror import RealArmMirror


def main() -> None:
    """Send the Gen3 to its zero configuration over the mirror link."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="ZMQ server host")
    parser.add_argument("--state-port", type=int, default=5555)
    parser.add_argument("--cmd-port", type=int, default=5556)
    parser.add_argument(
        "--yes", action="store_true", help="skip the move confirmation prompt"
    )
    args = parser.parse_args()

    mirror = RealArmMirror.connect(
        args.host,
        state_port=args.state_port,
        cmd_port=args.cmd_port,
        confirm_start=not args.yes,
    )
    mirror.zero()
    print("Kinova joints zeroed.")


if __name__ == "__main__":
    main()
