"""Runs metrics and qualitative sampling on saved checkpoints."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    _ = parser.parse_args()

    # TODO: load models & metrics, log to W&B
    pass


if __name__ == "__main__":
    main()
