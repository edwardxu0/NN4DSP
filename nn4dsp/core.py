import argparse

from .train import train


def _parse_args():
    parser = argparse.ArgumentParser(
        prog="NN4DSP",
        description="Using neural networks to synthesize DSP functions",
        epilog="Text at the bottom of help",
    )

    parser.add_argument(
        "task",
        type=str,
        help="Task to perform",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.task == "train":
        train()
    else:
        assert False
