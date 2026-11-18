"""Create a one-task local dataset without downloading benchmark data."""
import argparse
from pathlib import Path

from datasets import Dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    source = Path(__file__).with_name("task.py").read_text()
    Dataset.from_list([
        {"level": 1, "name": "koda_gelu_smoke", "op_type": "Activation/GELU", "code": source}
    ]).save_to_disk(str(args.output))
    print(f"Prepared 1 task at {args.output.resolve()}")


if __name__ == "__main__":
    main()
