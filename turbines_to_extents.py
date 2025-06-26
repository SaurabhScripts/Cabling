#!/usr/bin/env python3
"""Compute an extents YAML from a turbines YAML file."""
import argparse
from pathlib import Path
import sys

# allow running without installation
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from workflow import turbines_to_extents_yaml


def main(turbines_yaml: str, out_dir: str) -> None:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(exist_ok=True)
    extents_path = out_dir_path / "extents.yaml"
    turbines_to_extents_yaml(Path(turbines_yaml), extents_path)
    print(f"Created {extents_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create extents.yaml from turbines.yaml")
    parser.add_argument("turbines_yaml", help="Path to turbines.yaml file")
    parser.add_argument("--out-dir", default="temp", help="Directory to store extents.yaml")
    args = parser.parse_args()
    main(args.turbines_yaml, args.out_dir)
