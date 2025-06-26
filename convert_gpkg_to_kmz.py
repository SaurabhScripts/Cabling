#!/usr/bin/env python3
"""Convert a GeoPackage to a KMZ with all layers rendered."""
import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from workflow import gpkg_to_kmz


def main(input_gpkg: str, output_kmz: str) -> None:
    """Command line entry point."""
    gpkg_to_kmz(Path(input_gpkg), Path(output_kmz))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a GPKG file to KMZ.")
    parser.add_argument("input_gpkg", help="Path to input GeoPackage file (.gpkg)")
    parser.add_argument("output_kmz", help="Path for output KMZ file (.kmz)")
    args = parser.parse_args()
    main(args.input_gpkg, args.output_kmz)
