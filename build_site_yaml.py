#!/usr/bin/env python3
"""Command line helper to create a site YAML from turbine, substation and obstacle files."""
import argparse
from pathlib import Path
import sys

# allow running without installation
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from workflow import build_site_yaml


def main(turbine_file: str, substation_file: str, obstacles_gpkg: str, output_yaml: str) -> None:
    build_site_yaml(Path(turbine_file), Path(substation_file), Path(obstacles_gpkg), Path(output_yaml))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a site YAML from turbine Excel/CSV, substation KML/KMZ and obstacle GeoPackage")
    parser.add_argument("turbine_file", help="Turbine Excel or CSV with UTM coordinates")
    parser.add_argument("substation_file", help="Substation KML or KMZ")
    parser.add_argument("obstacles_gpkg", help="Obstacle GeoPackage")
    parser.add_argument("output_yaml", help="Output YAML path")
    args = parser.parse_args()
    main(args.turbine_file, args.substation_file, args.obstacles_gpkg, args.output_yaml)

