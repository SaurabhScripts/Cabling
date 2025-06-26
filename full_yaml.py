#!/usr/bin/env python3
"""Create a complete site YAML from raw turbine, substation and obstacle data."""
import argparse
from pathlib import Path
import sys

# Allow running without installing the package
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from workflow import build_site_yaml


def main(turbine_file: Path, substation_file: Path, obstacles_gpkg: Path, output_yaml: Path) -> None:
    """Generate ``output_yaml`` by combining all input files."""
    build_site_yaml(turbine_file, substation_file, obstacles_gpkg, output_yaml)
    print(f"Created site YAML at {output_yaml}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a full site YAML from turbine, substation and obstacle files"
    )
    parser.add_argument("turbine_file", help="Turbine Excel or CSV with UTM coordinates")
    parser.add_argument("substation_file", help="Substation KML or KMZ")
    parser.add_argument("obstacles_gpkg", help="Obstacle GeoPackage")
    parser.add_argument("output_yaml", help="Destination for the combined YAML")
    args = parser.parse_args()

    main(Path(args.turbine_file), Path(args.substation_file), Path(args.obstacles_gpkg), Path(args.output_yaml))
