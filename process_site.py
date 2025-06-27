#!/usr/bin/env python3
"""Command-line pipeline to build and clean a site YAML and generate a route.

This script mirrors the logic used by the FastAPI ``/process`` endpoint. It
combines turbine, substation and obstacle files into a single YAML definition,
optionally cleans invalid obstacles and runs the optimisation workflow to
produce a KMZ route.
"""
import argparse
from pathlib import Path
import sys

# Allow running without installing the package
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from workflow import (
    build_site_yaml,
    clean_site_yaml,
    generate_optimized_route,
)


def main(
    turbine_file: Path,
    substation_file: Path,
    obstacles_gpkg: Path,
    out_dir: Path,
    *,
    clean: bool = True,
    run_route: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    site_yaml = out_dir / "site.yaml"

    build_site_yaml(turbine_file, substation_file, obstacles_gpkg, site_yaml)
    print(f"Created site YAML: {site_yaml}")

    processed_yaml = site_yaml
    if clean:
        processed_yaml = clean_site_yaml(site_yaml)
        print(f"Cleaned YAML written to: {processed_yaml}")

    if run_route:
        route_kmz = out_dir / "route.kmz"
        generate_optimized_route(processed_yaml, route_kmz)
        print(f"Route KMZ written to: {route_kmz}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a site YAML and optionally run the optimisation",
    )
    parser.add_argument("turbine_file", help="Turbine Excel/CSV with UTM coordinates")
    parser.add_argument("substation_file", help="Substation KML or KMZ")
    parser.add_argument("obstacles_gpkg", help="Obstacle GeoPackage")
    parser.add_argument(
        "out_dir",
        help="Directory to store the generated files",
    )
    parser.add_argument(
        "--no-clean",
        action="store_false",
        dest="clean",
        help="Skip obstacle cleaning",
    )
    parser.add_argument(
        "--no-route",
        action="store_false",
        dest="run_route",
        help="Skip running the optimisation",
    )

    args = parser.parse_args()

    main(
        Path(args.turbine_file),
        Path(args.substation_file),
        Path(args.obstacles_gpkg),
        Path(args.out_dir),
        clean=args.clean,
        run_route=args.run_route,
    )
