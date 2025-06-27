"""Utility for creating a full ``site.yaml`` from raw input files.

This module re-implements the notebook-style workflow that converts the
original turbine spreadsheet, substation KMZ/KML and obstacle GeoPackage
into the YAML format used by the optimisation tools.
"""

from __future__ import annotations

import string
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import utm


def build_full_site_yaml(
    turbine_file: str | Path,
    substation_file: str | Path,
    obstacles_gpkg: str | Path,
    output_yaml: str | Path,
) -> Path:
    """Create a combined site YAML from turbine, substation and obstacle data."""

    t_file = Path(turbine_file)
    s_file = Path(substation_file)
    o_file = Path(obstacles_gpkg)
    out_path = Path(output_yaml)

    # ------------------------------------------------------------------
    # Turbines: load spreadsheet with UTM coordinates and convert to WGS84
    # ------------------------------------------------------------------
    if t_file.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(t_file, skiprows=1)
    else:
        df = pd.read_csv(t_file)

    df.columns = [
        "Index",
        "Loc_No",
        "Phase",
        "Status",
        "MEDA_Status",
        "Feeder",
        "Phase_Duplicate",
        "Zone",
        "Easting",
        "Northing",
    ]
    df = df[["Loc_No", "Easting", "Northing", "Zone"]]
    utm_zone_number = int(str(df["Zone"].iloc[0])[:2])
    utm_zone_letter = str(df["Zone"].iloc[0])[2]

    def _utm_to_latlon(row: pd.Series) -> pd.Series:
        lat, lon = utm.to_latlon(
            row["Easting"], row["Northing"], utm_zone_number, utm_zone_letter
        )
        return pd.Series({"Latitude": lat, "Longitude": lon})

    df[["Latitude", "Longitude"]] = df.apply(_utm_to_latlon, axis=1)

    # ------------------------------------------------------------------
    # Helper functions for formatting coordinates
    # ------------------------------------------------------------------
    def index_to_code(idx: int) -> str:
        letters: list[str] = []
        for _ in range(2):
            letters.append(chr(ord("A") + (idx % 26)))
            idx //= 26
        return "".join(reversed(letters))

    def dec_to_dms(val: float, is_lat: bool) -> str:
        deg = int(abs(val))
        minutes = (abs(val) - deg) * 60
        hemi = "N" if is_lat else "E"
        if val < 0:
            hemi = "S" if is_lat else "W"
        return f"{deg}\u00b0{minutes:.3f}'{hemi}"

    # Turbine block
    turbine_lines = [
        f"  {index_to_code(i)} {dec_to_dms(r.Latitude, True)} {dec_to_dms(r.Longitude, False)}"
        for i, r in df.iterrows()
    ]

    # ------------------------------------------------------------------
    # Extents block based on turbine coordinates
    # ------------------------------------------------------------------
    min_lat, max_lat = df["Latitude"].min(), df["Latitude"].max()
    min_lon, max_lon = df["Longitude"].min(), df["Longitude"].max()
    extent_lines = [
        f"  A {dec_to_dms(max_lat, True)} {dec_to_dms(min_lon, False)}",
        f"  B {dec_to_dms(max_lat, True)} {dec_to_dms(max_lon, False)}",
        f"  C {dec_to_dms(min_lat, True)} {dec_to_dms(max_lon, False)}",
        f"  D {dec_to_dms(min_lat, True)} {dec_to_dms(min_lon, False)}",
    ]

    # ------------------------------------------------------------------
    # Substations from KMZ/KML
    # ------------------------------------------------------------------
    def _load_kml_root(p: Path) -> ET.Element:
        if p.suffix.lower() == ".kmz":
            with zipfile.ZipFile(p, "r") as kmz:
                name = next(n for n in kmz.namelist() if n.lower().endswith(".kml"))
                with kmz.open(name) as kmlf:
                    return ET.parse(kmlf).getroot()
        return ET.parse(p).getroot()

    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    root = _load_kml_root(s_file)
    sub_lines = []
    for pm in root.findall(".//kml:Placemark", ns):
        name = pm.findtext("kml:name", default="", namespaces=ns)
        coord = pm.findtext(".//kml:coordinates", default="", namespaces=ns)
        if coord:
            lon, lat = map(float, coord.split(",")[:2])
            sub_lines.append(
                f"  [{name}] {dec_to_dms(lat, True)} {dec_to_dms(lon, False)}"
            )

    # ------------------------------------------------------------------
    # Obstacles from GeoPackage
    # ------------------------------------------------------------------
    gdf = gpd.read_file(o_file)
    obstacle_items: list[str] = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        if geom.geom_type == "Polygon":
            coords = list(geom.exterior.coords)
        elif geom.geom_type in {"LineString", "LinearRing"}:
            coords = list(geom.coords)
        else:
            continue

        lines = []
        for i, (lon, lat) in enumerate(coords):
            label = string.ascii_uppercase[i] if i < 26 else f"P{i}"
            lat_deg = int(lat)
            lat_min = (lat - lat_deg) * 60
            lon_deg = int(lon)
            lon_min = (lon - lon_deg) * 60
            lat_dir = "N" if lat >= 0 else "S"
            lon_dir = "E" if lon >= 0 else "W"
            lines.append(
                f"{label} {lat_deg}\u00b0{lat_min:.3f}''{lat_dir} {lon_deg}\u00b0{lon_min:.3f}''{lon_dir}"
            )
        if lines:
            obstacle_items.append(lines[0] + "\n  " + "\n  ".join(lines[1:]))

    # ------------------------------------------------------------------
    # Write final YAML
    # ------------------------------------------------------------------
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("OPERATOR: Serentica\n")
        f.write("TURBINE:\n  make: Vestas\n  model: V90/3000\n  power_MW: 3\n\n")
        f.write("LANDSCAPE_ANGLE: 0\n\n")

        f.write("EXTENTS: |-\n")
        for ln in extent_lines:
            f.write(f"{ln}\n")

        f.write("\nSUBSTATIONS: |-\n")
        for ln in sub_lines:
            f.write(f"{ln}\n")

        f.write("\nTURBINES: |-\n")
        for ln in turbine_lines:
            f.write(f"{ln}\n")

        f.write("\nOBSTACLES:\n")
        for item in obstacle_items:
            f.write(f"- '{item}'\n")

    return out_path


def post_process_site_yaml(site_yaml: str | Path, obstacles_yaml: str | Path) -> Path:
    """Post-process the raw ``site.yaml`` created by :func:`build_full_site_yaml`.

    This helper mirrors the manual notebook workflow used in the original
    project.  It appends the formatted obstacle blocks to ``site_yaml`` and then
    performs several cleaning steps:

    1. Remove self-intersecting or overlapping obstacle polygons.
    2. Drop duplicated last points and blank lines.
    3. Filter obstacles that fall outside the ``EXTENTS`` polygon.
    4. Validate each obstacle via :func:`workflow.clean_site_yaml`.

    Parameters
    ----------
    site_yaml:
        Path to the YAML file produced by :func:`build_full_site_yaml`.
    obstacles_yaml:
        YAML file containing an ``OBSTACLES`` list generated from a GeoPackage.

    Returns
    -------
    Path
        The path to the final cleaned YAML file.
    """

    from shapely.geometry import Polygon
    import re
    import yaml
    from .workflow import clean_site_yaml

    s_path = Path(site_yaml)
    o_path = Path(obstacles_yaml)

    # ------------------------------------------------------------------
    # Step 1: append the obstacle list into the site YAML
    # ------------------------------------------------------------------
    site_lines = s_path.read_text(encoding="utf-8").splitlines()
    clean_lines: list[str] = []
    for line in site_lines:
        if line.strip() == "OBSTACLES:":
            break
        clean_lines.append(line)

    obs_lines = o_path.read_text(encoding="utf-8").splitlines()
    combined = clean_lines + ["", "OBSTACLES:"] + obs_lines
    s_path.write_text("\n".join(combined) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    dms_pattern = re.compile(r"(\d+)°([\d\.]+)''?([NS])\s+(\d+)°([\d\.]+)''?([EW])")

    def parse_dms(dms: str) -> tuple[float, float]:
        m = dms_pattern.match(dms.strip())
        if not m:
            raise ValueError(f"Invalid DMS: {dms!r}")
        lat_d, lat_m, ns, lon_d, lon_m, ew = m.groups()
        lat = int(lat_d) + float(lat_m) / 60
        lon = int(lon_d) + float(lon_m) / 60
        if ns == "S":
            lat = -lat
        if ew == "W":
            lon = -lon
        return lat, lon

    def clean_block(block: str) -> str:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            return ""
        first = lines[0].split(" ", 1)[1]
        last = lines[-1].split(" ", 1)[1]
        if first == last:
            lines.pop()
        return lines[0] + "\n  " + "\n  ".join(lines[1:])

    def dms_to_decimal(coord: str) -> float:
        m = re.match(r"(\d+)°(\d+\.\d+)'([NSEW])", coord)
        if not m:
            raise ValueError(f"Cannot parse DMS string: {coord}")
        deg, minutes, hemi = m.groups()
        val = float(deg) + float(minutes) / 60
        if hemi in ("S", "W"):
            val = -val
        return val

    def parse_multiline(field: str) -> list[tuple[float, float]]:
        coords: list[tuple[float, float]] = []
        for line in field.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                lat = dms_to_decimal(parts[1])
                lon = dms_to_decimal(parts[2])
                coords.append((lon, lat))
        return coords

    # ------------------------------------------------------------------
    # Step 2: remove invalid or intersecting polygons
    # ------------------------------------------------------------------
    data = yaml.safe_load(s_path.read_text(encoding="utf-8"))
    obs_blocks: list[str] = data.get("OBSTACLES", [])
    polygons: list[Polygon] = []
    for blk in obs_blocks:
        coords = [parse_dms(ln.split(" ", 1)[1]) for ln in blk.splitlines() if ln]
        polygons.append(Polygon([(lon, lat) for lat, lon in coords]))

    bad: set[int] = set()
    for i, poly in enumerate(polygons):
        if not poly.is_valid:
            bad.add(i)
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
                bad.add(i)
                bad.add(j)

    cleaned_blocks = [blk for idx, blk in enumerate(obs_blocks) if idx not in bad]
    data["OBSTACLES"] = cleaned_blocks
    tmp_clear = s_path.with_name("final_Serentica_clear.yaml")
    with open(tmp_clear, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------------
    # Step 3: tidy up the obstacle blocks
    # ------------------------------------------------------------------
    data["OBSTACLES"] = [clean_block(b) for b in cleaned_blocks]
    tmp_final = s_path.with_name("final_Serentica_final.yaml")
    with open(tmp_final, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------------
    # Step 4: remove obstacles outside the EXTENTS polygon
    # ------------------------------------------------------------------
    extent_coords = parse_multiline(data.get("EXTENTS", ""))
    extent_poly = Polygon(extent_coords)
    filtered: list[str] = []
    for blk in data["OBSTACLES"]:
        poly = Polygon(parse_multiline(blk))
        if poly.within(extent_poly):
            filtered.append(blk)
    data["OBSTACLES"] = filtered
    tmp_extent = s_path.with_name("final_Serentica_final_extent.yaml")
    with open(tmp_extent, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    # ------------------------------------------------------------------
    # Step 5: final validation using workflow.clean_site_yaml
    # ------------------------------------------------------------------
    final_path = clean_site_yaml(tmp_extent)
    return final_path


__all__ = ["build_full_site_yaml", "post_process_site_yaml"]

