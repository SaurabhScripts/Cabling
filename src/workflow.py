from pathlib import Path
from typing import Dict, List, Optional
from zipfile import ZipFile
import re

import geopandas as gpd
import pandas as pd
from shapely.geometry import box, LineString, Point
from shapely.ops import unary_union
import requests
import folium
import yaml

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _overpass_query(bbox: str, key: str, values: List[str]) -> str:
    value_filter = "|".join(values)
    query = (
        f"[out:json][timeout:25];(way['{key}'~'{value_filter}']({bbox});"
        f"relation['{key}'~'{value_filter}']({bbox}););out geom;>;out skel qt;"
    )
    return query


def download_osm_layer(bbox: tuple[float, float, float, float], key: str, values: List[str]) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bbox
    bbox_str = f"{miny},{minx},{maxy},{maxx}"
    query = _overpass_query(bbox_str, key, values)
    response = requests.post(OVERPASS_URL, data={"data": query}, timeout=60)
    response.raise_for_status()
    data = response.json()

    features = []
    for elem in data.get("elements", []):
        if "geometry" not in elem:
            continue
        if elem["type"] == "way":
            coords = [(pt["lon"], pt["lat"]) for pt in elem["geometry"]]
            geom = LineString(coords)
        elif elem["type"] == "node":
            geom = Point(elem["lon"], elem["lat"])
        else:
            continue
        props = elem.get("tags", {})
        props["osmid"] = elem["id"]
        features.append({"type": "Feature", "geometry": geom, "properties": props})

    gdf = gpd.GeoDataFrame.from_features(features, crs=4326)
    return gdf


def create_extent(points: gpd.GeoDataFrame, buffer: float = 0.01) -> tuple[float, float, float, float]:
    bounds = points.total_bounds
    minx, miny, maxx, maxy = bounds
    minx -= buffer
    miny -= buffer
    maxx += buffer
    maxy += buffer
    return minx, miny, maxx, maxy


def merge_layers(layers: List[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Merge multiple GeoDataFrames into a single 4326 layer."""
    combined = gpd.GeoDataFrame(pd.concat(layers, ignore_index=True))
    combined.set_crs(epsg=4326, inplace=True)
    return combined


def buffer_and_union(gdf: gpd.GeoDataFrame, distance: float) -> gpd.GeoSeries:
    """Return a single geometry unioned from the layer buffered by *distance*."""
    buffered = gdf.to_crs(3857).buffer(distance).to_crs(4326)
    unioned = unary_union(buffered)
    return gpd.GeoSeries([unioned], crs=4326)


def export_kmz(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Write *gdf* to a KMZ archive at *path*."""
    kml_path = path.with_suffix('.kml')
    gdf.to_file(kml_path, driver='KML')
    with ZipFile(path, 'w') as zf:
        zf.write(kml_path, arcname=kml_path.name)
    kml_path.unlink()

def export_folium_map(layers: Dict[str, gpd.GeoDataFrame], path: Path) -> None:
    """Export a Folium map with multiple layers and layer control."""
    fmap = folium.Map()
    for name, layer in layers.items():
        if layer.empty:
            continue
        folium.GeoJson(layer, name=name).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    fmap.save(str(path))


def csv_to_yaml(csv_path: Path, yaml_path: Path) -> None:
    """Convert a CSV file to a YAML representation."""
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(records, f)


def load_csv_points(path: Path) -> gpd.GeoDataFrame:
    """Load a CSV with latitude/longitude columns into a GeoDataFrame."""
    df = pd.read_csv(path)
    lat_col = next((c for c in df.columns if c.lower() in {"lat", "latitude", "y"}), None)
    lon_col = next((c for c in df.columns if c.lower() in {"lon", "long", "longitude", "x"}), None)
    if not lat_col or not lon_col:
        raise ValueError("CSV must contain latitude/longitude columns")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=4326)
    return gdf


def generate_simple_route(turbines: gpd.GeoDataFrame, substation: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create simple lines from the substation to each turbine."""
    if turbines.empty or substation.empty:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    start = substation.geometry.iloc[0]
    lines = [LineString([start, pt]) for pt in turbines.geometry]
    return gpd.GeoDataFrame(geometry=lines, crs=4326)


dms_pattern = re.compile(r"(\d+)°([\d\.]+)'([NS])\s+(\d+)°([\d\.]+)'([EW])")


def _dms_to_decimal(dms: str) -> tuple[float, float]:
    """Convert a coordinate string like '12°34.5'N 45°6.7'E' to decimals."""
    m = dms_pattern.match(dms.strip())
    if not m:
        raise ValueError(f"Invalid DMS format: {dms!r}")
    dlat, mlat, ns, dlon, mlon, ew = m.groups()
    lat = int(dlat) + float(mlat) / 60
    lon = int(dlon) + float(mlon) / 60
    if ns == "S":
        lat = -lat
    if ew == "W":
        lon = -lon
    return lat, lon


def load_yaml_points(path: Path) -> gpd.GeoDataFrame:
    """Load a turbines YAML file into a GeoDataFrame."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    turbines_block = data.get("TURBINES")

    if isinstance(turbines_block, list):
        lines = turbines_block
    elif isinstance(turbines_block, str):
        lines = [ln.strip() for ln in turbines_block.splitlines() if ln.strip()]
    else:
        raise RuntimeError("TURBINES field is neither list nor block string")

    coords: list[tuple[float, float]] = []
    for ln in lines:
        parts = ln.split(maxsplit=1)
        if len(parts) != 2:
            continue
        _, dms_str = parts
        coords.append(_dms_to_decimal(dms_str))

    if not coords:
        return gpd.GeoDataFrame(geometry=[], crs=4326)

    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lons, lats),
        crs=4326,
    )
    return gdf



