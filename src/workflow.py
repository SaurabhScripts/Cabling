from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import box, LineString, Point
from shapely.ops import unary_union
import requests
import folium

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
    combined = gpd.GeoDataFrame(pd.concat(layers, ignore_index=True))
    combined.set_crs(epsg=4326, inplace=True)
    return combined


def buffer_and_union(gdf: gpd.GeoDataFrame, distance: float) -> gpd.GeoSeries:
    buffered = gdf.to_crs(3857).buffer(distance).to_crs(4326)
    unioned = unary_union(buffered)
    return gpd.GeoSeries([unioned], crs=4326)


def export_kmz(gdf: gpd.GeoDataFrame, path: Path) -> None:
    gdf.to_file(path, driver='KML')


def export_folium_map(gdf: gpd.GeoDataFrame, path: Path) -> None:
    fmap = folium.Map()
    folium.GeoJson(gdf).add_to(fmap)
    fmap.save(str(path))


