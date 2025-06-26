import geopandas as gpd

from .workflow import download_osm_layer


ROAD_TYPES = ["motorway", "trunk", "primary", "secondary"]
POWER_LINE_TYPES = ["line"]


def count_crossings(route: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, lines: gpd.GeoDataFrame) -> tuple[int, int]:
    route_lines = route.geometry.unary_union
    road_count = 0
    for geom in roads.geometry:
        if route_lines.intersects(geom):
            road_count += 1
    line_count = 0
    for geom in lines.geometry:
        if route_lines.intersects(geom):
            line_count += 1
    return road_count, line_count


def load_osm_roads(bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Download major road features from OpenStreetMap within *bbox*."""
    return download_osm_layer(bbox, "highway", ROAD_TYPES)


def load_osm_powerlines(bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Download power line features from OpenStreetMap within *bbox*."""
    return download_osm_layer(bbox, "power", POWER_LINE_TYPES)


