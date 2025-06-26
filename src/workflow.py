from pathlib import Path
from typing import Dict, List, Optional
from zipfile import ZipFile
import re

import geopandas as gpd
import fiona
import pandas as pd
from shapely.geometry import box, LineString, Point
from shapely.ops import unary_union
import requests
import folium
import yaml
import string
import simplekml

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def _overpass_query(bbox: str, key: str, values: List[str]) -> str:
    value_filter = "|".join(values)
    query = (
        f"[out:json][timeout:25];(way['{key}'~'{value_filter}']({bbox});"
        f"relation['{key}'~'{value_filter}']({bbox}););out geom;>;out skel qt;"
    )
    return query


def download_osm_layer(
    bbox: tuple[float, float, float, float], key: str, values: List[str]
) -> gpd.GeoDataFrame:
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


def turbines_to_extents_yaml(turbines_yaml: Path, extents_yaml: Path) -> None:
    """Create an ``extents.yaml`` file covering the turbine coordinates."""

    data = yaml.safe_load(turbines_yaml.read_text(encoding="utf-8"))
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
        raise RuntimeError("No valid turbine coordinates found")

    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    max_lat, min_lat = max(lats), min(lats)
    max_lon, min_lon = max(lons), min(lons)

    def dec_to_dms(val: float, is_lat: bool) -> str:
        deg = int(abs(val))
        minutes = (abs(val) - deg) * 60
        hemi = "N" if is_lat else "E"
        if val < 0:
            hemi = "S" if is_lat else "W"
        return f"{deg}\u00b0{minutes:.3f}'{hemi}"

    dms_A = dec_to_dms(max_lat, True), dec_to_dms(min_lon, False)
    dms_B = dec_to_dms(max_lat, True), dec_to_dms(max_lon, False)
    dms_C = dec_to_dms(min_lat, True), dec_to_dms(max_lon, False)
    dms_D = dec_to_dms(min_lat, True), dec_to_dms(min_lon, False)

    with open(extents_yaml, "w", encoding="utf-8") as f:
        f.write("EXTENTS: |-\n")
        f.write(f"  A {dms_A[0]} {dms_A[1]}\n")
        f.write(f"  B {dms_B[0]} {dms_B[1]}\n")
        f.write(f"  C {dms_C[0]} {dms_C[1]}\n")
        f.write(f"  D {dms_D[0]} {dms_D[1]}\n")

def create_extent(
    points: gpd.GeoDataFrame, buffer: float = 0.01
) -> tuple[float, float, float, float]:
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
    kml_path = path.with_suffix(".kml")
    gdf.to_file(kml_path, driver="KML")
    with ZipFile(path, "w") as zf:
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
    lat_col = next(
        (c for c in df.columns if c.lower() in {"lat", "latitude", "y"}), None
    )
    lon_col = next(
        (c for c in df.columns if c.lower() in {"lon", "long", "longitude", "x"}), None
    )
    if not lat_col or not lon_col:
        raise ValueError("CSV must contain latitude/longitude columns")
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=4326
    )
    return gdf


def generate_simple_route(
    turbines: gpd.GeoDataFrame, substation: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
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


def gpkg_to_obstacles_yaml(gpkg_path: Path, yaml_path: Path) -> None:
    """Convert obstacle polygons/lines from a GeoPackage to YAML format."""
    gdf = gpd.read_file(gpkg_path)

    items: list[str] = []
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
                f"{label} {lat_deg}\u00b0{lat_min:.3f}''{lat_dir} "
                f"{lon_deg}\u00b0{lon_min:.3f}''{lon_dir}"
            )

        if not lines:
            continue
        item_str = lines[0] + "\n  " + "\n  ".join(lines[1:])
        items.append(item_str)

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("OBSTACLES:\n")
        for item in items:
            f.write(f"- '{item}'\n")


def gpkg_to_kmz(gpkg_path: Path, kmz_path: Path) -> None:
    """Convert a GeoPackage with one or more layers to a KMZ archive."""

    if not gpkg_path.is_file():
        raise FileNotFoundError(f"Input GPKG file not found: {gpkg_path}")

    layers = fiona.listlayers(gpkg_path)
    if not layers:
        raise ValueError(f"No layers found in GeoPackage: {gpkg_path}")

    kml = simplekml.Kml()

    for layer in layers:
        gdf = gpd.read_file(gpkg_path, layer=layer)

        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            name = row.get("name") or f"{layer}_{idx}"

            if geom.geom_type == "Point":
                placemark = kml.newpoint(name=name, coords=[(geom.x, geom.y)])
            elif geom.geom_type in ("LineString", "MultiLineString"):
                coords = (
                    list(geom.coords)
                    if geom.geom_type == "LineString"
                    else [pt for part in geom for pt in part.coords]
                )
                placemark = kml.newlinestring(name=name, coords=coords)
            elif geom.geom_type in ("Polygon", "MultiPolygon"):
                polygons = [geom] if geom.geom_type == "Polygon" else geom
                for poly in polygons:
                    exterior = list(poly.exterior.coords)
                    placemark = kml.newpolygon(name=name, outerboundaryis=exterior)
            else:
                continue

            props = {k: v for k, v in row.items() if k not in {"geometry", "name"}}
            if props:
                placemark.description = "\n".join(f"{k}: {v}" for k, v in props.items())

    if str(kmz_path).lower().endswith(".kmz"):
        kml.savekmz(str(kmz_path))
    else:
        kml.save(str(kmz_path))


def export_H_to_kml(
    H,
    filepath: str | Path = "routes.kml",
    include_nodes: bool = False,
    project_to_wgs84: tuple[str, str] | None = None,
) -> None:
    """Export a routed graph *H* to KML.

    The function is a lightweight wrapper around :mod:`simplekml` that draws
    each edge of ``H`` and colours it by the ``cable`` attribute.  Optionally the
    coordinates can be reprojected on the fly using ``pyproj`` if ``H`` is in a
    projected CRS (e.g. UTM).

    Parameters
    ----------
    H:
        A networkx graph produced by ``interarray`` containing routing
        information.
    filepath:
        Destination ``.kml`` or ``.kmz`` file.
    include_nodes:
        If ``True`` all graph nodes are exported as placemarks.
    project_to_wgs84:
        Optional ``(src_crs, dst_crs)`` tuple for reprojection using
        ``pyproj.Transformer``.  When ``None`` the coordinates are written as-is.
    """

    import simplekml
    from pathlib import Path

    pos: dict[int, tuple[float, float]] = {}
    R = H.graph.get("R", 0)
    VertexC = H.graph.get("VertexC")
    fnT = H.graph.get("fnT")
    if VertexC is None:
        raise ValueError("Graph does not contain coordinate matrix 'VertexC'.")

    n_base = VertexC.shape[0] - R
    for i in range(n_base):
        pos[i] = tuple(VertexC[i])
    for idx, coord in enumerate(VertexC[-R:], start=-R):
        pos[idx] = tuple(coord)
    if fnT is not None:
        T = H.graph.get("T", 0)
        B = H.graph.get("B", 0)
        C = H.graph.get("C", 0)
        D = H.graph.get("D", 0)
        for n in range(T + B, T + B + C + D):
            pos[n] = tuple(VertexC[fnT[n]])

    if project_to_wgs84:
        from pyproj import Transformer

        transformer = Transformer.from_crs(*project_to_wgs84, always_xy=True)
        for n, (x, y) in pos.items():
            lon, lat = transformer.transform(x, y)
            pos[n] = (lon, lat)

    color_map = {
        0: simplekml.Color.blue,
        1: simplekml.Color.green,
        2: simplekml.Color.orange,
        3: simplekml.Color.red,
    }

    kml = simplekml.Kml()
    skipped: list[tuple[int, int]] = []

    for u, v, data in H.edges(data=True):
        if u not in pos or v not in pos:
            skipped.append((u, v))
            continue

        lon1, lat1 = pos[u]
        lon2, lat2 = pos[v]
        idx = data.get("cable", 0)
        line = kml.newlinestring(
            name=f"cable_{idx}", coords=[(lon1, lat1), (lon2, lat2)]
        )
        line.style.linestyle.color = color_map.get(idx, simplekml.Color.gray)
        line.style.linestyle.width = 3

    if include_nodes:
        for n, attrs in H.nodes(data=True):
            if n not in pos:
                continue
            lon, lat = pos[n]
            p = kml.newpoint(name=str(n), coords=[(lon, lat)])
            if n < 0:
                p.style.iconstyle.color = simplekml.Color.red
            else:
                p.style.iconstyle.scale = 0.5

    dst = Path(filepath)
    if dst.suffix.lower() == ".kmz":
        kml.savekmz(str(dst))
    else:
        kml.save(str(dst))


def generate_optimized_route(site_yaml: Path, output_kmz: Path) -> None:
    """Run the interarray optimisation workflow and export the route to KMZ.

    This is a convenience wrapper around a subset of the example code provided
    by the ``interarray`` package.  It requires several optional dependencies
    (``interarray`` and ``pyomo``).  If they are missing the function will raise
    an ``ImportError``.
    """

    import pyomo.environ as pyo
    from interarray.importer import L_from_yaml
    from interarray.mesh import make_planar_embedding
    from interarray.interarraylib import G_from_S
    from interarray.pathfinding import PathFinder
    from interarray.EW_presolver import EW_presolver
    import interarray.MILP.pyomo as omo
    from interarray.interface import assign_cables

    cable_costs = [75, 80, 90, 100]
    turbines_per_cable = [5, 11, 20, 43]
    cables = [
        (None, capacity, cost)
        for capacity, cost in zip(turbines_per_cable, cable_costs)
    ]
    capacity = max(turbines_per_cable)

    solver = pyo.SolverFactory("cbc")

    L = L_from_yaml(site_yaml, handle="Site")
    P, A = make_planar_embedding(L)

    S_pre = EW_presolver(A, capacity)
    model = omo.make_min_length_model(
        A, capacity, gateXings_constraint=False, gates_limit=False, branching=True
    )
    omo.warmup_model(model, S_pre)

    result = solver.solve(model, warmstart=model.warmed_by, tee=True)
    S = omo.S_from_solution(model, solver, result)
    G = G_from_S(S, A)
    H = PathFinder(G, planar=P, A=A).create_detours()
    assign_cables(H, cables)

    export_H_to_kml(
        H, filepath=output_kmz, project_to_wgs84=("EPSG:32643", "EPSG:4326")
    )


def build_site_yaml(
    turbine_file: Path,
    substation_file: Path,
    obstacles_gpkg: Path,
    output_yaml: Path,
) -> None:
    """Create a combined site YAML from turbine, substation and obstacle data.

    Parameters
    ----------
    turbine_file:
        Excel or CSV file containing turbine coordinates in UTM as produced by
        the ``read_turbine_excel`` helper.
    substation_file:
        ``.kml`` or ``.kmz`` file with substation placemarks.
    obstacles_gpkg:
        GeoPackage with obstacle polygons/lines.
    output_yaml:
        Destination for the resulting YAML file.
    """

    from .turbine_excel import read_turbine_excel
    import xml.etree.ElementTree as ET
    import zipfile
    import tempfile

    df = read_turbine_excel(turbine_file)

    def index_to_code(idx: int) -> str:
        letters = []
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
        return f"{deg}\u00B0{minutes:.3f}'{hemi}"

    # --- Turbine lines ---
    turbine_lines = [
        f"  {index_to_code(i)} {dec_to_dms(r.Latitude, True)} {dec_to_dms(r.Longitude, False)}"
        for i, r in df.iterrows()
    ]

    # --- Extent lines (A-D clockwise) ---
    min_lat, max_lat = df["Latitude"].min(), df["Latitude"].max()
    min_lon, max_lon = df["Longitude"].min(), df["Longitude"].max()
    extent_lines = [
        f"  A {dec_to_dms(max_lat, True)} {dec_to_dms(min_lon, False)}",
        f"  B {dec_to_dms(max_lat, True)} {dec_to_dms(max_lon, False)}",
        f"  C {dec_to_dms(min_lat, True)} {dec_to_dms(max_lon, False)}",
        f"  D {dec_to_dms(min_lat, True)} {dec_to_dms(min_lon, False)}",
    ]

    # --- Parse substations from KML/KMZ ---
    def parse_substations(path: Path) -> list[tuple[str, float, float]]:
        def load_root(p: Path) -> ET.Element:
            if p.suffix.lower() == ".kmz":
                with zipfile.ZipFile(p, "r") as kmz:
                    name = next(n for n in kmz.namelist() if n.lower().endswith(".kml"))
                    with kmz.open(name) as kmlf:
                        return ET.parse(kmlf).getroot()
            return ET.parse(p).getroot()

        ns = {"kml": "http://www.opengis.net/kml/2.2"}
        root = load_root(path)
        result = []
        for pm in root.findall(".//kml:Placemark", ns):
            name = pm.findtext("kml:name", default="", namespaces=ns)
            coord = pm.findtext(".//kml:coordinates", default="", namespaces=ns)
            if coord:
                lon, lat = map(float, coord.split(",")[:2])
                result.append((name, lat, lon))
        return result

    subs = parse_substations(substation_file)
    sub_lines = [
        f"  [{name}] {dec_to_dms(lat, True)} {dec_to_dms(lon, False)}"
        for name, lat, lon in subs
    ]

    # --- Obstacles ---
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_yaml = Path(tmpdir) / "obstacles.yaml"
        gpkg_to_obstacles_yaml(obstacles_gpkg, tmp_yaml)
        obs_data = yaml.safe_load(tmp_yaml.read_text(encoding="utf-8"))
        obstacles_list = obs_data.get("OBSTACLES", [])

    # --- Write final YAML ---
    with open(output_yaml, "w", encoding="utf-8") as f:
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
        for item in obstacles_list:
            f.write(f"- '{item}'\n")

