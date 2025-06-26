from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

from .turbine_excel import (
    read_turbine_excel,
    dataframe_to_kml,
    dataframe_to_yaml,
    dataframe_to_gdf,
)
from fastapi.staticfiles import StaticFiles
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
import tempfile

from .workflow import (
    create_extent,
    download_osm_layer,
    merge_layers,
    buffer_and_union,
    export_kmz,
    export_folium_map,
    csv_to_yaml,
    load_csv_points,
    generate_simple_route,
)
from .crossings import load_osm_roads, load_osm_powerlines, count_crossings

app = FastAPI()

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = static_dir / "index.html"
    return index_path.read_text()


@app.get("/map", response_class=HTMLResponse)
async def map_page():
    map_path = static_dir / "map.html"
    return map_path.read_text()


@app.post("/process/")
async def process_files(
    turbines: UploadFile = File(...),
    substation: UploadFile = File(...),
    obstacles: UploadFile = File(...),
):
    """Process uploaded files and return YAML strings, KMZ route and map."""
    with tempfile.TemporaryDirectory() as tmpdir:
        turbine_path = Path(tmpdir) / turbines.filename
        sub_path = Path(tmpdir) / substation.filename
        obstacle_path = Path(tmpdir) / obstacles.filename

        with open(turbine_path, "wb") as f:
            f.write(await turbines.read())
        with open(sub_path, "wb") as f:
            f.write(await substation.read())
        with open(obstacle_path, "wb") as f:
            f.write(await obstacles.read())

        turbines_gdf = load_csv_points(turbine_path)
        substation_gdf = load_csv_points(sub_path)
        obstacle_gdf = gpd.read_file(obstacle_path)

        # create YAML outputs
        turbine_yaml = Path(tmpdir) / "turbines.yaml"
        substation_yaml = Path(tmpdir) / "substation.yaml"
        csv_to_yaml(turbine_path, turbine_yaml)
        csv_to_yaml(sub_path, substation_yaml)

        # extent layer
        extent = create_extent(turbines_gdf)
        extent_gdf = gpd.GeoDataFrame(geometry=[box(*extent)], crs=4326)

        # obstacles processing similar to original example
        roads = load_osm_roads(extent)
        power = load_osm_powerlines(extent)
        obstacles_combined = merge_layers([obstacle_gdf, roads, power])
        obstacle_union = buffer_and_union(obstacles_combined, 20)

        # simple route generation
        route_gdf = generate_simple_route(turbines_gdf, substation_gdf)

        kmz_path = Path(tmpdir) / "route.kmz"
        export_kmz(route_gdf, kmz_path)

        # build map with all layers
        map_layers = {
            "Turbines": turbines_gdf,
            "Substation": substation_gdf,
            "Obstacles": obstacle_gdf,
            "Route": route_gdf,
            "Extent": extent_gdf,
        }
        map_path = Path(tmpdir) / "map.html"
        export_folium_map(map_layers, map_path)

        road_cross, power_cross = count_crossings(route_gdf, roads, power)

        return {
            "turbine_yaml": turbine_yaml.read_text(),
            "substation_yaml": substation_yaml.read_text(),
            "route_kmz": kmz_path.read_bytes().hex(),
            "map": map_path.read_text(),
            "road_crossings": road_cross,
            "power_crossings": power_cross,
        }


@app.post("/turbine-kml/")
async def turbine_kml(file: UploadFile = File(...)):
    """Convert a turbine Excel/CSV to KML and return the map."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / file.filename
        with open(fpath, "wb") as f:
            f.write(await file.read())

        df = read_turbine_excel(fpath)
        gdf = dataframe_to_gdf(df)

        kml_path = Path(tmpdir) / "turbines.kml"
        yaml_path = Path(tmpdir) / "turbines.yaml"
        dataframe_to_kml(df, kml_path)
        dataframe_to_yaml(df, yaml_path)

        map_path = Path(tmpdir) / "map.html"
        export_folium_map({"Turbines": gdf}, map_path)

        return {
            "kml": kml_path.read_bytes().hex(),
            "yaml": yaml_path.read_text(),
            "map": map_path.read_text(),
        }

