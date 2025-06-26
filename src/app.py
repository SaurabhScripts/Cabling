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
from zipfile import ZipFile

from .workflow import (
    create_extent,
    export_kmz,
    export_folium_map,
    csv_to_yaml,
    load_csv_points,
    generate_simple_route,
    load_yaml_points,
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
    turbines: UploadFile | None = File(None),
    substation: UploadFile | None = File(None),
    obstacles: UploadFile | None = File(None),
):
    """Process uploaded files and return YAML strings, KMZ route and map."""
    with tempfile.TemporaryDirectory() as tmpdir:
        if turbines:
            turbine_path = Path(tmpdir) / turbines.filename
            with open(turbine_path, "wb") as f:
                f.write(await turbines.read())
            turbines_gdf = load_csv_points(turbine_path)
        else:
            turbines_gdf = gpd.GeoDataFrame(geometry=[], crs=4326)

        if substation:
            sub_path = Path(tmpdir) / substation.filename
            with open(sub_path, "wb") as f:
                f.write(await substation.read())
            substation_gdf = load_csv_points(sub_path)
        else:
            substation_gdf = gpd.GeoDataFrame(geometry=[], crs=4326)

        if obstacles:
            obstacle_path = Path(tmpdir) / obstacles.filename
            with open(obstacle_path, "wb") as f:
                f.write(await obstacles.read())
            obstacle_gdf = gpd.read_file(obstacle_path)
        else:
            obstacle_gdf = gpd.GeoDataFrame(geometry=[], crs=4326)

        # create YAML outputs if data provided
        turbine_yaml = Path(tmpdir) / "turbines.yaml"
        substation_yaml = Path(tmpdir) / "substation.yaml"
        if turbines:
            csv_to_yaml(turbine_path, turbine_yaml)
            turbine_yaml_text = turbine_yaml.read_text()
        else:
            turbine_yaml_text = ""
        if substation:
            csv_to_yaml(sub_path, substation_yaml)
            substation_yaml_text = substation_yaml.read_text()
        else:
            substation_yaml_text = ""

        # extent layer based on available data
        extent_source = turbines_gdf
        if extent_source.empty and not substation_gdf.empty:
            extent_source = substation_gdf
        if extent_source.empty and not obstacle_gdf.empty:
            extent_source = obstacle_gdf
        if not extent_source.empty:
            extent = create_extent(extent_source)
            extent_gdf = gpd.GeoDataFrame(geometry=[box(*extent)], crs=4326)
        else:
            extent = (0, 0, 0, 0)
            extent_gdf = gpd.GeoDataFrame(geometry=[], crs=4326)

        # obstacles processing similar to original example
        roads = load_osm_roads(extent) if not extent_source.empty else gpd.GeoDataFrame(geometry=[], crs=4326)
        power = load_osm_powerlines(extent) if not extent_source.empty else gpd.GeoDataFrame(geometry=[], crs=4326)

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

        road_cross, power_cross = count_crossings(route_gdf, roads, power) if not route_gdf.empty else (0, 0)

        return {
            "turbine_yaml": turbine_yaml_text,
            "substation_yaml": substation_yaml_text,
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


@app.post("/upload")
async def upload_excel(file: UploadFile = File(...)):
    """Accept a geospatial file and return GeoJSON (and extent) for map display."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / file.filename
        with open(fpath, "wb") as f:
            f.write(await file.read())

        suffix = fpath.suffix.lower()
        if suffix in {".xlsx", ".xls", ".csv"}:
            df = read_turbine_excel(fpath)
            gdf = dataframe_to_gdf(df)
        elif suffix in {".yml", ".yaml"}:
            gdf = load_yaml_points(fpath)
        elif suffix == ".kmz":
            with ZipFile(fpath, "r") as zf:
                kml_files = [n for n in zf.namelist() if n.lower().endswith(".kml")]
                if not kml_files:
                    raise ValueError("KMZ archive contains no KML file")
                kml_path = Path(tmpdir) / kml_files[0]
                zf.extract(kml_files[0], tmpdir)
            gdf = gpd.read_file(kml_path)
        else:
            gdf = gpd.read_file(fpath)

        extent_gdf = gpd.GeoDataFrame(geometry=[box(*gdf.total_bounds)], crs=4326)

        return {
            "geojson": gdf.__geo_interface__,
            "extent": extent_gdf.__geo_interface__,
        }

