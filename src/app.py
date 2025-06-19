from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import geopandas as gpd
from pathlib import Path
import tempfile

from .workflow import create_extent, download_osm_layer, merge_layers, buffer_and_union, export_kmz, export_folium_map
from .crossings import load_osm_roads, load_osm_powerlines, count_crossings

app = FastAPI()

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = static_dir / "index.html"
    return index_path.read_text()


@app.post("/process/")
async def process_files(turbines: UploadFile = File(...), obstacles: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmpdir:
        turbine_path = Path(tmpdir) / turbines.filename
        obstacle_path = Path(tmpdir) / obstacles.filename
        with open(turbine_path, "wb") as f:
            f.write(await turbines.read())
        with open(obstacle_path, "wb") as f:
            f.write(await obstacles.read())

        turbines_gdf = gpd.read_file(turbine_path)
        obstacle_gdf = gpd.read_file(obstacle_path)

        extent = create_extent(turbines_gdf)
        roads = load_osm_roads(extent)
        power = load_osm_powerlines(extent)
        obstacles_combined = merge_layers([obstacle_gdf, roads, power])
        obstacle_union = buffer_and_union(obstacles_combined, 20)

        kmz_path = Path(tmpdir) / "obstacles.kmz"
        export_kmz(obstacle_union.to_frame(name="geometry"), kmz_path)
        map_path = Path(tmpdir) / "map.html"
        export_folium_map(obstacle_union.to_frame(name="geometry"), map_path)

        road_cross, power_cross = count_crossings(obstacle_union.to_frame(name="geometry"), roads, power)

        return {
            "kmz": kmz_path.read_bytes().hex(),
            "map": map_path.read_text(),
            "road_crossings": road_cross,
            "power_crossings": power_cross,
        }

