from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

from .turbine_excel import (
    read_turbine_excel,
    dataframe_to_kml,
    dataframe_to_yaml,
    dataframe_to_gdf,
)
from fastapi.staticfiles import StaticFiles
import shutil
from datetime import datetime
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
import tempfile
from zipfile import ZipFile

from .workflow import export_folium_map, load_yaml_points, generate_optimized_route
from creating_full_yaml import build_full_site_yaml

app = FastAPI()

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# directory for storing generated KMZ results
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
app.mount("/results", StaticFiles(directory=str(results_dir)), name="results")

# base directory for persisting uploaded files
template_dir = Path(__file__).resolve().parent.parent / "Template"
template_dir.mkdir(exist_ok=True)


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
    """Combine uploads into a site YAML and run the optimisation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = template_dir / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)

    turbine_path = session_dir / turbines.filename
    sub_path = session_dir / substation.filename
    obstacle_path = session_dir / obstacles.filename

    for upload, path in [
        (turbines, turbine_path),
        (substation, sub_path),
        (obstacles, obstacle_path),
    ]:
        with open(path, "wb") as f:
            f.write(await upload.read())

    site_yaml = session_dir / "site.yaml"
    build_full_site_yaml(turbine_path, sub_path, obstacle_path, site_yaml)

    with tempfile.TemporaryDirectory() as tmpdir:
        kmz_tmp = Path(tmpdir) / "route.kmz"
        generate_optimized_route(site_yaml, kmz_tmp)

        with ZipFile(kmz_tmp, "r") as zf:
            kml_files = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kml_files:
                raise RuntimeError("Optimisation output contains no KML")
            kml_path = Path(tmpdir) / kml_files[0]
            zf.extract(kml_files[0], tmpdir)
        gdf = gpd.read_file(kml_path)

        out_name = f"route_{timestamp}.kmz"
        out_path = results_dir / out_name
        shutil.move(kmz_tmp, out_path)

    return {"url": f"/results/{out_name}", "geojson": gdf.__geo_interface__}


@app.post("/run-final/")
async def run_final_route(site: UploadFile = File(...)):
    """Run the advanced interarray optimisation using an uploaded site YAML."""
    with tempfile.TemporaryDirectory() as tmpdir:
        site_path = Path(tmpdir) / site.filename
        with open(site_path, "wb") as f:
            f.write(await site.read())

        kmz_tmp = Path(tmpdir) / "route.kmz"
        generate_optimized_route(site_path, kmz_tmp)

        # also load the KMZ as GeoJSON for easier visualisation
        with ZipFile(kmz_tmp, "r") as zf:
            kml_files = [n for n in zf.namelist() if n.lower().endswith(".kml")]
            if not kml_files:
                raise RuntimeError("Optimisation output contains no KML")
            kml_path = Path(tmpdir) / kml_files[0]
            zf.extract(kml_files[0], tmpdir)
        gdf = gpd.read_file(kml_path)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"route_{ts}.kmz"
        out_path = results_dir / out_name
        shutil.move(kmz_tmp, out_path)

    return {"url": f"/results/{out_name}", "geojson": gdf.__geo_interface__}


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

        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        extent_gdf = gpd.GeoDataFrame(geometry=[box(*gdf.total_bounds)], crs=4326)

        return {
            "geojson": gdf.__geo_interface__,
            "extent": extent_gdf.__geo_interface__,
        }
