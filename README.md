# Cabling

This repository provides utilities for replicating a QGIS workflow in Python and a small FastAPI app for processing turbine and obstacle data.

## Features

- **Workflow functions** (`src/workflow.py`)
  - Download OSM layers inside an extent
  - Merge and buffer obstacles
  - Export KMZ and Folium maps
  - Convert obstacle GeoPackage files to YAML
- **Crossing utilities** (`src/crossings.py`)
  - Load road and power line data from OSM
  - Count the number of crossings with a cable route
- **Web API** (`src/app.py`)
  - Upload turbine and obstacle files (e.g., GeoPackage)
  - Generate combined obstacle layers and return KMZ/Folium map
  - Report road and power line crossing counts

### Running locally

Clone the repository and install the dependencies (requires Python 3.11). The
`requirements.txt` file includes `python-multipart`, which FastAPI needs to
handle file uploads. If you use a virtual environment, activate it before
installing the packages.

```bash
git clone <repo-url>
cd Cabling
pip install -r requirements.txt
```

Start the API:

```bash
python -m src
```

This will launch a FastAPI server on `http://localhost:8000`.
Open `http://localhost:8000` in your browser to use the interactive interface for uploading turbine, substation and obstacle files.
The interface sends the selected files to the `/process/` endpoint and displays the resulting map and crossing counts.

### Obstacles from GeoPackage

To convert obstacle data stored in a GeoPackage into the YAML format used by the
workflow, call `gpkg_to_obstacles_yaml`:

```python
from cabling import gpkg_to_obstacles_yaml

gpkg_to_obstacles_yaml("obstacles.gpkg", "obstacles.yaml")
```

The resulting `obstacles.yaml` can then be merged with the turbine and
substation data when building the site configuration.

If you need a quick KMZ representation of the obstacles for visual inspection,
use `gpkg_to_kmz` which applies simple styling so polygons are visible in most
KML viewers:

```python
from cabling import gpkg_to_kmz

gpkg_to_kmz("obstacles.gpkg", "obstacles.kmz")
```

You can also convert directly from the command line using
`convert_gpkg_to_kmz.py`:

```bash
./convert_gpkg_to_kmz.py obstacles.gpkg obstacles.kmz
```

### Generate extents from turbines

If your turbine locations are already stored as a YAML block, use
`turbines_to_extents_yaml` to create an `extents.yaml` bounding box. The
helper writes the result to a temporary directory (``temp`` by default):

```python
from pathlib import Path
from cabling import turbines_to_extents_yaml

turbines_to_extents_yaml("turbines.yaml", Path("temp") / "extents.yaml")
```

There is also a small command line wrapper ``turbines_to_extents.py``:

```bash
./turbines_to_extents.py turbines.yaml --out-dir temp
```

### Quick map viewer

Navigate to `/map` to access a simple viewer for plotting turbine and substation files on a Leaflet map. Upload turbine coordinates as `.xlsx`, `.csv`, or `.yaml` and optional substation data as `.kmz`. The `/upload` endpoint converts the data to GeoJSON and also returns a bounding-box layer when turbines are provided.
The routing endpoints (`/process/` and `/run-final/`) now return GeoJSON so the
generated cable layouts can be displayed alongside turbines and substations.
