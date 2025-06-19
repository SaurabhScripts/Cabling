# Cabling

This repository provides utilities for replicating a QGIS workflow in Python and a small FastAPI app for processing turbine and obstacle data.

## Features

- **Workflow functions** (`src/workflow.py`)
  - Download OSM layers inside an extent
  - Merge and buffer obstacles
  - Export KMZ and Folium maps
- **Crossing utilities** (`src/crossings.py`)
  - Load road and power line data from OSM
  - Count the number of crossings with a cable route
- **Web API** (`src/app.py`)
  - Upload turbine and obstacle files (e.g., GeoPackage)
  - Generate combined obstacle layers and return KMZ/Folium map
  - Report road and power line crossing counts

### Running locally

Install the dependencies (requires Python 3.11):

```bash
pip install geopandas folium osmnet fastapi uvicorn
```

Start the API:

```bash
python -m src
```

This will launch a FastAPI server on `http://localhost:8000` with an endpoint `/process/` that accepts `turbines` and `obstacles` files.
