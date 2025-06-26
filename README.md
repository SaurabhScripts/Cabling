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

### Quick map viewer

Navigate to `/map` to access a simple viewer for plotting turbine and substation files on a Leaflet map. Upload turbine coordinates as `.xlsx`, `.csv`, or `.yaml` and optional substation data as `.kmz`. The `/upload` endpoint converts the data to GeoJSON and also returns a bounding-box layer when turbines are provided.
