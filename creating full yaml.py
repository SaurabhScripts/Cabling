
# preprocessing 
#code for preprocessing turbine excel 
import pandas as pd
import yaml
import utm

# Load the Excel file
df = pd.read_excel(r"D:\Flask-Api\RE_Wind Route\Data\1. Ph 12 Kallam_Mastersheet_with MEDA_161024 18-10-2024.xlsx",skiprows=1)
# Load the Excel file
df = pd.read_excel(r"F:\Route optimisation RE\interarray\notebooks\Data\1. Ph 12 Kallam_Mastersheet_with MEDA_161024 18-10-2024 (1).xlsx", skiprows=1)

# Rename columns
df.columns = ["Index", "Loc_No", "Phase", "Status", "MEDA_Status",
              "Feeder", "Phase_Duplicate", "Zone", "Easting", "Northing"]

# Drop irrelevant columns
df = df[["Loc_No", "Easting", "Northing", "Zone"]]

# Extract the UTM zone (number and letter separately)
utm_zone_number = int(df["Zone"].iloc[0][:2])   # Extracts '43' from '43Q'
utm_zone_letter = df["Zone"].iloc[0][2]         # Extracts 'Q' from '43Q'

# Convert each UTM coordinate to Lat/Lon
def utm_to_latlon(row):
    lat, lon = utm.to_latlon(row["Easting"], row["Northing"], utm_zone_number, utm_zone_letter)
    return pd.Series([lat, lon])

df[["Latitude", "Longitude"]] = df.apply(utm_to_latlon, axis=1)

# Save the converted data
df.to_csv("Results/converted_wind_locations.csv", index=False)

# Display the converted values
print(df.head())
import pandas as pd
import yaml
from pathlib import Path

# --- Configuration ---
CSV_PATH    = Path(r"D:\Flask-Api\RE_Wind Route\Final_setup\converted_wind_locations.csv")  # Update to your CSV path
OUTPUT_YAML = Path(r"D:\Flask-Api\RE_Wind Route\Final_setup\turbines2.yaml")

# --- Helper to generate Excel-like codes: AA, AB, ..., AZ, BA, ... ZZ ---
def index_to_code(idx):
    """Convert 0-based index to two-letter code (A-Z)."""
    letters = []
    for _ in range(2):
        letters.append(chr(ord('A') + (idx % 26)))
        idx //= 26
    return ''.join(reversed(letters))

# --- Load CSV ---
df = pd.read_csv(CSV_PATH)

# --- DMS conversion helper ---
def dec_to_dms(dec):
    deg = int(dec)
    minutes = abs(dec - deg) * 60
    return deg, round(minutes, 3)

# --- Build lines with generated codes ---
lines = []
for i, row in df.iterrows():
    code = index_to_code(i)  # AA, AB, AC, ...
    lat_deg, lat_min = dec_to_dms(row["Latitude"])
    lon_deg, lon_min = dec_to_dms(row["Longitude"])
    lat_str = f"{lat_deg}°{lat_min:.3f}'N"
    lon_str = f"{lon_deg}°{lon_min:.3f}'E"
    lines.append(f"{code} {lat_str} {lon_str}")

# --- Write YAML with literal block style ---
with OUTPUT_YAML.open("w", encoding="utf-8") as f:
    f.write("TURBINES: |-\n")
    for ln in lines:
        f.write(f"  {ln}\n")

print("✅ turbines.yaml with sequential codes written to:", OUTPUT_YAML)


##
import yaml
import re
from pathlib import Path

# --- CONFIGURATION ---
TURBINES_YAML = Path(r"D:\Flask-Api\RE_Wind Route\Final_setup\turbines2.yaml")  # path to your turbines.yaml
EXTENTS_YAML  = Path(r"D:\Flask-Api\RE_Wind Route\Final_setup\extents.yaml")   # path where extents.yaml will be saved

# --- DMS parsing helper ---

# --- DMS parsing helper ---
dms_pattern = re.compile(r"(\d+)°([\d\.]+)'([NS])\s+(\d+)°([\d\.]+)'([EW])")
def dms_to_decimal(dms_str):
    """Convert D°M.MMM'N E°M.MMM'E to (lat, lon) decimal degrees."""
    m = dms_pattern.match(dms_str.strip())
    if not m:
        raise ValueError(f"Invalid DMS format: {dms_str!r}")
    dlat, mlat, ns, dlon, mlon, ew = m.groups()
    lat = int(dlat) + float(mlat)/60
    lon = int(dlon) + float(mlon)/60
    if ns == 'S':
        lat = -lat
    if ew == 'W':
        lon = -lon
    return lat, lon

def decimal_to_dms(lat, lon):
    """Convert decimal lat,lon to D°M.MMM'N D°M.MMM'E strings."""
    def _to_dms(dec, is_lat):
        hemi = 'N' if (is_lat and dec >= 0) else 'S' if is_lat else 'E' if dec >= 0 else 'W'
        dec_abs = abs(dec)
        deg = int(dec_abs)
        mins = (dec_abs - deg) * 60
        return f"{deg}°{mins:.3f}'{hemi}"
    return _to_dms(lat, True), _to_dms(lon, False)

# --- Load turbines.yaml ---
data = yaml.safe_load(TURBINES_YAML.read_text(encoding='utf-8'))
turbines_block = data.get("TURBINES")

# --- Normalize into list of lines ---
if isinstance(turbines_block, list):
    lines = turbines_block
elif isinstance(turbines_block, str):
    lines = [ln.strip() for ln in turbines_block.splitlines() if ln.strip()]
else:
    raise RuntimeError("TURBINES field is neither list nor block string")

# --- Parse each turbine coordinate ---
coords = []
for ln in lines:
    # Expect "Name DMS_lat DMS_lon"
    parts = ln.split(maxsplit=1)
    if len(parts) != 2:
        continue
    _, dms_str = parts
    lat, lon = dms_to_decimal(dms_str)
    coords.append((lat, lon))

if not coords:
    raise RuntimeError("No valid turbine coordinates found")

# --- Compute bounding box ---
lats = [lat for lat, _ in coords]
lons = [lon for _, lon in coords]
max_lat, min_lat = max(lats), min(lats)
min_lon, max_lon = min(lons), max(lons)

# --- Convert bbox corners to DMS ---
dms_A = decimal_to_dms(max_lat, min_lon)  # top-left (A)
dms_B = decimal_to_dms(max_lat, max_lon)  # top-right (B)
dms_C = decimal_to_dms(min_lat, max_lon)  # bottom-right (C)
dms_D = decimal_to_dms(min_lat, min_lon)  # bottom-left (D)

# --- Write extents.yaml ---
with EXTENTS_YAML.open("w", encoding='utf-8') as f:
    f.write("EXTENTS: |-\n")
    f.write(f"  A {dms_A[0]} {dms_A[1]}\n")
    f.write(f"  B {dms_B[0]} {dms_B[1]}\n")
    f.write(f"  C {dms_C[0]} {dms_C[1]}\n")
    f.write(f"  D {dms_D[0]} {dms_D[1]}\n")

print("✅ extents.yaml created at:", EXTENTS_YAML)
import zipfile

# Path to your KMZ file
kmz_file_path = r"F:\Route optimisation RE\interarray\notebooks\Data\Kallam_Serentica PSS 1 and 2 locs (1).kmz"
kml_extract_path = "Results/substations.kml"  # Extracted KML file

# Extract .kml from .kmz
with zipfile.ZipFile(kmz_file_path, 'r') as kmz:
    kml_filename = [name for name in kmz.namelist() if name.endswith('.kml')][0]
    with open(kml_extract_path, 'wb') as f:
        f.write(kmz.read(kml_filename))

print(f" Extracted {kml_filename} to {kml_extract_path}")
from pykml import parser
import yaml
import os

# Path to the extracted KML file
kml_file = r"F:\Route optimisation RE\interarray\notebooks\Results\substations.kml"

# Check if file exists
if not os.path.exists(kml_file):
    raise FileNotFoundError(f"KML file not found: {kml_file}")

# Read and parse the KML file
with open(kml_file, "rt", encoding="utf-8") as f:
    root = parser.parse(f).getroot()

substations = []

# Extract placemarks
for placemark in root.Document.Folder.Placemark:
    name = placemark.name.text
    coordinates = placemark.Point.coordinates.text.strip().split(",")

    if len(coordinates) >= 2:
        lon, lat = float(coordinates[0]), float(coordinates[1])
        substations.append(f"{name} {lat:.6f}°N {lon:.6f}°E")

# Ensure we extracted something
if not substations:
    print(" No substations found in the KML file!")

# Save to YAML file
yaml_data = {"SUBSTATIONS": substations}
yaml_file = r"F:\Route optimisation RE\interarray\notebooks\Results\substations.yaml"

with open(yaml_file, "w", encoding="utf-8") as f:
    yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

# Output results
print(f"Extracted {len(substations)} substations and saved to '{yaml_file}'")
for sub in substations:
    print(sub)
import geopandas as gpd
import string

# Paths - adjust as needed
gpkg_path = r'D:\Flask-Api\RE_Wind Route\Final_setup\final_obstacles.gpkg'
output_yaml = r'D:\Flask-Api\RE_Wind Route\Final_setup\obstacles_final.yaml'

# Read GeoPackage
gdf = gpd.read_file(gpkg_path)

formatted_items = []

for idx, row in gdf.iterrows():
    geom = row.geometry
    # Extract coordinate list based on geometry type
    if geom.geom_type == 'Polygon':
        coords = list(geom.exterior.coords)
    elif geom.geom_type in ['LineString', 'LinearRing']:
        coords = list(geom.coords)
    else:
        continue

    item_lines = []
    for i, (lon, lat) in enumerate(coords):
        # Labeling: A-Z then P26, P27, ...
        label = string.ascii_uppercase[i] if i < 26 else f"P{i}"
        
        # Convert latitude to deg & decimal minutes
        lat_deg = int(lat)
        lat_min = (lat - lat_deg) * 60
        lat_dir = 'N' if lat >= 0 else 'S'
        # Convert longitude to deg & decimal minutes
        lon_deg = int(lon)
        lon_min = (lon - lon_deg) * 60
        lon_dir = 'E' if lon >= 0 else 'W'
        
        # Use unicode degree symbol \u00B0
        line = f"{label} {lat_deg}\u00B0{lat_min:.3f}''{lat_dir} {lon_deg}\u00B0{lon_min:.3f}''{lon_dir}"
        item_lines.append(line)

    # Combine lines with blank lines and indentation
    item_str = item_lines[0] + "\n\n  " + "\n\n  ".join(item_lines[1:])
    formatted_items.append(item_str)

# Write to YAML with explicit UTF-8 encoding
with open(output_yaml, 'w', encoding='utf-8') as f:
    f.write(f"OBSTACLES: \n")
    for item in formatted_items:
        f.write(f"- '{item}'\n")

print(f"YAML file written to: {output_yaml}")
import pandas as pd
import yaml
from pathlib import Path

# --- Configuration: adjust these paths ---
CSV_PATH          = Path(r"D:\Flask-Api\RE_Wind Route\converted_wind_locations.csv")
SUBSTATIONS_YAML  = Path(r"D:\Flask-Api\RE_Wind Route\Final_setup\substations.yaml")
EXTENTS_YAML      = Path(r"D:\Flask-Api\RE_Wind Route\Final_setup\extents.yaml")
OUT_YAML          = Path(r"D:\Flask-Api\RE_Wind Route\Final_setup\site_full_block_style.yaml")

# --- Load turbines CSV ---
df = pd.read_csv(CSV_PATH)

# --- Load substation YAML ---
sub_data = yaml.safe_load(SUBSTATIONS_YAML.read_text(encoding='utf-8'))
sub_list = sub_data.get("SUBSTATIONS", [])

# --- Load extents YAML ---
ext_data = yaml.safe_load(EXTENTS_YAML.read_text(encoding='utf-8'))
ext_raw  = ext_data.get("EXTENTS", "").strip()  # multiline string
# Reconstruct as literal block
ext_lines = ext_raw.splitlines()
extents_block = "|-\n  " + "\n  ".join(ext_lines)

# --- Helpers ---
def index_to_code(idx):
    out = ""
    for _ in range(2):
        out = chr(ord('A') + (idx % 26)) + out
        idx //= 26
    return out

def dec_to_dms(v, is_lat):
    hemi = 'N' if is_lat else 'E'
    if v < 0:
        hemi = 'S' if is_lat else 'W'
    v = abs(v)
    d = int(v)
    m = (v - d) * 60
    return f"{d}°{m:.3f}'{hemi}"

# --- Build SUBSTATIONS block from sub_list ---
sub_lines = []
for ln in sub_list:
    parts = ln.rsplit(" ", 2)
    if len(parts) != 3:
        continue
    name, lat_str, lon_str = parts
    lat_val = float(lat_str[:-2]) * (1 if lat_str.endswith('N') else -1)
    lon_val = float(lon_str[:-2]) * (1 if lon_str.endswith('E') else -1)
    sub_lines.append(f"  [{name}] {dec_to_dms(lat_val, True)} {dec_to_dms(lon_val, False)}")
substations_block = "|-\n" + "\n".join(sub_lines)

# --- Build TURBINES block from CSV ---
turbine_lines = []
for i, row in df.iterrows():
    code = index_to_code(i)
    turbine_lines.append(f"  {code} {dec_to_dms(row['Latitude'], True)} {dec_to_dms(row['Longitude'], False)}")
turbines_block = "|-\n" + "\n".join(turbine_lines)

# --- Write combined YAML (without obstacles) ---
with OUT_YAML.open("w", encoding="utf-8") as f:
    f.write("OPERATOR: Serentica\n")
    f.write("TURBINE:\n  make: Vestas\n  model: V90/3000\n  power_MW: 3\n\n")
    f.write("LANDSCAPE_ANGLE: 0\n\n")
    f.write(f"EXTENTS: {extents_block}\n\n")
    f.write(f"SUBSTATIONS: {substations_block}\n\n")
    f.write(f"TURBINES: {turbines_block}\n")

print("✅ Updated site YAML with extents from extents.yaml written to:", OUT_YAML)
from pathlib import Path

# Paths - adjust if needed
site_yaml = Path(r"D:\Flask-Api\RE_Wind Route\Final_setup\site_full_block_style.yaml")
obs_yaml  = Path(r"D:\Flask-Api\RE_Wind Route\Final_setup\obstacles_final.yaml")

# Read site_full YAML lines
site_lines = site_yaml.read_text(encoding='utf-8').splitlines()

# Remove existing OBSTACLES section if present
clean_lines = []
in_obs = False
for line in site_lines:
    if line.strip() == "OBSTACLES:":
        in_obs = True
        break
    clean_lines.append(line)

# Read new obstacles entries (already formatted with '- ' and quotes)
obs_lines = obs_yaml.read_text(encoding='utf-8').splitlines()

# Build combined content
combined = clean_lines + ["", "OBSTACLES:"] + obs_lines

# Write back to site_full YAML
site_yaml.write_text("\n".join(combined) + "\n", encoding='utf-8')

print(f"✅ Appended obstacles from {obs_yaml.name} into {site_yaml.name}")
import yaml
import re
from shapely.geometry import Polygon
from shapely.ops import unary_union

def parse_dms(dms_str):
    """
    Parse "18°31.566''N 75°54.950''E" into (lat, lon).
    """
    pattern = r"(\d+)°([\d\.]+)''?([NS])\s+(\d+)°([\d\.]+)''?([EW])"
    m = re.match(pattern, dms_str.strip())
    if not m:
        raise ValueError(f"Invalid DMS: {dms_str!r}")
    lat_deg, lat_min, lat_dir, lon_deg, lon_min, lon_dir = m.groups()
    lat = int(lat_deg) + float(lat_min) / 60 * (1 if lat_dir=='N' else -1)
    lon = int(lon_deg) + float(lon_min) / 60 * (1 if lon_dir=='E' else -1)
    return lat, lon

# 1) Load the full YAML
with open(r'D:\Flask-Api\RE_Wind Route\Final_setup\site_full_block_style.yaml', 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

# 2) Build Shapely Polygons for each obstacle
obs_blocks = data.get('OBSTACLES', [])
polygons = []
for block in obs_blocks:
    coords = []
    for ln in block.strip().splitlines():
        # drop the label (e.g. "A ") and parse DMS
        dms = ln.split(' ', 1)[1]
        lat, lon = parse_dms(dms)
        coords.append((lon, lat))  # note: lon,lat order
    polygons.append(Polygon(coords))

# 3) Detect “bad” polygons
bad_idxs = set()

# 3a. Self-intersection:
for i, poly in enumerate(polygons):
    if not poly.is_valid:
        bad_idxs.add(i)

# 3b. Pairwise intersection:
n = len(polygons)
for i in range(n):
    for j in range(i+1, n):
        # They must only touch at shared vertices, so use intersects() minus touches()
        if polygons[i].intersects(polygons[j]) and not polygons[i].touches(polygons[j]):
            bad_idxs.add(i)
            bad_idxs.add(j)

print("Removing obstacle indices:", sorted(bad_idxs))

# 4) Filter out the bad blocks
clean_blocks = [
    blk for idx, blk in enumerate(obs_blocks)
    if idx not in bad_idxs
]

# 5) Write cleaned YAML
data['OBSTACLES'] = clean_blocks
with open(r'D:\Flask-Api\RE_Wind Route\Final_setup\final_Serentica_clear.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(data, f, sort_keys=False, allow_unicode=True)

print("Clean YAML written to site_clean.yaml")
import yaml
import re

# Custom presenter to output multiline strings using literal block style
def str_presenter(dumper, data):
    if '\n' in data:  # multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.SafeDumper.add_representer(str, str_presenter)

def clean_block(block):
    """
    Clean a multiline obstacle block by:
    - Removing blank lines
    - Dropping the duplicate last point if it matches the first
    - Rejoining lines with two-space indentation on continued lines
    """
    # Split into lines, strip, and filter out empties
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return ''
    # Drop last if duplicate of first
    first_dms = lines[0].split(' ', 1)[1]
    last_dms  = lines[-1].split(' ', 1)[1]
    if first_dms == last_dms:
        lines.pop()
    # Reassemble with two-space indent for continued lines
    return lines[0] + "\n  " + "\n  ".join(lines[1:])

# Paths
INPUT_YAML  = r"F:\Route optimisation RE\interarray\notebooks\Results\final_Serentica_clear.yaml"
OUTPUT_YAML = r"F:\Route optimisation RE\interarray\notebooks\Results\final_Serentica_final.yaml"

# Load the YAML file
with open(INPUT_YAML, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

# Clean OBSTACLES section if present
if 'OBSTACLES' in data and isinstance(data['OBSTACLES'], list):
    cleaned = []
    for blk in data['OBSTACLES']:
        cleaned_blk = clean_block(blk)
        cleaned.append(cleaned_blk)
    data['OBSTACLES'] = cleaned

# Write out the cleaned YAML
with open(OUTPUT_YAML, 'w', encoding='utf-8') as f:
    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

print(f"Cleaned YAML written to: {OUTPUT_YAML}")
import re
import yaml
from pathlib import Path
from shapely.geometry import Polygon

##############################################
# Helper: Convert DMS string to decimal degrees
##############################################
def dms_to_decimal(coord_str):
    """
    Convert a DMS string (e.g., "18°38.567'N") into decimal degrees.
    """
    match = re.match(r"(\d+)°(\d+\.\d+)'([NSEW])", coord_str)
    if match:
        deg, minutes, hemi = match.groups()
        dec = float(deg) + float(minutes) / 60
        if hemi in ('S', 'W'):
            dec = -dec
        return dec
    raise ValueError(f"Cannot parse DMS string: {coord_str}")

##############################################
# Helper: Parse a multiline field into a list of (lon, lat) tuples
##############################################
def parse_multiline_field(field_str):
    """
    Expects each line in the field to be formatted as:
      Label  latitude  longitude
    Returns a list of (lon, lat) tuples (since Shapely expects (x, y)).
    """
    lines = [line.strip() for line in field_str.splitlines() if line.strip()]
    coords = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            lat = dms_to_decimal(parts[1])
            lon = dms_to_decimal(parts[2])
            coords.append((lon, lat))
    return coords

##############################################
# 1. Read the YAML file
##############################################
# Change this path to your YAML file.
input_yaml_file = Path(r"F:\Route optimisation RE\interarray\notebooks\Results\final_Serentica_final.yaml")
with open(input_yaml_file, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

##############################################
# 2. Build the extent polygon from the EXTENTS block
##############################################
extents_str = data.get("EXTENTS", "")
extent_coords = parse_multiline_field(extents_str)
if len(extent_coords) < 3:
    raise ValueError("EXTENTS block does not contain enough points to form a polygon.")
extent_poly = Polygon(extent_coords)
print("Extent polygon:", extent_poly)

##############################################
# 3. Process OBSTACLES and filter out those outside the extent
##############################################
# OBSTACLES is expected to be a list of multiline strings.
raw_obstacles = data.get("OBSTACLES", [])
filtered_obstacles = []

def obstacle_to_polygon(obs_str):
    coords = parse_multiline_field(obs_str)
    if len(coords) < 3:
        return None
    return Polygon(coords)

for obs_str in raw_obstacles:
    poly = obstacle_to_polygon(obs_str)
    if poly is None:
        print("Skipping obstacle with insufficient vertices.")
        continue
    if poly.within(extent_poly):
        filtered_obstacles.append(obs_str)
    else:
        print("Removing obstacle not fully inside extent.")

# Update the YAML data with filtered obstacles.
data["OBSTACLES"] = filtered_obstacles

##############################################
# 4. Write the cleaned YAML to a new file
##############################################
output_yaml_file = Path(r"F:\Route optimisation RE\interarray\notebooks\Results\final_Serentica_final_extent.yaml")
with open(output_yaml_file, "w", encoding="utf-8") as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"Filtered YAML written to {output_yaml_file}")
import yaml
import tempfile
import shutil
from pathlib import Path
from interarray.importer import L_from_yaml
from interarray.mesh import make_planar_embedding

# Path to your combined YAML with OBSTACLES
combined_yaml_path = Path(r"F:\Route optimisation RE\interarray\notebooks\Results\final_Serentica_final_extent.yaml")

# Load full data
with open(combined_yaml_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

obstacles = data.get('OBSTACLES', [])

print("Testing each obstacle individually...")
for idx, obs in enumerate(obstacles):
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / "single_obs.yaml"
    # Build a minimal YAML with all sections but only one obstacle
    test_data = {
        **{k: v for k,v in data.items() if k != 'OBSTACLES'},
        'OBSTACLES': [obs]
    }
    with open(tmp_path, 'w', encoding='utf-8') as tf:
        yaml.safe_dump(test_data, tf, sort_keys=False, allow_unicode=True)
    try:
        L = L_from_yaml(tmp_path, handle=f"SingleObs{idx}")
        P, A = make_planar_embedding(L)
        # print(f"Obstacle {idx}: SUCCESS")
    except Exception as e:
        print(f"Obstacle {idx}: FAIL -> {e}--obstacle: {obs}")
    finally:
        shutil.rmtree(tmp_dir)

print("Testing complete.")
import yaml
import tempfile
import shutil
from pathlib import Path
from interarray.importer import L_from_yaml
from interarray.mesh import make_planar_embedding

# Path to your combined YAML with OBSTACLES
combined_yaml_path = Path(r"F:\Route optimisation RE\interarray\notebooks\Results\final_Serentica_final_extent.yaml")
output_yaml_path = combined_yaml_path.with_name("final_Serentica_no_bad_obstacles_processed_27June.yaml")

# Load full data
with open(combined_yaml_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

obstacles = data.get('OBSTACLES', [])

print("Testing each obstacle individually and removing failures...")
good_obstacles = []
failed_indices = []

for idx, obs in enumerate(obstacles):
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / "single_obs.yaml"
    # Build a minimal YAML with all sections but only this obstacle
    test_data = {k: v for k, v in data.items() if k != 'OBSTACLES'}
    test_data['OBSTACLES'] = [obs]
    with open(tmp_path, 'w', encoding='utf-8') as tf:
        yaml.safe_dump(test_data, tf, sort_keys=False, allow_unicode=True)
    try:
        L = L_from_yaml(tmp_path, handle=f"SingleObs{idx}")
        P, A = make_planar_embedding(L)
        # If no exception, keep this obstacle
        good_obstacles.append(obs)
    except Exception as e:
        failed_indices.append(idx)
        print(f"Removed obstacle {idx} due to error: {e}")
    finally:
        shutil.rmtree(tmp_dir)

# Write cleaned YAML
cleaned_data = {k: v for k, v in data.items() if k != 'OBSTACLES'}
cleaned_data['OBSTACLES'] = good_obstacles

with open(output_yaml_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cleaned_data, f, sort_keys=False, allow_unicode=True)

print(f"\nRemoved obstacles at indices: {failed_indices}")
print("Cleaned YAML written to:", output_yaml_path)
