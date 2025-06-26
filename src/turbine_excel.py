import pandas as pd
import utm
import simplekml
import geopandas as gpd
from pathlib import Path


def read_turbine_excel(path: Path) -> pd.DataFrame:
    """Load turbine Excel/CSV with UTM coords and convert to lat/lon."""
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        df = pd.read_excel(path, skiprows=1)
    else:
        df = pd.read_csv(path)
    df.columns = [
        "Index",
        "Loc_No",
        "Phase",
        "Status",
        "MEDA_Status",
        "Feeder",
        "Phase_Duplicate",
        "Zone",
        "Easting",
        "Northing",
    ]
    df = df[["Loc_No", "Easting", "Northing", "Zone"]]
    utm_zone_number = int(str(df["Zone"].iloc[0])[:2])
    utm_zone_letter = str(df["Zone"].iloc[0])[2]

    def utm_to_latlon(row):
        lat, lon = utm.to_latlon(
            row["Easting"], row["Northing"], utm_zone_number, utm_zone_letter
        )
        return pd.Series({"Latitude": lat, "Longitude": lon})

    df[["Latitude", "Longitude"]] = df.apply(utm_to_latlon, axis=1)
    return df


def dataframe_to_kml(df: pd.DataFrame, path: Path) -> None:
    kml = simplekml.Kml()

    def index_to_code(idx: int) -> str:
        letters = []
        for _ in range(2):
            letters.append(chr(ord("A") + (idx % 26)))
            idx //= 26
        return "".join(reversed(letters))

    for i, row in df.iterrows():
        code = index_to_code(i)
        name = f"Turbine {code}"
        desc = f"Loc_No: {row['Loc_No']}\nZone: {row['Zone']}"
        kml.newpoint(
            name=name,
            coords=[(row["Longitude"], row["Latitude"])],
            description=desc,
        )

    kml.save(str(path))


def dataframe_to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs=4326,
    )


def dataframe_to_yaml(df: pd.DataFrame, path: Path) -> None:
    def index_to_code(idx: int) -> str:
        letters = []
        for _ in range(2):
            letters.append(chr(ord('A') + (idx % 26)))
            idx //= 26
        return ''.join(reversed(letters))

    def dec_to_dms(dec: float) -> tuple[int, float]:
        deg = int(dec)
        minutes = abs(dec - deg) * 60
        return deg, round(minutes, 3)

    lines = []
    for i, row in df.iterrows():
        code = index_to_code(i)
        lat_deg, lat_min = dec_to_dms(row['Latitude'])
        lon_deg, lon_min = dec_to_dms(row['Longitude'])
        lat_str = f"{lat_deg}\u00B0{lat_min:.3f}'N"
        lon_str = f"{lon_deg}\u00B0{lon_min:.3f}'E"
        lines.append(f"{code} {lat_str} {lon_str}")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('TURBINES: |-\n')
        for line in lines:
            f.write(f"  {line}\n")
