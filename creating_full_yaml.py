from pathlib import Path
from src.workflow import build_site_yaml


def build_full_site_yaml(
    turbine_file: Path | str,
    substation_file: Path | str,
    obstacles_gpkg: Path | str,
    output_yaml: Path | str,
) -> Path:
    """Create a combined site YAML from turbine, substation and obstacle data."""

    t_file = Path(turbine_file)
    s_file = Path(substation_file)
    o_file = Path(obstacles_gpkg)
    out = Path(output_yaml)
    build_site_yaml(t_file, s_file, o_file, out)
    return out
