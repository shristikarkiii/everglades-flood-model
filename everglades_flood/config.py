"""Single source of truth for the study area, data sources and model settings.

Everything the pipeline depends on is declared here so a reader can see the
whole configuration of the model in one screen, and so no magic number is
buried in a processing step.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"

for _d in (RAW, INTERIM, OUTPUTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Study area
# ---------------------------------------------------------------------------

# Bounding box in EPSG:4326, covering Everglades National Park, the southern
# tip of Big Cypress, Florida Bay and the western fringe of Miami-Dade.
AOI_4326 = (-81.52, 24.85, -80.25, 25.89)  # (west, south, east, north)

# Analysis grid.
#
# Terrain derivatives are computed in a projected CRS, not in EPSG:4326. In a
# geographic CRS the pixel is measured in degrees, and a degree of longitude at
# this latitude is about 100 km against 111 km for a degree of latitude -- so
# slope comes out distorted by roughly 10% along one axis, and every distance
# is in units that are not metres. In UTM 17N the pixel is 30 m on both axes.
ANALYSIS_CRS = "EPSG:32617"  # WGS 84 / UTM zone 17N
PIXEL_SIZE = 30.0  # metres


# ---------------------------------------------------------------------------
# Data sources (all openly accessible, no credentials required)
# ---------------------------------------------------------------------------

# Copernicus DEM GLO-30, 1 arc-second, as Cloud-Optimised GeoTIFF on AWS.
COP_DEM_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
COP_DEM_TILES = [
    ("N24", "W081"),
    ("N24", "W082"),
    ("N25", "W081"),
    ("N25", "W082"),
]

# ESA WorldCover 10 m, 2021 v200.
WORLDCOVER_BASE = "https://esa-worldcover.s3.amazonaws.com/v200/2021/map"
WORLDCOVER_TILES = ["N24W081", "N24W084"]

# JRC Global Surface Water occurrence (Pekel et al. 2016), 1984-2021.
# Landsat-derived, so independent of Sentinel-1.
GSW_URL = (
    "https://storage.googleapis.com/global-surface-water/downloads2021/"
    "occurrence/occurrence_90W_30Nv1_4_2021.tif"
)

# FEMA National Flood Hazard Layer, public ArcGIS REST service.
FEMA_NFHL = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer"
FEMA_FLOOD_HAZARD_LAYER = 28


# ---------------------------------------------------------------------------
# Sentinel-1
# ---------------------------------------------------------------------------

STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Radiometrically terrain-corrected gamma0. Using RTC rather than raw GRD means
# calibration, terrain flattening and geocoding are already applied, and
# applied identically to every scene in the archive -- so none of it becomes a
# per-scene manual step, and none of it can drift between scenes.
S1_COLLECTION = "sentinel-1-rtc"
S1_START = "2023-01-01"
S1_END = "2023-12-31"
S1_POLARIZATION = "vv"

# Water detection.
#
# A fixed dB threshold does not survive contact with this archive. Open water
# is a specular reflector only while it is smooth; wind roughens the surface
# and Bragg scattering lifts the return substantially. Measured over ESA
# WorldCover's permanent-water class, the median VV gamma0 of the same water
# moved from -23.3 dB (27 Aug 2023) to -16.1 dB (6 Jan 2023) -- a 7 dB swing.
# A threshold of -18 dB would have labelled 100% of that water correctly in
# August and 4.8% of it in January.
#
# So the threshold is derived per scene by Otsu's method, which finds the split
# maximising between-class variance in that scene's own histogram.
OTSU_SEARCH_RANGE_DB = (-35.0, 0.0)
OTSU_BINS = 256

# Otsu assumes two modes. A scene clipping the corner of the AOI sees almost
# nothing but ocean, and Otsu then splits water against itself and returns
# nonsense -- measured water recall on such scenes was 30-62%, against 96-99%
# on scenes that see both land and water. These guards reject them.
MIN_SCENE_COVERAGE = 0.40      # fraction of the analysis grid with valid data
MIN_SCENE_LAND_FRACTION = 0.20  # fraction of that data over non-water land cover
MIN_OTSU_SEPARABILITY = 0.70   # between-class variance ratio
OTSU_PLAUSIBLE_RANGE_DB = (-28.0, -6.0)

# Reading every scene at full 10 m resolution would move tens of gigabytes for
# no benefit at a 30 m analysis grid. Scenes are read from the overview level
# closest to this resolution and then resampled onto the analysis grid.
S1_READ_RESOLUTION = 60.0  # metres

# A pixel needs this many accepted observations across the year before its
# water frequency is trusted. Roughly a quarter of the scenes returned by the
# search pass the guards above -- the rest are edge slivers -- so this is set
# against that reduced count, not against the raw scene total.
S1_MIN_OBSERVATIONS = 8


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Terrain derivatives on near-flat ground.
#
# The Everglades falls roughly 3 cm per kilometre. On that gradient a D8 flow
# router is close to undefined: the steepest-descent direction is decided by
# noise in the DEM rather than by topography. The pipeline still computes the
# usual derivatives, but TWI needs a floor on tan(beta), or it divides by
# effectively zero and a handful of cells acquire astronomical values that
# reduce everything else to salt-and-pepper once normalised.
MIN_SLOPE_RAD = 0.001  # ~0.057 degrees

# Cells draining more than this many upstream cells are treated as channels.
# 1000 cells at 30 m is 0.9 km^2.
CHANNEL_ACCUMULATION_CELLS = 1000

# Percentile clip applied before min-max normalisation, so a handful of extreme
# cells cannot compress every other value into the bottom of the range. This is
# the second half of the fix for the speckled output.
NORMALISE_PERCENTILES = (1.0, 99.0)

# Classification of the final index into five susceptibility bands. Quantiles
# rather than equal intervals: on a skewed index, equal intervals leave the
# upper classes nearly empty and the map unreadable.
N_CLASSES = 5
CLASS_LABELS = ["Very low", "Low", "Moderate", "High", "Very high"]
