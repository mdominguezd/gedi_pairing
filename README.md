# GEDI Pairing API (Earth Engine)

A FastAPI service that finds **disturbed GEDI feature-space pairs** inside a
burned-area polygon. You give it a GeoJSON AOI and a fire date; it returns, for
each post-fire GEDI footprint, its most similar pre-fire footprint — matched on
Sentinel-2, Sentinel-1, and topographic features plus spatial proximity.

Imagery features come from **Google Earth Engine**. GEDI shots come from
**gediDB** (GEDI L2A over S3).

---

## What it does

For a burned-area AOI and disturbance date, the `/disturbed_pairs` endpoint:

1. Pulls all GEDI shots in the AOI (gediDB) and labels each as pre- or
   post-fire relative to the disturbance date.
2. Extracts EE features (VV/VH, Sentinel-2 bands, optionally slope/aspect) for
   **every** shot once, in a batch (`precompute_features`).
3. Matches each disturbed (post-fire) shot to its most similar earlier
   undisturbed shot in the weighted feature space (`get_close_fs_pairs_fast`).
4. Returns a GeoJSON `FeatureCollection` of the pairs (geometry = the post-fire
   query shot), with the matched shot id, candidate list, feature vectors, and
   match distance.

---

## Requirements

### Python environment

Create the environment from the provided file (see `gee_gedi_env.yaml`):

```bash
mamba env create -f gee_gedi_env.yaml
mamba activate gee_gedi
```

### Your own modules (NOT installed by the environment)

The API imports code that is **not on PyPI/conda** — it is part of your
chapter-1 project and must be importable:

- `chap1_modules.geedal.geedal_utils` — provides `earthengine_init()` and
  `scale_features()`
- `chap1_modules.geedal.geedal_utils_on_steroids` — provides `extract_ee_values()`
- `chap1_modules.gedi_pairs` provides `pairing_algorithms_enhanced` and
`pairing_algorithms` which provide `precompute_features()` and
  `get_close_fs_pairs_fast()`

Put `pairing_algorithms.py` next to `gee_main.py`, or on the same path.

> Each package folder (`chap1_modules/`, `chap1_modules/geedal/`, …) needs an
> `__init__.py` (may be empty) or the imports will fail with `ModuleNotFoundError`.

### Earth Engine authentication

`geedal_utils.earthengine_init()` runs on each request, but EE credentials must
already exist in the server process. Authenticate once before serving:

```bash
earthengine authenticate          # interactive, or
# configure a service account inside earthengine_init() for headless use
```

Your EE project must have quota — the batch feature extraction issues many
requests.

### GediDB access

gediDB reads GEDI L2A from a public S3 bucket
(`dog-ext.gedidb.gedi-l2-l4-v002.0` at `https://s3.gfz-potsdam.de`). No key is
needed, but the machine must have outbound HTTPS to that host.

---

## Running the API

```bash
uvicorn gee_main:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000/** for the web UI (draw an AOI on the map or
upload/drop a GeoJSON; a `burn_date` in the file auto-fills the fire date), or
call the endpoint directly.

---

## Endpoint: `POST /disturbed_pairs`

### Request body

| Field | Type | Default | Meaning |
|---|---|---|---|
| `geojson` | object | — | Burned-area AOI (FeatureCollection / Feature / geometry) |
| `disturbance_date` | str | — | Fire date, `YYYY-MM-DD` |
| `img_start` / `img_end` | str | `2018-01-01` / `2019-01-01` | Imagery window for EE features |
| `gedi_start` / `gedi_end` | str | `2019-04-01` / `2024-04-01` | GEDI acquisition window |
| `max_distance` | float | 400 | Spatial search radius (m) |
| `weight_geo` | float | 0.75 | Geographic weight (0–1); 1 = spatial-proximity only |
| `use_baseline` | bool | true | Include Sentinel-1 + Sentinel-2 features |
| `use_slope` | bool | true | Include DEM slope/aspect |
| `use_embeddings` | bool | false | Include Google satellite-embedding features |

The projected CRS for distance calculations is **auto-computed** from the AOI
(local UTM zone) — you do not pass it.

### Example

```bash
curl -X POST http://localhost:8000/disturbed_pairs \
  -H "Content-Type: application/json" \
  -d '{
        "geojson": { "type": "Polygon", "coordinates": [[[ -6.1,37.4 ],[ -6.0,37.4 ],[ -6.0,37.5 ],[ -6.1,37.5 ],[ -6.1,37.4 ]]] },
        "disturbance_date": "2022-07-15",
        "img_start": "2021-06-01",
        "img_end": "2021-09-01",
        "max_distance": 400,
        "weight_geo": 0.75
      }'
```

### Response

A GeoJSON `FeatureCollection`. Each feature is a post-fire query shot with:

- `new_shot_num_1` — matched (pre-fire) shot id
- `possible_pairs` — all candidate ids considered (JSON string)
- `s2_feats`, `s1_feats` — standardized feature vectors for query and match (JSON strings)
- `match_dist` — distance in the weighted feature space
- geometry — the post-fire query footprint

---

## Calling it in a loop (Python)

```python
import requests, json, geopandas as gpd

gj = json.load(open("test_0.geojson"))
# burn_date lives in the geojson properties in this project:
burn = gj["features"][0]["properties"]["burn_date"]

r = requests.post("http://localhost:8000/disturbed_pairs",
                  json={"geojson": gj, "disturbance_date": str(burn)[:10]},
                  timeout=1800)          # EE extraction is slow
pairs = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
```

---

## Notes & gotchas

- **Slow requests.** Each call runs a full EE extraction; large AOIs take
  minutes. Use a generous client timeout.
- **Fire date must sit inside the GEDI window.** If `gedi_start` is after the
  fire, there are no pre-disturbance candidates and you get zero pairs.
- **Auto-CRS vs. old runs.** Distances use the AOI's local UTM zone. If you are
  reproducing an older run that used a fixed CRS, results will differ — match
  that run's CRS instead.
- **Feature scaling.** Standardization uses `geedal_utils.scale_features` (a
  fixed scaler). The fast matcher is equivalent to the per-query version only
  because that scaler is fixed; if it ever becomes a per-call fit, results
  change.
- **String shot ids.** GEDI shot numbers are 19-digit integers; the API returns
  them as strings. When comparing against parquet outputs, cast both sides to
  `str` (parquet may store them as `uint64`).