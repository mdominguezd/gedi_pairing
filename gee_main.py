# gee_main.py — standalone Earth-Engine pairing API
#
# Give it a burned-area GeoJSON + a disturbance (fire) date, it returns the
# DISTURBED feature-space pairs from your chapter-1 GEE pipeline.
#
# REQUIRES (must be importable on the PYTHONPATH, NOT provided here):
#   chap1_modules.geedal.geedal_utils              (earthengine_init, scale_features)
#   chap1_modules.geedal.geedal_utils_on_steroids  (extract_ee_values, ...)
#   pairing_algorithms.get_close_fs_pairs          (your uploaded pairing module)
# plus: earthengine-api, geemap, gedidb, geopandas, xarray, fastapi, uvicorn
#
# EE auth: this calls geedal_utils.earthengine_init(); make sure that function
# (and your EE credentials / project) work before serving.

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import shape

import ee
import gedidb as gdb

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --- your uploaded modules (must be on PYTHONPATH) ---
from chap1_modules.geedal import geedal_utils            # earthengine_init, scale_features
from chap1_modules.gedi_pairs import pairing_algorithms_enhanced 

app = FastAPI(title="GEDI pairing (GEE)")


# ─────────────────────────── helpers ───────────────────────────
def _to_gdf(geojson: dict) -> gpd.GeoDataFrame:
    t = geojson.get("type")
    if t == "FeatureCollection":
        return gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
    if t == "Feature":
        return gpd.GeoDataFrame(geometry=[shape(geojson["geometry"])], crs="EPSG:4326")
    return gpd.GeoDataFrame(geometry=[shape(geojson)], crs="EPSG:4326")


def _get_gedi(geojson, start_time, end_time, buffer_m=400, bbox=None,
              variables=("agbd", "rh:98")):
    """Pull GEDI shots via gedidb. If `bbox` (minx,miny,maxx,maxy in EPSG:4326)
    is given, query that box directly and skip the internal buffer."""
    from shapely.geometry import box
    if bbox is not None:
        roi = gpd.GeoDataFrame(geometry=[box(*bbox)], crs="EPSG:4326")
    else:
        gdf = _to_gdf(geojson)
        roi = gdf.to_crs(gdf.estimate_utm_crs()).buffer(buffer_m).to_crs("EPSG:4326")
        roi = gpd.GeoDataFrame(geometry=roi, crs="EPSG:4326")
    provider = gdb.GEDIProvider(
        storage_type="s3",
        s3_bucket="dog-ext.gedidb.gedi-l2-l4-v002.0",
        url="https://s3.gfz-potsdam.de",
    )
    shots = provider.get_data(
        variables=list(variables), query_type="bounding_box", geometry=roi,
        start_time=start_time, end_time=end_time, return_type="dataframe",
    )
    if shots is None or len(shots) == 0:
        return None
    return gpd.GeoDataFrame(
        shots,
        geometry=gpd.points_from_xy(shots["longitude"], shots["latitude"]),
        crs="EPSG:4326",
    )


def _to_ds(shots_gdf, idx="shot_number"):
    """Build the xarray candidate pool `ds` that get_close_fs_pairs expects:
    indexed by shot_number, with longitude/latitude/time coords/vars."""
    df = shots_gdf.drop(columns="geometry").copy()
    df[idx] = df[idx].astype("uint64")           # pairing does np.uint64(...) lookups
    df = df.set_index(idx)
    return df.to_xarray()


# ─────────────────────────── API ───────────────────────────
class PairRequest(BaseModel):
    geojson: dict                          # burned-area polygon
    disturbance_date: str                  # fire date, YYYY-MM-DD
    img_start: str = "2018-01-01"          # imagery window for EE features
    img_end: str = "2019-01-01"
    gedi_start: str = "2019-04-01"
    gedi_end: str | None = "2024-04-01"
    max_distance: float = 400
    weight_geo: float = 0.75
    crs: str = "EPSG:25830"
    use_baseline: bool = True
    use_slope: bool = True
    use_embeddings: bool = False
    n_jobs: int = 1


@app.post("/disturbed_pairs")
def disturbed_pairs(req: PairRequest):
    geedal_utils.earthengine_init()
    end = req.gedi_end or pd.Timestamp.now().strftime("%Y-%m-%d")

    aoi = _to_gdf(req.geojson)
    bbox = tuple(aoi.total_bounds)                  # (minx, miny, maxx, maxy) in EPSG:4326
    shots = _get_gedi(req.geojson, req.gedi_start, end, bbox=bbox)
    
    if shots is None or len(shots) < 2:
        return {"type": "FeatureCollection", "features": []}

    # label + build the columns the pairing needs
    fire = pd.Timestamp(req.disturbance_date)
    shots["time"] = pd.to_datetime(shots["time"])
    shots["disturbed"] = shots["time"] > fire
    shots["dist"] = fire                            # per-shot fire date (single event here)
    shots["shot_num_2"] = shots["shot_number"].astype("uint64")

    # query set = disturbed (post-fire) shots; candidate pool = ALL shots (feat_gdf)
    shots_2 = shots[shots["disturbed"]].copy().reset_index(drop=True)
    if len(shots_2) == 0:
        return {"type": "FeatureCollection", "features": []}

    # PHASE 1 — extract EE features for EVERY shot ONCE (batched, not per-query)
    feat_gdf, feat_cols = pairing_algorithms_enhanced.precompute_features(
        shots, dates=[req.img_start, req.img_end],
        use_baseline=req.use_baseline, use_slope=req.use_slope,
        use_embeddings=req.use_embeddings,
    )
    # PHASE 2 — match disturbed (post-fire) queries in memory, no EE in the loop
    paired = pairing_algorithms_enhanced.get_close_fs_pairs_fast(
        feat_gdf, feat_cols, shots_2=shots_2,
        crs=req.crs, max_distance=req.max_distance, weight_geo=req.weight_geo,
        disturbed=True, use_baseline=req.use_baseline, use_slope=req.use_slope,
    )

    # keep matched queries; serialize dict/list columns for GeoJSON
    paired = paired[paired["new_shot_num_1"].notna()].copy()
    if len(paired) == 0:
        return {"type": "FeatureCollection", "features": []}
    for c in ["possible_pairs", "s2_feats", "s1_feats", "all_pos_feats"]:
        if c in paired.columns:
            paired[c] = paired[c].apply(lambda v: json.dumps(v, default=str) if v is not None else None)
    for c in paired.columns:
        if pd.api.types.is_datetime64_any_dtype(paired[c]):
            paired[c] = paired[c].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    paired["new_shot_num_1"] = paired["new_shot_num_1"].astype(str)
    if "shot_number" in paired.columns:
        paired["shot_number"] = paired["shot_number"].astype(str)
    return json.loads(paired.to_json())


@app.get("/", response_class=HTMLResponse)
def landing():
    return """
<!doctype html><html><head><meta charset="utf-8"><title>GEDI pairing (GEE)</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css"/>
<style>
 :root{--bg:#0f1720;--card:#17212b;--line:#2a3743;--ink:#e6edf3;--mut:#8aa0b2;--acc:#2ea86f}
 *{box-sizing:border-box}body{font-family:system-ui,sans-serif;margin:0;background:var(--bg);color:var(--ink)}
 .wrap{max-width:820px;margin:0 auto;padding:1.5rem 1rem 3rem}h1{font-size:1.3rem}
 .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:1.1rem;margin-bottom:1rem}
 label{display:block;margin:.5rem 0 .2rem;font-size:.8rem;font-weight:600;color:var(--mut)}
 input,textarea{width:100%;padding:.5rem;border:1px solid var(--line);border-radius:8px;background:#0d141c;color:var(--ink)}
 textarea{min-height:80px;font-family:monospace;font-size:.78rem}
 .row{display:flex;gap:.9rem;flex-wrap:wrap}.row>div{flex:1;min-width:130px}
 #map{height:320px;border-radius:10px;border:1px solid var(--line)}
 .tabs{display:flex;gap:.4rem;margin:.5rem 0}.tab{padding:.35rem .8rem;border:1px solid var(--line);border-radius:999px;cursor:pointer;font-size:.8rem;color:var(--mut)}.tab.on{background:var(--acc);color:#fff;border-color:var(--acc)}
 button{margin-top:1rem;padding:.6rem 1.15rem;border:0;border-radius:9px;background:var(--acc);color:#fff;font-weight:600;cursor:pointer}
 button.ghost{background:transparent;border:1px solid var(--line);color:var(--ink)}
 .hide{display:none}.mut{color:var(--mut);font-size:.8rem}.ok{color:var(--acc)}
 pre{background:#0d141c;border:1px solid var(--line);padding:.9rem;border-radius:9px;font-size:.8rem;white-space:pre-wrap}
</style></head><body><div class="wrap">
<h1>🔥 GEDI disturbed-pair finder (GEE)</h1>
<div class="card">
  <label>Burned area</label>
  <div class="tabs"><div class="tab on" data-t="map">Draw</div><div class="tab" data-t="paste">GeoJSON</div></div>
  <div id="pane-map"><div id="map"></div></div>
  <div id="pane-paste" class="hide"><textarea id="gjtext" placeholder='{"type":"Polygon",...}'></textarea></div>
  <div id="aoi-state" class="mut">No AOI set.</div>
</div>
<div class="card">
  <label>Fire / disturbance date</label><input id="disturbance_date" placeholder="YYYY-MM-DD" value="2018-07-01">
  <div class="row">
    <div><label>Imagery start</label><input id="img_start" value="2018-01-01"></div>
    <div><label>Imagery end</label><input id="img_end" value="2019-01-01"></div>
  </div>
  <div class="row">
    <div><label>Max distance (m)</label><input id="max_distance" type="number" value="400"></div>
    <div><label>weight_geo</label><input id="weight_geo" type="number" step="0.05" value="0.75"></div>
  </div>
  <button id="run" disabled>Find disturbed pairs</button>
  <button id="dl" class="ghost hide">⬇ Download GeoJSON</button>
  <span id="status" class="mut"></span>
  <pre id="out" class="hide"></pre>
</div></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
<script>
let aoi=null,last=null;const $=id=>document.getElementById(id);
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.remove('on'));t.classList.add('on');const m=t.dataset.t==='map';$('pane-map').classList.toggle('hide',!m);$('pane-paste').classList.toggle('hide',m);if(m)setTimeout(()=>map.invalidateSize(),50);});
const map=L.map('map').setView([37.5,-6.0],6);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:19,attribution:'Esri'}).addTo(map);
const drawn=new L.FeatureGroup().addTo(map);
map.addControl(new L.Control.Draw({edit:{featureGroup:drawn},draw:{polygon:true,rectangle:true,marker:false,polyline:false,circle:false,circlemarker:false}}));
function setAOI(g,s){aoi=g;$('aoi-state').innerHTML='<span class="ok">✓ AOI set ('+s+')</span>';$('run').disabled=false;}
map.on(L.Draw.Event.CREATED,e=>{drawn.clearLayers();drawn.addLayer(e.layer);setAOI(e.layer.toGeoJSON().geometry,'drawn');});
$('gjtext').addEventListener('input',()=>{try{setAOI(JSON.parse($('gjtext').value),'geojson');}catch(e){}});
$('run').onclick=async()=>{
  if(!aoi)return;
  const body={geojson:aoi,disturbance_date:$('disturbance_date').value,
    img_start:$('img_start').value,img_end:$('img_end').value,
    max_distance:parseFloat($('max_distance').value),weight_geo:parseFloat($('weight_geo').value)};
  $('run').disabled=true;$('dl').classList.add('hide');$('out').classList.remove('hide');
  $('status').textContent='Running EE extraction… (can take minutes)';$('out').textContent='';
  const t0=performance.now();
  try{
    const r=await fetch('/disturbed_pairs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();last=d;
    const f=d.features||[];
    $('out').textContent='disturbed pairs: '+f.length+'   ·   '+((performance.now()-t0)/1000).toFixed(1)+'s';
    $('status').textContent='Done.';if(f.length)$('dl').classList.remove('hide');
  }catch(e){$('out').textContent='Error: '+e;$('status').textContent='';}
  $('run').disabled=false;
};
$('dl').onclick=()=>{if(!last)return;const b=new Blob([JSON.stringify(last)],{type:'application/geo+json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='disturbed_pairs.geojson';a.click();URL.revokeObjectURL(a.href);};
</script></body></html>
"""