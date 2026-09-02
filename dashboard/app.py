"""dashboard/app.py
================
HydroShield Intelligence — Flood Early-Warning Command Dashboard
Mission-Control GIS Platform for the SIH Flood Nowcasting System.

Run with:
    streamlit run dashboard/app.py
"""

import json
import time
import urllib.error
import urllib.request
from typing import Optional

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Page Config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HydroShield Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE    = "http://localhost:8000"
CENTER_LAT  = 25.5941
CENTER_LON  = 85.1376

SHELTERS = [
    {"name": "AIIMS Patna",                   "lat": 25.5606, "lon": 85.0441, "capacity": 800,  "route": "NH-30 via Phulwari"},
    {"name": "Gandhi Maidan Shelter Camp",    "lat": 25.6178, "lon": 85.1415, "capacity": 3500, "route": "Ashok Rajpath"},
    {"name": "Nalanda Medical College",       "lat": 25.5997, "lon": 85.1843, "capacity": 600,  "route": "Bailey Road East"},
    {"name": "Patna Junction Logistics Hub",  "lat": 25.6022, "lon": 85.1375, "capacity": 2000, "route": "Station Road (Priority Corridor)"},
]

RISK_STYLES = {
    "HIGH":   {"fillColor": "#EF4444", "color": "#DC2626", "fillOpacity": 0.65, "weight": 2},
    "MEDIUM": {"fillColor": "#F59E0B", "color": "#D97706", "fillOpacity": 0.50, "weight": 1.5},
    "LOW":    {"fillColor": "#10B981", "color": "#059669", "fillOpacity": 0.20, "weight": 1},
}

# ---------------------------------------------------------------------------
# Custom CSS — Mission Operations Theme & Tactical Dark Map Inversion
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Global canvas ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0B0F17 !important;
    font-family: 'Inter', sans-serif !important;
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] {
    background: #0D1321 !important;
    border-right: 1px solid #1E2D45 !important;
    display: block !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"], button[kind="header"] {
    display: block !important;
    z-index: 999999 !important;
    color: #06B6D4 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stDeployButton"],
[data-testid="stDecoration"], [data-testid="stToolbar"] { display: none !important; }

/* ── Hardware-accelerated tactical dark mode for standard OSM tiles ── */
.leaflet-container {
    background: #0B0F17 !important;
}
.leaflet-tile-pane {
    filter: brightness(0.6) invert(1) contrast(3) hue-rotate(200deg) saturate(0.3) brightness(0.7) !important;
}

/* ── Header bar ── */
.hs-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F172A 100%);
    border: 1px solid #2A364F;
    border-radius: 12px;
    padding: 18px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 24px rgba(6, 182, 212, 0.08);
}
.hs-brand { font-size: 1.35rem; font-weight: 700; letter-spacing: 0.05em;
            color: #F8FAFC; font-family: 'JetBrains Mono', monospace; }
.hs-brand span { color: #06B6D4; }
.hs-sector { font-size: 0.72rem; color: #64748B; letter-spacing: 0.08em;
             margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
.pill-live { display:inline-flex; align-items:center; gap:6px; padding:6px 14px;
             border-radius:20px; background:#0D2818; border:1px solid #10B981;
             font-size:0.78rem; color:#10B981; font-weight:600; }
.pill-dead { background:#2D0A0A; border-color:#EF4444; color:#EF4444; }

/* ── KPI Metric Cards ── */
.kpi-card {
    background: #161F30;
    border: 1px solid #2A364F;
    border-radius: 12px;
    padding: 20px 22px 16px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 16px rgba(0,0,0,0.4);
    height: 110px;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.kpi-red::before    { background: linear-gradient(90deg, #EF4444, #DC2626); box-shadow: 0 0 12px #EF444466; }
.kpi-amber::before  { background: linear-gradient(90deg, #F59E0B, #D97706); box-shadow: 0 0 12px #F59E0B55; }
.kpi-cyan::before   { background: linear-gradient(90deg, #06B6D4, #0891B2); box-shadow: 0 0 12px #06B6D455; }
.kpi-green::before  { background: linear-gradient(90deg, #10B981, #059669); box-shadow: 0 0 12px #10B98155; }
.kpi-label  { font-size: 0.70rem; color: #64748B; text-transform: uppercase;
              letter-spacing: 0.10em; font-weight: 600; margin-bottom: 8px; }
.kpi-value  { font-size: 1.80rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; line-height:1; }
.kpi-value.red    { color: #EF4444; text-shadow: 0 0 18px #EF444444; }
.kpi-value.amber  { color: #F59E0B; }
.kpi-value.cyan   { color: #06B6D4; }
.kpi-value.green  { color: #10B981; }
.kpi-sub { font-size: 0.68rem; color: #475569; margin-top: 4px; }

/* ── Alert cards (dispatch feed) ── */
.alert-card {
    background: #161F30;
    border: 1px solid #2A364F;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    border-left: 3px solid;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.alert-card.high   { border-left-color: #EF4444; }
.alert-card.medium { border-left-color: #F59E0B; }
.badge-critical { display:inline-block; background:#7F1D1D; color:#FCA5A5;
                  border:1px solid #EF4444; border-radius:4px; padding:2px 8px;
                  font-size:0.65rem; font-weight:700; letter-spacing:0.08em; margin-bottom:6px; }
.badge-advisory { display:inline-block; background:#451A03; color:#FCD34D;
                  border:1px solid #F59E0B; border-radius:4px; padding:2px 8px;
                  font-size:0.65rem; font-weight:700; letter-spacing:0.08em; margin-bottom:6px; }
.alert-cell { font-size:0.82rem; font-weight:600; color:#E2E8F0; margin-bottom:3px; }
.alert-ttf  { font-size:0.75rem; color:#94A3B8; margin-bottom:6px; }
.alert-action { font-size:0.72rem; color:#CBD5E1; background:#1E2D45;
                border-radius:6px; padding:6px 10px; margin-top:4px; }

/* ── Glassmorphic error banner ── */
.error-banner {
    background: rgba(127, 29, 29, 0.25);
    backdrop-filter: blur(12px);
    border: 1px solid #EF4444;
    border-radius: 12px;
    padding: 20px 24px;
    color: #FCA5A5;
    font-size: 0.90rem;
    margin: 16px 0;
}
.error-banner code { background: #2D0A0A; border-radius:4px; padding:2px 8px;
                     color:#F87171; font-family:'JetBrains Mono',monospace; font-size:0.82rem; }

/* ── Sidebar typography ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label { color: #94A3B8 !important; font-size:0.82rem; }
[data-testid="stSidebar"] h3 { color: #E2E8F0 !important; font-size:0.82rem;
    text-transform: uppercase; letter-spacing: 0.12em; border-bottom: 1px solid #1E2D45;
    padding-bottom: 6px; margin-bottom: 12px; }

/* ── Execute button ── */
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
    color: #F8FAFC !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em !important;
    width: 100% !important;
    box-shadow: 0 0 0 0 #2563EB44 !important;
    transition: box-shadow 0.2s, transform 0.1s !important;
}
.stButton > button:hover {
    box-shadow: 0 0 20px #2563EB88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background: #1E2D45 !important;
    color: #06B6D4 !important;
    border: 1px solid #06B6D4 !important;
    border-radius: 8px !important;
    width: 100% !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [role="slider"] {
    background: #2563EB !important;
    box-shadow: 0 0 10px #2563EB88 !important;
}

/* ── Scrollable dispatch feed ── */
.dispatch-scroll { max-height: 580px; overflow-y: auto; padding-right:4px; }
.dispatch-scroll::-webkit-scrollbar { width: 4px; }
.dispatch-scroll::-webkit-scrollbar-track { background: #0B0F17; }
.dispatch-scroll::-webkit-scrollbar-thumb { background: #2A364F; border-radius: 2px; }

/* ── Section separator ── */
.section-sep { border-top: 1px solid #1E2D45; margin: 14px 0 10px; }

/* ── Map section label ── */
.map-label { font-size:0.70rem; color:#475569; text-transform:uppercase;
             letter-spacing:0.10em; margin-bottom:6px; font-weight:600; }
.map-wrapper { border: 1px solid #2A364F; border-radius: 12px; overflow: hidden;
               box-shadow: 0 4px 24px rgba(0,0,0,0.5); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def api_health() -> Optional[dict]:
    try:
        with urllib.request.urlopen(f"{API_BASE}/api/v1/health", timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return None


def api_simulate(scenario_mm: float) -> Optional[dict]:
    try:
        body = json.dumps({"rainfall_scenario_mm": scenario_mm}).encode()
        req  = urllib.request.Request(
            f"{API_BASE}/api/v1/simulate", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Folium map builder (Direct OpenStreetMap + Zero CARTO Watermark)
# ---------------------------------------------------------------------------
def build_map(
    geojson_data:    Optional[dict],
    show_shelters:   bool,
    show_inundation: bool,
) -> folium.Map:
    m = folium.Map(
        location=[CENTER_LAT, CENTER_LON],
        zoom_start=13,
        tiles=None,          # Step 1: suppress ALL default tile layers
    )
    
    # Step 2: explicit raster_layers.TileLayer — correct folium submodule
    # overlay=False + control=False makes it the only, un-removable base layer.
    folium.raster_layers.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri Satellite",
        max_zoom=18,
        overlay=False,
        control=False,
    ).add_to(m)

    # ── Esri reference overlays: place names + road network ───────────────
    # These transparent PNG layers sit above the satellite imagery and render
    # city names, district labels, landmarks, roads, and highways without
    # obscuring the underlying imagery.
    folium.raster_layers.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Labels",
        name="Labels & Places",
        overlay=True,
        control=False,
    ).add_to(m)
    folium.raster_layers.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Roads",
        name="Road Network",
        overlay=True,
        control=False,
    ).add_to(m)

    # Step 3a: JS sweep — remove any cartocdn layer injected by streamlit-folium
    # after DOM load. Two timeouts (150ms + 600ms) cover both the initial render
    # and any deferred re-render streamlit-folium does on iframe resize.
    from branca.element import Element
    purge_js = Element("""
<script>
document.addEventListener("DOMContentLoaded", function () {
    function purgeCarto() {
        document.querySelectorAll('.folium-map').forEach(function (mapEl) {
            var inst = window[mapEl.id];
            if (inst && inst.eachLayer) {
                inst.eachLayer(function (layer) {
                    if (layer._url &&
                        (layer._url.indexOf('cartocdn') !== -1 ||
                         layer._url.indexOf('carto.com') !== -1)) {
                        inst.removeLayer(layer);
                    }
                });
            }
        });
    }
    setTimeout(purgeCarto, 150);
    setTimeout(purgeCarto, 600);
});
</script>
""")
    m.get_root().html.add_child(purge_js)

    # Step 3b: CSS — hide CARTO attribution links without nuking the whole bar
    carto_css = Element("""
<style>
.leaflet-control-attribution a[href*="carto"],
.leaflet-control-attribution a[href*="CARTO"] { display: none !important; }
</style>
""")
    m.get_root().header.add_child(carto_css)

    # ── GeoJSON risk polygons ──────────────────────────────────────────────
    if geojson_data:
        grid_group = folium.FeatureGroup(name="Risk Grid", show=True)
        for feat in geojson_data.get("features", []):
            props = feat.get("properties", {})
            level = props.get("alert_level", "LOW")
            style = RISK_STYLES.get(level, RISK_STYLES["LOW"])

            ttf = props.get("time_to_flood_hrs")
            ttf_str = f"{ttf} hrs" if ttf is not None else "—"
            risk_score = props.get("risk_score", 0)
            if risk_score is None:
                risk_score = 0

            tooltip_html = f"""
            <div style='font-family:Inter,sans-serif;font-size:12px;
                        background:#161F30;color:#E2E8F0;
                        padding:10px 14px;border-radius:8px;
                        border:1px solid #2A364F;min-width:200px;'>
              <b style='color:#06B6D4;font-size:13px'>{props.get("cell_id","—")}</b>
              <hr style='border-color:#2A364F;margin:6px 0'>
              <table style='width:100%;border-collapse:collapse;'>
                <tr><td style='color:#64748B;padding:2px 0'>Risk Index</td>
                    <td style='text-align:right;font-weight:700;
                        color:{"#EF4444" if level=="HIGH" else "#F59E0B" if level=="MEDIUM" else "#10B981"}'>
                        {risk_score:.1f} / 100</td></tr>
                <tr><td style='color:#64748B;padding:2px 0'>Alert Level</td>
                    <td style='text-align:right;font-weight:600;
                        color:{"#EF4444" if level=="HIGH" else "#F59E0B" if level=="MEDIUM" else "#10B981"}'>
                        {level}</td></tr>
                <tr><td style='color:#64748B;padding:2px 0'>Elevation</td>
                    <td style='text-align:right'>{props.get("elevation_m","—")} m</td></tr>
                <tr><td style='color:#64748B;padding:2px 0'>Slope</td>
                    <td style='text-align:right'>{props.get("slope_deg","—")}°</td></tr>
                <tr><td style='color:#64748B;padding:2px 0'>Dist. to Ganges</td>
                    <td style='text-align:right'>{props.get("dist_to_river_m","—")} m</td></tr>
                <tr><td style='color:#64748B;padding:2px 0'>Drainage Density</td>
                    <td style='text-align:right'>{props.get("drainage_density","—")} km/km²</td></tr>
                <tr><td style='color:#64748B;padding:2px 0'>Time to Flood</td>
                    <td style='text-align:right;color:#F59E0B;font-weight:600'>{ttf_str}</td></tr>
              </table>
            </div>"""

            popup_html = f"""
            <div style='font-family:Inter,sans-serif;font-size:12px;
                        background:#0F172A;color:#E2E8F0;
                        padding:14px 18px;border-radius:10px;
                        border:1px solid #2A364F;min-width:240px;
                        box-shadow:0 8px 24px rgba(0,0,0,0.6)'>
              <div style='font-size:15px;font-weight:700;color:#06B6D4;margin-bottom:8px'>
                📍 {props.get("cell_id","—")}
              </div>
              <div style='background:{"#7F1D1D" if level=="HIGH" else "#451A03" if level=="MEDIUM" else "#052e16"};
                          color:{"#FCA5A5" if level=="HIGH" else "#FCD34D" if level=="MEDIUM" else "#6EE7B7"};
                          border-radius:6px;padding:4px 10px;display:inline-block;
                          font-size:11px;font-weight:700;letter-spacing:0.08em;margin-bottom:10px'>
                {"🔴 CRITICAL ALERT" if level=="HIGH" else "🟡 ADVISORY" if level=="MEDIUM" else "🟢 NOMINAL"}
              </div>
              <table style='width:100%;border-collapse:collapse;font-size:11.5px'>
                <tr style='border-bottom:1px solid #1E2D45'>
                  <td style='color:#64748B;padding:4px 0'>Dynamic Risk Index</td>
                  <td style='text-align:right;font-weight:700;font-size:14px;
                      color:{"#EF4444" if level=="HIGH" else "#F59E0B" if level=="MEDIUM" else "#10B981"}'>
                      {risk_score:.1f}</td></tr>
                <tr><td style='color:#64748B;padding:4px 0'>Elevation</td>
                    <td style='text-align:right'>{props.get("elevation_m","—")} m</td></tr>
                <tr><td style='color:#64748B;padding:4px 0'>Slope Gradient</td>
                    <td style='text-align:right'>{props.get("slope_deg","—")}°</td></tr>
                <tr><td style='color:#64748B;padding:4px 0'>Dist. to Ganges River</td>
                    <td style='text-align:right'>{props.get("dist_to_river_m","—")} m</td></tr>
                <tr><td style='color:#64748B;padding:4px 0'>Drainage Density</td>
                    <td style='text-align:right'>{props.get("drainage_density","—")} km/km²</td></tr>
                <tr><td style='color:#64748B;padding:4px 0'>1h Rainfall</td>
                    <td style='text-align:right'>{round(props.get("rainfall_1h_mm",0),2) if props.get("rainfall_1h_mm") is not None else "—"} mm</td></tr>
                <tr><td style='color:#64748B;padding:4px 0'>Historical Floods</td>
                    <td style='text-align:right'>{props.get("historical_flood_count","—")}</td></tr>
                <tr style='border-top:1px solid #1E2D45'>
                  <td style='color:#F59E0B;padding:4px 0;font-weight:600'>Est. Time to Flood</td>
                  <td style='text-align:right;color:#F59E0B;font-weight:700'>{ttf_str}</td></tr>
              </table>
            </div>"""

            folium.GeoJson(
                feat,
                style_function=lambda f, s=style: {
                    "fillColor":   s["fillColor"],
                    "color":       s["color"],
                    "fillOpacity": s["fillOpacity"],
                    "weight":      s["weight"],
                },
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(grid_group)

        grid_group.add_to(m)

    # ── Force frame around the Patna 64-cell grid ───────────────
    m.fit_bounds([[25.55, 85.08], [25.65, 85.20]])

    # ── River inundation buffer ────────────────────────────────────────────
    if show_inundation:
        folium.CircleMarker(
            location=[CENTER_LAT + 0.042, CENTER_LON],
            radius=180, color="#06B6D4", weight=2,
            fill=True, fill_color="#06B6D4", fill_opacity=0.08,
            tooltip="<b style='color:#06B6D4'>Ganges River Inundation Buffer Zone</b>",
        ).add_to(m)
        folium.PolyLine(
            locations=[
                [CENTER_LAT + 0.038, CENTER_LON - 0.09],
                [CENTER_LAT + 0.041, CENTER_LON - 0.04],
                [CENTER_LAT + 0.043, CENTER_LON + 0.00],
                [CENTER_LAT + 0.045, CENTER_LON + 0.05],
                [CENTER_LAT + 0.042, CENTER_LON + 0.09],
            ],
            color="#06B6D4", weight=3, opacity=0.7,
            tooltip="<b style='color:#06B6D4'>Ganges River Corridor</b>",
            dash_array="8 4",
        ).add_to(m)

    # ── Shelter markers ────────────────────────────────────────────────────
    if show_shelters:
        for s in SHELTERS:
            popup_html = f"""
            <div style='font-family:Inter,sans-serif;font-size:12px;
                        background:#0F172A;color:#E2E8F0;
                        padding:12px 16px;border-radius:8px;
                        border:1px solid #10B981;min-width:220px'>
              <div style='font-size:13px;font-weight:700;color:#10B981;margin-bottom:6px'>
                🏥 {s["name"]}
              </div>
              <div style='color:#94A3B8;font-size:11px;margin-bottom:4px'>
                Capacity: <b style='color:#F8FAFC'>{s["capacity"]:,} persons</b>
              </div>
              <div style='color:#94A3B8;font-size:11px'>
                Route: <b style='color:#06B6D4'>{s["route"]}</b>
              </div>
              <div style='margin-top:8px;background:#052e16;border-radius:4px;
                          padding:4px 8px;font-size:10px;color:#6EE7B7;font-weight:600'>
                ✅ OPERATIONAL — Priority Evacuation Node
              </div>
            </div>"""

            folium.Marker(
                location=[s["lat"], s["lon"]],
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"🏥 {s['name']}",
                icon=folium.Icon(color="green", icon="plus-sign", prefix="glyphicon"),
            ).add_to(m)

    return m


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

# ── Health check ──────────────────────────────────────────────────────────
health_data = api_health()
api_live    = health_data is not None

# ── Header bar ────────────────────────────────────────────────────────────
pill_class = "pill-live" if api_live else "pill-dead"
pill_text  = "🟢 LIVE FEED CONNECTED" if api_live else "🔴 API DISCONNECTED"
st.markdown(f"""
<div class='hs-header'>
  <div>
    <div class='hs-brand'>🌊 HYDRO<span>SHIELD</span> INTELLIGENCE // FLOOD EARLY-WARNING COMMAND</div>
    <div class='hs-sector'>SECTOR: PATNA URBAN &amp; GANGES RIVERFRONT CORRIDOR [25.5941° N, 85.1376° E]</div>
  </div>
  <div class='{pill_class}'>{pill_text}</div>
</div>
""", unsafe_allow_html=True)

# ── API offline banner ─────────────────────────────────────────────────────
if not api_live:
    st.markdown(f"""
    <div class='error-banner'>
      ⚠️ <b>FastAPI Engine Unreachable</b> on <code>{API_BASE}</code><br>
      Run <code>uvicorn server.api:app --reload</code> from the project root to restore dynamic telemetry.
      Grid and simulation features are disabled until the backend is online.
    </div>
    """, unsafe_allow_html=True)

# ── Simulation Controls ────────────────────────────────────────────────────
with st.expander("⚙ Dynamic Simulation & Sensor Telemetry Override", expanded=True):
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        scenario_mm = st.slider("Precipitation Surge Intensity (mm/hr)", min_value=0, max_value=250, value=190, step=5)
    with c2:
        hydro_model = st.selectbox("Hydrological Model", ["Standard Monsoon Runoff (Baseline)", "Severe Cloudburst Event (+6h Forecast)", "Dam Breach / River Swell Surge (+24h Multi-Day)"])
    with c3:
        st.write("") # vertical spacing
        st.write("")
        execute = st.button("⚡ EXECUTE INFERENCE", disabled=not api_live, use_container_width=True)

show_shelters = True
show_inundation = True

# ── Session state: last simulation result ──────────────────────────────────
if "sim_result" not in st.session_state:
    st.session_state.sim_result = None

if execute and api_live:
    with st.spinner("🔄 Running inference pipeline..."):
        result = api_simulate(float(scenario_mm))
    if result:
        st.session_state.sim_result = result
    else:
        st.error("Simulation request failed. Check API logs.")

sim = st.session_state.sim_result

# ── KPI Row ────────────────────────────────────────────────────────────────
critical_count = sim["critical_cells_count"]   if sim else 0
medium_count   = sim["medium_cells_count"]     if sim else 0
basin_mm       = sim["scenario_intensity_mm"]  if sim else float(scenario_mm)

# Earliest inundation horizon
earliest_ttf = "Nominal (>12h)"
if sim and sim.get("alerts"):
    valid_ttf = [a["time_to_flood_hrs"] for a in sim["alerts"]
                 if a.get("time_to_flood_hrs") is not None]
    if valid_ttf:
        earliest_ttf = f"{min(valid_ttf):.1f} hrs Window"

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class='kpi-card kpi-red'>
      <div class='kpi-label'>Critical Alert Zones</div>
      <div class='kpi-value red'>{critical_count}</div>
      <div class='kpi-sub'>HIGH risk cells</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class='kpi-card kpi-amber'>
      <div class='kpi-label'>Moderate Vulnerability Zones</div>
      <div class='kpi-value amber'>{medium_count}</div>
      <div class='kpi-sub'>MEDIUM risk cells</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class='kpi-card kpi-cyan'>
      <div class='kpi-label'>Simulated Basin Precipitation</div>
      <div class='kpi-value cyan'>{basin_mm:.1f}</div>
      <div class='kpi-sub'>mm / hr scenario</div>
    </div>""", unsafe_allow_html=True)
with k4:
    ttf_color = "red" if critical_count > 0 else "amber" if medium_count > 0 else "green"
    ttf_accent = "kpi-red" if critical_count > 0 else "kpi-amber" if medium_count > 0 else "kpi-green"
    st.markdown(f"""
    <div class='kpi-card {ttf_accent}'>
      <div class='kpi-label'>Earliest Inundation Horizon</div>
      <div class='kpi-value {ttf_color}' style='font-size:1.25rem;padding-top:4px'>{earliest_ttf}</div>
      <div class='kpi-sub'>time-to-flood estimate</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ── Main split: Map (70%) | Dispatch Feed (30%) ───────────────────────────
map_col, feed_col = st.columns([7, 3], gap="medium")

with map_col:
    st.markdown("<div class='map-label'>🗺 GIS Command Map — Patna Flood Risk Grid</div>",
                unsafe_allow_html=True)
    geojson_data = sim["grid_geojson"] if sim else None
    if not geojson_data and api_live:
        try:
            with urllib.request.urlopen(f"{API_BASE}/api/v1/grid", timeout=5) as r:
                geojson_data = json.loads(r.read())
        except Exception:
            pass

    m = build_map(geojson_data, show_shelters, show_inundation)

    st.markdown("<div class='map-wrapper'>", unsafe_allow_html=True)
    st_folium(m, width=None, height=620, returned_objects=[])
    st.markdown("</div>", unsafe_allow_html=True)

    # Legend
    st.markdown("""
    <div style='display:flex;gap:20px;margin-top:10px;font-size:0.72rem;color:#64748B;'>
      <span><span style='color:#EF4444'>■</span> HIGH Risk (&ge;70)</span>
      <span><span style='color:#F59E0B'>■</span> MEDIUM Risk (40–70)</span>
      <span><span style='color:#10B981'>■</span> LOW Risk (&lt;40)</span>
      <span><span style='color:#06B6D4'>─ ─</span> Ganges Corridor</span>
      <span><span style='color:#10B981'>✛</span> Relief Shelter</span>
    </div>""", unsafe_allow_html=True)

with feed_col:
    st.markdown(
        "<div style='font-size:0.78rem;font-weight:700;color:#E2E8F0;"
        "text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px'>"
        "🚨 Priority Evacuation Dispatch Queue</div>",
        unsafe_allow_html=True,
    )

    alerts_all = []
    if sim and "grid_geojson" in sim and "features" in sim["grid_geojson"]:
        for feat in sim["grid_geojson"]["features"]:
            p = feat["properties"]
            if p.get("alert_level") in ("HIGH", "MEDIUM"):
                alerts_all.append(p)
        alerts_all.sort(key=lambda x: (
            x.get("time_to_flood_hrs") is None,
            x.get("time_to_flood_hrs") or 999,
        ))

    if not alerts_all:
        st.markdown("""
        <div style='background:#161F30;border:1px solid #2A364F;border-radius:10px;
                    padding:24px;text-align:center;color:#475569;font-size:0.82rem;'>
          <div style='font-size:1.8rem;margin-bottom:8px'>🟢</div>
          <b style='color:#10B981'>All Sectors Nominal</b><br>
          No HIGH or MEDIUM alerts in current scenario.<br>
          Increase precipitation intensity or execute simulation.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<div class='dispatch-scroll'>", unsafe_allow_html=True)
        for p in alerts_all:
            level   = p.get("alert_level", "LOW")
            ttf     = p.get("time_to_flood_hrs")
            ttf_str = f"Inundation within {ttf:.1f} hrs" if ttf is not None else "Window: TBD"
            card_cls  = "high"   if level == "HIGH"   else "medium"
            badge_cls = "badge-critical" if level == "HIGH" else "badge-advisory"
            badge_txt = "CRITICAL PRIORITY" if level == "HIGH" else "ADVISORY"
            action    = (
                "🔴 INITIATE EVACUATION &amp; DEPLOY FLOOD BARRIERS"
                if level == "HIGH"
                else "🟡 PRE-POSITION MOBILE PUMP UNITS &amp; CLEAR DRAINS"
            )
            dist = p.get("dist_to_river_m", 0)
            risk = p.get("risk_score", 0) or 0

            st.markdown(f"""
            <div class='alert-card {card_cls}'>
              <span class='{badge_cls}'>{badge_txt}</span>
              <div class='alert-cell'>
                {p.get("cell_id","—")} &nbsp;·&nbsp;
                <span style='color:#06B6D4'>Risk {risk:.1f}</span>
              </div>
              <div class='alert-ttf'>
                📍 {dist:.0f} m from Ganges &nbsp;|&nbsp; {ttf_str}
              </div>
              <div class='alert-action'>{action}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Export button ────────────────────────────────────────────────────
    if alerts_all:
        export_fc = {
            "type": "FeatureCollection",
            "name": "hydroshield_dispatch",
            "features": [
                f for f in (sim["grid_geojson"]["features"] if sim else [])
                if f["properties"].get("alert_level") in ("HIGH", "MEDIUM")
            ],
        }
        st.markdown("<div style='margin-top:12px'>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Dispatch GeoJSON",
            data=json.dumps(export_fc, indent=2),
            file_name=f"hydroshield_dispatch_{int(time.time())}.geojson",
            mime="application/geo+json",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)