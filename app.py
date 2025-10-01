# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests, io, math
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

DATA_URL = "https://data.nasa.gov/resource/gh4g-9sfh.csv"  # NASA Meteorite Landings (Socrata)

st.set_page_config(page_title="Meteor Madness", layout="wide")

@st.cache_data
def load_data(url=DATA_URL):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        st.error("Couldn't fetch dataset automatically. Place 'meteorite_landings.csv' in the app folder as fallback.")
        df = pd.read_csv("meteorite_landings.csv")
    # Normalize
    df.columns = [c.strip() for c in df.columns]
    # Ensure expected columns exist
    for col in ['reclat','reclong','mass','year']:
        if col not in df.columns:
            # try variants, but if not found leave and handle later
            pass
    # Try to convert year & mass and coords
    if 'year' in df.columns:
        df['year'] = pd.to_datetime(df['year'], errors='coerce').dt.year
    if 'mass' in df.columns:
        df['mass'] = pd.to_numeric(df['mass'], errors='coerce')  # grams
    # Keep only rows with coords
    lat_col = 'reclat' if 'reclat' in df.columns else None
    lon_col = 'reclong' if 'reclong' in df.columns else None
    if lat_col and lon_col:
        df = df.dropna(subset=[lat_col, lon_col])
        # convert to float
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        df = df.dropna(subset=[lat_col, lon_col])
    else:
        st.warning("Latitude/longitude columns not found in dataset. You may need to rename columns.")
    return df

def estimate_diameter_m(mass_g, density=3500):
    # Very rough sphere approximation: mass -> diameter (m). density in kg/m^3
    try:
        mass_kg = float(mass_g) / 1000.0
        if mass_kg <= 0:
            return None
        vol = mass_kg / density
        radius = (3 * vol / (4 * math.pi)) ** (1/3)
        return 2 * radius
    except:
        return None

def haversine(lat1, lon1, lat2, lon2):
    # returns distance in km
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# --- UI ---
st.title("Meteor Madness 🌠")
st.write("Interactive exploration of NASA's Meteorite Landings dataset.")

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
min_year = int(np.nanmin(df['year'].dropna())) if 'year' in df.columns and df['year'].dropna().size>0 else 0
max_year = int(np.nanmax(df['year'].dropna())) if 'year' in df.columns and df['year'].dropna().size>0 else 2025
year_range = st.sidebar.slider("Year range", min_value=min_year, max_value=max_year, value=(min_year, max_year))

min_mass = int(df['mass'].min()) if 'mass' in df.columns and df['mass'].dropna().size>0 else 0
max_mass = int(df['mass'].max()) if 'mass' in df.columns and df['mass'].dropna().size>0 else 10000000
mass_range = st.sidebar.slider("Mass range (grams)", min_value=0, max_value=max_mass, value=(0, max_mass))

# Nearest search (optional)
st.sidebar.header("Find nearest meteor")
use_coords = st.sidebar.checkbox("Enter coords to find nearest", value=False)
lat_input = None
lon_input = None
if use_coords:
    lat_input = st.sidebar.number_input("Your latitude (e.g. 24.47)", format="%.6f")
    lon_input = st.sidebar.number_input("Your longitude (e.g. 54.36)", format="%.6f")

# Filter dataframe
df_filtered = df.copy()
if 'year' in df.columns:
    df_filtered = df_filtered[(df_filtered['year'] >= year_range[0]) & (df_filtered['year'] <= year_range[1])]
if 'mass' in df.columns:
    df_filtered = df_filtered[(df_filtered['mass'] >= mass_range[0]) & (df_filtered['mass'] <= mass_range[1])]

st.sidebar.markdown(f"Total shown: **{len(df_filtered):,}**")

# Map creation
map_center = [0, 0]
if len(df_filtered) > 0 and 'reclat' in df_filtered.columns:
    map_center = [df_filtered['reclat'].median(), df_filtered['reclong'].median()]

m = folium.Map(location=map_center, zoom_start=2, tiles="CartoDB positron")
mc = MarkerCluster().add_to(m)

for _, row in df_filtered.iterrows():
    try:
        lat = float(row['reclat'])
        lon = float(row['reclong'])
    except:
        continue
    mass = float(row['mass']) if 'mass' in row and not pd.isna(row['mass']) else None
    size = 3
    if mass and mass > 0:
        size = max(3, (np.log10(mass + 1)) * 2)  # visual scale
    diam = estimate_diameter_m(mass) if mass else None
    popup_html = "<b>{}</b><br>".format(row.get('name', 'unknown'))
    popup_html += f"Year: {int(row['year']) if ('year' in row and not pd.isna(row['year'])) else 'unknown'}<br>"
    popup_html += f"Mass (g): {int(mass) if mass else 'unknown'}<br>"
    if 'recclass' in row and not pd.isna(row['recclass']):
        popup_html += f"Class: {row['recclass']}<br>"
    if diam:
        popup_html += f"Rough diameter ≈ {diam:.2f} m (assumed density 3500 kg/m³)<br>"
    folium.CircleMarker(
        location=[lat, lon],
        radius=size,
        color=None,
        fill=True,
        fill_opacity=0.7,
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(mc)

st.subheader("Map")
st.write("Tip: zoom in and click markers for details.")
st_data = st_folium(m, width=900, height=600)

# Right column: stats & table
col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Summary")
    st.metric("Meteorites shown", f"{len(df_filtered):,}")
    if 'mass' in df_filtered.columns and df_filtered['mass'].dropna().size>0:
        st.write("Mass (g) — min / median / max")
        st.write(df_filtered['mass'].agg(['min','median','max']).astype(float).round(1).to_frame().T)
    st.markdown("**Top 10 heaviest (sample)**")
    if 'mass' in df_filtered.columns:
        st.dataframe(df_filtered.sort_values(by='mass', ascending=False)[['name','year','mass','recclass','reclat','reclong']].head(10))
    else:
        st.write("Mass column not found")

with col2:
    st.subheader("Nearest meteor (optional)")
    if use_coords and lat_input is not None:
        # compute distances
        df_dist = df_filtered.copy()
        df_dist['dist_km'] = df_dist.apply(lambda r: haversine(lat_input, lon_input, float(r['reclat']), float(r['reclong'])), axis=1)
        nearest = df_dist.sort_values('dist_km').head(5)
        st.write(nearest[['name','year','mass','recclass','dist_km']].head(5).reset_index(drop=True))
    else:
        st.write("Enter coordinates in the sidebar to find nearest meteorite.")

# Download filtered CSV
csv = df_filtered.to_csv(index=False)
st.download_button("Download filtered CSV", csv, file_name="meteorites_filtered.csv", mime="text/csv")

st.markdown("---")
st.caption("Data: NASA Meteorite Landings (https://data.nasa.gov/Space-Science/Meteorite-Landings/gh4g-9sfh). Diameter estimate is a rough sphere approximation.")
