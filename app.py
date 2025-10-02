# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import requests, io, math
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

LOCAL_FILE = "meteorite-landings.csv"

st.set_page_config(page_title="Meteor Madness", layout="wide")

@st.cache_data
def load_data():
    df = None
    if os.path.exists(LOCAL_FILE):
        try:
            df = pd.read_csv(LOCAL_FILE)
        except Exception as e:
            st.error(f"Failed to read local file: {e}")
            df = None

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Year
    if "year" in df.columns:
        df["year"] = pd.to_datetime(df["year"], errors="coerce").dt.year

    # Mass
    if "mass" in df.columns:
        df["mass"] = pd.to_numeric(df["mass"], errors="coerce")

    # Coordinates
    if "reclat" in df.columns and "reclong" in df.columns:
        df["reclat"] = pd.to_numeric(df["reclat"], errors="coerce")
        df["reclong"] = pd.to_numeric(df["reclong"], errors="coerce")
        df = df.dropna(subset=["reclat", "reclong"])
    else:
        st.warning("Latitude/longitude columns not found in dataset.")

    return df

def estimate_diameter_m(mass_g, density=3500):
    """Very rough mass → diameter conversion."""
    try:
        mass_kg = float(mass_g) / 1000.0
        if mass_kg <= 0:
            return None
        vol = mass_kg / density
        radius = (3 * vol / (4 * math.pi)) ** (1/3)
        return 2 * radius
    except Exception:
        return None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# --- UI ---
st.title("Meteor Madness 🌠")
st.write("Interactive exploration of Meteorite Landings dataset.")

df = load_data()
if df.empty:
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
min_year = int(df["year"].min(skipna=True)) if "year" in df else 1800
max_year = int(df["year"].max(skipna=True)) if "year" in df else 2025
year_range = st.sidebar.slider("Year range", min_year, max_year, (min_year, max_year))

if min_year==max_year:
    st.sidebar.warning("Year range slider disabled (all data from same year).")

min_mass = int(df["mass"].min(skipna=True)) if "mass" in df else 0
max_mass = int(df["mass"].max(skipna=True)) if "mass" in df else 10000000
mass_range = st.sidebar.slider("Mass range (grams)", 0, max_mass, (0, max_mass))

# Nearest search (optional)
st.sidebar.header("Find nearest meteor")
use_coords = st.sidebar.checkbox("Enter coords to find nearest", value=False)
lat_input, lon_input = None, None
if use_coords:
    lat_input = st.sidebar.number_input("Your latitude", format="%.6f")
    lon_input = st.sidebar.number_input("Your longitude", format="%.6f")

# Filter data
df_filtered = df.copy()
if "year" in df.columns:
    df_filtered = df_filtered[df_filtered["year"].between(year_range[0], year_range[1])]
if "mass" in df.columns:
    df_filtered = df_filtered[df_filtered["mass"].between(mass_range[0], mass_range[1])]

st.sidebar.markdown(f"Total shown: **{len(df_filtered):,}**")

# Map
map_center = [df_filtered["reclat"].median(), df_filtered["reclong"].median()] if len(df_filtered) > 0 else [0, 0]
m = folium.Map(location=map_center, zoom_start=2, tiles="CartoDB positron")
mc = MarkerCluster().add_to(m)

for _, row in df_filtered.iterrows():
    lat, lon = row["reclat"], row["reclong"]
    if pd.isna(lat) or pd.isna(lon):
        continue
    mass = row["mass"] if "mass" in row else None
    size = max(3, (np.log10(mass + 1)) * 2) if mass and mass > 0 else 3
    diam = estimate_diameter_m(mass) if mass else None
    popup_html = f"<b>{row.get('name','unknown')}</b><br>"
    popup_html += f"Year: {int(row['year']) if not pd.isna(row.get('year')) else 'unknown'}<br>"
    popup_html += f"Mass (g): {int(mass) if mass else 'unknown'}<br>"
    if "recclass" in row and not pd.isna(row["recclass"]):
        popup_html += f"Class: {row['recclass']}<br>"
    if diam:
        popup_html += f"Rough diameter ≈ {diam:.2f} m<br>"

    folium.CircleMarker(
        location=[lat, lon],
        radius=size,
        fill=True,
        fill_opacity=0.7,
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(mc)

st.subheader("Map")
st.write("Tip: zoom in and click markers for details.")
st_data = st_folium(m, width=900, height=600)

# Stats and tables
col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Summary")
    st.metric("Meteorites shown", f"{len(df_filtered):,}")
    if "mass" in df_filtered:
        st.write("Mass (g) — min / median / max")
        st.write(df_filtered["mass"].agg(["min","median","max"]).round(1).to_frame().T)
        st.markdown("**Top 10 heaviest**")
        st.dataframe(df_filtered.sort_values(by="mass", ascending=False)[["name","year","mass","recclass","reclat","reclong"]].head(10))

with col2:
    st.subheader("Nearest meteor")
    if use_coords and lat_input and lon_input:
        df_dist = df_filtered.copy()
        df_dist["dist_km"] = df_dist.apply(lambda r: haversine(lat_input, lon_input, r["reclat"], r["reclong"]), axis=1)
        nearest = df_dist.sort_values("dist_km").head(5)
        st.write(nearest[["name","year","mass","recclass","dist_km"]].reset_index(drop=True))
    else:
        st.write("Enter coordinates in the sidebar to find nearest meteorite.")

# Download button
csv = df_filtered.to_csv(index=False)
st.download_button("Download filtered CSV", csv, "meteorites_filtered.csv", "text/csv")

st.markdown("---")
st.caption("Data: NASA Meteorite Landings dataset. Diameter estimate is a rough sphere approximation.")
