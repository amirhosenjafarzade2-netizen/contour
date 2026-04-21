import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import KDTree

def parse_bna(file_content):
    """
    Parses a BNA file to extract X, Y coordinates and their associated contour values.
    """
    points = []
    lines = file_content.decode("utf-8").splitlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # BNA headers are typically: "Name","Value",NumPoints
        if line.startswith('"'):
            parts = line.split(',')
            try:
                # Extract the contour value (second element)
                contour_value = float(parts[1].replace('"', ''))
                num_points = int(parts[2])
                i += 1
                
                # Read the next 'num_points' lines for coordinates
                for _ in range(num_points):
                    if i < len(lines):
                        coords = lines[i].split(',')
                        points.append({
                            'x': float(coords[0]),
                            'y': float(coords[1]),
                            'z': contour_value
                        })
                        i += 1
            except (ValueError, IndexError):
                i += 1
        else:
            i += 1
            
    return pd.DataFrame(points)

# Streamlit UI
st.title("BNA Contour Value Finder")
st.write("Upload a .bna file to query contour values at specific locations.")

uploaded_file = st.file_uploader("Choose a BNA file", type="bna")

if uploaded_file is not None:
    # Process the file
    df = parse_bna(uploaded_file.getvalue())
    
    if not df.empty:
        st.success(f"Successfully loaded {len(df)} points from {uploaded_file.name}")
        
        # Build a KDTree for fast nearest-neighbor lookup
        tree = KDTree(df[['x', 'y']].values)
        
        st.subheader("Query Location")
        col1, col2 = st.columns(2)
        
        with col1:
            input_x = st.number_input("Enter X coordinate", value=float(df['x'].mean()))
        with col2:
            input_y = st.number_input("Enter Y coordinate", value=float(df['y'].mean()))
            
        if st.button("Get Value"):
            # Find the nearest point in the BNA data
            dist, index = tree.query([input_x, input_y])
            result_value = df.iloc[index]['z']
            
            st.metric(label="Contour Value", value=f"{result_value}")
            st.info(f"Nearest data point found at distance: {dist:.2f}")
            
            # Optional: Show a small map/plot of points
            st.write("Data Preview (Sample):")
            st.dataframe(df.head())
    else:
        st.error("Could not find any valid contour data in the file.")
