"""
BNA Contour Explorer v3
────────────────────────
After uploading, the user picks a task from a visual hub.
Tasks:
  1. Interactive Map & Visualization
  2. Point Query (single / IDW / nearest)
  3. Batch CSV Query
  4. Radius / Zone Search
  5. Cross-Section / Transect Profile
  6. Contour Band Analysis
  7. Hotspot & Anomaly Detection
  8. File Comparison (two BNA files)
  9. 3-D Surface View
 10. Data Export & Report
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import griddata
from scipy import stats as scipy_stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BNA Contour Explorer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 700; }
code, .mono { font-family: 'JetBrains Mono', monospace; }

/* metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg,#0f172a,#1e293b);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 14px 18px;
    color: #f8fafc;
}
[data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #38bdf8 !important; font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; }

/* task cards */
.task-card {
    background: linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    cursor: pointer;
    transition: all .25s;
    min-height: 130px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.task-card:hover { border-color: #38bdf8; transform: translateY(-3px); box-shadow: 0 8px 32px rgba(56,189,248,.15); }
.task-card .icon { font-size: 2rem; margin-bottom: 8px; }
.task-card .title { font-weight: 700; color: #f1f5f9; font-size: .95rem; }
.task-card .desc  { color: #64748b; font-size: .75rem; margin-top: 4px; }

/* section header */
.section-header {
    background: linear-gradient(90deg,#0ea5e9,#6366f1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 1.6rem; font-weight: 700; margin-bottom: 4px;
}
.breadcrumb { color:#64748b; font-size:.85rem; margin-bottom:20px; }

/* history pill */
.history-pill {
    display:inline-block;
    background:#1e293b; border:1px solid #334155;
    border-radius:999px; padding:3px 12px;
    font-family:'JetBrains Mono',monospace; font-size:11px; color:#38bdf8;
    margin:2px;
}

/* upload zone */
.upload-hero {
    background: linear-gradient(135deg,#0f172a,#1a1f3a);
    border: 2px dashed #334155;
    border-radius: 20px;
    padding: 48px 24px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
for key, val in {
    "task": None,
    "query_history": [],
    "df": None,
    "df2": None,
    "tree": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

COLORSCALES = ["Viridis", "Plasma", "Cividis", "RdBu", "Turbo", "Inferno", "Magma"]

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Parsing BNA …")
def parse_bna(file_bytes: bytes, filename: str) -> pd.DataFrame:
    points, errors = [], []
    lines = file_bytes.decode("utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if line.startswith('"'):
            parts = line.split(',')
            try:
                z = float(parts[1].replace('"', '').strip())
                n = int(parts[2].strip())
                i += 1
                for _ in range(n):
                    if i < len(lines):
                        try:
                            cx, cy = lines[i].split(',')[:2]
                            points.append({'x': float(cx), 'y': float(cy), 'z': z})
                        except Exception as e:
                            errors.append(str(e))
                        i += 1
            except Exception as e:
                errors.append(str(e)); i += 1
        else:
            i += 1
    if errors:
        st.warning(f"Skipped {len(errors)} malformed line(s).")
    return pd.DataFrame(points)


def idw(df, qx, qy, k=6, p=2):
    kd = KDTree(df[['x','y']].values)
    d, idx = kd.query([qx, qy], k=min(k, len(df)))
    d = np.array(d, dtype=float)
    v = df.iloc[idx]['z'].values.astype(float)
    if d[0] == 0: return float(v[0])
    w = 1/(d**p); return float(np.dot(w,v)/w.sum())


def make_heatmap(df, colorscale="Viridis", title="Contour Surface", res=300):
    sample = df if len(df) <= 60_000 else df.sample(60_000, random_state=0)
    xi = np.linspace(sample['x'].min(), sample['x'].max(), res)
    yi = np.linspace(sample['y'].min(), sample['y'].max(), res)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((sample['x'], sample['y']), sample['z'], (XI, YI), method='linear')
    fig = go.Figure(go.Heatmap(
        z=ZI, x=xi, y=yi,
        colorscale=colorscale,
        colorbar=dict(title="Z", thickness=14),
    ))
    fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y",
                      height=520, margin=dict(l=20,r=20,t=40,b=20),
                      paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                      font=dict(color='#94a3b8'))
    return fig


def stat_summary(df):
    z = df['z']
    return {
        "Count":    len(z),
        "Min":      z.min(),
        "Max":      z.max(),
        "Mean":     z.mean(),
        "Median":   z.median(),
        "Std":      z.std(),
        "Skewness": float(scipy_stats.skew(z)),
        "Kurtosis": float(scipy_stats.kurtosis(z)),
    }


# ─────────────────────────────────────────────────────────────
# UPLOAD SCREEN
# ─────────────────────────────────────────────────────────────
def upload_screen():
    st.markdown('<h1 style="font-size:2.4rem;font-weight:800;color:#f1f5f9;margin-bottom:4px">🗺️ BNA Contour Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;font-size:1rem;margin-bottom:32px">Upload a BNA file to unlock all analysis tools</p>', unsafe_allow_html=True)

    col_up, col_info = st.columns([2,1], gap="large")

    with col_up:
        uploaded = st.file_uploader("Primary BNA file", type="bna", label_visibility="collapsed")
        st.caption("Drag & drop or click to browse  ·  .bna format")

        uploaded2 = st.file_uploader("Optional: second BNA file for comparison", type="bna")

    with col_info:
        st.markdown("""
        **What is a BNA file?**  
        A BNA (Atlas Boundary) file stores geographic contour/boundary data as named polylines with associated numeric values.

        **Supported tasks after upload:**
        - Interactive heatmap & 3-D surface
        - Point querying with IDW interpolation
        - Batch CSV lookup
        - Radius / zone search
        - Transect cross-section profiles
        - Contour band analysis
        - Hotspot & anomaly detection
        - Two-file comparison
        - Report & CSV export
        """)

    if uploaded:
        with st.spinner("Parsing…"):
            df = parse_bna(uploaded.getvalue(), uploaded.name)
            df["source"] = uploaded.name
        if df.empty:
            st.error("No valid data found."); return
        st.session_state.df   = df
        st.session_state.tree = KDTree(df[['x','y']].values)
        if uploaded2:
            df2 = parse_bna(uploaded2.getvalue(), uploaded2.name)
            df2["source"] = uploaded2.name
            st.session_state.df2 = df2
        st.success(f"✅ Loaded **{len(df):,}** points from `{uploaded.name}`")
        st.rerun()


# ─────────────────────────────────────────────────────────────
# TASK HUB
# ─────────────────────────────────────────────────────────────
TASKS = [
    ("🗺️", "Interactive Map",      "Heatmap, point cloud, contour lines"),
    ("🎯", "Point Query",           "Single-point lookup with interpolation"),
    ("📂", "Batch CSV Query",        "Query hundreds of points from a CSV"),
    ("🔵", "Radius / Zone Search",  "All points within a distance"),
    ("📏", "Transect Profile",      "Cross-section Z along a line"),
    ("🎨", "Contour Band Analysis", "Stats & area per contour level"),
    ("🔥", "Hotspot Detection",     "Find peaks, valleys & anomalies"),
    ("⚖️", "File Comparison",       "Diff two BNA files side-by-side"),
    ("🧊", "3-D Surface",           "Interactive 3-D perspective view"),
    ("📤", "Export & Report",       "Download CSV, stats report & charts"),
]

def task_hub():
    df = st.session_state.df
    c_a,c_b,c_c,c_d,c_e = st.columns(5)
    for col, (label, val) in zip(
        [c_a,c_b,c_c,c_d,c_e],
        [("Points",f"{len(df):,}"),("Z levels",f"{df['z'].nunique()}"),
         ("Z min",f"{df['z'].min():.3f}"),("Z max",f"{df['z'].max():.3f}"),
         ("Std",f"{df['z'].std():.3f}")]):
        col.metric(label, val)

    st.markdown("---")
    st.markdown('<p style="color:#94a3b8;font-size:.9rem;margin-bottom:16px">Select a task to begin ↓</p>', unsafe_allow_html=True)

    rows = [TASKS[:5], TASKS[5:]]
    for row in rows:
        cols = st.columns(5)
        for col, (icon, title, desc) in zip(cols, row):
            with col:
                st.markdown(f"""
                <div class="task-card">
                  <div class="icon">{icon}</div>
                  <div class="title">{title}</div>
                  <div class="desc">{desc}</div>
                </div>""", unsafe_allow_html=True)
                if st.button(f"Open", key=f"task_{title}"):
                    st.session_state.task = title
                    st.rerun()


def back_button():
    if st.button("← Back to task hub"):
        st.session_state.task = None
        st.rerun()


# ─────────────────────────────────────────────────────────────
# TASK 1 — Interactive Map
# ─────────────────────────────────────────────────────────────
def task_map():
    df = st.session_state.df
    st.markdown('<div class="section-header">🗺️ Interactive Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="breadcrumb">Home → Interactive Map</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    view     = c1.selectbox("View type", ["Heatmap (interpolated)", "Point Cloud", "Contour Lines"])
    cscale   = c2.selectbox("Colour scale", COLORSCALES)
    res      = c3.slider("Grid resolution", 100, 500, 250, 50, help="Higher = sharper but slower")

    sample = df if len(df) <= 60_000 else df.sample(60_000, random_state=0)

    if view == "Heatmap (interpolated)":
        fig = make_heatmap(sample, cscale, res=res)

    elif view == "Point Cloud":
        fig = px.scatter(sample, x='x', y='y', color='z',
                         color_continuous_scale=cscale, opacity=.6,
                         title="Point Cloud", height=540)
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                          font=dict(color='#94a3b8'))

    else:  # Contour lines
        xi = np.linspace(sample['x'].min(), sample['x'].max(), res)
        yi = np.linspace(sample['y'].min(), sample['y'].max(), res)
        XI, YI = np.meshgrid(xi, yi)
        ZI = griddata((sample['x'], sample['y']), sample['z'], (XI, YI), method='linear')
        fig = go.Figure(go.Contour(
            z=ZI, x=xi, y=yi,
            colorscale=cscale,
            contours=dict(showlabels=True, labelfont=dict(size=10, color='white')),
            colorbar=dict(title="Z"),
        ))
        fig.update_layout(title="Contour Lines", height=540,
                          paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                          font=dict(color='#94a3b8'))

    st.plotly_chart(fig, use_container_width=True)
    if len(df) > 60_000:
        st.caption(f"Displaying 60,000-point sample of {len(df):,} total.")


# ─────────────────────────────────────────────────────────────
# TASK 2 — Point Query
# ─────────────────────────────────────────────────────────────
def task_point_query():
    df   = st.session_state.df
    tree = st.session_state.tree
    st.markdown('<div class="section-header">🎯 Point Query</div>', unsafe_allow_html=True)
    st.markdown('<div class="breadcrumb">Home → Point Query</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    qx  = c1.number_input("X", value=float(df['x'].mean()), format="%.6f")
    qy  = c2.number_input("Y", value=float(df['y'].mean()), format="%.6f")
    method = c3.selectbox("Method", ["IDW interpolation","Nearest-neighbour","Linear griddata"])
    k   = c4.slider("k neighbours", 3, 20, 6, disabled=(method != "IDW interpolation"))

    if st.button("▶ Query point", type="primary"):
        dist, idx = tree.query([qx, qy])
        snap_z = float(df.iloc[idx]['z'])

        if method == "IDW interpolation":
            result_z = idw(df, qx, qy, k=k)
        elif method == "Nearest-neighbour":
            result_z = snap_z
        else:
            result_z = float(griddata((df['x'], df['y']), df['z'], [(qx, qy)], method='linear')[0])

        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Result Z", f"{result_z:.6f}")
        r2.metric("Method", method.split()[0])
        r3.metric("Distance to nearest", f"{dist:.4f}")
        r4.metric("Snap Z (nearest pt)", f"{snap_z:.6f}")

        st.session_state.query_history.append({"x":qx,"y":qy,"z":result_z,"method":method})

        # Show on map with surrounding context
        sample = df.sample(min(8000, len(df)), random_state=1)
        fig = px.scatter(sample, x='x', y='y', color='z',
                         color_continuous_scale='Viridis', opacity=.35, height=380)
        fig.add_scatter(x=[qx], y=[qy], mode='markers',
                        marker=dict(color='#f43f5e', size=14, symbol='x-thin', line=dict(width=3)),
                        name='Query')
        nn = df.iloc[idx]
        fig.add_scatter(x=[nn['x']], y=[nn['y']], mode='markers',
                        marker=dict(color='#fbbf24', size=10, symbol='circle'),
                        name='Nearest point')
        fig.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                          font=dict(color='#94a3b8'), margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    if st.session_state.query_history:
        st.markdown("#### Query history this session")
        hist_df = pd.DataFrame(st.session_state.query_history)
        st.dataframe(hist_df, use_container_width=True)
        if st.button("Clear history"):
            st.session_state.query_history = []
            st.rerun()


# ─────────────────────────────────────────────────────────────
# TASK 3 — Batch CSV Query
# ─────────────────────────────────────────────────────────────
def task_batch():
    df   = st.session_state.df
    tree = st.session_state.tree
    st.markdown('<div class="section-header">📂 Batch CSV Query</div>', unsafe_allow_html=True)
    st.markdown('<div class="breadcrumb">Home → Batch CSV Query</div>', unsafe_allow_html=True)
    st.info("Upload a CSV with columns **x** and **y**. Each row will be queried and Z values appended.")

    method = st.radio("Method", ["IDW interpolation","Nearest-neighbour"], horizontal=True)
    k = st.slider("k (IDW)", 3, 20, 6, disabled=(method != "IDW interpolation"))

    batch_file = st.file_uploader("Upload query CSV", type="csv")
    if not batch_file: return

    bdf = pd.read_csv(batch_file)
    if 'x' not in bdf.columns or 'y' not in bdf.columns:
        st.error("CSV must contain columns `x` and `y`."); return

    st.write(f"Preview ({len(bdf):,} rows):", bdf.head())

    if st.button("▶ Run batch query", type="primary"):
        with st.spinner(f"Querying {len(bdf):,} points…"):
            if method == "IDW interpolation":
                bdf['z'] = [idw(df, row.x, row.y, k=k) for row in bdf.itertuples()]
            else:
                dists, idxs = tree.query(bdf[['x','y']].values)
                bdf['z'] = df.iloc[idxs]['z'].values
                bdf['distance_to_nearest'] = dists

        st.success(f"Done — {len(bdf):,} points queried.")
        st.dataframe(bdf, use_container_width=True)

        fig = px.scatter(bdf, x='x', y='y', color='z',
                         color_continuous_scale='Plasma', height=400,
                         title="Batch query result map")
        fig.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                          font=dict(color='#94a3b8'))
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("⬇️ Download results CSV",
                           bdf.to_csv(index=False).encode(),
                           "batch_results.csv", "text/csv")


# ─────────────────────────────────────────────────────────────
# TASK 4 — Radius / Zone Search
# ─────────────────────────────────────────────────────────────
def task_radius():
    df   = st.session_state.df
    tree = st.session_state.tree
    st.markdown('<div class="section-header">🔵 Radius / Zone Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="breadcrumb">Home → Radius / Zone Search</div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    rx = c1.number_input("Centre X", value=float(df['x'].mean()), format="%.6f")
    ry = c2.number_input("Centre Y", value=float(df['y'].mean()), format="%.6f")
    default_r = float((df['x'].max()-df['x'].min())*0.05)
    radius = c3.number_input("Radius", value=default_r, min_value=0.0, format="%.4f")

    if st.button("▶ Search zone", type="primary"):
        idxs = tree.query_ball_point([rx, ry], r=radius)
        if not idxs:
            st.warning("No points in that radius. Try increasing it."); return

        rdf = df.iloc[idxs].copy()
        rdf['distance'] = np.sqrt((rdf['x']-rx)**2 + (rdf['y']-ry)**2)
        rdf = rdf.sort_values('distance').reset_index(drop=True)

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Points found", f"{len(rdf):,}")
        m2.metric("Z mean", f"{rdf['z'].mean():.4f}")
        m3.metric("Z min", f"{rdf['z'].min():.4f}")
        m4.metric("Z max", f"{rdf['z'].max():.4f}")

        col_map, col_hist = st.columns(2)
        with col_map:
            fig = px.scatter(rdf, x='x', y='y', color='z',
                             color_continuous_scale='Turbo', height=380,
                             title=f"Zone — radius {radius}")
            fig.add_scatter(x=[rx], y=[ry], mode='markers',
                            marker=dict(color='#f43f5e', size=14, symbol='x-thin', line=dict(width=3)),
                            name='Centre')
            # draw circle
            theta = np.linspace(0, 2*np.pi, 120)
            fig.add_scatter(x=rx+radius*np.cos(theta), y=ry+radius*np.sin(theta),
                            mode='lines', line=dict(color='#f43f5e', dash='dash'), name='Radius')
            fig.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                              font=dict(color='#94a3b8'), margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_hist:
            fig2 = px.histogram(rdf, x='z', nbins=40, height=380,
                                color_discrete_sequence=['#38bdf8'],
                                title="Z distribution inside zone")
            fig2.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                               font=dict(color='#94a3b8'))
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(rdf.head(300), use_container_width=True)
        st.download_button("⬇️ Download zone CSV", rdf.to_csv(index=False).encode(),
                           "zone_results.csv", "text/csv")


# ─────────────────────────────────────────────────────────────
# TASK 5 — Transect / Cross-Section Profile
# ─────────────────────────────────────────────────────────────
def task_transect():
    df = st.session_state.df
    st.markdown('<div class="section-header">📏 Transect Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="breadcrumb">Home → Transect Profile</div>', unsafe_allow_html=True)
    st.info("Define a start and end point. The tool samples Z values along the line and draws a cross-section profile.")

    c1,c2 = st.columns(2)
    with c1:
        x1 = st.number_input("Start X", value=float(df['x'].min()), format="%.6f")
        y1 = st.number_input("Start Y", value=float(df['y'].mean()), format="%.6f")
    with c2:
        x2 = st.number_input("End X",   value=float(df['x'].max()), format="%.6f")
        y2 = st.number_input("End Y",   value=float(df['y'].mean()), format="%.6f")

    n_samples = st.slider("Sample points along transect", 50, 500, 150)

    if st.button("▶ Generate profile", type="primary"):
        xs = np.linspace(x1, x2, n_samples)
        ys = np.linspace(y1, y2, n_samples)
        zs = []
        kd = KDTree(df[['x','y']].values)
        for qx, qy in zip(xs, ys):
            zs.append(idw(df, qx, qy, k=6))
        dist_along = np.sqrt((xs-x1)**2 + (ys-y1)**2)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Cross-section Profile","Transect on Map"))

        # profile
        fig.add_trace(go.Scatter(
            x=dist_along, y=zs, mode='lines+markers',
            line=dict(color='#38bdf8', width=2),
            marker=dict(size=4),
            fill='tozeroy', fillcolor='rgba(56,189,248,.15)',
            name='Z profile'), row=1, col=1)

        # map with line
        sample = df.sample(min(6000, len(df)), random_state=0)
        fig.add_trace(go.Scatter(
            x=sample['x'], y=sample['y'], mode='markers',
            marker=dict(color=sample['z'], colorscale='Viridis', size=3, opacity=.4),
            name='Data', showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[x1,x2], y=[y1,y2], mode='lines+markers',
            line=dict(color='#f43f5e', width=3),
            marker=dict(size=10), name='Transect'), row=1, col=2)

        fig.update_layout(height=440,
                          paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                          font=dict(color='#94a3b8'))
        fig.update_xaxes(gridcolor='#1e293b'); fig.update_yaxes(gridcolor='#1e293b')
        st.plotly_chart(fig, use_container_width=True)

        tdf = pd.DataFrame({'distance': dist_along, 'x': xs, 'y': ys, 'z': zs})
        m1,m2,m3 = st.columns(3)
        m1.metric("Transect length", f"{dist_along[-1]:.3f}")
        m2.metric("Z range along line", f"{max(zs)-min(zs):.4f}")
        m3.metric("Mean Z", f"{np.mean(zs):.4f}")
        st.download_button("⬇️ Download profile CSV", tdf.to_csv(index=False).encode(),
                           "transect.csv", "text/csv")


# ─────────────────────────────────────────────────────────────
# TASK 6 — Contour Band Analysis
# ─────────────────────────────────────────────────────────────
def task_band_analysis():
    df = st.session_state.df
    st.markdown('<div class="section-header">🎨 Contour Band Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="breadcrumb">Home → Contour Band Analysis</div>', unsafe_allow_html=True)

    z_levels = sorted(df['z'].unique())

    c1,c2 = st.columns(2)
    z_min_sel = c1.selectbox("Z lower bound (inclusive)", z_levels, index=0)
    z_max_sel = c2.selectbox("Z upper bound (inclusive)", z_levels, index=len(z_levels)-1)

    filtered = df[(df['z'] >= z_min_sel) & (df['z'] <= z_max_sel)]

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Points in band", f"{len(filtered):,}")
    m2.metric("% of total",     f"{100*len(filtered)/len(df):.1f}%")
    m3.metric("Z mean",         f"{filtered['z'].mean():.4f}" if len(filtered) else "—")
    m4.metric("Contour levels", f"{filtered['z'].nunique()}")

    # Per-level breakdown
    band_summary = (
        filtered.groupby('z')
        .agg(count=('z','count'), x_mean=('x','mean'), y_mean=('y','mean'))
        .reset_index()
    )

    col_bar, col_map = st.columns(2)
    with col_bar:
        fig = px.bar(band_summary, x='z', y='count',
                     color='count', color_continuous_scale='Viridis',
                     title="Points per contour level", height=360)
        fig.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                          font=dict(color='#94a3b8'))
        st.plotly_chart(fig, use_container_width=True)

    with col_map:
        fig2 = px.scatter(filtered, x='x', y='y', color='z',
                          color_continuous_scale='Plasma', opacity=.6,
                          title="Band footprint on map", height=360)
        fig2.update_traces(marker=dict(size=3))
        fig2.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                           font=dict(color='#94a3b8'))
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(band_summary, use_container_width=True)
    st.download_button("⬇️ Download band CSV", filtered.to_csv(index=False).encode(),
                       "band_filtered.csv", "text/csv")


# ─────────────────────────────────────────────────────────────
# TASK 7 — Hotspot & Anomaly Detection
# ─────────────────────────────────────────────────────────────
def task_hotspot():
    df = st.session_state.df
    st.markdown('<div class="section-header">🔥 Hotspot & Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="breadcrumb">Home → Hotspot Detection</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    threshold_pct = c1.slider("Anomaly threshold (top/bottom %)", 1, 20, 5)
    n_extremes     = c2.slider("Show N extreme points", 10, 200, 30)

    z = df['z']
    hi_thresh = np.percentile(z, 100 - threshold_pct)
    lo_thresh = np.percentile(z, threshold_pct)

    highs = df[z >= hi_thresh].copy(); highs['type'] = 'High'
    lows  = df[z <= lo_thresh].copy(); lows['type']  = 'Low'
    extremes = pd.concat([highs, lows])

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("High anomalies", f"{len(highs):,}")
    m2.metric("Low anomalies",  f"{len(lows):,}")
    m3.metric("High threshold", f"{hi_thresh:.4f}")
    m4.metric("Low threshold",  f"{lo_thresh:.4f}")

    sample = df.sample(min(8000, len(df)), random_state=2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample['x'], y=sample['y'], mode='markers',
        marker=dict(color=sample['z'], colorscale='Greys', size=3, opacity=.3),
        name='Background'))
    fig.add_trace(go.Scatter(
        x=highs['x'], y=highs['y'], mode='markers',
        marker=dict(color='#f43f5e', size=7, symbol='circle'),
        name=f'High (top {threshold_pct}%)'))
    fig.add_trace(go.Scatter(
        x=lows['x'], y=lows['y'], mode='markers',
        marker=dict(color='#38bdf8', size=7, symbol='triangle-down'),
        name=f'Low (bot {threshold_pct}%)'))
    fig.update_layout(title="Anomaly Map", height=480,
                      paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                      font=dict(color='#94a3b8'))
    st.plotly_chart(fig, use_container_width=True)

    col_h, col_l = st.columns(2)
    with col_h:
        st.markdown("**Top high-Z points**")
        st.dataframe(highs.nlargest(n_extremes, 'z')[['x','y','z']], use_container_width=True)
    with col_l:
        st.markdown("**Top low-Z points**")
        st.dataframe(lows.nsmallest(n_extremes, 'z')[['x','y','z']], use_container_width=True)

    st.download_button("⬇️ Download anomalies CSV", extremes.to_csv(index=False).encode(),
                       "anomalies.csv", "text/csv")


# ─────────────────────────────────────────────────────────────
# TASK 8 — File Comparison
# ─────────────────────────────────────────────────────────────
def task_comparison():
    df  = st.session_state.df
    df2 = st.session_state.df2
    st.markdown('<div class="section-header">⚖️ File Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="breadcrumb">Home → File Comparison</div>', unsafe_allow_html=True)

    if df2 is None:
        st.warning("Please upload a second BNA file on the upload screen to use this feature.")
        if st.button("← Go back and upload a second file"):
            st.session_state.df  = None
            st.session_state.df2 = None
            st.session_state.task = None
            st.rerun()
        return

    s1, s2 = stat_summary(df), stat_summary(df2)
    src1 = df['source'].iloc[0]; src2 = df2['source'].iloc[0]

    comp = pd.DataFrame({"Statistic": list(s1.keys()),
                         src1: list(s1.values()),
                         src2: list(s2.values())})
    comp['Δ'] = comp[src2] - comp[src1]
    st.dataframe(comp.set_index("Statistic").style.format("{:.4f}"), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(df, x='z', nbins=60, title=f"Z distribution — {src1}",
                           color_discrete_sequence=['#38bdf8'], height=320)
        fig.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                          font=dict(color='#94a3b8'))
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.histogram(df2, x='z', nbins=60, title=f"Z distribution — {src2}",
                           color_discrete_sequence=['#f43f5e'], height=320)
        fig.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                          font=dict(color='#94a3b8'))
        st.plotly_chart(fig, use_container_width=True)

    # overlay violin
    merged = pd.concat([
        df[['z']].assign(file=src1),
        df2[['z']].assign(file=src2)
    ])
    fig_v = px.violin(merged, x='file', y='z', box=True, color='file',
                      color_discrete_map={src1:'#38bdf8', src2:'#f43f5e'},
                      title="Z distribution comparison (violin)", height=360)
    fig_v.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                        font=dict(color='#94a3b8'))
    st.plotly_chart(fig_v, use_container_width=True)

    # Spatial overlap
    if st.checkbox("Show spatial overlap (may be slow for large files)"):
        s = df.sample(min(3000,len(df)), random_state=0)
        s2_ = df2.sample(min(3000,len(df2)), random_state=0)
        fig_s = go.Figure()
        fig_s.add_scatter(x=s['x'], y=s['y'], mode='markers',
                          marker=dict(color='#38bdf8', size=3, opacity=.5), name=src1)
        fig_s.add_scatter(x=s2_['x'], y=s2_['y'], mode='markers',
                          marker=dict(color='#f43f5e', size=3, opacity=.5), name=src2)
        fig_s.update_layout(title="Spatial footprint overlay", height=460,
                             paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                             font=dict(color='#94a3b8'))
        st.plotly_chart(fig_s, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# TASK 9 — 3-D Surface
# ─────────────────────────────────────────────────────────────
def task_3d():
    df = st.session_state.df
    st.markdown('<div class="section-header">🧊 3-D Surface View</div>', unsafe_allow_html=True)
    st.markdown('<div class="breadcrumb">Home → 3-D Surface</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    cscale = c1.selectbox("Colour scale", COLORSCALES, index=4)
    res    = c2.slider("Grid resolution", 60, 300, 120, 20)

    sample = df if len(df) <= 60_000 else df.sample(60_000, random_state=0)
    xi = np.linspace(sample['x'].min(), sample['x'].max(), res)
    yi = np.linspace(sample['y'].min(), sample['y'].max(), res)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((sample['x'], sample['y']), sample['z'], (XI, YI), method='linear')

    fig = go.Figure(go.Surface(
        z=ZI, x=XI, y=YI,
        colorscale=cscale,
        colorbar=dict(title="Z", thickness=14),
        lighting=dict(ambient=.6, diffuse=.8, specular=.4, roughness=.5),
        lightposition=dict(x=100, y=200, z=0),
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor='#0f172a', gridcolor='#1e293b', title='X'),
            yaxis=dict(backgroundcolor='#0f172a', gridcolor='#1e293b', title='Y'),
            zaxis=dict(backgroundcolor='#0f172a', gridcolor='#1e293b', title='Z'),
            bgcolor='#0f172a',
        ),
        paper_bgcolor='#0f172a',
        font=dict(color='#94a3b8'),
        height=600,
        margin=dict(l=0,r=0,t=20,b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("💡 Click & drag to rotate · Scroll to zoom · Double-click to reset")


# ─────────────────────────────────────────────────────────────
# TASK 10 — Export & Report
# ─────────────────────────────────────────────────────────────
def task_export():
    df = st.session_state.df
    st.markdown('<div class="section-header">📤 Export & Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="breadcrumb">Home → Export & Report</div>', unsafe_allow_html=True)

    src = df['source'].iloc[0]
    s = stat_summary(df)

    st.markdown("### Summary statistics")
    stat_df = pd.DataFrame({"Statistic": list(s.keys()), "Value": list(s.values())})
    st.dataframe(stat_df.set_index("Statistic").style.format("{:.6f}"), use_container_width=True)

    # Contour level table
    level_tbl = (df.groupby('z')
                   .agg(point_count=('z','count'),
                        x_min=('x','min'), x_max=('x','max'),
                        y_min=('y','min'), y_max=('y','max'))
                   .reset_index()
                   .rename(columns={'z':'contour_value'}))
    st.markdown("### Contour level table")
    st.dataframe(level_tbl, use_container_width=True)

    # Charts
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(df, x='z', nbins=60, title="Z distribution",
                           color_discrete_sequence=['#38bdf8'])
        fig.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                          font=dict(color='#94a3b8'), height=300)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig2 = px.box(df, y='z', title="Z box plot",
                      color_discrete_sequence=['#6366f1'])
        fig2.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#0f172a',
                           font=dict(color='#94a3b8'), height=300)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Downloads")
    dl1, dl2, dl3 = st.columns(3)
    dl1.download_button("⬇️ Full dataset CSV", df.to_csv(index=False).encode(),
                        "bna_data.csv", "text/csv")
    dl2.download_button("⬇️ Statistics CSV",  stat_df.to_csv(index=False).encode(),
                        "bna_stats.csv", "text/csv")
    dl3.download_button("⬇️ Level table CSV", level_tbl.to_csv(index=False).encode(),
                        "bna_levels.csv", "text/csv")

    # Plain-text report
    report = f"""BNA CONTOUR EXPLORER — REPORT
==============================
File:        {src}
Generated:   (current session)

STATISTICS
----------
{"".join(f"{k:<14}{v:.6f}\n" for k,v in s.items())}

CONTOUR LEVELS  ({df['z'].nunique()} unique)
--------------
{level_tbl.to_string(index=False)}
"""
    st.download_button("⬇️ Plain-text report", report.encode(),
                       "bna_report.txt", "text/plain")


# ─────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────
TASK_MAP = {
    "Interactive Map":      task_map,
    "Point Query":          task_point_query,
    "Batch CSV Query":      task_batch,
    "Radius / Zone Search": task_radius,
    "Transect Profile":     task_transect,
    "Contour Band Analysis":task_band_analysis,
    "Hotspot Detection":    task_hotspot,
    "File Comparison":      task_comparison,
    "3-D Surface":          task_3d,
    "Export & Report":      task_export,
}

# App title always visible
st.markdown('<h2 style="color:#f1f5f9;font-weight:800;margin-bottom:0">🗺️ BNA Contour Explorer</h2>', unsafe_allow_html=True)
st.markdown('<hr style="border-color:#1e293b;margin-top:8px">', unsafe_allow_html=True)

if st.session_state.df is None:
    upload_screen()
elif st.session_state.task is None:
    task_hub()
else:
    back_button()
    TASK_MAP[st.session_state.task]()
