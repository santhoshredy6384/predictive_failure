import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PredictMaint — Robotic Arm Monitor",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── base ── */
[data-testid="stAppViewContainer"] { background: #0b0f19; }
[data-testid="stSidebar"]          { background: #10151f; border-right: 1px solid #1e2535; }
[data-testid="stHeader"]           { background: transparent; }

/* ── hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── top nav bar ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 28px; background: #10151f;
    border-bottom: 1px solid #1e2535; margin-bottom: 24px;
    border-radius: 0 0 12px 12px;
}
.topbar-left { display: flex; align-items: center; gap: 12px; }
.topbar-logo { font-size: 1.7rem; }
.topbar-title { font-size: 1.1rem; font-weight: 800; color: #e2e8f0; letter-spacing: .02em; }
.topbar-sub   { font-size: 0.75rem; color: #64748b; margin-top: 1px; }
.topbar-badge {
    background: rgba(99,102,241,.15); color: #818cf8;
    border: 1px solid rgba(99,102,241,.35);
    border-radius: 20px; padding: 3px 12px; font-size: 0.72rem; font-weight: 600;
}

/* ── sidebar section heading ── */
.sb-head {
    font-size: .7rem; font-weight: 700; color: #475569;
    letter-spacing: .1em; text-transform: uppercase;
    margin: 18px 0 8px;
}

/* ── sensor card row ── */
.sensor-grid {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 10px; margin-bottom: 24px;
}
.sensor-card {
    background: #10151f; border: 1px solid #1e2535;
    border-radius: 10px; padding: 14px 10px;
    text-align: center; position: relative; overflow: hidden;
}
.sensor-card::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; border-radius: 10px 10px 0 0;
}
.sc-blue::before   { background: #3b82f6; }
.sc-violet::before { background: #8b5cf6; }
.sc-amber::before  { background: #f59e0b; }
.sc-teal::before   { background: #14b8a6; }
.sc-rose::before   { background: #f43f5e; }
.sc-cyan::before   { background: #06b6d4; }
.sc-green::before  { background: #22c55e; }
.sensor-icon  { font-size: 1.3rem; margin-bottom: 4px; }
.sensor-label { font-size: .68rem; color: #64748b; margin-bottom: 4px; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
.sensor-value { font-size: 1.25rem; font-weight: 800; color: #e2e8f0; line-height: 1; }
.sensor-unit  { font-size: .65rem; color: #475569; margin-top: 2px; }

/* ── result banner ── */
.result-banner {
    border-radius: 12px; padding: 24px 32px;
    text-align: center; margin-bottom: 20px;
    border: 1px solid;
}
.result-title { font-size: 1.6rem; font-weight: 800; margin: 0 0 4px; }
.result-sub   { font-size: 1rem; color: #94a3b8; margin: 0; }

/* ── prob bar ── */
.prob-wrap { background: #1e2535; border-radius: 8px; height: 18px; overflow: hidden; margin: 10px 0; }
.prob-fill  { height: 100%; border-radius: 8px; transition: width .6s ease; }

/* ── stat card ── */
.stat-row { display: flex; gap: 12px; margin-bottom: 20px; }
.stat-card {
    flex: 1; background: #10151f; border: 1px solid #1e2535;
    border-radius: 10px; padding: 16px; text-align: center;
}
.stat-card .sv { font-size: 1.5rem; font-weight: 800; color: #818cf8; }
.stat-card .sl { font-size: .75rem; color: #64748b; margin-top: 2px; font-weight: 600; }

/* ── section header ── */
.sec-head {
    font-size: .72rem; font-weight: 700; color: #475569;
    letter-spacing: .1em; text-transform: uppercase;
    border-bottom: 1px solid #1e2535; padding-bottom: 8px;
    margin: 28px 0 16px;
}

/* ── tab styling ── */
[data-baseweb="tab-list"] {
    background: #10151f !important;
    border-radius: 10px; padding: 4px; gap: 4px;
    border: 1px solid #1e2535;
}
[data-baseweb="tab"] {
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: .85rem !important; color: #64748b !important;
    padding: 8px 20px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #1e2b45 !important; color: #818cf8 !important;
}

/* ── dataframe dark ── */
[data-testid="stDataFrame"] { border: 1px solid #1e2535; border-radius: 8px; }

/* ── button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff; border: none; border-radius: 8px;
    font-weight: 700; padding: 10px 28px; font-size: .9rem;
    transition: opacity .2s;
}
[data-testid="stButton"] > button:hover { opacity: .85; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP NAV BAR
# ─────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-logo">🦾</div>
    <div>
      <div class="topbar-title">PredictMaint</div>
      <div class="topbar-sub">Robotic Arm Health Prediction System</div>
    </div>
  </div>
  <div class="topbar-badge">Random Forest · AI Diagnostics</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA & MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model  = joblib.load('random_forest_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found — run predictive_maintenance.py first.")
        return None, None

@st.cache_data
def load_data():
    for candidate in ["preprocessed_training.csv", "preprocessed_testing.csv",
                      "preprocessed_dataset.csv", "raw_data.csv"]:
        if os.path.exists(candidate):
            data = pd.read_csv(candidate)
            if "start_date" in data.columns:
                data["start_date"] = pd.to_datetime(data["start_date"])
            return data
    st.error("Data file not found — run predictive_maintenance.py first.")
    return None

@st.cache_data
def compute_confusion_matrix_data():
    model, scaler = load_artifacts()
    for candidate in ["preprocessed_testing.csv", "preprocessed_training.csv"]:
        if os.path.exists(candidate):
            df = pd.read_csv(candidate)
            break
    else:
        return None, None, None

    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
        df = df.dropna(subset=["start_date"])
        df["age_days"] = (datetime.now() - df["start_date"]).dt.days

    feat = ["temperature","vibration","torque","pressure","volt","rotate","age_days"]
    if any(c not in df.columns for c in feat) or "label" not in df.columns:
        return None, None, None

    X_scaled = scaler.transform(df[feat])
    y_pred   = model.predict(X_scaled)
    y_true   = df["label"]
    cm       = confusion_matrix(y_true, y_pred)
    report   = classification_report(y_true, y_pred,
                   target_names=["Healthy","Warning","Critical"],
                   zero_division=0, output_dict=True)
    return cm, y_true, report

try:
    model, scaler = load_artifacts()
    data          = load_data()
except Exception as e:
    st.error(f"Initialisation error: {e}")
    st.stop()

if model is None or data is None:
    st.stop()

FEATURES = ["temperature","vibration","torque","pressure","volt","rotate","age_days"]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-head">Input Mode</div>', unsafe_allow_html=True)
    input_mode = st.radio("Input Mode", ["Select Robot ID", "Manual Simulation"],
                          label_visibility="collapsed")

    st.markdown('<div class="sb-head">Robot Configuration</div>', unsafe_allow_html=True)

    selected_row = None
    arm_id       = None
    arm_rows     = None

    if input_mode == "Select Robot ID":
        available_ids = sorted(data["robotic_arm_id"].unique())
        arm_id        = st.selectbox("Robotic Arm ID", available_ids)
        arm_rows      = data[data["robotic_arm_id"] == arm_id]
        if not arm_rows.empty:
            selected_row = (arm_rows.sort_values("start_date").iloc[-1]
                            if "start_date" in arm_rows.columns
                            else arm_rows.iloc[-1])
    else:
        st.markdown('<div class="sb-head">Sensor Parameters</div>', unsafe_allow_html=True)
        manual_input = {
            "temperature": st.slider("🌡 Temperature (°C)",   20.0,  100.0,  60.0),
            "vibration":   st.slider("📳 Vibration (Hz)",      0.0,  100.0,  30.0),
            "torque":      st.slider("🔩 Torque (Nm)",          0.0,  100.0,  40.0),
            "pressure":    st.slider("💨 Pressure (Bar)",       0.0,  100.0,  30.0),
            "volt":        st.slider("⚡ Voltage (V)",        150.0,  300.0, 220.0),
            "rotate":      st.slider("🔄 Rotation (RPM)",       0.0, 3000.0,1500.0),
            "age_days":    st.slider("📅 Age (Days)",              0,   5000,   500),
        }
        selected_row = pd.Series(manual_input)
        arm_id = "MANUAL-SIM"

    # ── fleet summary ──
    st.markdown('<div class="sb-head">Fleet Overview</div>', unsafe_allow_html=True)
    total_arms = data["robotic_arm_id"].nunique() if "robotic_arm_id" in data.columns else "—"
    total_rows = len(data)
    st.markdown(f"""
    <div style="background:#0b0f19;border:1px solid #1e2535;border-radius:8px;padding:12px 14px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
        <span style="color:#64748b;font-size:.78rem;">Total Arms</span>
        <span style="color:#818cf8;font-weight:700;">{total_arms}</span>
      </div>
      <div style="display:flex;justify-content:space-between;">
        <span style="color:#64748b;font-size:.78rem;">Total Records</span>
        <span style="color:#818cf8;font-weight:700;">{total_rows}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_diag, tab_analytics, tab_history = st.tabs(
    ["🔍  Diagnostics", "📊  Model Analytics", "📋  History"])

# ══════════════════════════════════════════════
# TAB 1 — DIAGNOSTICS
# ══════════════════════════════════════════════
with tab_diag:
    if selected_row is None:
        st.info("Select a Robotic Arm from the sidebar to begin.")
    else:
        # ── arm header ──
        col_hd, col_ts = st.columns([3, 1])
        with col_hd:
            st.markdown(f"""
            <h2 style="color:#e2e8f0;margin:0;font-size:1.4rem;font-weight:800;">
              🦾 Arm ID — <span style="color:#818cf8;">{arm_id}</span>
            </h2>
            <p style="color:#64748b;margin:4px 0 0;font-size:.82rem;">
              Latest sensor snapshot · {datetime.now().strftime("%d %b %Y, %H:%M")}
            </p>
            """, unsafe_allow_html=True)
        with col_ts:
            st.markdown(f"""
            <div style="text-align:right;padding-top:4px;">
              <span style="background:#1e2535;border:1px solid #2d3748;border-radius:6px;
                           padding:6px 14px;font-size:.78rem;color:#94a3b8;">
                Mode: <b style="color:#818cf8;">{input_mode}</b>
              </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sec-head">Live Sensor Readings</div>', unsafe_allow_html=True)

        # ── sensor cards ──
        sensors = [
            ("🌡", "Temperature", f"{selected_row['temperature']:.1f}", "°C",  "sc-blue"),
            ("📳", "Vibration",   f"{selected_row['vibration']:.1f}",   "Hz",  "sc-violet"),
            ("🔩", "Torque",      f"{selected_row['torque']:.1f}",      "Nm",  "sc-amber"),
            ("💨", "Pressure",    f"{selected_row['pressure']:.1f}",    "Bar", "sc-teal"),
            ("⚡", "Voltage",     f"{selected_row['volt']:.1f}",        "V",   "sc-rose"),
            ("🔄", "Rotation",    f"{selected_row['rotate']:.0f}",      "RPM", "sc-cyan"),
            ("📅", "Age",         f"{int(selected_row['age_days'])}",   "Days","sc-green"),
        ]
        cards_html = '<div class="sensor-grid">'
        for icon, label, val, unit, cls in sensors:
            cards_html += f"""
            <div class="sensor-card {cls}">
              <div class="sensor-icon">{icon}</div>
              <div class="sensor-label">{label}</div>
              <div class="sensor-value">{val}</div>
              <div class="sensor-unit">{unit}</div>
            </div>"""
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

        st.markdown('<div class="sec-head">AI Diagnostics</div>', unsafe_allow_html=True)

        # ── run button ──
        run_diag = st.button("⚡  Run AI Diagnostics", use_container_width=False)

        if run_diag:
            with st.spinner("Analysing sensor patterns with Random Forest…"):
                input_df     = selected_row[FEATURES].to_frame().T
                input_scaled = scaler.transform(input_df)
                probs        = model.predict_proba(input_scaled)[0]

                classes  = model.classes_
                prob_map = {cls: p for cls, p in zip(classes, probs)}

                failure_prob  = prob_map.get(1.0, 0.0) + prob_map.get(2.0, 0.0)
                healthy_prob  = prob_map.get(0.0, 0.0)
                warning_prob  = prob_map.get(1.0, 0.0)
                critical_prob = prob_map.get(2.0, 0.0)

                if failure_prob >= 0.80:
                    status      = "CRITICAL — Immediate Action Required"
                    s_color     = "#ef4444"
                    s_bg        = "rgba(239,68,68,.08)"
                    s_border    = "#ef4444"
                    s_icon      = "🚨"
                elif failure_prob >= 0.50:
                    status      = "WARNING — Maintenance Needed Soon"
                    s_color     = "#f59e0b"
                    s_bg        = "rgba(245,158,11,.08)"
                    s_border    = "#f59e0b"
                    s_icon      = "⚠️"
                else:
                    status      = "HEALTHY — Operating Optimally"
                    s_color     = "#22c55e"
                    s_bg        = "rgba(34,197,94,.08)"
                    s_border    = "#22c55e"
                    s_icon      = "✅"

                # result banner
                st.markdown(f"""
                <div class="result-banner"
                     style="background:{s_bg};border-color:{s_border};">
                  <div class="result-title" style="color:{s_color};">
                    {s_icon} &nbsp; {status}
                  </div>
                  <p class="result-sub">Failure Probability: <b style="color:{s_color};">{failure_prob*100:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

                # probability gauge bar
                fill_color = s_color
                st.markdown(f"""
                <div style="margin:4px 0 18px;">
                  <div style="display:flex;justify-content:space-between;font-size:.72rem;color:#64748b;margin-bottom:4px;">
                    <span>0%</span><span>Failure Risk</span><span>100%</span>
                  </div>
                  <div class="prob-wrap">
                    <div class="prob-fill"
                         style="width:{failure_prob*100:.1f}%;background:{fill_color};"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # three-class probability breakdown
                col_g, col_b = st.columns([1, 1])

                with col_g:
                    # matplotlib gauge
                    fig, ax = plt.subplots(figsize=(4.5, 2.5),
                                           facecolor="#10151f", subplot_kw=dict(polar=False))
                    ax.set_facecolor("#10151f")
                    categories = ["Healthy", "Warning", "Critical"]
                    values     = [healthy_prob, warning_prob, critical_prob]
                    bar_colors = ["#22c55e", "#f59e0b", "#ef4444"]
                    bars = ax.barh(categories, values, color=bar_colors,
                                   height=0.5, edgecolor="none")
                    for bar, val in zip(bars, values):
                        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                                f"{val*100:.1f}%",
                                va="center", ha="left",
                                color="#e2e8f0", fontsize=9, fontweight="bold")
                    ax.set_xlim(0, 1.2)
                    ax.set_xlabel("Probability", color="#64748b", fontsize=8)
                    ax.tick_params(colors="#94a3b8", labelsize=9)
                    ax.spines[:].set_color("#1e2535")
                    ax.xaxis.set_tick_params(color="#1e2535")
                    plt.tight_layout(pad=0.5)
                    st.pyplot(fig)
                    plt.close(fig)

                with col_b:
                    st.markdown('<div class="sec-head" style="margin-top:8px;">Breakdown</div>',
                                unsafe_allow_html=True)
                    for label, prob, color in [
                        ("Healthy",  healthy_prob,  "#22c55e"),
                        ("Warning",  warning_prob,  "#f59e0b"),
                        ("Critical", critical_prob, "#ef4444"),
                    ]:
                        st.markdown(f"""
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;margin-bottom:10px;">
                          <span style="color:#94a3b8;font-size:.85rem;">{label}</span>
                          <span style="background:rgba(255,255,255,.06);
                                       border:1px solid {color}44;
                                       color:{color};font-weight:700;
                                       border-radius:6px;padding:2px 12px;
                                       font-size:.85rem;">{prob*100:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — MODEL ANALYTICS
# ══════════════════════════════════════════════
with tab_analytics:
    cm, y_true, report = compute_confusion_matrix_data()

    if cm is None:
        st.info("Run predictive_maintenance.py first to generate performance data.")
    else:
        labels = ["Healthy", "Warning", "Critical"]
        acc    = report.get("accuracy", 0)

        # ── KPI row ──
        st.markdown('<div class="sec-head">Model KPIs</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        def kpi(col, val, label):
            col.markdown(f"""
            <div class="stat-card">
              <div class="sv">{val}</div>
              <div class="sl">{label}</div>
            </div>""", unsafe_allow_html=True)

        kpi(k1, f"{acc*100:.1f}%", "Overall Accuracy")
        kpi(k2, f"{report['Healthy']['f1-score']*100:.1f}%",  "Healthy F1")
        kpi(k3, f"{report['Warning']['f1-score']*100:.1f}%",  "Warning F1")
        kpi(k4, f"{report['Critical']['f1-score']*100:.1f}%", "Critical F1")

        # ── confusion matrix + report ──
        st.markdown('<div class="sec-head">Confusion Matrix & Classification Report</div>',
                    unsafe_allow_html=True)
        col_cm, col_rep = st.columns([1, 1])

        with col_cm:
            fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor="#10151f")
            ax.set_facecolor("#10151f")
            sns.heatmap(
                cm, annot=True, fmt="d",
                cmap=sns.dark_palette("#818cf8", as_cmap=True),
                xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor="#0b0f19",
                annot_kws={"size": 13, "color": "#fff", "weight": "bold"},
                ax=ax,
            )
            ax.set_xlabel("Predicted", color="#94a3b8", fontsize=10)
            ax.set_ylabel("Actual",    color="#94a3b8", fontsize=10)
            ax.set_title("Confusion Matrix", color="#e2e8f0",
                         fontsize=12, fontweight="bold", pad=14)
            ax.tick_params(colors="#94a3b8")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_rep:
            report_rows = []
            for cls in labels:
                r = report.get(cls, {})
                report_rows.append({
                    "Class":     cls,
                    "Precision": f"{r.get('precision',0)*100:.1f}%",
                    "Recall":    f"{r.get('recall',0)*100:.1f}%",
                    "F1-Score":  f"{r.get('f1-score',0)*100:.1f}%",
                    "Support":   int(r.get('support', 0)),
                })
            st.dataframe(
                pd.DataFrame(report_rows).set_index("Class"),
                use_container_width=True,
            )

            # per-class F1 bar chart
            fig2, ax2 = plt.subplots(figsize=(5, 2.8), facecolor="#10151f")
            ax2.set_facecolor("#10151f")
            cls_names = ["Healthy", "Warning", "Critical"]
            f1_vals   = [report[c]["f1-score"] for c in cls_names]
            bar_clrs  = ["#22c55e", "#f59e0b", "#ef4444"]
            bar2 = ax2.bar(cls_names, [v*100 for v in f1_vals],
                           color=bar_clrs, width=0.45, edgecolor="none")
            for b, v in zip(bar2, f1_vals):
                ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8,
                         f"{v*100:.1f}%", ha="center", color="#e2e8f0",
                         fontsize=8.5, fontweight="bold")
            ax2.set_ylim(0, 115)
            ax2.set_ylabel("F1-Score (%)", color="#64748b", fontsize=8)
            ax2.tick_params(colors="#94a3b8", labelsize=8)
            ax2.spines[:].set_color("#1e2535")
            ax2.set_title("Per-Class F1 Score", color="#e2e8f0",
                          fontsize=9, fontweight="bold")
            plt.tight_layout(pad=0.4)
            st.pyplot(fig2)
            plt.close(fig2)

        # ── label distribution ──
        st.markdown('<div class="sec-head">Dataset Label Distribution</div>',
                    unsafe_allow_html=True)
        dist_col1, dist_col2 = st.columns(2)
        with dist_col1:
            dist = y_true.value_counts().sort_index()
            label_map = {0: "Healthy", 1: "Warning", 2: "Critical"}
            fig3, ax3 = plt.subplots(figsize=(4.5, 3), facecolor="#10151f")
            ax3.set_facecolor("#10151f")
            wedge_clrs = ["#22c55e", "#f59e0b", "#ef4444"]
            wedge_lbls = [label_map.get(i, str(i)) for i in dist.index]
            wedges, texts, auts = ax3.pie(
                dist.values, labels=None, autopct="%1.1f%%",
                colors=wedge_clrs[:len(dist)],
                startangle=140,
                pctdistance=0.75,
                wedgeprops=dict(linewidth=2, edgecolor="#0b0f19"),
            )
            for at in auts:
                at.set_color("#fff"); at.set_fontsize(9)
            ax3.legend(wedges, wedge_lbls, loc="lower center",
                       ncol=3, frameon=False,
                       labelcolor="#94a3b8", fontsize=8)
            ax3.set_title("Class Distribution", color="#e2e8f0",
                          fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

        with dist_col2:
            dist_df = pd.DataFrame({
                "Class":   wedge_lbls,
                "Samples": dist.values,
                "Share":   [f"{v/dist.values.sum()*100:.1f}%" for v in dist.values],
            })
            st.dataframe(dist_df.set_index("Class"), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════
with tab_history:
    if input_mode != "Select Robot ID" or arm_rows is None or arm_rows.empty:
        st.info("Switch to **Select Robot ID** mode in the sidebar to see historical records.")
    else:
        st.markdown(f'<div class="sec-head">History for Arm {arm_id}</div>', unsafe_allow_html=True)
        show_df = arm_rows.copy()
        if "start_date" in show_df.columns:
            show_df = show_df.sort_values("start_date", ascending=False)
        st.dataframe(show_df, use_container_width=True)

        # ── sensor trend ──
        if "start_date" in show_df.columns and len(show_df) > 1:
            st.markdown('<div class="sec-head">Sensor Trends Over Time</div>',
                        unsafe_allow_html=True)
            trend_df = show_df.sort_values("start_date")
            sensor_cols = ["temperature", "vibration", "torque", "pressure", "volt", "rotate"]
            available   = [c for c in sensor_cols if c in trend_df.columns]

            fig4, axes = plt.subplots(
                len(available), 1,
                figsize=(9, 2.2 * len(available)),
                facecolor="#10151f", sharex=True
            )
            if len(available) == 1:
                axes = [axes]

            line_colors = ["#3b82f6","#8b5cf6","#f59e0b","#14b8a6","#f43f5e","#06b6d4"]
            for ax_i, (col, lc) in enumerate(zip(available, line_colors)):
                ax = axes[ax_i]
                ax.set_facecolor("#10151f")
                ax.plot(trend_df["start_date"], trend_df[col],
                        color=lc, linewidth=1.8, marker="o",
                        markersize=3, markerfacecolor=lc)
                ax.fill_between(trend_df["start_date"], trend_df[col],
                                alpha=0.12, color=lc)
                ax.set_ylabel(col.capitalize(), color="#94a3b8", fontsize=8)
                ax.tick_params(colors="#64748b", labelsize=7)
                ax.spines[:].set_color("#1e2535")

            axes[-1].tick_params(axis="x", labelrotation=25, colors="#64748b")
            plt.suptitle(f"Sensor History — Arm {arm_id}",
                         color="#e2e8f0", fontsize=10, fontweight="bold", y=1.01)
            plt.tight_layout(pad=0.6)
            st.pyplot(fig4)
            plt.close(fig4)
