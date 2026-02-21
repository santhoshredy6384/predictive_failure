import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Set page config
st.set_page_config(
    page_title="Robotic Arm Monitor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Robotic Arm Health Prediction System")
st.markdown("Monitor real-time health status of robotic arms based on sensor data.")

# =====================================================
# DATA & MODEL LOADING
# =====================================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('random_forest_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found! Please run 'predictive_maintenance.py' first to generate model artifacts.")
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
    st.error("Data file not found! Please run 'predictive_maintenance.py' first.")
    return None

try:
    model, scaler = load_artifacts()
    data = load_data()
except Exception as e:
    st.error(f"Error initializing app: {e}")
    st.stop()

if model is not None and data is not None:
    
    # AI Agent Interface
    st.markdown("### 🔍 Diagnostics Interface")
    
    # Sidebar for inputs helps organize the UI
    st.sidebar.header("Configuration")
    input_mode = st.sidebar.radio("Input Mode:", ["Select Existing Robot ID", "Manual Simulation"])

    # Features expected by the model
    feature_names = [
        "temperature","vibration","torque",
        "pressure","volt","rotate","age_days"
    ]

    selected_row = None
    arm_id = None

    if input_mode == "Select Existing Robot ID":
        available_ids = sorted(data["robotic_arm_id"].unique())
        arm_id = st.sidebar.selectbox("Select Robotic Arm ID:", available_ids)
        
        # Get the latest entry for this exam
        arm_rows = data[data["robotic_arm_id"] == arm_id]
        if not arm_rows.empty:
            # Sort by date if possible, else take last
            if "start_date" in arm_rows.columns:
                selected_row = arm_rows.sort_values('start_date').iloc[-1]
            else:
                selected_row = arm_rows.iloc[-1]
    
    else: # Manual Simulation
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Adjust Sensor Parameters:**")
        
        # Create a dictionary for manual input
        manual_input = {}
        manual_input["temperature"] = st.sidebar.slider("Temperature (°C)", 20.0, 100.0, 60.0)
        manual_input["vibration"] = st.sidebar.slider("Vibration (Hz)", 0.0, 100.0, 30.0)
        manual_input["torque"] = st.sidebar.slider("Torque (Nm)", 0.0, 100.0, 40.0)
        manual_input["pressure"] = st.sidebar.slider("Pressure (Bar)", 0.0, 100.0, 30.0)
        manual_input["volt"] = st.sidebar.slider("Voltage (V)", 150.0, 300.0, 220.0)
        manual_input["rotate"] = st.sidebar.slider("Rotation (RPM)", 0.0, 3000.0, 1500.0)
        manual_input["age_days"] = st.sidebar.slider("Age (Days)", 0, 5000, 500)
        
        selected_row = pd.Series(manual_input)
        arm_id = "MANUAL-SIM"

    # Main Dashboard Logic
    if selected_row is not None:
        
        # Display Current Sensor Readings
        st.subheader(f"📊 Sensor Readings (ID: {arm_id})")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temperature", f"{selected_row['temperature']:.1f} °C")
        c2.metric("Vibration", f"{selected_row['vibration']:.1f} Hz")
        c3.metric("Torque", f"{selected_row['torque']:.1f} Nm")
        c4.metric("Age", f"{int(selected_row['age_days'])} Days")
        
        c5, c6, c7 = st.columns(3)
        c5.metric("Pressure", f"{selected_row['pressure']:.1f} Bar")
        c6.metric("Voltage", f"{selected_row['volt']:.1f} V")
        c7.metric("Rotation", f"{selected_row['rotate']:.0f} RPM")
        
        st.divider()

        # RUN PREDICTION
        if st.button("Run AI Diagnostics", type="primary"):
            
            with st.spinner("Analyzing sensor patterns..."):
                # Prepare data for model
                input_df = selected_row[feature_names].to_frame().T
                
                # Scale features
                input_scaled = scaler.transform(input_df)
                
                # Predict
                probs = model.predict_proba(input_scaled)[0]
                
                # Calculate Failure Probability (Warning + Critical)
                # Model classes are typically [0.0, 1.0, 2.0] for Healthy, Warning, Critical
                # But we need to handle cases where not all classes are present
                
                failure_prob = 0.0
                classes = model.classes_
                
                # Map probabilities to class labels
                prob_map = {cls: p for cls, p in zip(classes, probs)}
                
                # Sum probabilities for Warning (1.0) and Critical (2.0)
                if 1.0 in prob_map:
                    failure_prob += prob_map[1.0]
                if 2.0 in prob_map:
                    failure_prob += prob_map[2.0]
                
                # Determine Status Label (Status Logic from predictive_maintenance.py)
                status = "HEALTHY"
                status_color = "green"
                
                if failure_prob >= 0.80:
                    status = "CRITICAL (Immediate Action)"
                    status_color = "#ff4b4b" # Streamlit Red
                elif failure_prob >= 0.50:
                    status = "WARNING (Maintenance Needed)"
                    status_color = "#ffa421" # Streamlit Orange
                else:
                    status = "HEALTHY (Optimal)"
                    status_color = "#00c853" # Green
                
                # Display Results
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; border: 2px solid {status_color}; text-align: center; margin-bottom: 20px;">
                    <h2 style="color: {status_color}; margin:0;">{status}</h2>
                    <h3 style="margin:5px;">Failure Probability: {failure_prob*100:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(failure_prob)
                
                # Detailed breakdown
                st.write("**Detailed Probability Distribution:**")
                chart_data = pd.DataFrame({
                    "Status": ["Healthy", "Warning", "Critical"],
                    "Probability": [
                        prob_map.get(0.0, 0.0), 
                        prob_map.get(1.0, 0.0), 
                        prob_map.get(2.0, 0.0)
                    ]
                })
                st.bar_chart(chart_data.set_index("Status"))

        # Historical View (Only for existing IDs)
        if input_mode == "Select Existing Robot ID":
            with st.expander("View Historical Data for this Arm"):
                st.dataframe(arm_rows.sort_values('start_date', ascending=False))

    # =====================================================
    # CONFUSION MATRIX SECTION
    # =====================================================
    st.divider()
    st.subheader("📉 Model Performance — Confusion Matrix")

    @st.cache_data
    def compute_confusion_matrix():
        test_path = None
        for candidate in ["preprocessed_testing.csv", "preprocessed_training.csv"]:
            if os.path.exists(candidate):
                test_path = candidate
                break
        if test_path is None:
            return None, None, None

        df = pd.read_csv(test_path)
        if "start_date" in df.columns:
            df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
            df = df.dropna(subset=["start_date"])
            df["age_days"] = (datetime.now() - df["start_date"]).dt.days

        feat = ["temperature", "vibration", "torque", "pressure", "volt", "rotate", "age_days"]
        missing = [c for c in feat if c not in df.columns]
        if missing:
            return None, None, None

        X = df[feat]
        y_true = df["label"] if "label" in df.columns else None
        if y_true is None:
            return None, None, None

        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred,
                                       target_names=["Healthy", "Warning", "Critical"],
                                       zero_division=0, output_dict=True)
        return cm, y_true, report

    cm, y_true, report = compute_confusion_matrix()

    if cm is not None:
        labels = ["Healthy", "Warning", "Critical"]
        col_cm, col_rep = st.columns([1, 1])

        with col_cm:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax
            )
            ax.set_xlabel("Predicted Label", fontsize=12)
            ax.set_ylabel("True Label", fontsize=12)
            ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_rep:
            st.markdown("**Classification Report**")
            report_rows = []
            for cls in ["Healthy", "Warning", "Critical"]:
                r = report.get(cls, {})
                report_rows.append({
                    "Class": cls,
                    "Precision": f"{r.get('precision', 0):.2f}",
                    "Recall": f"{r.get('recall', 0):.2f}",
                    "F1-Score": f"{r.get('f1-score', 0):.2f}",
                    "Support": int(r.get('support', 0))
                })
            st.dataframe(pd.DataFrame(report_rows).set_index("Class"), use_container_width=True)

            acc = report.get("accuracy", 0)
            st.metric("Overall Accuracy", f"{acc*100:.2f}%")
    else:
        st.info("Run 'predictive_maintenance.py' first to generate preprocessed data for the confusion matrix.")
