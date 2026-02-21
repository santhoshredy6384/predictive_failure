import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def load_and_clean(filepath):
    """Load CSV, remove duplicates, coerce numerics, drop NaN rows, compute age_days."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found.")
        sys.exit(1)

    df = df.drop_duplicates()

    numeric_cols = ["temperature", "vibration", "torque", "pressure", "volt", "rotate"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)

    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df = df.dropna(subset=["start_date"])
    df["age_days"] = (datetime.now() - df["start_date"]).dt.days

    return df.reset_index(drop=True)


def calculate_risk_score(row):
    """Deterministic risk score based on sensor deviations and machine age."""
    score = 0.0
    score += max(0, (row["temperature"] - 60) * 0.03)
    score += max(0, (row["vibration"]   - 30) * 0.04)
    score += max(0, (row["torque"]      - 40) * 0.03)
    score += row["age_days"] / 1000.0
    return score


def create_label(risk_val):
    if risk_val > 4:
        return 2   # Critical
    elif risk_val > 2:
        return 1   # Warning
    else:
        return 0   # Healthy


def diagnose_fit(train_acc, test_acc, gap_threshold=0.10, low_acc_threshold=0.80):
    """Return (diagnosis_string, gap) given train/test accuracies."""
    gap = train_acc - test_acc
    if train_acc < low_acc_threshold and test_acc < low_acc_threshold:
        return "UNDERFITTING", gap
    elif gap > gap_threshold:
        return "OVERFITTING", gap
    else:
        return "GOOD FIT", gap


# =====================================================
# LOAD & PREPROCESS BOTH DATASETS
# =====================================================

print("\n" + "=" * 60)
print("  PREDICTIVE MAINTENANCE — MODEL TRAINING & EVALUATION")
print("=" * 60)

print("\n[+] Loading training data  : D:\\Clg\\Learnthon\\Project\\set\\raw_training.csv")
train_df = load_and_clean(r"D:\Clg\Learnthon\Project\set\raw_training.csv")
train_df["risk"]  = train_df.apply(calculate_risk_score, axis=1)
train_df["label"] = train_df["risk"].apply(create_label)

print("[+] Loading testing data   : D:\\Clg\\Learnthon\\Project\\set\\raw_testing.csv")
test_df = load_and_clean(r"D:\Clg\Learnthon\Project\set\raw_testing.csv")
test_df["risk"]  = test_df.apply(calculate_risk_score, axis=1)
test_df["label"] = test_df["risk"].apply(create_label)

FEATURES = ["temperature", "vibration", "torque", "pressure", "volt", "rotate", "age_days"]

X_train = train_df[FEATURES]
y_train = train_df["label"]
X_test  = test_df[FEATURES]
y_test  = test_df["label"]

print(f"\n  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")

print(f"\n  Training label distribution (0=Healthy, 1=Warning, 2=Critical):")
print(f"  {y_train.value_counts().sort_index().to_dict()}")
print(f"\n  Testing  label distribution:")
print(f"  {y_test.value_counts().sort_index().to_dict()}")

# Save processed files for reference
train_df.to_csv("preprocessed_training.csv", index=False)
test_df.to_csv("preprocessed_testing.csv",   index=False)
print("\n  Preprocessed files saved: preprocessed_training.csv, preprocessed_testing.csv")

# =====================================================
# SCALE — fit ONLY on training data to avoid data leakage
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)      # transform only — no leakage

# =====================================================
# STEP 1: INITIAL MODEL (unconstrained RandomForest)
# =====================================================
print("\n" + "=" * 60)
print("  STEP 1 — INITIAL MODEL (unconstrained RandomForest)")
print("=" * 60)

model_init = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,        # fully grown trees → likely to overfit
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
model_init.fit(X_train_scaled, y_train)

train_acc_init = accuracy_score(y_train, model_init.predict(X_train_scaled))
test_acc_init  = accuracy_score(y_test,  model_init.predict(X_test_scaled))
diagnosis_init, gap_init = diagnose_fit(train_acc_init, test_acc_init)

print(f"\n  Train Accuracy  : {train_acc_init:.4f}")
print(f"  Test  Accuracy  : {test_acc_init:.4f}")
print(f"  Gap (Train-Test): {gap_init:.4f}")

print("\n" + "=" * 60)
print("  DIAGNOSIS")
print("=" * 60)

if diagnosis_init == "UNDERFITTING":
    print(f"\n  [!] UNDERFITTING detected")
    print(f"      Both Train ({train_acc_init:.4f}) and Test ({test_acc_init:.4f}) accuracy are low.")
    print(f"      Model is too simple / has not learned the data patterns.")
elif diagnosis_init == "OVERFITTING":
    print(f"\n  [!] OVERFITTING detected")
    print(f"      Train: {train_acc_init:.4f}  >>  Test: {test_acc_init:.4f}  (gap = {gap_init:.4f})")
    print(f"      Model memorises training data and generalises poorly to new data.")
else:
    print(f"\n  [+] GOOD FIT")
    print(f"      Train: {train_acc_init:.4f} | Test: {test_acc_init:.4f} | Gap: {gap_init:.4f}")

# =====================================================
# STEP 2: APPLY CORRECTION IF NEEDED
# =====================================================
print("\n" + "=" * 60)
print("  STEP 2 — CORRECTION")
print("=" * 60)

if diagnosis_init == "UNDERFITTING":
    print("\n  [*] Fix: Increasing model complexity")
    print("      → n_estimators=300, no depth limit, min_samples_split=2")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

elif diagnosis_init == "OVERFITTING":
    print("\n  [*] Fix: Regularising the model")
    print("      → max_depth=12, min_samples_split=10, min_samples_leaf=5")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

else:
    print("\n  [+] No correction needed — keeping initial model.")
    model = model_init

# =====================================================
# STEP 3: FINAL EVALUATION
# =====================================================
train_acc_final = accuracy_score(y_train, model.predict(X_train_scaled))
test_acc_final  = accuracy_score(y_test,  model.predict(X_test_scaled))
diagnosis_final, gap_final = diagnose_fit(train_acc_final, test_acc_final)

print("\n" + "=" * 60)
print("  STEP 3 — FINAL RESULTS")
print("=" * 60)
print(f"\n  Train Accuracy  : {train_acc_final:.4f}")
print(f"  Test  Accuracy  : {test_acc_final:.4f}")
print(f"  Gap (Train-Test): {gap_final:.4f}")
print(f"  Final Diagnosis : {diagnosis_final}")

LABEL_NAMES  = ["Healthy", "Warning", "Critical"]
y_pred_test  = model.predict(X_test_scaled)
y_pred_train = model.predict(X_train_scaled)

print("\n  --- Classification Report — TRAINING SET ---")
print(classification_report(y_train, y_pred_train, target_names=LABEL_NAMES, zero_division=0))

print("  --- Classification Report — TEST SET ---")
print(classification_report(y_test, y_pred_test, target_names=LABEL_NAMES, zero_division=0))

print("  --- Confusion Matrix — TEST SET ---")
cm    = confusion_matrix(y_test, y_pred_test)
cm_df = pd.DataFrame(
    cm,
    index  =[f"True {l}"  for l in LABEL_NAMES],
    columns=[f"Pred {l}"  for l in LABEL_NAMES]
)
print(cm_df.to_string())

# Per-class TP / TN / FP / FN
print("\n  --- Per-Class TP / TN / FP / FN ---")
total = cm.sum()
for i, cls in enumerate(LABEL_NAMES):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = total - TP - FN - FP
    print(f"  {cls:>8}  |  TP={TP:6d}  TN={TN:6d}  FP={FP:6d}  FN={FN:6d}")

# Build custom annotations: diagonal=TP, off-diagonal=FP/FN
annot = np.empty_like(cm, dtype=object)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if i == j:
            annot[i, j] = f"{cm[i,j]}\n(TP)"
        else:
            annot[i, j] = f"{cm[i,j]}\n(FN/FP)"

# Plot confusion matrix with TP/TN/FP/FN annotations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: annotated heatmap
sns.heatmap(
    cm, annot=annot, fmt="", cmap="Blues",
    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
    linewidths=0.5, ax=axes[0]
)
axes[0].set_xlabel("Predicted Label", fontsize=12)
axes[0].set_ylabel("True Label", fontsize=12)
axes[0].set_title("Confusion Matrix — Test Set\n(diagonal=TP, off-diagonal=FN/FP)", fontsize=12, fontweight="bold")

# Right: per-class TP/TN/FP/FN bar chart
tp_vals, tn_vals, fp_vals, fn_vals = [], [], [], []
for i in range(len(LABEL_NAMES)):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = total - TP - FN - FP
    tp_vals.append(TP); tn_vals.append(TN)
    fp_vals.append(FP); fn_vals.append(FN)

x = np.arange(len(LABEL_NAMES))
width = 0.2
axes[1].bar(x - 1.5*width, tp_vals, width, label="TP", color="#2196F3")
axes[1].bar(x - 0.5*width, tn_vals, width, label="TN", color="#4CAF50")
axes[1].bar(x + 0.5*width, fp_vals, width, label="FP", color="#FF9800")
axes[1].bar(x + 1.5*width, fn_vals, width, label="FN", color="#F44336")
axes[1].set_xticks(x)
axes[1].set_xticklabels(LABEL_NAMES)
axes[1].set_title("Per-Class TP / TN / FP / FN", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Count")
axes[1].legend()

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
print("\n[+] Confusion matrix saved: confusion_matrix.png")
plt.show()

# =====================================================
# STEP 4: CROSS-VALIDATION (5-fold on training data)
# =====================================================
print("\n  --- 5-Fold Cross-Validation on Training Data ---")
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy", n_jobs=-1)
print(f"  CV Scores : {np.round(cv_scores, 4)}")
print(f"  CV Mean   : {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}")

# =====================================================
# SAVE MODEL & SCALER
# =====================================================
print("\n[+] Saving model and scaler...")
joblib.dump(model,  "random_forest_model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("    Saved: random_forest_model.joblib")
print("    Saved: scaler.joblib")

# =====================================================
# INTERACTIVE PREDICTION LOOP
# =====================================================
print("\n" + "=" * 60)
print("  AI MONITORING SYSTEM READY")
print("=" * 60)

all_data      = pd.concat([train_df, test_df]).drop_duplicates(subset=["robotic_arm_id"], keep="last")
available_ids = sorted(all_data["robotic_arm_id"].unique())
print(f"\n  Available Robotic Arms: {min(available_ids)} to {max(available_ids)}"
      f"  (Total: {len(available_ids)})\n")


def predict_and_report(arm_id):
    arm_rows = all_data[all_data["robotic_arm_id"] == arm_id]
    if arm_rows.empty:
        print(f"  ID {arm_id} not found in dataset.\n")
        return

    latest    = arm_rows.sort_values("start_date").iloc[-1]
    sample_df = latest[FEATURES].to_frame().T
    sample_sc = scaler.transform(sample_df)
    probs     = model.predict_proba(sample_sc)[0]

    failure_prob = sum(
        probs[i] for i, cls in enumerate(model.classes_) if cls in [1, 2]
    )

    if failure_prob >= 0.80:
        status = "CRITICAL (Immediate Maintenance Required)"
    elif failure_prob >= 0.50:
        status = "WARNING (Schedule Maintenance)"
    else:
        status = "HEALTHY (Optimal Operation)"

    print(f"\n  Report for Robotic Arm {arm_id}")
    print("  " + "-" * 35)
    print(f"  Operational Since   : {latest['start_date']}")
    print(f"  Current Status      : {status}")
    print(f"  Failure Probability : {failure_prob * 100:.2f}%")
    print("  " + "-" * 35 + "\n")


# --test flag for non-interactive CI / quick-check runs
if len(sys.argv) > 1 and sys.argv[1] == "--test":
    print("\n  [Test mode] Running prediction for Arm ID 54...")
    predict_and_report(54)
    sys.exit(0)

while True:
    try:
        user_input = input("Enter Robotic Arm ID (-1 to exit): ").strip()
        if user_input == "-1":
            print("System stopped.")
            break
        if not user_input.isdigit():
            print("  Please enter a valid numeric ID.\n")
            continue
        predict_and_report(int(user_input))
    except KeyboardInterrupt:
        print("\nSystem stopped.")
        break
    except Exception as e:
        print(f"  Error: {e}\n")
