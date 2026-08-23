# -*- coding: utf-8 -*-
"""
=============================================================================
 MASTER PIPELINE
 "Privacy-Utility Trade-offs in Healthcare Machine Learning: A Differential
  Privacy and Ensemble Averaging Approach"
=============================================================================

Runs, end to end, every experiment reported in the paper:

  SECTION A - UCI Heart Disease (primary benchmark, N=920)
              -> Non-private LR/RF baselines, Single DP-LR, Averaged DP
                 (Config A: m=5, Config B: m=10), 5-fold CV + 20-run
                 variability, Rényi-DP privacy accounting, epsilon sweep.
              -> Reproduces Table II, Table III, and Figures 1-9.

  SECTION B - Pima Indians Diabetes (generalizability check, N=768)
              -> Same non-private / single-DP / averaged-DP(m=5, m=10)
                 comparison. Reproduces Table IV (Section 5.5).

  SECTION C - Breast Cancer Wisconsin (generalizability check, N=569)
              -> Same comparison, robust (median) data_norm, averaged over
                 10 independent DP trials. Reproduces Table V (Section 5.5).

Each section is a self-contained function (own imports of variable names)
so the three datasets never share or overwrite state. Run this script
top-to-bottom to regenerate every number and every figure that appears in
the paper.

-----------------------------------------------------------------------------
DATASETS (not bundled in this repo - see README.md):
  HEART_DATASET_PATH  -> UCI Heart Disease combined CSV (N=920).
                         Env var, default "heart_disease_uci.csv".
  PIMA_DATASET_PATH   -> Pima Indians Diabetes CSV (N=768), header or not.
                         Env var, default "pima-indians-diabetes.csv".
  Breast Cancer Wisconsin needs no download - it ships with scikit-learn.
-----------------------------------------------------------------------------
"""

import subprocess, sys, os

def _ensure_packages():
    required = ["diffprivlib", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn"]
    for pkg in required:
        try:
            __import__(pkg.replace("-", "_") if pkg != "scikit-learn" else "sklearn")
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

_ensure_packages()

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
mpl_rc = matplotlib.rcParams
mpl_rc["font.family"] = "sans-serif"      # portable: avoids 'Arial not found' warnings
mpl_rc["axes.spines.top"] = False
mpl_rc["axes.spines.right"] = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve, confusion_matrix)
from diffprivlib.models import LogisticRegression as DPLR
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# SECTION A - UCI HEART DISEASE (primary benchmark)
# =============================================================================
def run_heart_disease_pipeline():
    print("\n" + "=" * 70)
    print(" SECTION A: UCI Heart Disease (N=920) - primary benchmark")
    print("=" * 70)

    # ---- A1. Load dataset (portable: Colab upload OR DATASET_PATH) ----------
    import zipfile
    extract_dir = "dataset_heart"
    os.makedirs(extract_dir, exist_ok=True)

    try:
        from google.colab import files  # noqa: F401
        in_colab = True
    except ImportError:
        in_colab = False

    if in_colab:
        uploaded = files.upload()
        zip_path = list(uploaded.keys())[0]
        if zip_path.endswith(".zip"):
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)
        else:
            import shutil
            shutil.copy(zip_path, os.path.join(extract_dir, zip_path))
    else:
        dataset_path = os.environ.get("HEART_DATASET_PATH",
                                       os.environ.get("DATASET_PATH", "heart_disease_uci.csv"))
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Could not find '{dataset_path}'. Set HEART_DATASET_PATH to your local "
                f"heart_disease_uci.csv, e.g.\n  export HEART_DATASET_PATH=/path/to/heart_disease_uci.csv"
            )
        import shutil
        if os.path.isdir(extract_dir):
            shutil.rmtree(extract_dir)
        os.makedirs(extract_dir, exist_ok=True)
        if dataset_path.endswith(".zip"):
            with zipfile.ZipFile(dataset_path, "r") as z:
                z.extractall(extract_dir)
        else:
            shutil.copy(dataset_path, os.path.join(extract_dir, os.path.basename(dataset_path)))

    files_in_dir = os.listdir(extract_dir)
    file_name = next((f for f in files_in_dir if f.endswith((".csv", ".xlsx"))), None)
    if file_name is None:
        raise ValueError("No CSV/XLSX file found for the Heart Disease dataset!")
    file_path = os.path.join(extract_dir, file_name)
    df = pd.read_csv(file_path) if file_name.endswith(".csv") else pd.read_excel(file_path)
    print("Shape:", df.shape)

    # ---- A2. Preprocess -------------------------------------------------------
    target_col = df.columns[-1]
    if len(sorted(df[target_col].unique())) > 2:
        df[target_col] = (df[target_col] > 0).astype(int)

    categorical_cols = [c for c in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "dataset"]
                        if c in df.columns]
    exclude_cols = categorical_cols + [target_col, "id"]
    numeric_cols = [c for c in df.columns if c not in exclude_cols]

    df_num = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[numeric_cols]), columns=numeric_cols)
    df_cat = pd.DataFrame(SimpleImputer(strategy="most_frequent").fit_transform(df[categorical_cols]), columns=categorical_cols)
    df_clean = pd.concat([df_num, df_cat, df[[target_col]].reset_index(drop=True)], axis=1)

    X = pd.get_dummies(df_clean.drop(columns=[target_col]), drop_first=True)
    y = df_clean[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    data_norm = float(np.max(np.linalg.norm(X_train, axis=1))) * 1.0001
    print("data_norm (L2 bound):", round(data_norm, 6))

    # ---- A3. Non-private baselines (5-fold CV) --------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def cv_metrics(model, X, y, cv):
        accs, f1s, aucs = [], [], []
        for tr, te in cv.split(X, y):
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            prob = model.predict_proba(X[te])[:, 1]
            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, zero_division=0))
            aucs.append(roc_auc_score(y[te], prob))
        return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s), np.mean(aucs), np.std(aucs)

    lr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    lr_acc, lr_acc_sd, lr_f1, lr_f1_sd, lr_auc, lr_auc_sd = cv_metrics(lr, X_train, y_train, skf)
    lr.fit(X_train, y_train)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    print(f"LR  (non-private, 5-fold CV): Acc={lr_acc:.4f}\u00b1{lr_acc_sd:.4f}  F1={lr_f1:.4f}  AUC={lr_auc:.4f}\u00b1{lr_auc_sd:.4f}")

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf_acc, rf_acc_sd, rf_f1, rf_f1_sd, rf_auc, rf_auc_sd = cv_metrics(rf, X_train, y_train, skf)
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    print(f"RF  (non-private, 5-fold CV): Acc={rf_acc:.4f}\u00b1{rf_acc_sd:.4f}  F1={rf_f1:.4f}  AUC={rf_auc:.4f}\u00b1{rf_auc_sd:.4f}")

    # ---- A4. Single DP-LR -------------------------------------------------------
    dp_single = DPLR(epsilon=1.0, data_norm=data_norm, max_iter=1000)
    dp_single.fit(X_train, y_train)
    y_prob_dp = dp_single.predict_proba(X_test)[:, 1]
    print(f"Single DP LR (\u03b5=1.0), one draw: Acc={accuracy_score(y_test, dp_single.predict(X_test)):.4f}  AUC={roc_auc_score(y_test, y_prob_dp):.4f}")

    # ---- A5. Averaged DP: Config A (m=5) and Config B (m=10) --------------------
    def train_averaged_dp(m, eps_run, X_train, y_train):
        coefs, intercepts = [], []
        for _ in range(m):
            model = DPLR(epsilon=eps_run, data_norm=data_norm, max_iter=1000)
            model.fit(X_train, y_train)
            coefs.append(model.coef_.ravel().copy())
            intercepts.append(np.array(model.intercept_).ravel().copy())
        return np.mean(np.vstack(coefs), axis=0), float(np.mean(np.vstack(intercepts)))

    def eval_avg(coef_avg, intercept_avg, X_test, y_test):
        logits = X_test.dot(coef_avg) + intercept_avg
        probs = 1.0 / (1.0 + np.exp(-logits))
        pred = (probs >= 0.5).astype(int)
        return (accuracy_score(y_test, pred), f1_score(y_test, pred, zero_division=0),
                roc_auc_score(y_test, probs), probs)

    coefA, interA = train_averaged_dp(5, 1.0, X_train, y_train)
    accA, f1A, aucA, probsA = eval_avg(coefA, interA, X_test, y_test)
    print(f"Avg DP Config A (m=5,  \u03b5_run=1.0), one draw: Acc={accA:.4f}  AUC={aucA:.4f}")

    coefB, interB = train_averaged_dp(10, 0.5, X_train, y_train)
    accB, f1B, aucB, probsB = eval_avg(coefB, interB, X_test, y_test)
    print(f"Avg DP Config B (m=10, \u03b5_run=0.5), one draw: Acc={accB:.4f}  AUC={aucB:.4f}")

    # ---- A6. Statistical variability: 20 independent runs (Table III) -----------
    N_EXPERIMENTS = 20

    def repeated_dp_experiment(m, eps_run, n_exp=20):
        accs, f1s, aucs = [], [], []
        for _ in range(n_exp):
            coef_avg, intercept_avg = train_averaged_dp(m, eps_run, X_train, y_train)
            acc, f1, auc, _ = eval_avg(coef_avg, intercept_avg, X_test, y_test)
            accs.append(acc); f1s.append(f1); aucs.append(auc)
        return dict(acc_mean=np.mean(accs), acc_sd=np.std(accs),
                    f1_mean=np.mean(f1s), f1_sd=np.std(f1s),
                    auc_mean=np.mean(aucs), auc_sd=np.std(aucs),
                    acc_list=accs, f1_list=f1s, auc_list=aucs)

    print(f"Running variability analysis ({N_EXPERIMENTS} repeats per configuration)...")
    var_single = repeated_dp_experiment(1, 1.0, N_EXPERIMENTS)
    var_A = repeated_dp_experiment(5, 1.0, N_EXPERIMENTS)
    var_B = repeated_dp_experiment(10, 0.5, N_EXPERIMENTS)

    table3 = pd.DataFrame([
        {"Model": "LR (non-private)", "m": "\u2014", "\u03b5_run": "\u2014", "\u03b5_total": "\u2014",
         "Accuracy (\u00b1SD)": f"{lr_acc:.4f} \u00b1 {lr_acc_sd:.4f}", "F1 (\u00b1SD)": f"{lr_f1:.4f} \u00b1 {lr_f1_sd:.4f}", "AUC (\u00b1SD)": f"{lr_auc:.4f} \u00b1 {lr_auc_sd:.4f}"},
        {"Model": "RF (non-private)", "m": "\u2014", "\u03b5_run": "\u2014", "\u03b5_total": "\u2014",
         "Accuracy (\u00b1SD)": f"{rf_acc:.4f} \u00b1 {rf_acc_sd:.4f}", "F1 (\u00b1SD)": f"{rf_f1:.4f} \u00b1 {rf_f1_sd:.4f}", "AUC (\u00b1SD)": f"{rf_auc:.4f} \u00b1 {rf_auc_sd:.4f}"},
        {"Model": "Single DP LR", "m": 1, "\u03b5_run": 1.0, "\u03b5_total": 1.0,
         "Accuracy (\u00b1SD)": f"{var_single['acc_mean']:.4f} \u00b1 {var_single['acc_sd']:.4f}", "F1 (\u00b1SD)": f"{var_single['f1_mean']:.4f} \u00b1 {var_single['f1_sd']:.4f}", "AUC (\u00b1SD)": f"{var_single['auc_mean']:.4f} \u00b1 {var_single['auc_sd']:.4f}"},
        {"Model": "Avg DP Config A (m=5)", "m": 5, "\u03b5_run": 1.0, "\u03b5_total": 5.0,
         "Accuracy (\u00b1SD)": f"{var_A['acc_mean']:.4f} \u00b1 {var_A['acc_sd']:.4f}", "F1 (\u00b1SD)": f"{var_A['f1_mean']:.4f} \u00b1 {var_A['f1_sd']:.4f}", "AUC (\u00b1SD)": f"{var_A['auc_mean']:.4f} \u00b1 {var_A['auc_sd']:.4f}"},
        {"Model": "Avg DP Config B (m=10)", "m": 10, "\u03b5_run": 0.5, "\u03b5_total": 5.0,
         "Accuracy (\u00b1SD)": f"{var_B['acc_mean']:.4f} \u00b1 {var_B['acc_sd']:.4f}", "F1 (\u00b1SD)": f"{var_B['f1_mean']:.4f} \u00b1 {var_B['f1_sd']:.4f}", "AUC (\u00b1SD)": f"{var_B['auc_mean']:.4f} \u00b1 {var_B['auc_sd']:.4f}"},
    ])
    print("\n=== TABLE III: Model Performance Comparison with Standard Deviation ===")
    display(table3)
    table3.to_csv("table3_heart_disease_performance.csv", index=False)

    # ---- A7. Privacy accounting: Basic composition vs Rényi DP (Table II) -------
    def laplace_rdp(alpha, eps_run):
        if alpha <= 1:
            return eps_run
        t1 = alpha * np.exp((alpha - 1) * eps_run)
        t2 = (alpha - 1) * np.exp(-alpha * eps_run)
        val = (t1 + t2) / (2 * alpha - 1)
        return np.log(val) / (alpha - 1) if val > 0 else np.inf

    def rdp_to_dp(rdp_val, alpha, delta):
        return rdp_val + np.log(1.0 / delta) / (alpha - 1)

    def compute_rdp_epsilon(m, eps_run, delta=1e-6):
        alphas = np.arange(1.01, 50, 0.01)
        return min(rdp_to_dp(m * laplace_rdp(a, eps_run), a, delta) for a in alphas)

    rdp_single = compute_rdp_epsilon(1, 1.0)
    rdp_A = compute_rdp_epsilon(5, 1.0)
    rdp_B = compute_rdp_epsilon(10, 0.5)
    table2 = pd.DataFrame([
        {"Configuration": "Single DP (m=1)", "m": 1, "\u03b5_run": 1.0, "\u03b5_total Basic": 1.0, "\u03b5_total R\u00e9nyi DP (\u03b4=1e-6)": round(rdp_single, 4)},
        {"Configuration": "Avg DP Config A (m=5)", "m": 5, "\u03b5_run": 1.0, "\u03b5_total Basic": 5.0, "\u03b5_total R\u00e9nyi DP (\u03b4=1e-6)": round(rdp_A, 4)},
        {"Configuration": "Avg DP Config B (m=10)", "m": 10, "\u03b5_run": 0.5, "\u03b5_total Basic": 5.0, "\u03b5_total R\u00e9nyi DP (\u03b4=1e-6)": round(rdp_B, 4)},
    ])
    print("\n=== TABLE II: Privacy Accounting - Basic Composition vs R\u00e9nyi DP ===")
    display(table2)
    table2.to_csv("table2_privacy_accounting.csv", index=False)

    # ---- A8. Epsilon sweep (averaged over repeats for a smooth curve) -----------
    eps_list = [0.1, 0.5, 1.0, 2.0]
    M_SWEEP, N_REPEATS = 5, 15
    eps_rows = []
    for eps in eps_list:
        accs, f1s, aucs = [], [], []
        for _ in range(N_REPEATS):
            coef_avg, intercept_avg = train_averaged_dp(M_SWEEP, eps, X_train, y_train)
            acc, f1, auc, _ = eval_avg(coef_avg, intercept_avg, X_test, y_test)
            accs.append(acc); f1s.append(f1); aucs.append(auc)
        eps_rows.append({"epsilon": eps, "m": M_SWEEP,
                          "accuracy": np.mean(accs), "acc_sd": np.std(accs),
                          "f1": np.mean(f1s), "f1_sd": np.std(f1s),
                          "auc": np.mean(aucs), "auc_sd": np.std(aucs)})
        print(f"  \u03b5={eps}: Acc={np.mean(accs):.4f}\u00b1{np.std(accs):.4f}  AUC={np.mean(aucs):.4f}\u00b1{np.std(aucs):.4f}")
    df_eps = pd.DataFrame(eps_rows)
    df_eps.to_csv("epsilon_sweep_heart_disease.csv", index=False)

    # ---- A9. Figures --------------------------------------------------------------
    # Figure 1: performance with error bars
    labels = ["LR\n(non-private)", "RF\n(non-private)", "Single DP\n(m=1)", "Avg DP-A\n(m=5)", "Avg DP-B\n(m=10)"]
    acc_v = [lr_acc, rf_acc, var_single["acc_mean"], var_A["acc_mean"], var_B["acc_mean"]]
    acc_sd_v = [lr_acc_sd, rf_acc_sd, var_single["acc_sd"], var_A["acc_sd"], var_B["acc_sd"]]
    f1_v = [lr_f1, rf_f1, var_single["f1_mean"], var_A["f1_mean"], var_B["f1_mean"]]
    f1_sd_v = [lr_f1_sd, rf_f1_sd, var_single["f1_sd"], var_A["f1_sd"], var_B["f1_sd"]]
    auc_v = [lr_auc, rf_auc, var_single["auc_mean"], var_A["auc_mean"], var_B["auc_mean"]]
    auc_sd_v = [lr_auc_sd, rf_auc_sd, var_single["auc_sd"], var_A["auc_sd"], var_B["auc_sd"]]

    x = np.arange(len(labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w, acc_v, w, yerr=acc_sd_v, capsize=4, label="Accuracy", color="#2E86AB")
    ax.bar(x, f1_v, w, yerr=f1_sd_v, capsize=4, label="F1 Score", color="#E07B39")
    ax.bar(x + w, auc_v, w, yerr=auc_sd_v, capsize=4, label="AUC", color="#3BB273")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Score"); ax.set_ylim(0.4, 1.05)
    ax.set_title("Figure 1. UCI Heart Disease \u2014 Model Performance with SD")
    ax.legend(loc="lower right"); ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("figure1_heart_performance.pdf", bbox_inches="tight")
    plt.savefig("figure1_heart_performance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: privacy accounting
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xc = np.arange(3); wc = 0.32
    ax.bar(xc - wc/2, [1.0, 5.0, 5.0], wc, label="Basic Composition", color="#C0392B", alpha=0.85)
    ax.bar(xc + wc/2, [round(rdp_single,4), round(rdp_A,4), round(rdp_B,4)], wc, label="R\u00e9nyi DP (\u03b4=1e-6)", color="#2E86AB", alpha=0.85)
    ax.set_xticks(xc); ax.set_xticklabels(["Single DP\n(m=1)", "Avg DP-A\n(m=5)", "Avg DP-B\n(m=10)"])
    ax.set_ylabel("Total Privacy Budget (\u03b5)")
    ax.set_title("Figure 2. Privacy Accounting: Basic Composition vs. R\u00e9nyi DP")
    ax.legend(); ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("figure2_privacy_accounting.pdf", bbox_inches="tight")
    plt.savefig("figure2_privacy_accounting.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: ROC curves
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for probs, name, style in [(y_prob_lr, f"LR non-private (AUC={lr_auc:.3f})", {"color": "navy"}),
                                (y_prob_rf, f"RF non-private (AUC={rf_auc:.3f})", {"color": "forestgreen"}),
                                (y_prob_dp, f"Single DP (AUC={roc_auc_score(y_test, y_prob_dp):.3f})", {"color": "darkorange", "linestyle": "--"}),
                                (probsA, f"Avg DP-A (AUC={aucA:.3f})", {"color": "royalblue", "linestyle": "-."}),
                                (probsB, f"Avg DP-B (AUC={aucB:.3f})", {"color": "purple", "linestyle": ":"})]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, lw=2, label=name, **style)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Figure 3. ROC Curves \u2014 UCI Heart Disease")
    ax.legend(loc="lower right", fontsize=9); ax.grid(linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig("figure3_roc_heart.pdf", bbox_inches="tight")
    plt.savefig("figure3_roc_heart.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: variability boxplot
    fig, ax = plt.subplots(figsize=(7.5, 5))
    box_data = {"Single DP\n(m=1)": var_single["acc_list"], "Avg DP-A\n(m=5)": var_A["acc_list"], "Avg DP-B\n(m=10)": var_B["acc_list"]}
    bp = ax.boxplot(box_data.values(), positions=range(3), patch_artist=True, widths=0.45)
    for patch, color in zip(bp["boxes"], ["#E07B39", "#2E86AB", "#3BB273"]):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticks(range(3)); ax.set_xticklabels(box_data.keys())
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Figure 4. Accuracy Distribution over {N_EXPERIMENTS} Independent Runs")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("figure4_variability_boxplot.pdf", bbox_inches="tight")
    plt.savefig("figure4_variability_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 5 / 9: epsilon sweep
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(df_eps["epsilon"], df_eps["accuracy"], yerr=df_eps["acc_sd"], marker="o", capsize=4, label="Accuracy", color="#2E86AB")
    ax.errorbar(df_eps["epsilon"], df_eps["f1"], yerr=df_eps["f1_sd"], marker="s", capsize=4, label="F1 Score", color="#E07B39")
    ax.errorbar(df_eps["epsilon"], df_eps["auc"], yerr=df_eps["auc_sd"], marker="^", capsize=4, label="AUC", color="#3BB273")
    ax.set_xscale("log"); ax.set_xlabel("Privacy Budget \u03b5 (log scale)"); ax.set_ylabel("Score")
    ax.set_title(f"Figure 9. Privacy\u2013Utility Trade-off: Averaged DP (m={M_SWEEP})\n(mean \u00b1 SD over {N_REPEATS} repeats)")
    ax.legend(loc="lower right"); ax.grid(True, linestyle=":", alpha=0.5); ax.set_ylim(0.3, 1.0)
    plt.tight_layout()
    plt.savefig("figure9_epsilon_sweep.pdf", bbox_inches="tight")
    plt.savefig("figure9_epsilon_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure: accuracy-AUC trade-off scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    sc_models = ["LR\n(non-private)", "RF\n(non-private)", "Single DP", "Avg DP-A (m=5)", "Avg DP-B (m=10)"]
    sc_acc = [lr_acc, rf_acc, var_single["acc_mean"], var_A["acc_mean"], var_B["acc_mean"]]
    sc_auc = [lr_auc, rf_auc, var_single["auc_mean"], var_A["auc_mean"], var_B["auc_mean"]]
    colors = ["navy", "forestgreen", "darkorange", "royalblue", "purple"]
    markers = ["o", "s", "D", "^", "v"]
    for i, m in enumerate(sc_models):
        ax.scatter(sc_acc[i], sc_auc[i], color=colors[i], marker=markers[i], s=130, label=m, zorder=5)
    ax.plot(sc_acc, sc_auc, "--", color="gray", alpha=0.5)
    ax.set_xlabel("Accuracy"); ax.set_ylabel("AUC")
    ax.set_title("Figure 8. Accuracy\u2013AUC Trade-off (Heart Disease)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig("figure8_accuracy_auc_tradeoff.pdf", bbox_inches="tight")
    plt.savefig("figure8_accuracy_auc_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nSection A done. Figures saved: figure1-4, 8, 9 (heart disease).")

    return {"table2": table2, "table3": table3, "epsilon_sweep": df_eps}


# =============================================================================
# SECTION B - PIMA INDIANS DIABETES (generalizability, Section 5.5 / Table IV)
# =============================================================================
def run_pima_pipeline():
    print("\n" + "=" * 70)
    print(" SECTION B: Pima Indians Diabetes (N=768) - Table IV")
    print("=" * 70)

    data_path = os.environ.get("PIMA_DATASET_PATH", "pima-indians-diabetes.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find '{data_path}'. Set PIMA_DATASET_PATH to your local "
            f"Pima Indians Diabetes CSV, e.g.\n  export PIMA_DATASET_PATH=/path/to/pima-indians-diabetes.csv"
        )
    column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    with open(data_path) as f:
        first_line = f.readline()
    has_header = not first_line.strip().split(",")[0].replace(".", "", 1).isdigit()
    df = pd.read_csv(data_path) if has_header else pd.read_csv(data_path, header=None, names=column_names)
    if has_header:
        df.columns = column_names
    print("Shape:", df.shape)

    zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_as_missing] = df[zero_as_missing].replace(0, np.nan)
    df[zero_as_missing] = SimpleImputer(strategy="median").fit_transform(df[zero_as_missing])

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    data_norm = np.sqrt((X_train.values ** 2).sum(axis=1)).max()
    print("data_norm (L2 bound):", data_norm)

    def evaluate(model, name):
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        return {"Model": name, "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1": f1_score(y_test, pred, zero_division=0),
                "AUC": roc_auc_score(y_test, proba)}

    results = []
    nonpriv = LogisticRegression(max_iter=1000)
    nonpriv.fit(X_train, y_train)
    results.append(evaluate(nonpriv, "Non-private LR"))

    dp_single = DPLR(epsilon=1.0, data_norm=data_norm, max_iter=1000)
    dp_single.fit(X_train, y_train)
    results.append(evaluate(dp_single, "Single DP LR (\u03b5=1.0)"))

    def train_averaged(m, eps_run):
        coefs, intercepts = [], []
        for _ in range(m):
            model = DPLR(epsilon=eps_run, data_norm=data_norm, max_iter=1000)
            model.fit(X_train, y_train)
            coefs.append(model.coef_.copy()); intercepts.append(model.intercept_.copy())
        avg = DPLR(epsilon=eps_run, data_norm=data_norm, max_iter=1000)
        avg.fit(X_train, y_train)
        avg.coef_ = np.mean(coefs, axis=0)
        avg.intercept_ = np.mean(intercepts, axis=0)
        return avg

    avg_A = train_averaged(5, 1.0)
    results.append(evaluate(avg_A, "Averaged DP (Config A: m=5, \u03b5=1.0)"))
    avg_B = train_averaged(10, 0.5)
    results.append(evaluate(avg_B, "Averaged DP (Config B: m=10, \u03b5=0.5)"))

    table4 = pd.DataFrame(results)
    table4[["Accuracy", "Precision", "Recall", "F1", "AUC"]] = table4[["Accuracy", "Precision", "Recall", "F1", "AUC"]].round(4)
    print("\n=== TABLE IV: Pima Indians Diabetes \u2014 Model Performance Comparison ===")
    display(table4)
    table4.to_csv("table4_pima_diabetes.csv", index=False)

    # Figure: bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(table4)); w = 0.25
    for i, m in enumerate(["Accuracy", "F1", "AUC"]):
        ax.bar(x + i * w, table4[m], w, label=m)
    ax.set_xticks(x + w); ax.set_xticklabels(table4["Model"], rotation=20, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Pima Indians Diabetes \u2014 Non-private vs DP vs Averaged-DP")
    ax.legend()
    plt.tight_layout()
    plt.savefig("table4_pima_barchart.pdf", bbox_inches="tight")
    plt.savefig("table4_pima_barchart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Section B done.")
    return {"table4": table4}


# =============================================================================
# SECTION C - BREAST CANCER WISCONSIN (generalizability, Section 5.5 / Table V)
# =============================================================================
def run_breast_cancer_pipeline():
    print("\n" + "=" * 70)
    print(" SECTION C: Breast Cancer Wisconsin (N=569) - Table V")
    print("=" * 70)

    from sklearn.datasets import load_breast_cancer
    N_TRIALS = 10

    data = load_breast_cancer(as_frame=True)
    df = data.frame
    print("Shape:", df.shape)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # Robust (median, not max) L2 norm bound -- see README for why.
    norms = np.sqrt((X_train.values ** 2).sum(axis=1))
    data_norm = np.median(norms)
    print(f"L2 norm stats -> max: {norms.max():.2f}, median: {data_norm:.2f}")

    def evaluate(model):
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        return (accuracy_score(y_test, pred), precision_score(y_test, pred, zero_division=0),
                recall_score(y_test, pred, zero_division=0), f1_score(y_test, pred, zero_division=0),
                roc_auc_score(y_test, proba))

    def train_averaged_dp(m, eps_run, seed_base):
        coefs, intercepts = [], []
        for i in range(m):
            model = DPLR(epsilon=eps_run, data_norm=data_norm, max_iter=2000, random_state=seed_base * 1000 + i)
            model.fit(X_train, y_train)
            coefs.append(model.coef_.copy()); intercepts.append(model.intercept_.copy())
        avg = DPLR(epsilon=eps_run, data_norm=data_norm, max_iter=2000, random_state=seed_base)
        avg.fit(X_train, y_train)
        avg.coef_ = np.mean(coefs, axis=0)
        avg.intercept_ = np.mean(intercepts, axis=0)
        return avg

    nonpriv = LogisticRegression(max_iter=2000)
    nonpriv.fit(X_train, y_train)
    nonpriv_metrics = evaluate(nonpriv)

    cols = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    rows_single, rows_A, rows_B = [], [], []
    for trial in range(N_TRIALS):
        single = DPLR(epsilon=1.0, data_norm=data_norm, max_iter=2000, random_state=trial)
        single.fit(X_train, y_train)
        rows_single.append(evaluate(single))

        avgA = train_averaged_dp(5, 1.0, trial)
        rows_A.append(evaluate(avgA))

        avgB = train_averaged_dp(10, 0.5, trial + 500)
        rows_B.append(evaluate(avgB))

    df_single = pd.DataFrame(rows_single, columns=cols)
    df_A = pd.DataFrame(rows_A, columns=cols)
    df_B = pd.DataFrame(rows_B, columns=cols)

    def summarize(d, name):
        mean, std = d.mean(), d.std()
        return {"Model": name, **{c: f"{mean[c]:.4f} \u00b1 {std[c]:.4f}" for c in cols}}

    table5 = pd.DataFrame([
        {"Model": "Non-private LR", **{c: f"{v:.4f}" for c, v in zip(cols, nonpriv_metrics)}},
        summarize(df_single, f"Single DP LR (\u03b5=1.0), n={N_TRIALS} trials"),
        summarize(df_A, f"Averaged DP Config A (m=5, \u03b5=1.0), n={N_TRIALS} trials"),
        summarize(df_B, f"Averaged DP Config B (m=10, \u03b5=0.5), n={N_TRIALS} trials"),
    ])
    print("\n=== TABLE V: Breast Cancer Wisconsin \u2014 Model Performance Comparison ===")
    display(table5)
    table5.to_csv("table5_breast_cancer.csv", index=False)

    # Figure: mean +/- std accuracy bar chart
    means = [nonpriv_metrics[0], df_single["Accuracy"].mean(), df_A["Accuracy"].mean(), df_B["Accuracy"].mean()]
    stds = [0, df_single["Accuracy"].std(), df_A["Accuracy"].std(), df_B["Accuracy"].std()]
    labels = ["Non-private", "Single DP", "Avg DP (A)", "Avg DP (B)"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, means, yerr=stds, capsize=6, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title(f"Breast Cancer Wisconsin \u2014 Accuracy (mean \u00b1 std, {N_TRIALS} trials)")
    plt.tight_layout()
    plt.savefig("table5_breast_cancer_barchart.pdf", bbox_inches="tight")
    plt.savefig("table5_breast_cancer_barchart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Section C done.")
    return {"table5": table5}


# =============================================================================
# MAIN
# =============================================================================
def main():
    results = {}
    results.update(run_heart_disease_pipeline())
    results.update(run_pima_pipeline())
    results.update(run_breast_cancer_pipeline())

    print("\n" + "=" * 70)
    print(" ALL SECTIONS COMPLETE")
    print("=" * 70)
    print("Reproduces: Table II, Table III, Figures 1-4/8/9 (Heart Disease);")
    print("            Table IV (Pima Indians Diabetes);")
    print("            Table V (Breast Cancer Wisconsin).")
    print("All tables saved as .csv, all figures saved as .pdf (vector) + .png.")
    return results


if __name__ == "__main__":
    main()
