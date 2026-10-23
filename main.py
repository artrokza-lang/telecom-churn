# -*- coding: utf-8 -*-
"""
Telecom Customer Churn Prediction
Adapted for VS Code + Python 3.12 (no Google Colab dependency).

Project layout:
    C:\\Users\\User\\Desktop\\CodingML\\telecom\\
        ├── main.py                 <- this script
        ├── data\\
        │   └── telecom_users.csv   <- dataset (recommended location)
        ├── requirements.txt
        └── churn_results\\          <- output folder (created automatically)
"""

import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")


# ==================== CONFIG ====================
# Project root folder
PROJECT_DIR = Path(r"C:\Users\User\Desktop\CodingML\telecom")

# Data folder — recommended location for the dataset
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset filename expected by the pipeline
DATA_FILENAME = "telecom_users.csv"

# Primary dataset location (recommended)
DATA_FILE = DATA_DIR / DATA_FILENAME

# Legacy location in the project root (kept for backward compatibility)
LEGACY_DATA_FILE = PROJECT_DIR / DATA_FILENAME

# Output folder for plots, models and metrics
RESULTS_DIR = PROJECT_DIR / "churn_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Fixed random seed for reproducibility
RANDOM_STATE = 42

# Try modern seaborn style, fall back to a generic one if unavailable
try:
    plt.style.use("seaborn-v0_8-darkgrid")
except OSError:
    plt.style.use("ggplot")
sns.set_palette("husl")
pd.set_option("display.max_columns", None)


def save_fig(name: str) -> None:
    """Save the current matplotlib figure to RESULTS_DIR and show it."""
    path = RESULTS_DIR / f"{name}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  [saved] {path}")
    plt.show()
    plt.close()


def _resolve_data_file() -> Path:
    """Return the dataset path: data/ first, then project root.

    Keeps backward compatibility so existing setups (CSV in project root)
    continue to work without moving files.
    """
    for path in (DATA_FILE, LEGACY_DATA_FILE):
        if path.exists():
            return path
    return DATA_FILE  # default path — used in the error message


# ==================== 1. LOAD DATA ====================
def load_data() -> pd.DataFrame:
    """Load the raw CSV file and perform a first sanity check."""
    print("=" * 60)
    print("1. LOADING AND INITIAL DATA OVERVIEW")
    print("=" * 60)

    data_file = _resolve_data_file()
    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found.\n"
            f"  Expected: {DATA_FILE.resolve()}\n"
            f"  Or (legacy): {LEGACY_DATA_FILE.resolve()}\n\n"
            f"Place '{DATA_FILENAME}' into the data/ folder.\n"
            f"See data/README.md for download instructions."
        )

    print(f"Reading: {data_file}")
    df = pd.read_csv(data_file)

    # Drop columns that are not useful for modelling
    df = df.drop(
        columns=["Unnamed: 0", "customerID", "Provider", "Age"],
        errors="ignore",
    )

    # Convert TotalCharges to numeric (empty strings become NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    print(f"Dataset shape: {df.shape}")
    print("Missing values before processing:\n",
          df.isnull().sum()[df.isnull().sum() > 0])
    return df


# ==================== 2. MISSING VALUES ====================
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in TotalCharges using tenure-group medians."""
    print("\n" + "=" * 60)
    print("2. HANDLING MISSING VALUES")
    print("=" * 60)

    n_missing = df["TotalCharges"].isnull().sum()
    print(f"Missing in TotalCharges: {n_missing} "
          f"({df['TotalCharges'].isnull().mean() * 100:.2f}%)")

    if n_missing:
        miss = df[df["TotalCharges"].isnull()]
        print(f"  tenure of missing rows: "
              f"min={miss['tenure'].min()}, "
              f"mean={miss['tenure'].mean():.1f}, "
              f"max={miss['tenure'].max()}")

        # First try median within the same tenure group
        df["TotalCharges"] = df.groupby("tenure")["TotalCharges"].transform(
            lambda s: s.fillna(s.median())
        )
        # Fallback to global median if any gaps remain
        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["TotalCharges"].median()
        )

    print("Missing values after processing:", df.isnull().sum().sum())
    return df


# ==================== 3. ENCODING ====================
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns: yes/no -> 0/1, other categories -> LabelEncoder."""
    print("\n" + "=" * 60)
    print("3. ENCODING CATEGORICAL FEATURES")
    print("=" * 60)

    out = df.copy()

    # Target variable: "Yes" -> 1, "No" -> 0
    out["Churn"] = out["Churn"].map({"Yes": 1, "No": 0})

    for col in out.select_dtypes(include="object").columns:
        # Get unique non-null values as lowercase strings for detection
        str_vals = out[col].dropna().astype(str).str.strip().str.lower().unique()
        is_binary_yes_no = (
            any(v == "yes" for v in str_vals)
            and any(v == "no" for v in str_vals)
        )

        if is_binary_yes_no:
            # Binary yes/no column -> 1/0
            out[col] = (
                out[col].astype(str).str.strip().str.lower()
                .map({"yes": 1, "no": 0})
            )
            print(f"  binary encoded: {col}")
        else:
            # Multi-class column -> integer labels
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col].fillna("missing").astype(str))
            print(f"  label encoded: {col} ({len(le.classes_)} classes)")

    return out


# ==================== 4. NUMERIC ANOMALIES ====================
def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infinities and remaining NaNs in numeric columns."""
    print("\n" + "=" * 60)
    print("4. NUMERIC MISSING VALUES AND INFINITIES")
    print("=" * 60)

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        # Convert +/-inf to NaN so they can be filled with the median
        inf_mask = np.isinf(df[col])
        if inf_mask.any():
            df.loc[inf_mask, col] = np.nan
            print(f"  {col}: {inf_mask.sum()} inf -> NaN")

        # Fill any remaining NaN with the median
        if df[col].isnull().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
            print(f"  {col}: missing -> median {med:.2f}")

    print(f"Total NaN: {int(df.isnull().sum().sum())}")
    return df


# ==================== 5. FEATURE ENGINEERING ====================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a small set of engineered features."""
    print("\n" + "=" * 60)
    print("5. FEATURE ENGINEERING")
    print("=" * 60)

    # Monthly-to-tenure ratio (spending intensity per month of tenure)
    df["Monthly_to_tenure_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # Total charges per month of tenure (historical average spending)
    df["Total_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Whether the customer has both OnlineSecurity and TechSupport
    df["Has_Security_And_Support"] = (
        (df.get("OnlineSecurity", 0) == 1)
        & (df.get("TechSupport", 0) == 1)
    ).astype(int)

    # Number of additional services the customer subscribes to
    service_cols = [
        "OnlineSecurity", "TechSupport", "OnlineBackup",
        "DeviceProtection", "StreamingTV", "StreamingMovies",
    ]
    existing = [c for c in service_cols if c in df.columns]
    if existing:
        df["Service_Count"] = df[existing].sum(axis=1)

    # Flag for very new customers (less than 3 months)
    df["Is_New_Customer"] = (df["tenure"] < 3).astype(int)

    # Monthly charges relative to the median
    med_monthly = df["MonthlyCharges"].median()
    df["Monthly_Charges_Ratio"] = (
        df["MonthlyCharges"] / med_monthly if med_monthly else 0.0
    )

    # Sanity cleanup for engineered features (no NaN / inf allowed)
    new_features = [
        "Monthly_to_tenure_ratio", "Total_per_tenure",
        "Service_Count", "Monthly_Charges_Ratio",
    ]
    for f in new_features:
        if f in df.columns:
            if df[f].isnull().any():
                df[f] = df[f].fillna(df[f].median())
            inf_mask = np.isinf(df[f])
            if inf_mask.any():
                df.loc[inf_mask, f] = np.nan
                df[f] = df[f].fillna(df[f].median())

    print(f"Engineered features added: "
          f"{len([c for c in new_features if c in df.columns])}")
    return df


# ==================== 6. TRAIN/TEST PREPARATION ====================
def prepare_data(df: pd.DataFrame):
    """Split into train/test, apply scaling, and return all modelling artifacts."""
    print("\n" + "=" * 60)
    print("6. DATA PREPARATION")
    print("=" * 60)

    X = df.drop(columns="Churn")
    y = df["Churn"]

    # Drop rows with missing target (safety net)
    if y.isnull().any():
        keep = y.notnull()
        X, y = X[keep], y[keep]

    assert X.isnull().sum().sum() == 0, "NaN found in X!"
    assert y.isnull().sum() == 0, "NaN found in y!"

    print(f"Features: {X.shape[1]}, rows: {X.shape[0]}")
    print(f"Class balance: {y.value_counts(normalize=True).round(3).to_dict()}")

    # Stratified split keeps the class ratio in both subsets
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Standardize features (fit only on train to avoid data leakage)
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(
        scaler.fit_transform(X_tr), columns=X.columns, index=X_tr.index
    )
    X_te_s = pd.DataFrame(
        scaler.transform(X_te), columns=X.columns, index=X_te.index
    )

    print(f"train={X_tr.shape}, test={X_te.shape}")
    print(f"train balance: {y_tr.value_counts(normalize=True).round(3).to_dict()}")
    print(f"test balance: {y_te.value_counts(normalize=True).round(3).to_dict()}")
    return X_tr_s, X_te_s, y_tr, y_te, scaler, X.columns


# ==================== 7. MODEL TRAINING ====================
def train_models(X_tr, y_tr) -> dict:
    """Run GridSearchCV for each base model and return the best estimators."""
    print("\n" + "=" * 60)
    print("7. MODEL TRAINING (GridSearchCV)")
    print("=" * 60)

    # Hyperparameter grids for each model
    param_grids = {
        "RandomForest": {
            "n_estimators": [50, 100],
            "max_depth": [10, 20],
            "min_samples_leaf": [2, 4],
        },
        "LogisticRegression": {
            "C": [0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["liblinear"],
        },
        "XGBoost": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1],
        },
        "GradientBoosting": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
        },
    }

    # Base estimators (hyperparameters will be tuned by GridSearchCV)
    base_models = {
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "LogisticRegression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }

    # Stratified K-fold keeps class ratio in every fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    trained = {}
    for name, model in base_models.items():
        print(f"\n-> {name} ...")
        try:
            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,          # use all CPU cores
                refit=True,
            )
            gs.fit(X_tr, y_tr)
            trained[name] = gs.best_estimator_
            print(f"  best: {gs.best_params_}  "
                  f"(CV ROC-AUC={gs.best_score_:.4f})")
        except Exception as e:
            print(f"  [error] {e}")

    return trained


# ==================== 8. EVALUATION ====================
def evaluate_models(models: dict, X_te, y_te):
    """Compute metrics for each model, plot comparison, return the best one."""
    print("\n" + "=" * 60)
    print("8. MODEL EVALUATION")
    print("=" * 60)

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    results_df = pd.DataFrame(
        index=list(models.keys()), columns=metrics, dtype=float
    )

    for name, model in models.items():
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        results_df.loc[name] = [
            accuracy_score(y_te, y_pred),
            precision_score(y_te, y_pred, zero_division=0),
            recall_score(y_te, y_pred, zero_division=0),
            f1_score(y_te, y_pred, zero_division=0),
            roc_auc_score(y_te, y_proba),
        ]

    print(results_df.round(4))

    # Best model = highest ROC-AUC on the test set
    best_name = results_df["ROC-AUC"].astype(float).idxmax()
    best_model = models[best_name]
    print(f"\n*** BEST MODEL: {best_name}  "
          f"(ROC-AUC={results_df.loc[best_name, 'ROC-AUC']:.4f})")

    # Side-by-side bar chart for all metrics
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, m in enumerate(metrics):
        ax = axes[i // 3, i % 3]
        data = results_df[m].sort_values(ascending=False)
        bars = ax.barh(data.index, data.values,
                       color=sns.color_palette("husl", len(data)))
        # Highlight the best model in red
        bars[list(data.index).index(best_name)].set_color("red")
        ax.set_title(f"{m} comparison")
        ax.set_xlabel(m)
        for b in bars:
            ax.text(b.get_width() + 0.005,
                    b.get_y() + b.get_height() / 2,
                    f"{b.get_width():.3f}",
                    va="center", fontsize=8)
    plt.suptitle("Model performance comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig("01_models_comparison")

    return results_df, best_name, best_model


# ==================== 9. BEST MODEL DETAILS ====================
def detail_best(best_name: str, best_model, X_te, y_te) -> None:
    """Print classification report and plot confusion matrix + ROC curve."""
    print("\n" + "=" * 60)
    print("9. DETAILED EVALUATION OF THE BEST MODEL")
    print("=" * 60)

    y_pred = best_model.predict(X_te)
    y_proba = best_model.predict_proba(X_te)[:, 1]

    print(f"Model: {best_name}\n")
    print(classification_report(y_te, y_pred,
                                target_names=["No churn", "Churn"]))

    # Confusion matrix heatmap
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No churn", "Churn"],
                yticklabels=["No churn", "Churn"])
    plt.title(f"Confusion matrix - {best_name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    save_fig("02_confusion_matrix")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    auc = roc_auc_score(y_te, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve - {best_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    save_fig("03_roc_curve")


# ==================== 10. FEATURE IMPORTANCE ====================
def feature_importance(best_name: str, best_model, columns):
    """Plot and print top features by importance.

    Supports tree-based models (feature_importances_) and linear models
    (|coef_| as a proxy for importance).
    """
    # Extract importance values: tree models or linear coefficients
    if hasattr(best_model, "feature_importances_"):
        values = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        values = np.abs(best_model.coef_[0])
    else:
        print("\n(Model supports neither feature_importances_ nor coef_)")
        return None

    print("\n" + "=" * 60)
    print("10. FEATURE IMPORTANCE")
    print("=" * 60)

    imp = pd.DataFrame({
        "feature": columns,
        "importance": values,
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp.head(15), x="importance", y="feature")
    plt.title(f"Top 15 features - {best_name}")
    plt.tight_layout()
    save_fig("04_feature_importance")

    print("Top 10:")
    for _, row in imp.head(10).iterrows():
        print(f"  {row['feature']:>28}: {row['importance']:.4f}")
    return imp


# ==================== 11. BUSINESS RECOMMENDATIONS ====================
def business_recommendations() -> None:
    """Print high-level business recommendations derived from the analysis."""
    print("\n" + "=" * 60)
    print("11. BUSINESS RECOMMENDATIONS")
    print("=" * 60)

    recs = [
        ("Focus on new customers",
         "Most churn occurs within the first 3 months. Improve onboarding "
         "and offer special conditions during the first 90 days."),
        ("Security service bundles",
         "Customers without OnlineSecurity and TechSupport churn 2-3x more "
         "often. Create attractive security bundles."),
        ("Flexible contracts",
         "Month-to-month customers churn more often. Incentivize yearly "
         "contracts with discounts."),
        ("Price control",
         "High MonthlyCharges correlate with churn. Revisit pricing policy."),
        ("Personalized retention",
         "Use the model to identify at-risk customers and offer proactive "
         "retention deals."),
    ]
    for title, text in recs:
        print(f"\n- {title}\n  {text}")


# ==================== 12. SAVING ARTIFACTS ====================
def save_artifacts(best_name: str, best_model,
                   scaler: StandardScaler, results_df: pd.DataFrame) -> None:
    """Persist the best model, scaler and metric table to disk."""
    print("\n" + "=" * 60)
    print("12. SAVING RESULTS")
    print("=" * 60)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    joblib.dump(best_model,
                RESULTS_DIR / f"best_model_{best_name}_{stamp}.pkl")
    joblib.dump(scaler, RESULTS_DIR / "scaler.pkl")
    results_df.round(4).to_csv(RESULTS_DIR / f"results_{stamp}.csv")

    print(f"Model, scaler and metrics saved to: {RESULTS_DIR.resolve()}")


# ==================== MAIN ====================
def main() -> None:
    """End-to-end pipeline execution."""
    df = load_data()
    df = handle_missing(df)
    df = encode_categorical(df)
    df = clean_numeric(df)
    df = add_features(df)

    X_tr, X_te, y_tr, y_te, scaler, cols = prepare_data(df)

    models = train_models(X_tr, y_tr)
    results_df, best_name, best_model = evaluate_models(models, X_te, y_te)

    detail_best(best_name, best_model, X_te, y_te)
    feature_importance(best_name, best_model, cols)
    business_recommendations()
    save_artifacts(best_name, best_model, scaler, results_df)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()