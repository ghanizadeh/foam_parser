import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ==========================================================
# Utility functions
# ==========================================================
def a20_index(y_true, y_pred):
    ratio = y_pred / y_true
    return np.mean((ratio >= 0.8) & (ratio <= 1.2))


# ==========================================================
# Page config
# ==========================================================
st.set_page_config(
    page_title="Permeability Prediction – Optimized Random Forest",
    layout="wide"
)

st.title("🪨 Permeability Prediction (Gap-Optimized Random Forest)")


# ==========================================================
# Upload data
# ==========================================================
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("Dataset loaded successfully")

# ================================
# Custom Ratio Generator Section
# ================================
with st.container(border=True):
    st.subheader("⚗️ Optional: Create Custom Ratio Features (e.g., **APG/TDS**)")

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    st.markdown("Select numerators and denominators to automatically create ratio features:")

    col1, col2 = st.columns(2)

    numerators = col1.multiselect(
        "Select Numerator Columns",
        options=numeric_columns
    )

    denominators = col2.multiselect(
        "Select Denominator Columns",
        options=numeric_columns
    )

    ratio_warnings = []
    new_ratio_cols = {}

    if numerators and denominators:

        for num in numerators:
            for den in denominators:
                ratio_name = f"{num}/{den}"

                # Skip duplicates
                if ratio_name in df.columns:
                    continue

                # Create ratio with NaN where denominator is zero
                df[ratio_name] = df[num] / df[den].replace(0, np.nan)

                # Count zeros
                zero_count = (df[den] == 0).sum()

                if zero_count > 0:
                    ratio_warnings.append(
                        f"⚠️ Ratio **{ratio_name}**: {zero_count} rows have denominator = 0 → filled with NaN."
                    )

                new_ratio_cols[ratio_name] = ratio_name

        # Show created ratio names
        if new_ratio_cols:
            st.success(f"✅ Created {len(new_ratio_cols)} ratio features:")
            st.write(list(new_ratio_cols.keys()))

        # Show warnings
        if ratio_warnings:
            for warning in ratio_warnings:
                st.warning(warning)

    else:
        st.info("Select at least one numerator and one denominator to generate ratio features.")

df_original = df.copy()

# ==========================================================
# Feature / target selection
# ==========================================================
st.subheader("🎯 Feature & Target Selection")

columns = df.columns.tolist()

default_features = [
    c for c in columns
    if c in [
        "Rebound hardness (HLD)",
        "Corrected Vp (m/s)",
        "Corrected Vs (m/s)",
        "Quartz"
    ]
]

features = st.multiselect(
    "Select input features",
    options=columns,
    default=default_features
)

target = st.selectbox(
    "Select target variable",
    options=[c for c in columns if c not in features],
    index=columns.index("Permeability (md)")
    if "Permeability (md)" in columns else 0
)

if len(features) == 0:
    st.warning("Please select at least one feature")
    st.stop()


# ==========================================================
# Clean data
# ==========================================================
df = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()

st.subheader("📊 Dataset Summary")
st.dataframe(df.describe().T)


# ==========================================================
# Target transformation
# ==========================================================
st.subheader("🔄 Target Transformation")

log_target = st.checkbox(
    "Apply log10 transform to target (recommended for permeability)",
    value=True
)

X = df[features].values
y_raw = df[target].values

if log_target:
    y = np.log10(np.maximum(y_raw, 1e-12))
else:
    y = y_raw.copy()


# ==========================================================
# Validation settings
# ==========================================================
st.subheader("⚙️ Validation Settings")

test_size = st.slider(
    "Test size (%)",
    min_value=10,
    max_value=40,
    value=25
) / 100


# ==========================================================
# RF optimization settings
# ==========================================================
st.subheader("🧠 Random Forest Optimization (Gap-Focused)")

n_estimators = st.slider(
    "Number of trees",
    min_value=100,
    max_value=800,
    value=600,
    step=50
)

max_depth_list = [10, 12, 14]
min_samples_leaf_list = [4, 5, 6]
max_features_list = [0.5, 0.6, 0.7]


# ==========================================================
# Train & optimize
# ==========================================================
if st.button("🚀 Train & Optimize Random Forest"):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42
    )
    st.info(f"Size of Training data: {len(X_train)}")
    st.info(f"Size of Test data: {len(X_test)}")
    best_model = None
    best_gap = np.inf

    with st.spinner("Optimizing Random Forest (minimizing generalization gap)..."):
        for depth in max_depth_list:
            for leaf in min_samples_leaf_list:
                for feat in max_features_list:

                    rf = RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=depth,
                        min_samples_leaf=leaf,
                        max_features=feat,
                        bootstrap=True,
                        random_state=42,
                        n_jobs=-1
                    )

                    rf.fit(X_train, y_train)

                    r2_tr = r2_score(y_train, rf.predict(X_train))
                    r2_te = r2_score(y_test, rf.predict(X_test))
                    gap = abs(r2_tr - r2_te)

                    if gap < best_gap:
                        best_gap = gap
                        best_model = rf

    # ======================================================
    # Predictions
    # ======================================================
    y_train_pred = best_model.predict(X_train)
    y_test_pred  = best_model.predict(X_test)

    if log_target:
        y_train_true = 10 ** y_train
        y_test_true  = 10 ** y_test
        y_train_pred_orig = 10 ** y_train_pred
        y_test_pred_orig  = 10 ** y_test_pred
    else:
        y_train_true = y_train
        y_test_true  = y_test
        y_train_pred_orig = y_train_pred
        y_test_pred_orig  = y_test_pred


    # ======================================================
    # Metrics (CORRECT SCALES)
    # ======================================================
    metrics_df = pd.DataFrame({
        "Metric": ["R² (model scale)", "MSE (md²)", "RMSE (md)", "MAE (md)", "a20 index"],
        "Train": [
            r2_score(y_train, y_train_pred),
            mean_squared_error(y_train_true, y_train_pred_orig),
            np.sqrt(mean_squared_error(y_train_true, y_train_pred_orig)),
            mean_absolute_error(y_train_true, y_train_pred_orig),
            a20_index(y_train_true, y_train_pred_orig)
        ],
        "Test": [
            r2_score(y_test, y_test_pred),
            mean_squared_error(y_test_true, y_test_pred_orig),
            np.sqrt(mean_squared_error(y_test_true, y_test_pred_orig)),
            mean_absolute_error(y_test_true, y_test_pred_orig),
            a20_index(y_test_true, y_test_pred_orig)
        ]
    })

    st.subheader("📊 Model Performance Metrics")
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns

    st.dataframe(
        metrics_df.style.format({col: "{:.4f}" for col in numeric_cols})
    )



    # ======================================================
    # Predicted vs Measured plot
    # ======================================================
    st.subheader("📈 Predicted vs Measured Permeability")

    global_min = min(
        y_train_true.min(), y_train_pred_orig.min(),
        y_test_true.min(), y_test_pred_orig.min()
    )
    global_max = max(
        y_train_true.max(), y_train_pred_orig.max(),
        y_test_true.max(), y_test_pred_orig.max()
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    axes[0].scatter(y_train_true, y_train_pred_orig,
                    facecolors="none", edgecolors="black", marker="o")
    axes[0].plot([global_min, global_max], [global_min, global_max], "k--")
    axes[0].set_title("Train")
    axes[0].set_xlabel("Observed Permeability (md)")
    axes[0].set_ylabel("Predicted Permeability (md)")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].scatter(y_test_true, y_test_pred_orig,
                    facecolors="none", edgecolors="black", marker="^")
    axes[1].plot([global_min, global_max], [global_min, global_max], "k--")
    axes[1].set_title("Test")
    axes[1].set_xlabel("Observed Permeability (md)")
    axes[1].set_ylabel("Predicted Permeability (md)")
    axes[1].tick_params(axis="y", labelleft=True)
    axes[1].set_aspect("equal", adjustable="box")

    plt.tight_layout()
    st.pyplot(fig)


    # ======================================================
    # SHAP analysis
    # ======================================================
    st.subheader("🧠 SHAP Explainability (Random Forest)")

    X_train_df = pd.DataFrame(X_train, columns=features)
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_train_df)

    fig_bar = plt.figure(figsize=(8, 4))
    shap.summary_plot(shap_values, X_train_df, plot_type="bar", show=False)
    st.pyplot(fig_bar)

    fig_bee = plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_train_df, show=False)
    st.pyplot(fig_bee)

    #dep_feature = st.selectbox("SHAP dependency feature", features)
    #fig_dep = plt.figure(figsize=(6, 5))
    #shap.dependence_plot(dep_feature, shap_values, X_train_df, show=False)
    # st.pyplot(fig_dep)

    st.success("✅ Training, evaluation, and SHAP analysis completed.")
 

    # ======================================================
    # A) EVALUATION PREDICTION (ONLY LABELED DATA ~251)
    # ======================================================
    st.subheader("🧪 Evaluation Prediction (Measured Samples Only)")

    df_eval = df_original.copy()
    st.info(f"Original dataset size: {len(df_eval)} rows")
    df_eval = df_eval[features + [target]].replace([np.inf, -np.inf], np.nan)
    df_eval = df_eval.dropna(subset=features + [target])

    df_eval = df_eval.reset_index(drop=True)

    X_eval = df_eval[features].values

    y_eval_pred = best_model.predict(X_eval)
    if log_target:
        y_eval_pred = 10 ** y_eval_pred

    df_eval = df_eval.reset_index(drop=True)
    df_eval["Predicted Permeability (md)"] = y_eval_pred

    eval_display_cols = features + [
        target,
        "Predicted Permeability (md)"
    ]

    st.dataframe(
        df_eval[eval_display_cols],
        use_container_width=True
    )
    st.info(f"Evaluation samples: {len(df_eval)} rows")

    eval_csv = df_eval.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Evaluation Predictions (Measured Only)",
        data=eval_csv,
        file_name="permeability_evaluation_predictions.csv",
        mime="text/csv"
    )
 
