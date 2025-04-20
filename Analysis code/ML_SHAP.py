import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(f"SHAP version: {shap.__version__}")

# --- Configuration ---
FILE_PATH = r'E:\Project\ML\PH_LCC_ALL.xlsx'
MODEL_SAVE_DIR = r'E:\Project\ML\saved_models'
SHAP_SAVE_DIR = r'E:\Project\ML\shap_visualizations'
RANDOM_STATE = 42
MAX_SAMPLES = 200

# --- Define new feature groups ---
feature_groups = {
    "Group1_PH": ["Time1_PH", "Time2_PH"],
    "Group2_PH": ["Time3_PH", "Time4_PH", "Time5_PH", "Time6_PH"],
    "Group3_LCC": ["Time1_LCC", "Time2_LCC"],
    "Group4_LCC": ["Time3_LCC", "Time4_LCC", "Time5_LCC", "Time6_LCC"],
    "Group5_LCC": ["Time7_LCC", "Time8_LCC", "Time9_LCC"],
}

# --- 1. Load Models and Preprocessors ---
print("--- 1. Loading Models, Preprocessors and Feature Names ---")
try:
    model_xgb = joblib.load(os.path.join(MODEL_SAVE_DIR, 'best_xgb_model.joblib'))
    scaler = joblib.load(os.path.join(MODEL_SAVE_DIR, 'scaler.joblib'))
    feature_names = joblib.load(os.path.join(MODEL_SAVE_DIR, 'feature_names.joblib'))
    print("Loading successful.")
except Exception as e:
    print(f"Error: {str(e)}")
    exit()

# --- 2. Data Preparation ---
print("\n--- 2. Reloading and Preparing Data ---")
data = pd.read_excel(FILE_PATH)
X = data.drop(columns=['ID', 'yield'])
y = data['yield']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)

X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=feature_names)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

# --- Create Grouped Datasets ---
def create_grouped_data(df):
    grouped_df = pd.DataFrame()
    for group_name, features in feature_groups.items():
        valid_features = [f for f in features if f in df.columns]
        if not valid_features:
            raise ValueError(f"Group {group_name} has no valid features")
        grouped_df[group_name] = df[valid_features].mean(axis=1)
    return grouped_df

X_grouped_train = create_grouped_data(X_train_scaled)
X_grouped_test = create_grouped_data(X_test_scaled)

# --- 3. SHAP Visualization ---
print("\n--- 3. Generating SHAP Visualizations ---")

print("Initializing explainer...")
explainer = shap.TreeExplainer(model_xgb)
sample_indices = np.random.choice(len(X_train_scaled), min(300, len(X_train_scaled)), replace=False)

# Calculate SHAP values for original features
shap_values = explainer.shap_values(X_train_scaled.iloc[sample_indices])
print("Original SHAP values shape:", np.array(shap_values).shape)

# --- Aggregate SHAP values by feature groups ---
group_shap = np.zeros((len(sample_indices), len(feature_groups)))
for i, (group_name, features) in enumerate(feature_groups.items()):
    feature_indices = [X_train_scaled.columns.get_loc(f) for f in features if f in X_train_scaled.columns]
    group_shap[:, i] = np.sum(shap_values[:, feature_indices], axis=1)

# --- Visualization 1: Grouped Summary Plot ---
plt.figure(figsize=(10, 6))
shap.summary_plot(
    group_shap,
    X_grouped_train.iloc[sample_indices],
    feature_names=list(feature_groups.keys()),
    plot_type="dot",
    show=False
)
plt.title("SHAP Summary Plot (Feature Groups)")
plt.gcf().set_size_inches(8, 6)
plt.savefig(os.path.join(SHAP_SAVE_DIR, 'XGBoost_SHAP_Summary.pdf'),
            bbox_inches='tight', dpi=600)
plt.close()

# --- Visualization 2: Grouped Heatmap ---
n_samples_heatmap = min(500, X_test_scaled.shape[0])
X_test_subset = X_test_scaled.iloc[:n_samples_heatmap]

# Calculate SHAP values for test set
shap_values_test = explainer.shap_values(X_test_subset)
group_shap_test = np.zeros((n_samples_heatmap, len(feature_groups)))
for i, (group_name, features) in enumerate(feature_groups.items()):
    feature_indices = [X_test_subset.columns.get_loc(f) for f in features if f in X_test_subset.columns]
    if len(feature_indices) == 0:
        raise ValueError(f"Group {group_name} has no valid features")
    group_shap_test[:, i] = np.sum(shap_values_test[:, feature_indices], axis=1)

# Create Explanation object
shap_explanation = shap.Explanation(
    values=group_shap_test,
    base_values=explainer.expected_value,
    data=X_grouped_test.iloc[:n_samples_heatmap].values,
    feature_names=list(feature_groups.keys())
)

# Plot heatmap
plt.figure(figsize=(12, 8))
shap.plots.heatmap(
    shap_explanation,
    instance_order=np.argsort(shap_values_test.sum(axis=1)),
    feature_order=np.argsort(np.abs(group_shap_test).mean(0))[::-1],
    show=False
)
plt.title("SHAP Heatmap (Feature Groups)")
plt.savefig(os.path.join(SHAP_SAVE_DIR, 'XGBoost_SHAP_Heatmap.pdf'),
            bbox_inches='tight', dpi=600)
plt.close()

print("\n--- SHAP Analysis Completed ---")