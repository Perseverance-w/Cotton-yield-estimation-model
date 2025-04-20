import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression as MLR
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import optuna
import joblib
import os
import sklearn

print(f"Optuna version: {optuna.__version__}")
print(f"XGBoost version: {xgb.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version: {np.__version__}")

# --- Configuration ---
FILE_PATH = r'E:\Project\ML\PH_LCC_ALL.xlsx'
MODEL_SAVE_DIR = r'E:\Project\ML\saved_models'
VIS_SAVE_DIR = r'E:\Project\ML\visualizations'
N_TRIALS_PLS = 30  # Number of optimization trials for PLSR
N_TRIALS_XGB = 100  # Number of optimization trials for XGBoost
CV_FOLDS = 5  # Cross-validation folds
RANDOM_STATE = 42  # Random seed for reproducibility

# --- Create Directories if they don't exist ---
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(VIS_SAVE_DIR, exist_ok=True)

# --- 1. Data Loading and Preparation ---
print("--- 1. Loading and Preparing Data ---")
# Check file existence and load data
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"File not found: {FILE_PATH}")
data = pd.read_excel(FILE_PATH)

# Separate features and target
X = data.drop(columns=['ID', 'yield'])
y = data['yield']
feature_names = X.columns.tolist()  # Get feature names

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

# Data standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
print("Data loading and preprocessing completed.")
print(f"Training set size: {X_train_scaled.shape}, Test set size: {X_test_scaled.shape}")

# --- 2. Evaluation Metrics and Helper Functions ---
print("\n--- 2. Defining Evaluation Metrics and Helper Functions ---")
def calculate_metrics(y_true, y_pred):
    """Calculate regression evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

def print_metrics(model_name, train_metrics, test_metrics):
    """Print model evaluation metrics"""
    print(f"--- {model_name} Model Evaluation Results ---")
    print("Training set:")
    print(f"  MSE: {train_metrics[0]:.4f}, RMSE: {train_metrics[1]:.4f}, MAE: {train_metrics[2]:.4f}, R²: {train_metrics[3]:.4f}")
    print("Test set:")
    print(f"  MSE: {test_metrics[0]:.4f}, RMSE: {test_metrics[1]:.4f}, MAE: {test_metrics[2]:.4f}, R²: {test_metrics[3]:.4f}")
    print("-" * (len(model_name) + 25))

def visualize_and_save(model_name, y_train_actual, y_train_pred, y_test_actual, y_test_pred,
                      train_metrics, test_metrics, color_train, color_test, filename):
    """Visualize predictions vs actual values and save as PDF"""
    data_train = pd.DataFrame({'True': y_train_actual, 'Predicted': y_train_pred, 'Dataset': 'Train'})
    data_test = pd.DataFrame({'True': y_test_actual, 'Predicted': y_test_pred, 'Dataset': 'Test'})
    data_combined = pd.concat([data_train, data_test])
    palette = {'Train': color_train, 'Test': color_test}

    plt.figure(figsize=(10, 8), dpi=300)
    g = sns.JointGrid(data=data_combined, x="True", y="Predicted", hue="Dataset", height=8, palette=palette)
    g.plot_joint(sns.scatterplot, alpha=0.7, s=100)
    sns.regplot(data=data_train, x="True", y="Predicted", scatter=False, ax=g.ax_joint,
                color=color_train, line_kws={'label':'Train Reg. Line'})
    sns.regplot(data=data_test, x="True", y="Predicted", scatter=False, ax=g.ax_joint,
                color=color_test, line_kws={'label':'Test Reg. Line'})
    g.plot_marginals(sns.histplot, kde=False, element='step', alpha=0.7)
    ax = g.ax_joint
    min_val = min(data_combined['True'].min(), data_combined['Predicted'].min())*0.95
    max_val = max(data_combined['True'].max(), data_combined['Predicted'].max())*1.05
    ax.plot([min_val, max_val], [min_val, max_val], c="black", alpha=0.7, linestyle='--', label='y=x')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    # Add metrics annotations
    ax.text(0.95, 0.15, f'Train $R^2$ = {train_metrics[3]:.3f}', transform=ax.transAxes,
            fontsize=12, va='bottom', ha='right', bbox=dict(boxstyle="round,pad=0.3", ec="grey", fc="white", alpha=0.8))
    ax.text(0.95, 0.08, f'Test $R^2$ = {test_metrics[3]:.3f}', transform=ax.transAxes,
            fontsize=12, va='bottom', ha='right', bbox=dict(boxstyle="round,pad=0.3", ec="grey", fc="white", alpha=0.8))
    ax.text(0.05, 0.95, f'Model = {model_name}', transform=ax.transAxes,
            fontsize=12, va='top', ha='left', bbox=dict(boxstyle="round,pad=0.3", ec="grey", fc="white", alpha=0.8))

    ax.set_xlabel("True Yield", fontsize=14)
    ax.set_ylabel("Predicted Yield", fontsize=14)
    ax.legend(fontsize=12)
    plt.suptitle(f'{model_name} Model Performance', y=1.02, fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(VIS_SAVE_DIR, f'{filename}.pdf')
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"{model_name} visualization saved to: {save_path}")
    plt.close()

# --- 3. Model Training and Evaluation ---
print("\n--- 3. Training and Evaluating Models ---")

# === 3.1 Multiple Linear Regression (MLR) ===
print("Training MLR model...")
model_mlr = MLR()
model_mlr.fit(X_train_scaled, y_train)
y_train_pred_mlr = model_mlr.predict(X_train_scaled)
y_test_pred_mlr = model_mlr.predict(X_test_scaled)
train_metrics_mlr = calculate_metrics(y_train, y_train_pred_mlr)
test_metrics_mlr = calculate_metrics(y_test, y_test_pred_mlr)
print_metrics("MLR", train_metrics_mlr, test_metrics_mlr)
visualize_and_save("MLR", y_train, y_train_pred_mlr, y_test, y_test_pred_mlr,
                   train_metrics_mlr, test_metrics_mlr, '#D98380', '#8E0f31', 'Perf_MLR')
joblib.dump(model_mlr, os.path.join(MODEL_SAVE_DIR, 'best_mlr_model.joblib'))
print("MLR model saved.")

# === 3.2 Partial Least Squares Regression (PLSR) - Optuna Optimization ===
print("\nOptimizing PLSR model with Optuna...")
def objective_pls(trial):
    max_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
    upper_bound = min(max_components, 30)
    if upper_bound < 1: upper_bound = 1
    n_components = trial.suggest_int('n_components', 1, upper_bound)
    model = PLSRegression(n_components=n_components, scale=False)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=CV_FOLDS,
                            scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -np.mean(scores)

study_pls = optuna.create_study(direction='minimize')
study_pls.optimize(objective_pls, n_trials=N_TRIALS_PLS, show_progress_bar=True)
best_params_pls = study_pls.best_params
print(f"Best PLSR parameters: {best_params_pls}")

model_pls = PLSRegression(**best_params_pls, scale=False)
model_pls.fit(X_train_scaled, y_train)
y_train_pred_pls = model_pls.predict(X_train_scaled).flatten()
y_test_pred_pls = model_pls.predict(X_test_scaled).flatten()
train_metrics_pls = calculate_metrics(y_train, y_train_pred_pls)
test_metrics_pls = calculate_metrics(y_test, y_test_pred_pls)
print_metrics("PLSR (Optuna Optimized)", train_metrics_pls, test_metrics_pls)
visualize_and_save("PLSR (Optuna)", y_train, y_train_pred_pls, y_test, y_test_pred_pls,
                   train_metrics_pls, test_metrics_pls, '#f68aa2', '#2db1ba', 'Perf_PLSR_Optimized')
joblib.dump(model_pls, os.path.join(MODEL_SAVE_DIR, 'best_pls_model.joblib'))
print("Optimized PLSR model saved.")

# === 3.3 XGBoost - Optuna Optimization ===
print("\nOptimizing XGBoost model with Optuna...")
def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror', 'eval_metric': 'rmse',
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-4, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
        'seed': RANDOM_STATE, 'nthread': -1
    }
    if params['booster'] == 'dart':
        params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        params['rate_drop'] = trial.suggest_float('rate_drop', 1e-4, 0.5, log=True)
        params['skip_drop'] = trial.suggest_float('skip_drop', 1e-4, 0.5, log=True)

    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=CV_FOLDS,
                            scoring='neg_root_mean_squared_error', n_jobs=-1)
    valid_scores = scores[~np.isnan(scores)]
    if len(valid_scores) == 0:
        return float('inf')
    rmse_cv = -np.mean(valid_scores)
    return rmse_cv

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS_XGB, show_progress_bar=True)
best_params_xgb = study_xgb.best_params
print(f"Best XGBoost parameters: {best_params_xgb}")

final_params_xgb = {'objective': 'reg:squarederror', 'eval_metric': 'rmse',
                   'seed': RANDOM_STATE, 'nthread': -1, **best_params_xgb}
model_xgb = xgb.XGBRegressor(**final_params_xgb)
model_xgb.fit(X_train_scaled, y_train)

y_train_pred_xgb = model_xgb.predict(X_train_scaled)
y_test_pred_xgb = model_xgb.predict(X_test_scaled)
train_metrics_xgb = calculate_metrics(y_train, y_train_pred_xgb)
test_metrics_xgb = calculate_metrics(y_test, y_test_pred_xgb)
print_metrics("XGBoost (Optuna Optimized)", train_metrics_xgb, test_metrics_xgb)
visualize_and_save("XGBoost (Optuna)", y_train, y_train_pred_xgb, y_test, y_test_pred_xgb,
                   train_metrics_xgb, test_metrics_xgb, '#CED48C', '#194F67', 'Perf_XGBoost_Optimized')
joblib.dump(model_xgb, os.path.join(MODEL_SAVE_DIR, 'best_xgb_model.joblib'))
print("Optimized XGBoost model saved.")

# --- 4. Save Preprocessor and Feature Names ---
print("\n--- 4. Saving Preprocessor and Feature List ---")
joblib.dump(scaler, os.path.join(MODEL_SAVE_DIR, 'scaler.joblib'))
joblib.dump(feature_names, os.path.join(MODEL_SAVE_DIR, 'feature_names.joblib'))
print(f"Scaler and feature names saved to: {MODEL_SAVE_DIR}")

print("\n--- Model Training and Evaluation Script Completed ---")