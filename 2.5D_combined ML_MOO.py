#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
combined_rf_optimization_v2.py
==============================

Key Steps:
1. Load data from specified Excel files (2.5D_SJR.xlsx, 2.5D_Thermal_lidless.xlsx).
2. Identify target (objective) columns using keywords.
3. Dynamically identify categorical input features by checking for text content.
4. Perform one-hot encoding for categorical features globally.
5. Train Random Forest models for each target:
   - Scale input features using a global StandardScaler.
   - Tune 'max_depth' hyperparameter.
   - Generate diagnostic plots.
6. Train XGBoost models for each target (for comparison and diagnostics):
   - Use the same scaled input features.
   - Tune 'max_depth' hyperparameter.
   - Generate diagnostic plots.
7. Generate combined feature importance plots (RF vs XGB).
8. Define a Pymoo optimization problem using the trained RF models as surrogates.
9. Run NSGA-II to find Pareto-optimal solutions.
10. Plot optimization results (RadViz, 2D Pareto fronts) and save solutions to CSV.
    - Plotting adapts to the number of detected and modeled targets.
11. Generate combined plot of all target distributions.
"""

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api import types as ptypes
from scipy.optimize import curve_fit
import itertools
import os
import re

from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# --- Configuration ---
FILE_CONFIGS = [
    {'name': '2.5D_SJR.xlsx'},
    {'name': '2.5D_Thermal_lidless.xlsx'}
]

TARGET_KEYWORDS = ["Theta", "DeltaW", "stress", "Warpage"]  # Keywords to identify target columns
RF_MAX_DEPTH_RANGE = [3, 5, 8, 10, 12, 15]
XGB_MAX_DEPTH_RANGE = [2, 3, 4, 5, 6, 8]
N_ESTIMATORS_DEFAULT = 100
TOP_N_FEATURES = 10

PLOTS_DIR = "../../Packaging/2.5D packaging project/optimization_plots_combined_v2"
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Plots will be saved to '{PLOTS_DIR}/' directory.")

#--- Plot configs ---
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.color": "gray"
})

# --- Helper Functions ---

def sanitize_filename_component(name):
    """Sanitizes a string to be used as a filename component."""
    name = str(name)
    name = re.sub(r'[<>:"/\\|?*]+', '_', name)  # Remove forbidden characters
    name = name.replace(' ', '_')  # Replace spaces with underscores
    return name


def build_column_defaults(dfs, cols_for_defaults):
    """Computes default (median) values for specified columns across multiple dataframes."""
    defaults = {}
    for c in cols_for_defaults:
        vals = []
        for df_item in dfs:
            if c in df_item.columns:
                # Attempt to convert to numeric, coercing errors to NaN
                numeric_col = pd.to_numeric(df_item[c], errors='coerce')
                valid_numeric_vals = numeric_col.dropna()  # Drop NaNs before calculating median
                if not valid_numeric_vals.empty:
                    vals.extend(valid_numeric_vals.tolist())
        # Use median of all collected valid numeric values, or 0.0 if none found
        defaults[c] = np.median(vals) if vals else 0.0
    return defaults


def get_best_param_from_r2_vc(estimator_class, base_params, X_train, y_train,
                              param_name, param_range, cv=5, n_jobs=-1, current_target_name_for_log=""):
    """
    Determines the best hyperparameter value using R² validation curve.
    """
    log_prefix = f"Target '{current_target_name_for_log}': " if current_target_name_for_log else ""
    model_name_for_log = estimator_class.__name__
    print(
        f"{log_prefix}Determining best '{param_name}' for {model_name_for_log} using R² validation curve (range: {param_range})...")

    current_tuning_params = base_params.copy()
    if param_name in current_tuning_params:
        del current_tuning_params[param_name]  # Remove param being tuned from base

    estimator = estimator_class(**current_tuning_params)
    try:
        # Ensure inputs are NumPy arrays for scikit-learn compatibility
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train

        if np.any(np.isnan(X_train_np)) or np.any(np.isinf(X_train_np)):
            print(f"{log_prefix}Warning: NaNs or Infs found in X_train for {param_name} VC of {model_name_for_log}.")
        if np.any(np.isnan(y_train_np)) or np.any(np.isinf(y_train_np)):
            print(f"{log_prefix}Warning: NaNs or Infs found in y_train for {param_name} VC of {model_name_for_log}.")

        _, test_scores = validation_curve(
            estimator, X_train_np, y_train_np, param_name=param_name, param_range=param_range,
            cv=cv, scoring="r2", n_jobs=n_jobs, error_score=np.nan  # Use NaN for errors to allow nanmean
        )
        raw_test_scores_mean = np.nanmean(test_scores, axis=1)  # Mean R² across CV folds

        if np.all(np.isnan(raw_test_scores_mean)):
            print(
                f"{log_prefix}Warning: All CV R² scores are NaN for {param_name} ({model_name_for_log}). Defaulting to first param: {param_range[0] if param_range else 'N/A'}.")
            return param_range[0] if param_range and len(param_range) > 0 else None

        best_param_idx = np.nanargmax(raw_test_scores_mean)  # Index of best mean R²
        param_range_array = np.array(param_range)
        best_param_value = param_range_array[best_param_idx]
        best_score = raw_test_scores_mean[best_param_idx]

        print(
            f"{log_prefix}Best {param_name} for {model_name_for_log} from R² VC: {best_param_value} (CV R² score: {best_score:.4f})")
        return best_param_value
    except Exception as e:
        print(f"{log_prefix}ERROR during R² validation_curve for {param_name} ({model_name_for_log}): {e}")
        if isinstance(X_train, pd.DataFrame): print(f"X_train dtypes:\n{X_train.dtypes}")
        return param_range[0] if param_range and len(param_range) > 0 else None


# --- Plotting Helper Functions ---

def plot_target_distribution(target_series, full_target_name_for_title, ax_hist, ax_box):
    """Plots histogram and box plot of the target variable on given axes."""
    try:
        data_for_hist = target_series.dropna()
        if data_for_hist.empty: raise ValueError("Target series empty for hist")
        ax_hist.hist(data_for_hist, bins='auto', edgecolor='k', alpha=0.7)
        ax_hist.set_title(f'Histogram - {full_target_name_for_title}',fontweight='bold')
        ax_hist.set_xlabel('Target Value');
        ax_hist.set_ylabel('Frequency')
        ax_hist.grid(True, linestyle='--', alpha=0.7)
    except Exception as e:
        ax_hist.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax_hist.transAxes)
        ax_hist.set_title(f'Histogram - {full_target_name_for_title}\nError')

    try:
        data_for_box = target_series.dropna()
        if data_for_box.empty: raise ValueError("Target series empty for boxplot")
        ax_box.boxplot(data_for_box, vert=False, widths=0.7, patch_artist=True, medianprops={'color': 'black'})
        ax_box.set_title(f'Box Plot - {full_target_name_for_title}',fontweight="bold")
        ax_box.set_yticklabels([]);
        ax_box.set_xlabel('Target Value')
        ax_box.grid(True, linestyle='--', alpha=0.7)
    except Exception as e:
        ax_box.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax_box.transAxes)
        ax_box.set_title(f'Box Plot - {full_target_name_for_title}\nError')


def _plot_single_validation_curve(ax, estimator_instance, X, y, param_name, param_range, scoring, title_suffix, cv=5,
                                  n_jobs=-1):
    """Helper to plot a single validation curve on a given Axes object."""
    try:
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y

        train_scores, test_scores = validation_curve(
            estimator_instance, X_np, y_np, param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring, n_jobs=n_jobs, error_score=np.nan
        )
    except Exception as e:
        ax.text(0.5, 0.5, f'VC Error:\n{e}', ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_title(f"Validation Curve {title_suffix}\nError");
        return

    raw_train_scores_mean = np.nanmean(train_scores, axis=1)
    raw_test_scores_mean = np.nanmean(test_scores, axis=1)
    train_scores_std = np.nanstd(train_scores, axis=1)
    test_scores_std = np.nanstd(test_scores, axis=1)

    plot_train_scores_mean, plot_test_scores_mean = raw_train_scores_mean, raw_test_scores_mean
    ylabel = f"Score ({scoring})"
    if scoring == "neg_mean_squared_error":
        plot_train_scores_mean, plot_test_scores_mean = -raw_train_scores_mean, -raw_test_scores_mean  # Invert for positive MSE
        ylabel = "Mean Squared Error (MSE)"
        max_val = 0
        if not np.all(np.isnan(plot_train_scores_mean)): max_val = np.nanmax(
            plot_train_scores_mean[~np.isnan(plot_train_scores_mean)])
        if not np.all(np.isnan(plot_test_scores_mean)): max_val = max(max_val, np.nanmax(
            plot_test_scores_mean[~np.isnan(plot_test_scores_mean)]))
        ax.set_ylim(bottom=0, top=max_val * 1.1 if max_val > 0 else 1.0)  # Ensure MSE plot starts at 0
    elif scoring == "r2":
        ylabel = "R² Score";
        # ax.set_ylim(-0.1, 1.1)  # R² range
        # Dynamically adjust y-axis limits based on min/max of mean ± std for both training and validation scores
        min_y = min(np.nanmin(raw_train_scores_mean - train_scores_std),np.nanmin(raw_test_scores_mean - test_scores_std))
        max_y = max(np.nanmax(raw_train_scores_mean + train_scores_std),np.nanmax(raw_test_scores_mean + test_scores_std))
        ax.set_ylim(min_y - 0.05, max_y + 0.05)

    param_range_array = np.array(param_range)
    # Use numerical index for x-axis if param_range contains non-numeric types, else use actual param values
    plot_param_range = np.arange(len(param_range_array)) if not all(
        isinstance(pr, (int, float)) for pr in param_range_array) else param_range_array.astype(float)
    tick_labels = [str(pr) for pr in param_range_array]

    ax.plot(plot_param_range, plot_train_scores_mean, label="Training score", color="darkorange", marker='o', lw=2)
    ax.fill_between(plot_param_range, plot_train_scores_mean - train_scores_std,
                    plot_train_scores_mean + train_scores_std, alpha=0.2, color="darkorange")
    ax.plot(plot_param_range, plot_test_scores_mean, label="Cross-validation score", color="navy", marker='o', lw=2)
    ax.fill_between(plot_param_range, plot_test_scores_mean - test_scores_std, plot_test_scores_mean + test_scores_std,
                    alpha=0.2, color="navy")

    ax.set_xticks(plot_param_range);
    ax.set_xticklabels(tick_labels, rotation=30, ha='right')
    ax.set_title(f"Validation Curve {title_suffix} ({param_name})",fontweight='bold')
    ax.set_xlabel(str(param_name));
    ax.set_ylabel(ylabel)
    ax.legend(loc="best");
    ax.grid(True)


def _plot_single_learning_curve(ax, estimator_instance, X, y, scoring, title_suffix, cv=5, n_jobs=-1,
                                train_sizes=np.linspace(.1, 1.0, 5)):
    """Helper to plot a single learning curve on a given Axes object."""
    try:
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator_instance, X_np, y_np, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes,
            random_state=42, error_score=np.nan
        )
    except Exception as e:
        ax.text(0.5, 0.5, f'LC Error:\n{e}', ha='center', va='center', transform=ax.transAxes, fontsize=8)
        ax.set_title(f"Learning Curve {title_suffix}\nError");
        return

    raw_train_scores_mean = np.nanmean(train_scores, axis=1)
    raw_test_scores_mean = np.nanmean(test_scores, axis=1)
    train_scores_std = np.nanstd(train_scores, axis=1)
    test_scores_std = np.nanstd(test_scores, axis=1)

    plot_train_scores_mean, plot_test_scores_mean = raw_train_scores_mean, raw_test_scores_mean
    ylabel = f"Score ({scoring})"
    if scoring == "neg_mean_squared_error":
        plot_train_scores_mean, plot_test_scores_mean = -raw_train_scores_mean, -raw_test_scores_mean
        ylabel = "Mean Squared Error (MSE)"
        max_val = 0
        if not np.all(np.isnan(plot_train_scores_mean)): max_val = np.nanmax(
            plot_train_scores_mean[~np.isnan(plot_train_scores_mean)])
        if not np.all(np.isnan(plot_test_scores_mean)): max_val = max(max_val, np.nanmax(
            plot_test_scores_mean[~np.isnan(plot_test_scores_mean)]))
        ax.set_ylim(bottom=0, top=max_val * 1.1 if max_val > 0 else 1.0)
    elif scoring == "r2":
        ylabel = "R² Score";
        # ax.set_ylim(-0.1, 1.1)
        # Dynamically adjust y-axis limits based on min/max of mean ± std for both training and validation scores
        min_y = min(np.nanmin(raw_train_scores_mean - train_scores_std),
                    np.nanmin(raw_test_scores_mean - test_scores_std))
        max_y = max(np.nanmax(raw_train_scores_mean + train_scores_std),
                    np.nanmax(raw_test_scores_mean + test_scores_std))
        ax.set_ylim(min_y - 0.05, max_y + 0.05)

    ax.plot(train_sizes_abs, plot_train_scores_mean, 'o-', color="darkorange", label="Training score", lw=2)
    ax.fill_between(train_sizes_abs, plot_train_scores_mean - train_scores_std,
                    plot_train_scores_mean + train_scores_std, alpha=0.1, color="darkorange")
    ax.plot(train_sizes_abs, plot_test_scores_mean, 'o-', color="navy", label="Cross-validation score", lw=2)
    ax.fill_between(train_sizes_abs, plot_test_scores_mean - test_scores_std, plot_test_scores_mean + test_scores_std,
                    alpha=0.1, color="navy")

    ax.set_title(f"Learning Curve {title_suffix}",fontweight='bold')
    ax.set_xlabel("Training examples");
    ax.set_ylabel(ylabel)
    ax.legend(loc="best");
    ax.grid(True)
    ax.set_xlim(left=0, right=train_sizes_abs.max() * 1.05 if train_sizes_abs.size > 0 else 1)


def _plot_single_actual_vs_predicted(ax, y_true, y_pred, dataset_name_str):
    """Helper to plot a single Actual vs. Predicted scatter plot."""
    if y_true is None or y_pred is None:
        ax.text(0.5, 0.5, 'Data N/A', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"Actual vs. Predicted - {dataset_name_str}\nData N/A");
        return

    # Remove NaNs that might arise from failed predictions or issues in y_true/y_pred
    y_true_clean = y_true[~np.isnan(y_true) & ~np.isnan(y_pred)]
    y_pred_clean = y_pred[~np.isnan(y_true) & ~np.isnan(y_pred)]
    if len(y_true_clean) == 0:
        ax.text(0.5, 0.5, 'No valid data points', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"Actual vs. Predicted - {dataset_name_str}\nNo Valid Data",fontweight='bold');
        return

    ax.scatter(y_true_clean, y_pred_clean, alpha=0.6, edgecolors='k', s=50, label="Data points")
    all_vals = np.concatenate([y_true_clean, y_pred_clean])
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    margin = (max_val - min_val) * 0.05 if (max_val - min_val) > 0 else 0.1
    plot_min, plot_max = min_val - margin, max_val + margin
    if plot_min == plot_max: plot_min -= 0.5; plot_max += 0.5  # Handle case where all points are same

    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', lw=2, label="Ideal (y=x)")  # Perfect prediction line
    ax.set_xlabel("Actual Values");
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"Actual vs. Predicted - {dataset_name_str}",fontweight='bold')
    ax.legend(loc="best");
    ax.grid(True)
    ax.set_xlim(plot_min, plot_max);
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect('equal', adjustable='box')  # Ensure square plot with equal scaling

def generate_combined_actual_vs_predicted_grid(scatter_info_list, output_path):
    """
    Draws all Actual vs. Predicted plots into a single 2x3 figure.
    scatter_info_list: List of tuples -> (y_true, y_pred, dataset_name)
    output_path: Path to save the combined figure
    """
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    axs = axs.flatten()
    for i, (y_true, y_pred, dataset_name) in enumerate(scatter_info_list):
        if i >= 6: break  # Limit to 6 plots
        _plot_single_actual_vs_predicted(axs[i], y_true, y_pred, dataset_name)

    fig.suptitle("Actual vs. Predicted - All Targets", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved combined Actual vs. Predicted plot to {output_path}")

def generate_diagnostic_subplots(estimator_class, base_params,
                                 X_train, y_train, X_test, y_test,
                                 vc_param_name, vc_param_range,
                                 best_param_value_for_lc,  # Best param value determined from VC
                                 main_plot_title, output_plot_filename_base):
    """Generates a 2x3 subplot figure with VCs, LCs, and Actual vs. Predicted plots."""
    print(f"  Generating diagnostic plot: {main_plot_title}")
    fig, axs = plt.subplots(2, 3, figsize=(24, 14))  # 2 rows, 3 columns of subplots

    # --- Validation Curves (Row 1, Col 1 & 2) ---
    vc_estimator_params_for_plot = base_params.copy()
    if vc_param_name in vc_estimator_params_for_plot: del vc_estimator_params_for_plot[vc_param_name]
    vc_estimator = estimator_class(**vc_estimator_params_for_plot)
    _plot_single_validation_curve(axs[0, 0], vc_estimator, X_train, y_train, vc_param_name, vc_param_range, "r2",
                                  "(R²)")
    _plot_single_validation_curve(axs[0, 1], vc_estimator, X_train, y_train, vc_param_name, vc_param_range,
                                  "neg_mean_squared_error", "(MSE)")

    # --- Model for Learning Curves and Predictions (using best param from VC) ---
    model_params_for_lc_pred = base_params.copy()
    initial_model_fit_error = False
    if best_param_value_for_lc is not None:
        model_params_for_lc_pred[vc_param_name] = best_param_value_for_lc
    else:
        print(f"    Warning: best_param_value_for_lc for {vc_param_name} is None. Using first from range for LC/Pred.")
        if vc_param_range and len(vc_param_range) > 0:
            model_params_for_lc_pred[vc_param_name] = vc_param_range[0]
        else:  # Should not happen if vc_param_range is always populated
            initial_model_fit_error = True;
            print(
                f"    Error: vc_param_range empty for {vc_param_name}. Cannot create model for LC/Pred.")

    # Ensure n_estimators is set, defaulting if necessary
    if 'n_estimators' not in model_params_for_lc_pred:
        model_params_for_lc_pred['n_estimators'] = base_params.get('n_estimators', N_ESTIMATORS_DEFAULT)

    model_for_lc_pred = None
    if not initial_model_fit_error:
        try:
            model_for_lc_pred = estimator_class(**model_params_for_lc_pred)
        except Exception as e_inst:
            print(f"    Error instantiating model for LC/Pred plots: {e_inst}")
            initial_model_fit_error = True

    y_train_pred, y_test_pred = None, None
    predictions_available = False
    actual_fitting_error = False

    if model_for_lc_pred is not None:
        try:
            model_for_lc_pred.fit(X_train, y_train)
            y_train_pred = model_for_lc_pred.predict(X_train)
            y_test_pred = model_for_lc_pred.predict(X_test)
            predictions_available = True
        except Exception as e_fit:
            print(f"    Error fitting model for LC/Pred plots: {e_fit}");
            actual_fitting_error = True
    else:  # If model_for_lc_pred is None due to instantiation error
        actual_fitting_error = True  # Consider it a fitting phase error

    # --- Actual vs. Predicted - Training Data (Row 1, Col 3) ---
    if predictions_available:
        _plot_single_actual_vs_predicted(axs[0, 2], y_train, y_train_pred, "Training Data")
    else:
        err_msg = 'Model fit/instantiation failed' if (initial_model_fit_error or actual_fitting_error) else 'Pred. N/A'
        axs[0, 2].text(0.5, 0.5, err_msg, ha='center', va='center', transform=axs[0, 2].transAxes)
        axs[0, 2].set_title(f"Actual vs. Predicted - Training\n{err_msg}",fontweight='bold')

    # --- Learning Curves (Row 2, Col 1 & 2) ---
    if model_for_lc_pred is not None and not actual_fitting_error:  # Only plot if model was successfully fit
        lc_title_suffix_r2 = f"(R², {vc_param_name}={model_params_for_lc_pred.get(vc_param_name, 'N/A')}, N_est={model_params_for_lc_pred.get('n_estimators', 'N/A')})"
        lc_title_suffix_mse = f"(MSE, {vc_param_name}={model_params_for_lc_pred.get(vc_param_name, 'N/A')}, N_est={model_params_for_lc_pred.get('n_estimators', 'N/A')})"
        _plot_single_learning_curve(axs[1, 0], model_for_lc_pred, X_train, y_train, "r2", lc_title_suffix_r2)
        _plot_single_learning_curve(axs[1, 1], model_for_lc_pred, X_train, y_train, "neg_mean_squared_error",
                                    lc_title_suffix_mse)
    else:
        err_msg = 'LC not plotted (model error)'
        axs[1, 0].text(0.5, 0.5, err_msg, ha='center', va='center');
        axs[1, 0].set_title(f"Learning Curve (R²)\n{err_msg}")
        axs[1, 1].text(0.5, 0.5, err_msg, ha='center', va='center');
        axs[1, 1].set_title(f"Learning Curve (MSE)\n{err_msg}")

    # --- Actual vs. Predicted - Test Data (Row 2, Col 3) ---
    if predictions_available:
        _plot_single_actual_vs_predicted(axs[1, 2], y_test, y_test_pred, "Test Data")
    else:
        err_msg = 'Model fit/instantiation failed' if (initial_model_fit_error or actual_fitting_error) else 'Pred. N/A'
        axs[1, 2].text(0.5, 0.5, err_msg, ha='center', va='center', transform=axs[1, 2].transAxes)
        axs[1, 2].set_title(f"Actual vs. Predicted - Test\n{err_msg}")

    fig.suptitle(main_plot_title, fontsize=18,fontweight='bold')
    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95],pad=2.0)  # Adjust layout to prevent title overlap
    except UserWarning as e:  # Catch potential UserWarning from tight_layout
        print(f"    UserWarning during tight_layout: {e}")

    output_filename = os.path.join(PLOTS_DIR, f"{output_plot_filename_base}_diagnostic_plots.png")
    plt.savefig(output_filename);
    print(f"  Saved diagnostic plots to {output_filename}")
    plt.close(fig)


def plot_combined_top_n_feature_importances(model1, model2, model1_name, model2_name,
                                            feature_names, top_n, plot_title_suffix, ax):
    """Plots combined feature importances from two models (e.g., RF and XGB) on a given ax."""
    if not feature_names:  # Check if feature names list is provided
        ax.text(0.5, 0.5, "Feature names missing", ha='center', va='center');
        ax.set_title(f"FI - {plot_title_suffix}\nError");
        return

    # Get feature importances if models and attributes exist
    m1_fi = model1.feature_importances_ if model1 and hasattr(model1, 'feature_importances_') else None
    m2_fi = model2.feature_importances_ if model2 and hasattr(model2, 'feature_importances_') else None

    if m1_fi is None and m2_fi is None:  # If neither model provided importances
        ax.text(0.5, 0.5, f'{model1_name} & {model2_name}\nNo importances', ha='center', va='center');
        ax.set_title(f"FI - {plot_title_suffix}\nError");
        return

    len_features = len(feature_names)
    imp_data = {'feature': list(feature_names)}  # Start with feature names
    m1_valid, m2_valid = False, False  # Flags for valid importances

    if m1_fi is not None and len(m1_fi) == len_features:
        imp_data[model1_name] = m1_fi;
        m1_valid = True
    else:  # Pad with zeros if model1 importances are missing/invalid
        imp_data[model1_name] = np.zeros(len_features);

    if m2_fi is not None and len(m2_fi) == len_features:
        imp_data[model2_name] = m2_fi;
        m2_valid = True
    else:  # Pad with zeros if model2 importances are missing/invalid
        imp_data[model2_name] = np.zeros(len_features);

    if not m1_valid and not m2_valid:  # If no valid data from either model
        ax.text(0.5, 0.5, 'No valid FI data', ha='center', va='center');
        ax.set_title(
            f"FI - {plot_title_suffix}\nError");
        return

    df_imp = pd.DataFrame(imp_data)
    # Combine scores for ranking, prioritizing valid data
    df_imp['combined_score'] = df_imp[model1_name] + df_imp[model2_name] if m1_valid and m2_valid else (
        df_imp[model1_name] if m1_valid else df_imp[model2_name])
    top_features_df = df_imp.sort_values(by='combined_score', ascending=False).head(top_n)

    if top_features_df.empty:
        ax.text(0.5, 0.5, 'No features to display', ha='center', va='center');
        ax.set_title(f"Top {top_n} FI - {plot_title_suffix}\nNo data");
        return

    index = np.arange(len(top_features_df))  # y-positions for bars
    bar_width = 0.35  # Width of each bar

    # Plot bars for each model if data is valid
    if m1_valid: ax.barh(index - bar_width / 2 if m2_valid else index, top_features_df[model1_name],
                         bar_width if m2_valid else bar_width * 1.5, label=model1_name, color='skyblue')
    if m2_valid: ax.barh(index + bar_width / 2 if m1_valid else index, top_features_df[model2_name],
                         bar_width if m1_valid else bar_width * 1.5, label=model2_name, color='lightcoral')

    ax.set_xlabel('Feature Importance');
    ax.set_ylabel('Features')
    ax.set_title(f"Top {top_n} Feature Importances - {plot_title_suffix}",fontweight='bold')
    ax.set_yticks(index);
    ax.set_yticklabels(top_features_df['feature'])  # Feature names as y-ticks
    ax.legend();
    ax.invert_yaxis()  # Display most important at the top


def train_evaluate_final_model(model_class, model_params, X_train, y_train, X_test, y_test, model_name_with_target):
    """Trains the final model with best params and evaluates it."""
    print(f"  Training and evaluating final {model_name_with_target}...")
    model = model_class(**model_params)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"    ERROR during final model fitting for {model_name_with_target}: {e}")
        return None  # Return None if fitting fails

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    print(f"  --- {model_name_with_target} Final Evaluation ---")
    print(f"    Training -> MSE: {train_mse:.4f}, R2: {train_r2:.4f}")
    print(f"    Test     -> MSE: {test_mse:.4f}, R2: {test_r2:.4f}")
    return model


def radviz_projection(objs):
    """Projects multi-objective data into 2D for RadViz plot."""
    if objs is None or objs.shape[0] == 0: return np.empty((0, 2))  # Handle empty input
    m = objs.shape[1]  # Number of objectives
    if m == 0: return np.empty((objs.shape[0], 2))  # Handle zero objectives

    # Normalize objectives to [0, 1] range (inverted: 0 is best, 1 is worst for projection)
    obj_min = objs.min(axis=0);
    obj_range = np.ptp(objs, axis=0)  # Peak-to-peak (max - min)
    obj_range_safe = np.where(obj_range == 0, 1e-9, obj_range)  # Avoid division by zero
    norm = (objs - obj_min) / obj_range_safe;  # Standard normalization
    norm = 1.0 - norm  # Invert: higher original value -> smaller normalized value (closer to anchor)

    # Define anchor points for objectives on a unit circle
    angles = 2 * np.pi * np.arange(m) / m
    S_matrix = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (m x 2) matrix of anchor coords

    # Calculate RadViz projection
    numerator = norm @ S_matrix  # Weighted sum of anchor coords
    denominator = norm.sum(axis=1, keepdims=True)  # Sum of normalized objective values
    denominator_safe = np.where(denominator == 0, 1e-12, denominator)  # Avoid division by zero
    return numerator / denominator_safe


def exp_func_type1(x, a, b, c):
    """Exponential function for fitting Pareto curves: a * exp(b*x) + c"""
    x_safe = np.clip(x * b, -700, 700);  # Prevent overflow in exp
    return a * np.exp(x_safe) + c


# --- Main Script Logic ---

def run_combined_workflow():
    print("--- Starting Combined RF Training and Optimization Workflow (v2) ---")
    all_target_series_for_global_plot = []  # For combined distribution plot
    all_feature_importance_data = []  # For combined FI plot

    # 1. Load Excel data
    print("\n1. Loading data...")
    raw_dfs = []
    for config in FILE_CONFIGS:
        fname = config['name']
        try:
            df = pd.read_excel(fname)
            raw_dfs.append(df);
            print(f"Successfully loaded {fname} with {len(df.columns)} columns and {len(df)} rows.")
        except Exception as e:
            print(f"Error loading {fname}: {e}. Proceeding without it.");
            raw_dfs.append(pd.DataFrame())  # Append empty DataFrame on error
    original_dfs_for_cat_analysis = [df.copy() for df in raw_dfs]  # Keep copies for analysis

    # Identify all potential target columns from all files
    initial_objective_names = []
    target_info_map = {}  # Stores which file a target primarily comes from (for naming)
    for idx, df_orig in enumerate(original_dfs_for_cat_analysis):
        if df_orig.empty: continue

        file_name_prefix_for_target = "unknownfile"
        if idx < len(FILE_CONFIGS):
            original_file_name = FILE_CONFIGS[idx]['name']
            file_name_prefix_for_target = sanitize_filename_component(os.path.splitext(original_file_name)[0])

        for col_name in df_orig.columns:
            is_target = any(keyword.lower() in col_name.lower() for keyword in TARGET_KEYWORDS)
            if is_target:
                if col_name not in initial_objective_names:  # Add if new
                    initial_objective_names.append(col_name)
                    target_info_map[col_name] = {'df_index': idx, 'file_name_prefix': file_name_prefix_for_target}
    initial_objective_names = sorted(list(set(initial_objective_names)))  # Unique sorted list
    if not initial_objective_names:
        raise SystemExit("CRITICAL: No target columns identified based on keywords. Exiting.")
    print(f"Identified potential objectives ({len(initial_objective_names)}): {initial_objective_names}")

    # Dynamically identify categorical columns based on text content
    print("\nIdentifying categorical features dynamically based on text content...")
    globally_identified_cat_cols_set = set()
    for df_idx, df_orig in enumerate(original_dfs_for_cat_analysis):
        current_file_name = FILE_CONFIGS[df_idx]['name'] if df_idx < len(FILE_CONFIGS) else f"DF_Index_{df_idx}"
        if df_orig.empty:
            print(f"Info: DataFrame from '{current_file_name}' is empty. Skipping for categorical detection.")
            continue

        # print(f"Processing DataFrame from '{current_file_name}' for categorical features...")
        for col_name in df_orig.columns:
            if col_name in initial_objective_names:  # Skip if it's a target column
                continue

            is_categorical_by_text = False
            if df_orig[col_name].notna().any():  # Only check if column has non-null values
                if df_orig[col_name].dtype == 'object':  # 'object' dtype often indicates text or mixed types
                    try:
                        # Check if any non-NaN value in the column is a Python string
                        if df_orig[col_name].dropna().apply(lambda x: isinstance(x, str)).any():
                            is_categorical_by_text = True
                            # print(f"  Column '{col_name}' in '{current_file_name}' (dtype: object) identified as categorical due to string instances.")
                    except Exception as e:
                        print(
                            f"  Warning: Error checking string instances in object column '{col_name}' from '{current_file_name}': {e}.")

            if is_categorical_by_text:
                globally_identified_cat_cols_set.add(col_name)

    final_cat_cols_to_process = sorted(list(globally_identified_cat_cols_set))
    if not final_cat_cols_to_process:
        print("Info: No categorical columns were dynamically identified based on text content across all files.")
    else:
        print(
            f"Dynamically identified categorical input columns based on text content ({len(final_cat_cols_to_process)}): {final_cat_cols_to_process}")

    # Prepare for One-Hot Encoding (OHE)
    CAT_COL_DETAILS = {}  # Stores categories and OHE names for each categorical column
    globally_generated_ohe_col_names = []  # All unique OHE column names
    for col_name in final_cat_cols_to_process:
        all_vals = []  # Collect all unique values for this cat col from all DFs
        for df_orig in original_dfs_for_cat_analysis:
            if col_name in df_orig.columns: all_vals.extend(df_orig[col_name].dropna().astype(str).tolist())
        unique_cats = sorted(list(set(all_vals)))  # Unique sorted categories
        if not unique_cats: print(
            f"Warn: No unique categories found for '{col_name}'. Skipping OHE for this column."); continue

        # Create OHE column names: original_col_name + _ + category_value
        ohe_names = [f"{sanitize_filename_component(col_name)}_{sanitize_filename_component(s_val)}" for s_val in
                     unique_cats]
        CAT_COL_DETAILS[col_name] = {'categories': unique_cats, 'one_hot_names': ohe_names}
        globally_generated_ohe_col_names.extend(ohe_names)
    globally_generated_ohe_col_names = sorted(list(set(globally_generated_ohe_col_names)))  # Unique sorted OHE names
    print(f"Total unique OHE column names to be generated globally: {len(globally_generated_ohe_col_names)}")

    # Perform OHE on each DataFrame and align columns
    processed_dfs_after_ohe = []  # List to store DFs after OHE
    target_to_ohe_df_map = {}  # Maps target name to its corresponding OHE'd DataFrame
    for df_idx, df_orig in enumerate(original_dfs_for_cat_analysis):
        if df_orig.empty: processed_dfs_after_ohe.append(df_orig.copy()); continue  # Keep empty DF as is
        df_updated = df_orig.copy()

        # Separate categorical columns present in this specific DF
        cats_in_this_df = [c for c in final_cat_cols_to_process if c in df_updated.columns]
        non_cat_part = df_updated.drop(columns=cats_in_this_df,
                                       errors='ignore') if cats_in_this_df else df_updated.copy()

        ohe_gen_part = pd.DataFrame(index=non_cat_part.index)  # Empty DF for OHE columns
        if cats_in_this_df:
            cat_part_to_encode = df_updated[cats_in_this_df].astype(str)  # Ensure string type for get_dummies
            s_prefixes = {col: sanitize_filename_component(col) for col in cats_in_this_df}  # Prefixes for OHE names
            ohe_gen_part = pd.get_dummies(cat_part_to_encode, columns=cats_in_this_df, prefix=s_prefixes,
                                          prefix_sep='_', dummy_na=False)  # Perform OHE

        # Reindex OHE part to include ALL globally generated OHE columns, filling missing with 0
        ohe_reindexed = ohe_gen_part.reindex(columns=globally_generated_ohe_col_names, fill_value=0)
        df_full_updated = pd.concat([non_cat_part, ohe_reindexed], axis=1)  # Combine non-categorical and OHE parts
        processed_dfs_after_ohe.append(df_full_updated)

    # Map each target to its processed DataFrame (the one it originated from)
    for target_name, info in target_info_map.items():
        target_to_ohe_df_map[target_name] = processed_dfs_after_ohe[info['df_index']]
    print("One-hot encoding and DataFrame column alignment complete.")

    # Define the global set of input features for models (all non-target columns after OHE)
    temp_rf_feat_set = set()
    for df_ohe in processed_dfs_after_ohe:
        if not df_ohe.empty:
            for col in df_ohe.columns:
                if col not in initial_objective_names: temp_rf_feat_set.add(col)
    rf_input_features = sorted(list(temp_rf_feat_set))  # Final sorted list of all input features
    print(f"Total global input features for models (rf_input_features): {len(rf_input_features)}")
    if not rf_input_features: raise SystemExit("CRITICAL: No input features identified for models. Exiting.")

    # Identify continuous and original categorical variables for Pymoo optimization
    pymoo_opt_vars_continuous = sorted([col for col in rf_input_features if
                                        col not in globally_generated_ohe_col_names and col not in final_cat_cols_to_process])
    pymoo_opt_vars_categorical_original = sorted(final_cat_cols_to_process)  # Original names before OHE
    pymoo_opt_vars_all = pymoo_opt_vars_continuous + pymoo_opt_vars_categorical_original  # All variables Pymoo will handle
    print(
        f"Continuous variables for Pymoo ({len(pymoo_opt_vars_continuous)}): {pymoo_opt_vars_continuous[:min(5, len(pymoo_opt_vars_continuous))]}...")
    print(
        f"Original categorical variables for Pymoo ({len(pymoo_opt_vars_categorical_original)}): {pymoo_opt_vars_categorical_original}")

    # Store Pymoo index for categorical variables within CAT_COL_DETAILS
    for cat_name, details in CAT_COL_DETAILS.items():
        if cat_name in pymoo_opt_vars_all: details['pymoo_index'] = pymoo_opt_vars_all.index(cat_name)

    # Build default values (medians) for all RF input features
    defaults_for_rf = build_column_defaults(processed_dfs_after_ohe, rf_input_features)
    print(f"Computed defaults for {len(defaults_for_rf)} model input features.")

    # 2. Fit RF and XGBoost models for each objective
    print("\n2. Fitting RF and XGBoost surrogates with hyperparameter tuning...")
    fitted_rf_models_dict = {}  # Stores trained RF models for Pymoo
    global_problem_scaler = None  # Scaler for input features, fitted once

    for target_idx, target_name in enumerate(initial_objective_names):
        print(f"\n  --- Processing Target: {target_name} ({target_idx + 1}/{len(initial_objective_names)}) ---")
        if target_name not in target_to_ohe_df_map: print(f"  Warn: No OHE DF for '{target_name}'. Skip."); continue
        df_for_target = target_to_ohe_df_map[target_name]  # Get the processed DF for this target
        if df_for_target.empty or target_name not in df_for_target.columns: print(
            f"  Skipping '{target_name}': DataFrame is empty or target column is missing."); continue

        # Prepare X (features) and y (target) for this specific target
        current_input_cols = [col for col in df_for_target.columns if col in rf_input_features]
        X_intermediate = df_for_target[current_input_cols]
        # Reindex to ensure all global features are present, fill missing with NaN initially
        X_reindexed = X_intermediate.reindex(columns=rf_input_features, fill_value=np.nan)
        X_filled = X_reindexed.fillna(defaults_for_rf)  # Fill NaNs with global defaults
        y = df_for_target[target_name].fillna(
            df_for_target[target_name].median())  # Fill NaNs in target with its median

        if X_filled.empty or y.empty: print(f"  Skipping '{target_name}': X or y is empty after processing."); continue
        if X_filled.isnull().values.any(): X_filled = X_filled.fillna(0); print(
            f"  Warn: X for {target_name} had NaNs after reindex/default fill. Filled remaining with 0.")

        # Split data into training and testing sets
        X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_filled, y, test_size=0.2, random_state=42)
        if X_train_orig.empty: print(f"  Skipping '{target_name}': X_train is empty after split."); continue

        # Fit the global scaler on the training data of the *first* target processed
        if global_problem_scaler is None:
            print(f"    Fitting global_problem_scaler on X_train_orig of first target '{target_name}'.")
            global_problem_scaler = StandardScaler().fit(X_train_orig)

        # Scale features
        Xs_train = global_problem_scaler.transform(X_train_orig)
        Xs_test = global_problem_scaler.transform(X_test_orig)

        # Prepare names for plots and file
        file_prefix = "file"
        if target_name in target_info_map and 'file_name_prefix' in target_info_map[target_name]:
            file_prefix = target_info_map[target_name]['file_name_prefix']
        plot_target_name_for_title = f"{file_prefix}_{sanitize_filename_component(target_name)}"
        all_target_series_for_global_plot.append((df_for_target[target_name], plot_target_name_for_title))

        # --- Random Forest Model ---
        print(f"\n    ----- Random Forest for Target: {target_name} -----")
        rf_base_params = {'random_state': 42, 'n_estimators': N_ESTIMATORS_DEFAULT, 'n_jobs': -1}
        best_rf_depth = get_best_param_from_r2_vc(RandomForestRegressor, rf_base_params, Xs_train, y_train, "max_depth",
                                                  RF_MAX_DEPTH_RANGE, current_target_name_for_log=target_name)

        trained_rf_model = None
        if best_rf_depth is not None:
            final_rf_params = {**rf_base_params, 'max_depth': best_rf_depth}
            generate_diagnostic_subplots(RandomForestRegressor, final_rf_params, Xs_train, y_train, Xs_test, y_test,
                                         "max_depth", RF_MAX_DEPTH_RANGE, best_rf_depth,
                                         f"RF Diagnostics - {plot_target_name_for_title}",
                                         f"{plot_target_name_for_title}_RF")
            trained_rf_model = train_evaluate_final_model(RandomForestRegressor, final_rf_params, Xs_train, y_train,
                                                          Xs_test, y_test, f"Final RF ({plot_target_name_for_title})")
            if trained_rf_model: fitted_rf_models_dict[target_name] = trained_rf_model  # Store if successful
        else:
            print(
                f"    Skipping RF training for '{target_name}' due to hyperparameter (max_depth) determination error.")

        # --- XGBoost Model ---
        print(f"\n    ----- XGBoost for Target: {target_name} -----")
        xgb_base_params = {'random_state': 42, 'n_estimators': N_ESTIMATORS_DEFAULT, 'n_jobs': -1,
                           'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
        best_xgb_depth = get_best_param_from_r2_vc(XGBRegressor, xgb_base_params, Xs_train, y_train, "max_depth",
                                                   XGB_MAX_DEPTH_RANGE, current_target_name_for_log=target_name)

        trained_xgb_model = None
        if best_xgb_depth is not None:
            final_xgb_params = {**xgb_base_params, 'max_depth': best_xgb_depth}
            generate_diagnostic_subplots(XGBRegressor, final_xgb_params, Xs_train, y_train, Xs_test, y_test,
                                         "max_depth", XGB_MAX_DEPTH_RANGE, best_xgb_depth,
                                         f"XGBoost Diagnostics - {plot_target_name_for_title}",
                                         f"{plot_target_name_for_title}_XGB")
            trained_xgb_model = train_evaluate_final_model(XGBRegressor, final_xgb_params, Xs_train, y_train, Xs_test,
                                                           y_test, f"Final XGBoost ({plot_target_name_for_title})")
        else:
            print(
                f"    Skipping XGBoost training for '{target_name}' due to hyperparameter (max_depth) determination error.")

        # Store data for combined feature importance plot
        all_feature_importance_data.append(
            (trained_rf_model, trained_xgb_model, list(X_filled.columns), plot_target_name_for_title))

    # Filter for successfully trained RF models for Pymoo
    active_objective_names_for_pymoo = [name for name in initial_objective_names if name in fitted_rf_models_dict]
    active_rf_models_for_pymoo = [fitted_rf_models_dict[name] for name in active_objective_names_for_pymoo]

    if not active_objective_names_for_pymoo or global_problem_scaler is None:
        raise SystemExit(
            "CRITICAL: No RF models were successfully trained for Pymoo, or global_problem_scaler is missing. Exiting.")
    print(
        f"\nRF models for Pymoo ready. Active objectives ({len(active_objective_names_for_pymoo)}): {active_objective_names_for_pymoo}")

    # 3. Define the multi-objective problem for Pymoo
    print("\n3. Defining MOO problem...")
    var_bounds_list = []  # List of [min, max] for each Pymoo variable
    all_orig_dfs_for_bounds = [df.copy() for df in original_dfs_for_cat_analysis if
                               not df.empty]  # Use original DFs for bounds

    for var_name in pymoo_opt_vars_all:  # Iterate through all variables Pymoo will optimize
        if var_name in pymoo_opt_vars_continuous:  # Continuous variable
            min_v, max_v = np.inf, -np.inf;
            found = False
            for df_src in all_orig_dfs_for_bounds:  # Check all source DFs
                if var_name in df_src.columns and pd.api.types.is_numeric_dtype(df_src[var_name]):
                    num_data = pd.to_numeric(df_src[var_name], errors='coerce').dropna()
                    if not num_data.empty: min_v = min(min_v, num_data.min()); max_v = max(max_v,
                                                                                           num_data.max()); found = True
            if not found or min_v == np.inf:  # Default bounds if not found
                min_v, max_v = 0.0, 1.0;
                print(f"Warn: Bounds for continuous var '{var_name}' not found. Defaulting to [0,1].")
            elif min_v >= max_v:  # Ensure min < max
                dlt = max(0.01, abs(max_v * 0.01));
                min_v, max_v = min(min_v, max_v) - dlt, max(min_v, max_v) + dlt
            var_bounds_list.append([min_v, max_v])
        elif var_name in CAT_COL_DETAILS:  # Categorical variable (original name)
            # Bounds are indices [0, num_categories - 1]
            var_bounds_list.append([0, len(CAT_COL_DETAILS[var_name]['categories']) - 1])
        else:  # Should not happen if logic is correct
            var_bounds_list.append([0.0, 1.0]);
            print(f"Err: Var '{var_name}' type unknown for bounds. Default [0,1].")
    bounds_np_array = np.array(var_bounds_list, dtype=float)

    class SurrogateProblem(Problem):
        def __init__(self):
            super().__init__(n_var=len(pymoo_opt_vars_all),
                             n_obj=len(active_objective_names_for_pymoo),  # Number of active objectives
                             n_constr=0,  # No constraints defined here
                             xl=bounds_np_array[:, 0],  # Lower bounds
                             xu=bounds_np_array[:, 1])  # Upper bounds

        def _evaluate(self, X_batch, out, *args, **kwargs):  # X_batch from Pymoo
            n_sols = X_batch.shape[0]
            # Create a DataFrame matching rf_input_features structure for model prediction
            X_rf = pd.DataFrame(columns=rf_input_features, index=range(n_sols))

            # Populate continuous variables
            for cont_name in pymoo_opt_vars_continuous:
                if cont_name in X_rf.columns:
                    X_rf[cont_name] = X_batch[:, pymoo_opt_vars_all.index(cont_name)]

            # Populate OHE columns from categorical choices
            for cat_name, details in CAT_COL_DETAILS.items():
                if 'pymoo_index' not in details:  # Should have pymoo_index if it's an opt var
                    # Fill with default if this categorical var isn't optimized (e.g. only one category)
                    for ohe_n in details['one_hot_names']:
                        if ohe_n in X_rf.columns: X_rf[ohe_n] = defaults_for_rf.get(ohe_n, 0.0)
                    continue

                # Get integer choices from Pymoo, clip and round
                int_choices = np.clip(np.round(X_batch[:, details['pymoo_index']]).astype(int), 0,
                                      len(details['categories']) - 1)
                # Set all OHE columns for this category to 0 first
                for ohe_n in details['one_hot_names']:
                    if ohe_n in X_rf.columns: X_rf[ohe_n] = 0.0
                # Set the chosen OHE column to 1
                for sol_i, choice_i in enumerate(int_choices):
                    chosen_ohe = details['one_hot_names'][choice_i]
                    if chosen_ohe in X_rf.columns: X_rf.loc[sol_i, chosen_ohe] = 1.0

            # Fill any remaining columns (e.g., features not part of Pymoo vars) with defaults
            for col_fill in rf_input_features:
                if col_fill not in X_rf.columns or X_rf[col_fill].isnull().all():
                    X_rf[col_fill] = defaults_for_rf.get(col_fill, 0.0)
                elif X_rf[col_fill].isnull().any():  # If some NaNs remain after population
                    X_rf[col_fill].fillna(defaults_for_rf.get(col_fill, 0.0), inplace=True)

            X_rf_ordered = X_rf[rf_input_features]  # Ensure correct column order
            if global_problem_scaler is None:
                print("Err: global_problem_scaler is None in _evaluate.");
                out["F"] = np.full((n_sols, len(active_objective_names_for_pymoo)), np.nan);
                return
            try:
                X_scaled = global_problem_scaler.transform(X_rf_ordered)  # Scale features
            except Exception as e:
                print(
                    f"Err scaling in _evaluate: {e}. X_rf_ordered dtypes:\n{X_rf_ordered.dtypes}\nNulls:\n{X_rf_ordered.isnull().sum()}");
                out["F"] = np.full((n_sols, len(active_objective_names_for_pymoo)), np.nan);
                return

            # Predict objectives using the trained RF models
            out["F"] = np.column_stack([model.predict(X_scaled) for model in active_rf_models_for_pymoo])

    problem = SurrogateProblem()
    print("MOO problem defined.")

    # 4. Run NSGA-II
    print(f"\n4. Running NSGA-II for {len(active_objective_names_for_pymoo)} objectives...")
    if not active_objective_names_for_pymoo:  # Should not happen due to earlier check
        print("Skipping NSGA-II: No active objectives.");
        res = None
    else:
        algo = NSGA2(pop_size=100, eliminate_duplicates=True)  # NSGA-II algorithm
        res = minimize(problem, algo, termination=get_termination("n_gen", 100), seed=42, verbose=True)
        print("NSGA-II optimization finished.")

    pareto_F = res.F if res and hasattr(res, 'F') else None  # Objective values of Pareto front
    pareto_X = res.X if res and hasattr(res, 'X') else None  # Decision variable values of Pareto front
    if pareto_F is None or len(pareto_F) == 0: print(
        "Pareto front is empty. Exiting early from plotting/saving."); return

    # 5. Plotting and CSV Output for Optimization Results
    XY_radviz = radviz_projection(pareto_F) if pareto_F is not None else None  # Project for RadViz
    print("\n5a. Generating RadViz plots (Optimization)...")
    if XY_radviz is not None and XY_radviz.shape[0] > 0 and pareto_F is not None and pareto_F.shape[0] > 0:
        if XY_radviz.ndim < 2 or XY_radviz.shape[1] < 2:  # Check RadViz output validity
            print("Warn: RadViz projection resulted in insufficient dimensions. Skipping RadViz plots.")
        else:
            num_obj_plot = pareto_F.shape[1]
            if num_obj_plot < 2:  # RadViz needs at least 2 objectives
                print("Skip Radviz: Need >= 2 objectives.")
            else:
                # Define anchor points for RadViz polygon
                angles = 2 * np.pi * np.arange(num_obj_plot) / num_obj_plot
                anchors = np.stack([np.cos(angles), np.sin(angles)], axis=1)

                # Create one RadViz plot per objective, colored by that objective's values
                for i, name in enumerate(active_objective_names_for_pymoo):
                    fig, ax = plt.subplots(figsize=(9, 9))
                    sc = ax.scatter(XY_radviz[:, 0], XY_radviz[:, 1], c=pareto_F[:, i], cmap="viridis", s=60, ec="k",
                                    alpha=0.7, label="Pareto Designs")
                    # Draw the RadViz polygon (e.g., triangle for 3 obj, square for 4 obj)
                    poly_x = np.append(anchors[:, 0], anchors[0, 0]);  # Close the polygon
                    poly_y = np.append(anchors[:, 1], anchors[0, 1])
                    ax.plot(poly_x, poly_y, "--", c="r", lw=1.5, label="Objective Anchors")
                    ax.scatter(anchors[:, 0], anchors[:, 1], marker="^", c="r", s=120)  # Mark anchors
                    for il, lab in enumerate(active_objective_names_for_pymoo):  # Label anchors
                        ax.text(anchors[il, 0] * 1.08, anchors[il, 1] * 1.08, lab, c="m", ha="center", va="bottom", bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, boxstyle='round'),
                                fontsize=10)
                    ax.set_title(f"RadViz (Color by {name})");
                    ax.set_xlabel("RadViz X");
                    ax.set_ylabel("RadViz Y");
                    ax.axhline(0, c='k', lw=0.5, ls='--');  # Center lines
                    ax.axvline(0, c='k', lw=0.5, ls='--');
                    ax.set_aspect('equal');  # Equal aspect ratio
                    plt.colorbar(sc, ax=ax, label=name,location='bottom');  # Color bar for objective values
                    ax.legend(loc="upper right")
                    plt.tight_layout(pad=2.0)
                    sfname = sanitize_filename_component(name);  # Sanitize name for filename
                    plt.savefig(os.path.join(PLOTS_DIR, f"radviz_opt_{sfname}.png"));
                    plt.close(fig)
                    print(f"Saved RadViz opt plot: radviz_opt_{sfname}.png")
    else:
        print("Skipping RadViz (Opt) plots due to no data or insufficient dimensions.")

    # --- Combine All RadViz Plots into One Image ---
    combined_radviz_paths = []
    for name in active_objective_names_for_pymoo:
        sfname = sanitize_filename_component(name)
        img_path = os.path.join(PLOTS_DIR, f"radviz_opt_{sfname}.png")
        if os.path.exists(img_path):
            combined_radviz_paths.append(img_path)

    if len(combined_radviz_paths) == 4:  # Expecting 4 RadViz plots
        import matplotlib.image as mpimg

        fig_combined, axs_combined = plt.subplots(2, 2, figsize=(14, 12))
        axs_combined = axs_combined.flatten()

        for ax, img_path in zip(axs_combined, combined_radviz_paths):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')

        fig_combined.suptitle("Combined RadViz Optimization Plots", fontsize=18,fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        combined_path = os.path.join(PLOTS_DIR, "combined_radviz_plots.png")
        fig_combined.savefig(combined_path, dpi=300)
        plt.close(fig_combined)
        print(f"Saved combined RadViz plot to {combined_path}")
    else:
        print("Skipped combined RadViz image: 4 RadViz plots not found.")

    print("\n5b. Generating 2D scatter plots (Pareto Fronts)...")
    if pareto_F is not None and len(pareto_F) > 0 and pareto_F.shape[1] >= 2:
        # Create scatter plots for every pair of objectives
        for i1, i2 in itertools.combinations(range(pareto_F.shape[1]), 2):
            obj1_name_s, obj2_name_s = active_objective_names_for_pymoo[i1], active_objective_names_for_pymoo[i2]
            plot_title_s = f"{obj1_name_s} vs {obj2_name_s}"

            fig_scatter_s, ax_scatter_s = plt.subplots(figsize=(9, 8))
            ax_scatter_s.scatter(pareto_F[:, i1], pareto_F[:, i2], color='tab:blue',
                                 s=60, edgecolor='black', label='Pareto Front Points', zorder=5)

            # Attempt to fit and plot an exponential curve to the Pareto front
            if len(pareto_F) > 1:  # Need at least 2 points for a line, more for a curve
                df_for_interp = pd.DataFrame({'x': pareto_F[:, i1], 'y': pareto_F[:, i2]})
                # Sort and remove duplicates for curve fitting (monotonic x)
                unique_pf_points = df_for_interp.sort_values(by=['x', 'y']).drop_duplicates(subset=['x'], keep='first')
                interp_x_final = unique_pf_points['x'].values
                interp_y_final = unique_pf_points['y'].values

                if len(interp_x_final) >= 3:  # Need at least 3 points for robust curve_fit
                    try:
                        # Initial guess and bounds for exponential fit
                        initial_guess_exp1 = [np.max(interp_y_final), -0.1, np.min(interp_y_final)]
                        bounds_exp1 = ([0, -np.inf, -np.inf], [np.inf, 0, np.inf])  # a>0, b<0
                        params_exp1, _ = curve_fit(exp_func_type1, interp_x_final, interp_y_final,
                                                   p0=initial_guess_exp1, bounds=bounds_exp1, maxfev=5000)
                        x_smooth_exp1 = np.linspace(interp_x_final.min(), interp_x_final.max(), 300)
                        y_smooth_exp1 = exp_func_type1(x_smooth_exp1, *params_exp1)
                        ax_scatter_s.plot(x_smooth_exp1, y_smooth_exp1, linestyle='--', color='red',
                                          lw=2, label='Pareto Curve (Exp: ae^(bx)+c)', zorder=3)
                    except (RuntimeError, ValueError) as e_exp_fit:  # Fallback if curve fit fails
                        print(
                            f"Exponential fit failed for plot '{plot_title_s}': {e_exp_fit}. Plotting linear fallback.")
                        if len(interp_x_final) >= 2:  # Plot simple line if at least 2 points
                            ax_scatter_s.plot(interp_x_final, interp_y_final, '-', color='darkcyan', lw=1.5,
                                              label='Pareto Front (Line Fallback)', zorder=1)
                elif len(interp_x_final) >= 2:  # If only 2 unique x points, plot a line
                    ax_scatter_s.plot(interp_x_final, interp_y_final, '-', color='darkcyan', lw=1.5,
                                      label='Pareto Front (Line)', zorder=1)

            ax_scatter_s.set_xlabel(f"{obj1_name_s}");
            ax_scatter_s.set_ylabel(f"{obj2_name_s}")
            ax_scatter_s.set_title(plot_title_s,fontweight='bold');
            ax_scatter_s.legend(fontsize='small', loc='best')
            ax_scatter_s.grid(True, linestyle='--', alpha=0.5);
            plt.tight_layout(pad=2.0)

            stitle = sanitize_filename_component(plot_title_s).replace('vs', '_VS_');  # Sanitize for filename
            plt.savefig(os.path.join(PLOTS_DIR, f"scatter_opt_{stitle}.png"));
            plt.close(fig_scatter_s)
            print(f"Saved Scatter opt plot: scatter_opt_{stitle}.png")

    # --- Combine All Scatter Optimization Plots into One Image ---
    scatter_opt_paths = []
    for fname in os.listdir(PLOTS_DIR):
        if fname.startswith("scatter_opt_") and fname.endswith(".png"):
            scatter_opt_paths.append(os.path.join(PLOTS_DIR, fname))

    scatter_opt_paths.sort()  # Ensure consistent order
    selected_paths = scatter_opt_paths[:6]  # Up to 6 plots

    if selected_paths:
        import matplotlib.image as mpimg

        fig_scatteropt_combined, axs_combined = plt.subplots(2, 3, figsize=(20, 12))
        axs_combined = axs_combined.flatten()

        for ax, img_path in zip(axs_combined, selected_paths):
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.axis('off')

        for j in range(len(selected_paths), 6):
            axs_combined[j].axis('off')

        fig_scatteropt_combined.suptitle("Combined Scatter Optimization Plots", fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        combined_so_path = os.path.join(PLOTS_DIR, "combined_scatter_opt_plots.png")
        fig_scatteropt_combined.savefig(combined_so_path, dpi=300)
        plt.close(fig_scatteropt_combined)
        print(f"Saved combined Scatter Optimization plot to {combined_so_path}")

    else:
        print("Skipping 2D Scatter (Opt) plots due to no data or insufficient objectives.")

    print("\n5c. Mapping Pareto Solutions & Saving to CSV (Optimization)...")
    if pareto_F is not None and pareto_X is not None and len(pareto_F) == len(pareto_X):
        # Create DataFrame for decision variables (X)
        X_df_out = pd.DataFrame(pareto_X, columns=pymoo_opt_vars_all).copy()
        # Convert categorical integer choices back to original string values
        for cat_name, details in CAT_COL_DETAILS.items():
            if cat_name in X_df_out.columns:  # If this categorical var was optimized
                int_choices = np.clip(np.round(X_df_out[cat_name].values).astype(int), 0,
                                      len(details['categories']) - 1)
                X_df_out[cat_name] = [details['categories'][c] for c in int_choices]

        # Create DataFrames for objective values (F) and RadViz coordinates
        obj_cols = [f"Obj_{sanitize_filename_component(n)}" for n in active_objective_names_for_pymoo]
        F_df = pd.DataFrame(pareto_F, columns=obj_cols)
        radviz_df = pd.DataFrame(XY_radviz, columns=["RadViz_X", "RadViz_Y"]) if XY_radviz is not None and len(
            XY_radviz) == len(F_df) else pd.DataFrame(index=F_df.index)  # Empty if RadViz failed

        # Combine all results into one DataFrame
        final_df_list = []
        if not radviz_df.empty: final_df_list.append(radviz_df)
        final_df_list.append(F_df)
        final_df_list.append(X_df_out)

        results_df = pd.concat(final_df_list, axis=1)
        csv_p = os.path.join(PLOTS_DIR, "pareto_solutions_designs_opt.csv")
        results_df.to_csv(csv_p, index_label="Sol_Index");  # Save to CSV
        print(f"Saved Pareto solutions and designs to {csv_p}")
    else:
        print("Skipping CSV output (Opt) due to missing Pareto data.")

    # 6. Generate Combined Diagnostic Plots (at the end)
    # Plot distributions of all original target variables
    num_all_targets = len(all_target_series_for_global_plot)
    if num_all_targets > 0:
        fig_all_dist, axs_all_dist = plt.subplots(num_all_targets, 2, figsize=(14, 5 * num_all_targets), squeeze=False)
        fig_all_dist.suptitle("All Target Variable Distributions (Original Data)", fontsize=16, y=0.99)
        for i, (series, title) in enumerate(all_target_series_for_global_plot):
            plot_target_distribution(series, title, axs_all_dist[i, 0], axs_all_dist[i, 1])
        try:
            fig_all_dist.tight_layout(rect=[0, 0, 1, 0.98],pad=2.0)
        except UserWarning as e:
            print(f"Warn tight_layout (all_dist): {e}")
        dist_path = os.path.join(PLOTS_DIR, "all_targets_distributions.png")
        fig_all_dist.savefig(dist_path);
        print(f"\nSaved combined target distributions plot to {dist_path}");
        plt.close(fig_all_dist)

    # Plot combined feature importances for all targets
    num_fi_plots = len(all_feature_importance_data)
    if num_fi_plots > 0:
        fig_fi, axs_fi = plt.subplots(num_fi_plots, 1, figsize=(12, 7 * num_fi_plots), squeeze=False)
        fig_fi.suptitle("Combined Feature Importances (RF vs XGB) per Target", fontsize=16, y=0.99,fontweight='bold')
        for i, (rf_model, xgb_model, f_names, suffix) in enumerate(all_feature_importance_data):
            plot_combined_top_n_feature_importances(rf_model, xgb_model, "RandomForest", "XGBoost", f_names,
                                                    TOP_N_FEATURES, suffix, axs_fi[i, 0])
        try:
            fig_fi.tight_layout(rect=[0, 0, 1, 0.98],pad=2.0)
        except UserWarning as e:
            print(f"Warn tight_layout (all_fi): {e}")
        fi_path = os.path.join(PLOTS_DIR, "all_targets_feature_importances.png")
        fig_fi.savefig(fi_path);
        print(f"Saved combined feature importances plot to {fi_path}");
        plt.close(fig_fi)

    print("\n--- Script finished. ---")


if __name__ == "__main__":
    # Ensure the script can find the Excel files if they are not in the same directory
    # For example, by placing them in the same directory as the script,
    # or by providing full paths in FILE_CONFIGS.
    # Example: FILE_CONFIGS = [{'name': '/path/to/your/2.5D_SJR.xlsx'}, ...]

    # For this script to run, you need to have the following files in the same directory
    # (or provide correct paths):
    # - 2.5D_SJR.xlsx
    # - 2.5D_Thermal_lidless.xlsx

    run_combined_workflow()
