import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import json
import os
from typing import Literal, Tuple, Dict, Any

def load_settings():
    """Central configuration for all module parameters, columns, and thresholds."""
    return {
        # ----------------------------------------
        # 1. Column Names & Labels
        # ----------------------------------------
        'col_turbidity': 'turbidity',
        'col_visual_clarity': 'visual_clarity_m',
        'col_black_disc': 'black_disc_m',
        'col_tube': 'clarity_tube_m',
        'col_quality': 'quality_code',
        'col_date': 'Date_Time',
        'censor_suffix': '_censor',
        'value_suffix': '_value',
        
        # ----------------------------------------
        # 2. Data Processing Rules
        # ----------------------------------------
        'shmak_to_black_disc_max': 0.5,
        'censor_lower_multiplier': 0.5,
        'censor_upper_multiplier': 1.1,
        'min_sample_size': 10,
        
        # ----------------------------------------
        # 3. Model Parameters
        # ----------------------------------------
        'default_equation_type': 'power', # Options: 'power', 'exponential'
        'quantreg_lower_q': 0.05,
        'quantreg_upper_q': 0.95,
        'ols_alpha': 0.10,
        
        # ----------------------------------------
        # 4. Evaluation Classification Thresholds
        # ----------------------------------------
        # Ordered strictly as: (Very Good, Good, Satisfactory)
        'r2_thresholds': (0.70, 0.60, 0.30),
        'nse_thresholds': (0.65, 0.50, 0.35),
        'pbias_thresholds': (15.0, 20.0, 30.0),
        'pseudo_r1_thresholds': (0.60, 0.45, 0.30),
        'calibration_thresholds': (2.5, 5.0, 10.0), # Max absolute deviation from 50%
        
        'rating_map': {4: "Very Good", 3: "Good", 2: "Satisfactory", 1: "Unsatisfactory"},
        'min_approval_score': 3, # Minimum overall score (1-4) required to approve imputation
        
        # ----------------------------------------
        # 5. Imputation & I/O
        # ----------------------------------------
        'imputed_quality_code': 300,
        'model_output_dir': 'models',
        'report_output_dir': 'reports'
    }

settings = load_settings()


# ==========================================
# MODULE 1: DATA INGESTION & PREPROCESSING
# ==========================================

def load_data(file_path: str) -> pd.DataFrame:
    """Loads water quality data from CSV or Excel files."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
        
def extract_censoring_flags(
    df: pd.DataFrame, 
    input_col: str, 
    value_col: str = None, 
    censor_col: str = None
) -> pd.DataFrame:
    """
    Parses a mixed-format column (e.g., '<0.1', '2.5') into a clean numeric 
    value column and a separate censoring flag column.
    """
    result_df = df.copy()
    
    if value_col is None:
        value_col = f"{input_col}{settings['value_suffix']}"
    if censor_col is None:
        censor_col = f"{input_col}{settings['censor_suffix']}"
        
    str_series = result_df[input_col].astype(str).str.strip()
    result_df[censor_col] = str_series.str.extract(r'(^[<>])').fillna("")
    numeric_str = str_series.str.replace(r'[<>\s]', '', regex=True)
    result_df[value_col] = pd.to_numeric(numeric_str, errors='coerce')
    
    return result_df
        
def filter_by_quality_code(
    df: pd.DataFrame,
    quality_col: str,
    good_codes: list
) -> pd.DataFrame:
    """Removes data points that do not meet the acceptable quality standards."""
    original_len = len(df)
    filtered_df = df[df[quality_col].isin(good_codes)].copy()
    removed_count = original_len - len(filtered_df)
    print(f"Removed {removed_count} records with unacceptable quality codes.")
    
    return filtered_df.reset_index(drop=True)

def reshape_long_to_wide(
    df: pd.DataFrame,
    date_col: str,
    variable_col: str,
    value_col: str,
    censor_col: str = None,
    name_mapping: dict = None
) -> pd.DataFrame:
    """
    Converts long-form environmental data to wide format, where each 
    variable becomes its own column. Preserves date/time for timeseries plotting.
    """
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    
    if name_mapping:
        df_copy[variable_col] = df_copy[variable_col].replace(name_mapping)
        
    val_cols = [value_col]
    if censor_col:
        val_cols.append(censor_col)
        
    wide_df = df_copy.pivot_table(
        index=date_col, 
        columns=variable_col, 
        values=val_cols,
        aggfunc='first'
    )
    
    if isinstance(wide_df.columns, pd.MultiIndex):
        new_cols = []
        for val_type, var_name in wide_df.columns:
            if val_type == value_col:
                new_cols.append(var_name)
            else:
                new_cols.append(f"{var_name}_{censor_col}")
        wide_df.columns = new_cols
        
    return wide_df.reset_index()

def merge_clarity_measurements(
    df: pd.DataFrame, 
    black_disc_col: str = settings['col_black_disc'], 
    tube_col: str = settings['col_tube'],
    merged_col: str = settings['col_visual_clarity']
) -> pd.DataFrame:
    """Merges clarity tube observations with black disc observations based on thresholds."""
    data = df.copy()
    data[merged_col] = data[black_disc_col]
    
    valid_tube_mask = (data[tube_col] <= settings['shmak_to_black_disc_max']) & (data[black_disc_col].isna())
    data.loc[valid_tube_mask, merged_col] = data.loc[valid_tube_mask, tube_col]
    
    return data

def process_censored_data(
    df: pd.DataFrame,
    value_col: str,
    censor_col: str,
    method: Literal["ignore", "rule_based"] = "rule_based",
    a_lower: float = settings['censor_lower_multiplier'],
    b_upper: float = settings['censor_upper_multiplier']
) -> pd.DataFrame:
    """Handles censored values prior to model fitting."""
    data = df.copy()
    data[censor_col] = data[censor_col].fillna("").astype(str).str.strip()
    
    if method == "ignore":
        return data[~data[censor_col].isin(["<", ">"])].reset_index(drop=True)
        
    elif method == "rule_based":
        lower_mask = data[censor_col] == "<"
        data.loc[lower_mask, value_col] = data.loc[lower_mask, value_col].astype(float) * a_lower
        
        upper_mask = data[censor_col] == ">"
        data.loc[upper_mask, value_col] = data.loc[upper_mask, value_col].astype(float) * b_upper
        
        return data
    else:
        raise ValueError(f"Unsupported censoring method: {method}")

def validate_sample_size(
    df: pd.DataFrame, 
    min_samples: int = settings['min_sample_size'],
    x_col: str = settings['col_turbidity'], 
    y_col: str = settings['col_visual_clarity']
) -> bool:
    """Validates if a site has enough paired uncensored observations for fitting."""
    valid_pairs = df[[x_col, y_col]].dropna()
    return len(valid_pairs) >= min_samples


# ==========================================
# MODULE 2: MODEL FITTING ENGINE 
# ==========================================
def fit_quantile_regression(
    df: pd.DataFrame, 
    x_col: str = settings['col_turbidity'], 
    y_col: str = settings['col_visual_clarity'],
    equation_type: str = settings['default_equation_type'],
    lower_q: float = settings['quantreg_lower_q'],
    upper_q: float = settings['quantreg_upper_q']
) -> Dict[str, Any]:
    """Fits a relationship using Quantile Regression and includes parameter significance."""
    fit_data = df[[x_col, y_col]].dropna().copy()
    fit_data['ln_y'] = np.log(fit_data[y_col])
    
    if equation_type == "power":
        fit_data['model_x'] = np.log(fit_data[x_col])
    elif equation_type == "exponential":
        fit_data['model_x'] = fit_data[x_col]
    else:
        raise ValueError("equation_type must be 'power' or 'exponential'")
    
    model = smf.quantreg('ln_y ~ model_x', fit_data)
    
    res_50 = model.fit(q=0.50)
    res_lower = model.fit(q=lower_q)
    res_upper = model.fit(q=upper_q)
    
    pi_percentage = int(round((upper_q - lower_q) * 100))
    
    return {
        'method': 'quantile_regression',
        'equation_type': equation_type,
        'n_samples': len(fit_data),
        'pi_percentage': pi_percentage,
        'median': {
            'intercept': res_50.params['Intercept'],
            'intercept_pvalue': res_50.pvalues['Intercept'],
            'slope': res_50.params['model_x'],
            'slope_pvalue': res_50.pvalues['model_x']
        },
        'lower_PI': {
            'quantile': lower_q, 
            'intercept': res_lower.params['Intercept'],
            'intercept_pvalue': res_lower.pvalues['Intercept'],
            'slope': res_lower.params['model_x'],
            'slope_pvalue': res_lower.pvalues['model_x']
        },
        'upper_PI': {
            'quantile': upper_q, 
            'intercept': res_upper.params['Intercept'],
            'intercept_pvalue': res_upper.pvalues['Intercept'],
            'slope': res_upper.params['model_x'],
            'slope_pvalue': res_upper.pvalues['model_x']
        }
    }

def fit_ols_with_smearing(
    df: pd.DataFrame, 
    x_col: str = settings['col_turbidity'], 
    y_col: str = settings['col_visual_clarity'],
    equation_type: str = settings['default_equation_type'],
    alpha: float = settings['ols_alpha']
) -> Dict[str, Any]:
    """Fits a relationship using OLS with Duan's Smearing Estimator and includes parameter significance."""
    fit_data = df[[x_col, y_col]].dropna().copy()
    fit_data['ln_y'] = np.log(fit_data[y_col])
    
    if equation_type == "power":
        fit_data['model_x'] = np.log(fit_data[x_col])
    elif equation_type == "exponential":
        fit_data['model_x'] = fit_data[x_col]
    else:
        raise ValueError("equation_type must be 'power' or 'exponential'")
    
    X = sm.add_constant(fit_data['model_x'])
    model = sm.OLS(fit_data['ln_y'], X)
    results = model.fit()
    
    residuals = results.resid
    cf = np.mean(np.exp(residuals))
    
    n_samples = len(fit_data)
    mean_model_x = fit_data['model_x'].mean()
    ss_xx = np.sum((fit_data['model_x'] - mean_model_x)**2)
    mse = results.mse_resid 
    
    t_crit = stats.t.ppf(1 - alpha/2, df=results.df_resid)
    
    return {
        'method': 'ols_smearing',
        'equation_type': equation_type,
        'n_samples': n_samples,
        'alpha': alpha,
        'pi_percentage': int((1 - alpha) * 100),
        'mean_fit': {
            'intercept': results.params['const'],
            'intercept_pvalue': results.pvalues['const'],
            'slope': results.params['model_x'],
            'slope_pvalue': results.pvalues['model_x'],
            'model_f_pvalue': results.f_pvalue # Overall significance of the OLS model
        },
        'smearing_factor_cf': cf,
        'r_squared': results.rsquared,
        'pi_components': {
            'mse': mse,
            'mean_model_x': mean_model_x,
            'ss_xx': ss_xx,
            't_crit': t_crit
        }
    }


def predict_clarity(
    turbidity: pd.Series, 
    model_params: Dict[str, Any],
    return_pi: bool = True
) -> pd.DataFrame:
    """Predicts visual clarity from turbidity using fitted model parameters."""
    equation_type = model_params.get('equation_type', settings['default_equation_type'])
    
    if equation_type == "power":
        model_x = np.log(turbidity)
    elif equation_type == "exponential":
        model_x = turbidity
    else:
        raise ValueError("Unknown equation type in model parameters.")
        
    result_df = pd.DataFrame(index=turbidity.index)
    
    if model_params['method'] == 'quantile_regression':
        median_params = model_params['median']
        ln_y_pred = median_params['intercept'] + median_params['slope'] * model_x
        result_df['predicted_clarity'] = np.exp(ln_y_pred)
        
        if return_pi:
            lower_params = model_params['lower_PI']
            upper_params = model_params['upper_PI']
            
            ln_y_lower = lower_params['intercept'] + lower_params['slope'] * model_x
            ln_y_upper = upper_params['intercept'] + upper_params['slope'] * model_x
            
            result_df['lower_pi'] = np.exp(ln_y_lower)
            result_df['upper_pi'] = np.exp(ln_y_upper)
            
    elif model_params['method'] == 'ols_smearing':
        params = model_params['mean_fit']
        cf = model_params['smearing_factor_cf']
        
        ln_y_pred = params['intercept'] + params['slope'] * model_x
        result_df['predicted_clarity'] = cf * np.exp(ln_y_pred)
        
        if return_pi:
            pi_comp = model_params['pi_components']
            n = model_params['n_samples']
            
            leverage = (model_x - pi_comp['mean_model_x'])**2 / pi_comp['ss_xx']
            se_pred = np.sqrt(pi_comp['mse'] * (1 + (1/n) + leverage))
            
            moe = pi_comp['t_crit'] * se_pred
            
            result_df['lower_pi'] = np.exp(ln_y_pred - moe)
            result_df['upper_pi'] = np.exp(ln_y_pred + moe)
            
    else:
        raise ValueError("Unknown model fitting method.")
        
    return result_df


# ==========================================
# MODULE 3: PERFORMANCE EVALUATION & SCREENING
# ==========================================
def _align_and_clean_data(obs: pd.Series, pred: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Aligns observations and predictions, dropping any NaNs."""
    df = pd.DataFrame({'obs': obs, 'pred': pred}).dropna()
    if len(df) == 0:
        raise ValueError("No valid paired observation/prediction data available for evaluation.")
    return df['obs'], df['pred']

def calculate_r2(obs: pd.Series, pred: pd.Series) -> float:
    mean_obs = np.mean(obs)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - mean_obs) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

def calculate_nse(obs: pd.Series, pred: pd.Series) -> float:
    mean_obs = np.mean(obs)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - mean_obs) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

def calculate_pbias(obs: pd.Series, pred: pd.Series) -> float:
    sum_obs = np.sum(obs)
    return (np.sum(obs - pred) / sum_obs) * 100 if sum_obs != 0 else 0.0


def calculate_pseudo_r1(obs: pd.Series, pred: pd.Series) -> float:
    obs_median = np.median(obs)
    sad_model = np.sum(np.abs(obs - pred))
    sad_null = np.sum(np.abs(obs - obs_median))
    return 1 - (sad_model / sad_null) if sad_null != 0 else 0.0

def calculate_rmse(obs: pd.Series, pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((obs - pred) ** 2)))

def calculate_mae(obs: pd.Series, pred: pd.Series) -> float:
    return float(np.mean(np.abs(obs - pred)))

def calculate_median_calibration(obs: pd.Series, pred: pd.Series) -> Dict[str, float]:
    n = len(obs)
    if n == 0:
        return {'percent_above': 0.0, 'percent_below': 0.0, 'percent_exact': 0.0}
    
    return {
        'percent_above': float(np.sum(obs > pred) / n * 100),
        'percent_below': float(np.sum(obs < pred) / n * 100),
        'percent_exact': float(np.sum(obs == pred) / n * 100)
    }

# ==========================================
#  Classification Logic
# ==========================================

def classify_r2(r2: float) -> Tuple[int, str]:
    t1, t2, t3 = settings['r2_thresholds']
    if r2 >= t1: return 4, settings['rating_map'][4]
    if r2 > t2: return 3, settings['rating_map'][3]
    if r2 > t3: return 2, settings['rating_map'][2]
    return 1, settings['rating_map'][1]

def classify_nse(nse: float) -> Tuple[int, str]:
    t1, t2, t3 = settings['nse_thresholds']
    if nse > t1: return 4, settings['rating_map'][4]
    if nse > t2: return 3, settings['rating_map'][3]
    if nse > t3: return 2, settings['rating_map'][2]
    return 1, settings['rating_map'][1]

def classify_pbias(pbias: float) -> Tuple[int, str]:
    abs_pbias = abs(pbias)
    t1, t2, t3 = settings['pbias_thresholds']
    if abs_pbias < t1: return 4, settings['rating_map'][4]
    if abs_pbias < t2: return 3, settings['rating_map'][3]
    if abs_pbias < t3: return 2, settings['rating_map'][2]
    return 1, settings['rating_map'][1]

def classify_pseudo_r1(r1: float) -> Tuple[int, str]:
    t1, t2, t3 = settings['pseudo_r1_thresholds']
    if r1 >= t1: return 4, settings['rating_map'][4]
    if r1 >= t2: return 3, settings['rating_map'][3]
    if r1 >= t3: return 2, settings['rating_map'][2]
    return 1, settings['rating_map'][1]

def classify_median_calibration(percent_above: float, percent_below: float) -> Tuple[int, str]:
    deviation = max(abs(percent_above - 50.0), abs(percent_below - 50.0))
    t1, t2, t3 = settings['calibration_thresholds']
    if deviation <= t1: return 4, settings['rating_map'][4]
    if deviation <= t2: return 3, settings['rating_map'][3]
    if deviation <= t3: return 2, settings['rating_map'][2]
    return 1, settings['rating_map'][1]

def get_overall_rating(scores: list[int]) -> str:
    """Returns the overall rating based on the worst-of-three rule."""
    min_score = min(scores)
    return settings['rating_map'][min_score]

# ==========================================
# 4. Main Orchestrator
# ==========================================

def evaluate_model_performance(obs: pd.Series, pred: pd.Series) -> Dict[str, Any]:
    # 1. Prepare data
    y_obs, y_pred = _align_and_clean_data(obs, pred)
    
    # 2. Calculate Mean-centric Metrics
    r2 = calculate_r2(y_obs, y_pred)
    nse = calculate_nse(y_obs, y_pred)
    pbias = calculate_pbias(y_obs, y_pred)
    
    # 3. Calculate Median-centric & Absolute Metrics
    r1 = calculate_pseudo_r1(y_obs, y_pred)
    rmse = calculate_rmse(y_obs, y_pred)
    mae = calculate_mae(y_obs, y_pred)
    calibration = calculate_median_calibration(y_obs, y_pred)
    
    # 4. Classify Metrics
    r2_score, r2_rating = classify_r2(r2)
    nse_score, nse_rating = classify_nse(nse)
    pbias_score, pbias_rating = classify_pbias(pbias)
    
    r1_score, r1_rating = classify_pseudo_r1(r1)
    calib_score, calib_rating = classify_median_calibration(
        calibration['percent_above'], 
        calibration['percent_below']
    )
    
    # 5. Determine Overall Rating
    all_scores = [r2_score, nse_score, pbias_score, r1_score, calib_score]
    min_score = min(all_scores)
    
    return {
        'metrics': {
            'R2': float(r2),
            'NSE': float(nse),
            'PBIAS': float(pbias),
            'Pseudo_R1': float(r1),
            'Calibration_Above': calibration['percent_above'],
            'Calibration_Below': calibration['percent_below'],
            'Calibration_Exact': calibration['percent_exact'],
            'RMSE': rmse,
            'MAE': mae
        },
        'ratings': {
            'R2_rating': r2_rating,
            'NSE_rating': nse_rating,
            'PBIAS_rating': pbias_rating,
            'Pseudo_R1_rating': r1_rating,
            'Calibration_rating': calib_rating,
            'Overall_rating': settings['rating_map'][min_score]
        },
        'approved_for_imputation': min_score >= settings['min_approval_score']
    }

# ==========================================
# MODULE 4: IMPUTATION & DATA TAGGING
# ==========================================

def impute_missing_clarity(
    df: pd.DataFrame, 
    predictions: pd.DataFrame, 
    evaluation_results: dict,
    clarity_col: str = settings['col_visual_clarity'],
    quality_col: str = settings['col_quality']
) -> pd.DataFrame:
    """
    Imputes missing visual clarity records using model predictions,
    provided the model meets the performance acceptance criteria.
    """
    result_df = df.copy()
    
    if not evaluation_results.get('approved_for_imputation', False):
        print("Model did not meet minimum performance criteria. Skipping imputation.")
        return result_df
        
    missing_mask = result_df[clarity_col].isna() & predictions['predicted_clarity'].notna()
    result_df.loc[missing_mask, clarity_col] = predictions.loc[missing_mask, 'predicted_clarity']
    
    if quality_col not in result_df.columns:
        result_df[quality_col] = np.nan
        
    # Tag imputed records using the central setting
    result_df.loc[missing_mask, quality_col] = settings['imputed_quality_code']
    
    return result_df


# ==========================================
# MODULE 5: MODEL SERIALIZATION & REUSE
# ==========================================

def save_site_model(
    site_id: str,
    model_params: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    output_dir: str = settings['model_output_dir']
) -> str:
    """
    Saves fitted model parameters and metadata to a structured JSON file 
    for reproducible predictions on future datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{site_id}_clarity_model.json")
    
    artifact = {
        'site_id': site_id,
        'model_parameters': model_params,
        'evaluation_metrics': evaluation_results
    }
    
    with open(file_path, 'w') as f:
        json.dump(artifact, f, indent=4)
        
    return file_path

def load_site_model(file_path: str) -> Dict[str, Any]:
    """Reloads saved parameter bundles to transform new data streams."""
    with open(file_path, 'r') as f:
        return json.load(f)
    
    
# ==========================================
# MODULE 6: INTERACTIVE DASHBOARDS (UPDATED)
# ==========================================

def generate_site_report_plot(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    site_id: str,
    date_col: str = settings['col_date'],
    clarity_col: str = settings['col_visual_clarity'],
    turbidity_col: str = settings['col_turbidity'],
    quality_col: str = settings['col_quality'],
    save_html: bool = True,
    output_dir: str = settings['report_output_dir']
) -> go.Figure:
    """
    Renders an interactive Plotly dashboard with two subplots:
    1. A log-log regression scatter plot with prediction bands.
    2. A time-series plot of observed clarity and imputed clarity with PI error bars.
    """
    plot_df = pd.concat([df, predictions], axis=1)
    
    scatter_df = plot_df.sort_values(by=turbidity_col) 
    ts_df = plot_df.sort_values(by=date_col).dropna(subset=[date_col]) 

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Turbidity vs Visual Clarity Regression: Site {site_id}',
            f'Visual Clarity Time-Series & Imputations: Site {site_id}'
        ),
        vertical_spacing=0.12
    )
    
    # ---------------------------------------------------------
    # ROW 1: REGRESSION SCATTER PLOT
    # ---------------------------------------------------------
    imputation_code = settings['imputed_quality_code']
    obs_mask_sc = scatter_df.get(quality_col, pd.Series([0]*len(scatter_df))) != imputation_code
    imputed_mask_sc = scatter_df.get(quality_col, pd.Series([0]*len(scatter_df))) == imputation_code
    
    fig.add_trace(go.Scatter(
        x=scatter_df.loc[obs_mask_sc, turbidity_col], 
        y=scatter_df.loc[obs_mask_sc, clarity_col],
        mode='markers', name='Field Observations',
        marker=dict(color='blue', opacity=0.6),
        legendgroup='regression'
    ), row=1, col=1)
    
    if imputed_mask_sc.any():
        fig.add_trace(go.Scatter(
            x=scatter_df.loc[imputed_mask_sc, turbidity_col], 
            y=scatter_df.loc[imputed_mask_sc, clarity_col],
            mode='markers', name=f'Imputed Data (Code {imputation_code})',
            marker=dict(color='red', symbol='cross', size=8),
            legendgroup='regression'
        ), row=1, col=1)
        
    line_df = scatter_df.dropna(subset=['predicted_clarity'])
    if not line_df.empty:
        if 'lower_pi' in line_df.columns and 'upper_pi' in line_df.columns:
            fig.add_trace(go.Scatter(
                x=pd.concat([line_df[turbidity_col], line_df[turbidity_col][::-1]]),
                y=pd.concat([line_df['upper_pi'], line_df['lower_pi'][::-1]]),
                fill='toself', fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='90% Prediction Interval', showlegend=True,
                legendgroup='regression'
            ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=line_df[turbidity_col], y=line_df['predicted_clarity'],
            mode='lines', name='Fitted Regression Curve',
            line=dict(color='black', dash='dash'),
            legendgroup='regression'
        ), row=1, col=1)

    # ---------------------------------------------------------
    # ROW 2: TIME-SERIES PLOT WITH ERROR BARS
    # ---------------------------------------------------------
    obs_mask_ts = ts_df.get(quality_col, pd.Series([0]*len(ts_df))) != imputation_code
    imputed_mask_ts = ts_df.get(quality_col, pd.Series([0]*len(ts_df))) == imputation_code
    
    fig.add_trace(go.Scatter(
        x=ts_df.loc[obs_mask_ts, date_col], 
        y=ts_df.loc[obs_mask_ts, clarity_col],
        mode='lines+markers', name='Observed Clarity',
        line=dict(color='blue', width=1, dash='dot'),
        marker=dict(size=6, opacity=0.6),
        legendgroup='timeseries'
    ), row=2, col=1)

    imputed_ts = ts_df.loc[imputed_mask_ts]
    if not imputed_ts.empty:
        error_plus = imputed_ts['upper_pi'] - imputed_ts['predicted_clarity']
        error_minus = imputed_ts['predicted_clarity'] - imputed_ts['lower_pi']
        
        fig.add_trace(go.Scatter(
            x=imputed_ts[date_col], 
            y=imputed_ts['predicted_clarity'],
            mode='markers', name='Imputed Clarity (±90% PI)',
            marker=dict(color='red', symbol='cross', size=8),
            error_y=dict(
                type='data', symmetric=False,
                array=error_plus, arrayminus=error_minus,
                color='rgba(255, 0, 0, 0.4)', thickness=1.5, width=4
            ),
            legendgroup='timeseries'
        ), row=2, col=1)

    # ---------------------------------------------------------
    # LAYOUT FORMATTING
    # ---------------------------------------------------------
    fig.update_layout(
        height=900, template='plotly_white',
        title_text=f'Visual Clarity Imputation Report: Site {site_id}'
    )
    
    fig.update_xaxes(title_text='Turbidity', type='log', row=1, col=1)
    fig.update_yaxes(title_text='Visual Clarity (m)', type='log', row=1, col=1)
    
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Visual Clarity (m)', type='log', row=2, col=1)
    
    if save_html:
        os.makedirs(output_dir, exist_ok=True)
        html_path = os.path.join(output_dir, f"{site_id}_report.html")
        fig.write_html(html_path)
        
    return fig