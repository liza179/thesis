import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.linear_model import BayesianRidge, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from bayesian_quantile_transformer import BayesianQuantileTransformer
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from rich.console import Console
from rich.table import Table

def reconstruct_model_data(input_csv='merged_data.csv', start_date='2016-01-31', hp_lambda=14400, horizon=6):
    df = pd.read_csv(input_csv, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    df['cpi_yoy'] = np.log(1 + df['cpi_yoy'] / 100)
    df['rem_12'] = np.log(1 + df['rem_12'] / 100)

    log_output_gap = np.log(df['output_gap'].dropna())
    cycle, trend = hpfilter(log_output_gap, lamb=hp_lambda)
    
    df['gap_trend'] = np.nan
    df.loc[log_output_gap.index, 'gap_trend'] = trend.values

    df_filtered = df[df['date'] >= start_date].copy().reset_index(drop=True)
    df_filtered['target'] = df_filtered['cpi_yoy'].shift(-horizon) - df_filtered['cpi_yoy']
    
    df_filtered['cpi_log_return_1_months'] = df_filtered['cpi_yoy'] - df_filtered['cpi_yoy'].shift(1)
    df_filtered['cpi_log_return_6_months'] = df_filtered['cpi_yoy'] - df_filtered['cpi_yoy'].shift(6)
    df_filtered['cpi_log_return_12_months'] = df_filtered['cpi_yoy'] - df_filtered['cpi_yoy'].shift(12)
    
    df_filtered['rem_1_months'] = df_filtered['rem_12'].shift(1) - df_filtered['cpi_yoy'].shift(1)
    df_filtered['rem_6_months'] = df_filtered['rem_12'].shift(6) - df_filtered['cpi_yoy'].shift(6)
    df_filtered['rem_12_months'] = df_filtered['rem_12'].shift(12) - df_filtered['cpi_yoy'].shift(12)

    df_filtered = df_filtered.dropna(subset=['target']).reset_index(drop=True)

    feature_cols = ['cpi_log_return_1_months', 'cpi_log_return_6_months', 'cpi_log_return_12_months', 'rem_1_months', 'rem_6_months', 'rem_12_months']
    for col in feature_cols:
        if col in df_filtered.columns:
            df_filtered[col] = df_filtered[col].ffill()
            df_filtered[col] = df_filtered[col].bfill()

    final_cols = ['date', 'cpi_yoy', 'target', 'cpi_log_return_1_months', 'cpi_log_return_6_months', 'cpi_log_return_12_months', 'rem_1_months', 'rem_6_months', 'rem_12_months']

    target = df_filtered['target']
    print(target.head())
    print(target.tail())
    for q in [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
        print(f"Q{q}: {np.percentile(target, q)}")

    return df_filtered[final_cols]




def gaussian_rank_correlation(x, y):
    """
    Compute Gaussian rank correlation (Spearman's rho computed via normal scores).
    Both x and y are transformed to normal distributions, then Pearson correlation is computed.
    This is robust to outliers and captures monotonic relationships.
    """
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)

    x_transformed = BayesianQuantileTransformer().fit_transform(x).ravel()
    y_transformed = BayesianQuantileTransformer().fit_transform(y).ravel()

    return pearsonr(x_transformed, y_transformed)[0]


def bootstrap_metrics(y_true, y_pred, n_bootstrap=300, random_state=42):
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    r2_samples = []
    gauss_corr_samples = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        y_true_boot = y_true.iloc[indices]
        y_pred_boot = y_pred.iloc[indices]
        r2_samples.append(r2_score(y_true_boot, y_pred_boot))
        gauss_corr_samples.append(gaussian_rank_correlation(y_true_boot, y_pred_boot))

    return {
        'r2_ci_low': np.percentile(r2_samples, 2.5),
        'r2_ci_high': np.percentile(r2_samples, 97.5),
        'gauss_corr_ci_low': np.percentile(gauss_corr_samples, 2.5),
        'gauss_corr_ci_high': np.percentile(gauss_corr_samples, 97.5),
    }


def train_model(df, model_type='bayesian_ridge', clip_x_percentiles=None, clip_y_percentiles=None, clip_y_absolute=None, feature_transform='standard', ridge_alpha=200.0, lasso_alpha=0.01, feature_subset=None, transform_y=False):
    all_feature_cols = ['cpi_log_return_1_months', 'cpi_log_return_6_months', 'cpi_log_return_12_months', 'rem_1_months', 'rem_6_months', 'rem_12_months']

    # Use feature subset if specified, otherwise use all features
    if feature_subset is not None:
        feature_cols = feature_subset
    else:
        feature_cols = all_feature_cols

    df_clean = df.dropna(subset=all_feature_cols + ['target']).copy()

    if clip_x_percentiles:
        for col in feature_cols:
            p_low, p_high = np.percentile(df_clean[col], clip_x_percentiles)
            df_clean[col] = df_clean[col].clip(p_low, p_high)

    X = df_clean[feature_cols]
    y = df_clean['target']

    # Random 50/50 split to avoid temporal distribution shift
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * 0.5)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    # Clip y_train if requested
    if clip_y_absolute is not None:
        y_low, y_high = clip_y_absolute
        y_train_clipped = y_train.clip(y_low, y_high)
    elif clip_y_percentiles:
        y_low, y_high = np.percentile(y_train, clip_y_percentiles)
        y_train_clipped = y_train.clip(y_low, y_high)
    else:
        y_train_clipped = y_train

    # Optionally transform y_train to Gaussian (for optimizing Gaussian correlation)
    if transform_y:
        y_scaler = BayesianQuantileTransformer()
        y_train_transformed = y_scaler.fit_transform(y_train_clipped.values.reshape(-1, 1)).ravel()
        y_train_for_fitting = y_train_transformed
    else:
        y_train_for_fitting = y_train_clipped

    # Apply feature transformation
    if feature_transform == 'standard':
        scaler = StandardScaler()
    elif feature_transform == 'bayesian_quantile':
        scaler = BayesianQuantileTransformer()
    else:
        raise ValueError(f"Unknown feature_transform: {feature_transform}")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == 'lasso':
        model = Lasso(alpha=lasso_alpha, max_iter=10000, random_state=42)
    elif model_type == 'ridge':
        model = Ridge(alpha=ridge_alpha, random_state=42)
    elif model_type == 'bayesian_ridge':
        model = BayesianRidge()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train_scaled, y_train_for_fitting)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    y_train_pred = pd.Series(y_train_pred, index=y_train.index)
    y_test_pred = pd.Series(y_test_pred, index=y_test.index)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_gauss_corr = gaussian_rank_correlation(y_train, y_train_pred)
    test_gauss_corr = gaussian_rank_correlation(y_test, y_test_pred)

    train_bootstrap = bootstrap_metrics(y_train, y_train_pred)
    test_bootstrap = bootstrap_metrics(y_test, y_test_pred)

    result = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_gauss_corr': train_gauss_corr,
        'test_gauss_corr': test_gauss_corr,
        'train_bootstrap': train_bootstrap,
        'test_bootstrap': test_bootstrap,
        'model': model,
        'coefficients': dict(zip(feature_cols, model.coef_)),
        'feature_cols': feature_cols,
    }

    if model_type == 'lasso_cv':
        result['lasso_alpha'] = model.alpha_

    return result


if __name__ == "__main__":
    df = reconstruct_model_data()
    console = Console()

    print("="*80)
    print("MODEL SPECIFICATION")
    print("="*80)
    print("\nTarget: target = log(cpi_yoy)[t+horizon] - log(cpi_yoy)[t]")
    print("        horizon-month ahead CPI inflation log-return (default: 12 months)\n")

    print("Features:")
    features = [
        ('cpi_log_return_1_months', 'log(cpi_yoy)[t] - log(cpi_yoy)[t-1]'),
        ('cpi_log_return_6_months', 'log(cpi_yoy)[t] - log(cpi_yoy)[t-6]'),
        ('cpi_log_return_12_months', 'log(cpi_yoy)[t] - log(cpi_yoy)[t-12]'),
        ('rem_1_months', 'log(rem_12)[t-1] - log(cpi_yoy)[t-1]'),
        ('rem_6_months', 'log(rem_12)[t-6] - log(cpi_yoy)[t-6]'),
        ('rem_12_months', 'log(rem_12)[t-12] - log(cpi_yoy)[t-12]'),
    ]
    for feat, desc in features:
        print(f"  {feat:30s}: {desc}")

    print("\n" + "="*80)
    print("FEATURE TRANSFORMATION ANALYSIS")
    print("="*80)
    print()

    feature_cols = ['cpi_log_return_1_months', 'cpi_log_return_6_months', 'cpi_log_return_12_months',
                    'rem_1_months', 'rem_6_months', 'rem_12_months']
    df_clean = df.dropna(subset=feature_cols + ['target']).copy()
    y = df_clean['target'].values

    table = Table(title="Feature Transformation Correlations (Gaussian rank correlation)",
                  show_header=True, header_style="bold magenta")
    table.add_column("Feature", style="cyan")
    table.add_column("Raw", justify="right")
    table.add_column("Clip 1%", justify="right")
    table.add_column("Clip 2%", justify="right")
    table.add_column("Clip 3%", justify="right")
    table.add_column("BayesQuant", justify="right")
    table.add_column("Power", justify="right")
    table.add_column("Best", style="green", justify="right")

    for feat in feature_cols:
        X_raw = df_clean[feat].values

        # Raw
        corr_raw = abs(gaussian_rank_correlation(X_raw, y))

        # Clip 1%
        p1, p99 = np.percentile(X_raw, [1, 99])
        X_clip1 = X_raw.clip(p1, p99)
        corr_clip1 = abs(gaussian_rank_correlation(X_clip1, y))

        # Clip 2%
        p2, p98 = np.percentile(X_raw, [2, 98])
        X_clip2 = X_raw.clip(p2, p98)
        corr_clip2 = abs(gaussian_rank_correlation(X_clip2, y))

        # Clip 3%
        p3, p97 = np.percentile(X_raw, [3, 97])
        X_clip3 = X_raw.clip(p3, p97)
        corr_clip3 = abs(gaussian_rank_correlation(X_clip3, y))

        # BayesianQuantile
        try:
            X_bayes = BayesianQuantileTransformer().fit_transform(X_raw.reshape(-1, 1)).ravel()
            corr_bayes = abs(gaussian_rank_correlation(X_bayes, y))
        except:
            corr_bayes = 0.0

        # PowerTransformer
        try:
            X_power = PowerTransformer(method='yeo-johnson').fit_transform(X_raw.reshape(-1, 1)).ravel()
            corr_power = abs(gaussian_rank_correlation(X_power, y))
        except:
            corr_power = 0.0

        correlations = {
            'Raw': corr_raw,
            'Clip 1%': corr_clip1,
            'Clip 2%': corr_clip2,
            'Clip 3%': corr_clip3,
            'BayesQuant': corr_bayes,
            'Power': corr_power
        }

        best_transform = max(correlations, key=correlations.get)

        table.add_row(
            feat,
            f"{corr_raw:.4f}",
            f"{corr_clip1:.4f}",
            f"{corr_clip2:.4f}",
            f"{corr_clip3:.4f}",
            f"{corr_bayes:.4f}",
            f"{corr_power:.4f}",
            best_transform
        )

    console.print(table)

    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)
    print("Split: 50/50 random (seed=42)")
    print()

    # Economically sensible baselines and models
    ablations = [
        # Baseline 1: Pure expectations (forward-looking)
        ('Expectations [rem_1]', {'model_type': 'ridge', 'feature_transform': 'bayesian_quantile', 'ridge_alpha': 200.0, 'feature_subset': ['rem_1_months']}),

        # Baseline 2: Pure momentum (backward-looking)
        ('Momentum [cpi_1]', {'model_type': 'ridge', 'feature_transform': 'bayesian_quantile', 'ridge_alpha': 200.0, 'feature_subset': ['cpi_log_return_1_months']}),

        # Baseline 3: All CPI lags (pure AR model)
        ('AR model (CPI lags only)', {'model_type': 'ridge', 'feature_transform': 'bayesian_quantile', 'ridge_alpha': 200.0, 'feature_subset': ['cpi_log_return_1_months', 'cpi_log_return_6_months', 'cpi_log_return_12_months']}),

        # Baseline 4: All REM lags (pure expectations)
        ('Expectations (REM lags)', {'model_type': 'ridge', 'feature_transform': 'bayesian_quantile', 'ridge_alpha': 200.0, 'feature_subset': ['rem_1_months', 'rem_6_months', 'rem_12_months']}),

        # Ridge models with different regularization strengths
        ('Ridge α=100 (weak reg)', {'model_type': 'ridge', 'feature_transform': 'bayesian_quantile', 'ridge_alpha': 100.0}),
        ('Ridge α=200 (baseline)', {'model_type': 'ridge', 'feature_transform': 'bayesian_quantile', 'ridge_alpha': 200.0}),
        ('Ridge α=400 (strong reg)', {'model_type': 'ridge', 'feature_transform': 'bayesian_quantile', 'ridge_alpha': 400.0}),

        # Lasso models with different regularization strengths
        ('Lasso α=0.01 (weak reg)', {'model_type': 'lasso', 'feature_transform': 'bayesian_quantile', 'lasso_alpha': 0.01}),
        ('Lasso α=0.02 (baseline)', {'model_type': 'lasso', 'feature_transform': 'bayesian_quantile', 'lasso_alpha': 0.02}),
        ('Lasso α=0.04 (strong reg)', {'model_type': 'lasso', 'feature_transform': 'bayesian_quantile', 'lasso_alpha': 0.04}),
    ]

    results_list = []
    for name, params in ablations:
        results = train_model(df, **params)
        results_list.append((name, results))

    # print("\nModel Coefficients:")
    # for model_name, model_results in results_list:
    #     print(f"\n{model_name}:")
    #     for feat, coef in model_results['coefficients'].items():
    #         print(f"  {feat:30s}: {coef:10.6f}")
    # print()

    table = Table(show_header=True, header_style="bold magenta", title="Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Train R²", justify="right")
    table.add_column("Test R²", justify="right")
    table.add_column("Train Gauss Corr", justify="right")
    table.add_column("Test Gauss Corr", justify="right")

    for name, res in results_list:
        train_r2_ci = f"{res['train_r2']:.3f} [{res['train_bootstrap']['r2_ci_low']:.3f}, {res['train_bootstrap']['r2_ci_high']:.3f}]"
        test_r2_ci = f"{res['test_r2']:.3f} [{res['test_bootstrap']['r2_ci_low']:.3f}, {res['test_bootstrap']['r2_ci_high']:.3f}]"
        train_gauss_ci = f"{res['train_gauss_corr']:.3f} [{res['train_bootstrap']['gauss_corr_ci_low']:.3f}, {res['train_bootstrap']['gauss_corr_ci_high']:.3f}]"
        test_gauss_ci = f"{res['test_gauss_corr']:.3f} [{res['test_bootstrap']['gauss_corr_ci_low']:.3f}, {res['test_bootstrap']['gauss_corr_ci_high']:.3f}]"

        table.add_row(name, train_r2_ci, test_r2_ci, train_gauss_ci, test_gauss_ci)

    console.print(table)
