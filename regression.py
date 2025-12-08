import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.linear_model import BayesianRidge, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from rich.console import Console
from rich.table import Table

def reconstruct_model_data(input_csv='merged_data.csv', start_date='2016-01-31', hp_lambda=14400, horizon=12):
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
    
    return df_filtered[final_cols]




def bootstrap_metrics(y_true, y_pred, n_bootstrap=300, random_state=42):
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    r2_samples = []
    pearson_samples = []
    spearman_samples = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        y_true_boot = y_true.iloc[indices]
        y_pred_boot = y_pred.iloc[indices]
        r2_samples.append(r2_score(y_true_boot, y_pred_boot))
        pearson_samples.append(pearsonr(y_true_boot, y_pred_boot)[0])
        spearman_samples.append(spearmanr(y_true_boot, y_pred_boot)[0])

    return {
        'r2_ci_low': np.percentile(r2_samples, 2.5),
        'r2_ci_high': np.percentile(r2_samples, 97.5),
        'pearson_ci_low': np.percentile(pearson_samples, 2.5),
        'pearson_ci_high': np.percentile(pearson_samples, 97.5),
        'spearman_ci_low': np.percentile(spearman_samples, 2.5),
        'spearman_ci_high': np.percentile(spearman_samples, 97.5),
    }


def train_model(df, model_type='bayesian_ridge'):
    feature_cols = ['cpi_log_return_1_months', 'cpi_log_return_6_months', 'cpi_log_return_12_months', 'rem_1_months', 'rem_6_months', 'rem_12_months']
    df_clean = df.dropna(subset=feature_cols + ['target']).copy()

    X = df_clean[feature_cols]
    y = df_clean['target']

    split_idx = int(len(X) * 0.5)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == 'lasso':
        model = Lasso(alpha=0.1, max_iter=10000, random_state=42)
    elif model_type == 'ridge':
        model = Ridge(alpha=0.1, random_state=42)
    elif model_type == 'bayesian_ridge':
        model = BayesianRidge()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    y_train_pred = pd.Series(y_train_pred, index=y_train.index)
    y_test_pred = pd.Series(y_test_pred, index=y_test.index)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_pearson = pearsonr(y_train, y_train_pred)[0]
    test_pearson = pearsonr(y_test, y_test_pred)[0]
    train_spearman = spearmanr(y_train, y_train_pred)[0]
    test_spearman = spearmanr(y_test, y_test_pred)[0]

    train_bootstrap = bootstrap_metrics(y_train, y_train_pred)
    test_bootstrap = bootstrap_metrics(y_test, y_test_pred)

    result = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_pearson': train_pearson,
        'test_pearson': test_pearson,
        'train_spearman': train_spearman,
        'test_spearman': test_spearman,
        'train_bootstrap': train_bootstrap,
        'test_bootstrap': test_bootstrap,
        'model': model,
        'coefficients': dict(zip(feature_cols, model.coef_)),
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
        print(f"  {feat:20s}: {desc}")

    print("\nModel: LassoCV with StandardScaler")
    print("Split: 50/50 chronological (first half train, second half test)")
    print("="*80)
    print("\n")

    ablations = [
        ('BayesianRidge', {'model_type': 'bayesian_ridge'}),
        ('Ridge (alpha=0.1)', {'model_type': 'ridge'}),
        ('Lasso (alpha=0.1)', {'model_type': 'lasso'}),
    ]

    results_list = []
    for name, params in ablations:
        results = train_model(df, **params)
        results_list.append((name, results))

    print("="*80)
    print("MODEL COEFFICIENTS")
    print("="*80)
    for name, results in results_list:
        print(f"\n{name}:")
        if 'lasso_alpha' in results:
            print(f"  Alpha: {results['lasso_alpha']:.6f}")
        print("  Feature coefficients:")
        for feat, coef in results['coefficients'].items():
            print(f"    {feat:25s}: {coef:10.6f}")
    print("\n")

    table = Table(show_header=True, header_style="bold magenta", title="Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Train R²", justify="right")
    table.add_column("Test R²", justify="right")
    table.add_column("Train Pearson", justify="right")
    table.add_column("Test Pearson", justify="right")
    table.add_column("Train Spearman", justify="right")
    table.add_column("Test Spearman", justify="right")

    for name, res in results_list:
        train_r2_ci = f"{res['train_r2']:.3f} [{res['train_bootstrap']['r2_ci_low']:.3f}, {res['train_bootstrap']['r2_ci_high']:.3f}]"
        test_r2_ci = f"{res['test_r2']:.3f} [{res['test_bootstrap']['r2_ci_low']:.3f}, {res['test_bootstrap']['r2_ci_high']:.3f}]"
        train_pearson_ci = f"{res['train_pearson']:.3f} [{res['train_bootstrap']['pearson_ci_low']:.3f}, {res['train_bootstrap']['pearson_ci_high']:.3f}]"
        test_pearson_ci = f"{res['test_pearson']:.3f} [{res['test_bootstrap']['pearson_ci_low']:.3f}, {res['test_bootstrap']['pearson_ci_high']:.3f}]"
        train_spearman_ci = f"{res['train_spearman']:.3f} [{res['train_bootstrap']['spearman_ci_low']:.3f}, {res['train_bootstrap']['spearman_ci_high']:.3f}]"
        test_spearman_ci = f"{res['test_spearman']:.3f} [{res['test_bootstrap']['spearman_ci_low']:.3f}, {res['test_bootstrap']['spearman_ci_high']:.3f}]"

        table.add_row(name, train_r2_ci, test_r2_ci, train_pearson_ci, test_pearson_ci, train_spearman_ci, test_spearman_ci)

    console.print(table)
