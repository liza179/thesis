import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.linear_model import Ridge, BayesianRidge, LassoCV
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from bayesian_quantile_transformer import BayesianQuantileTransformer
from rich.console import Console
from rich.table import Table

def reconstruct_model_data(input_csv='merged_data.csv', start_date='2016-01-31', hp_lambda=14400):
    df = pd.read_csv(input_csv, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    df['cpi_yoy_raw'] = df['cpi_yoy']
    df['policy_rate_raw'] = df['policy_rate']
    df['market_rate_raw'] = df['market_rate']
    df['rem_12_raw'] = df['rem_12']

    df['cpi_yoy'] = np.log(df['cpi_yoy'])
    df['market_rate'] = np.log(df['market_rate'])
    df['policy_rate'] = np.log(df['policy_rate'])
    df['rem_12'] = np.log(df['rem_12'])

    log_output_gap = np.log(df['output_gap'].dropna())
    cycle, trend = hpfilter(log_output_gap, lamb=hp_lambda)
    cycle_pct = cycle * 100
    trend_pct = trend * 100

    df['gap'] = np.nan
    df['gap_trend'] = np.nan
    df.loc[log_output_gap.index, 'gap'] = cycle_pct.values
    df.loc[log_output_gap.index, 'gap_trend'] = trend_pct.values
    df['emae_index'] = df['output_gap']

    df['fx_log'] = np.log(df['fx'])
    df['d_fx_log'] = df['fx_log'].diff()

    df['cpi_yoy_lag1'] = df['cpi_yoy'].shift(1)
    df['rem_cpi_ratio'] = df['rem_12'] - df['cpi_yoy_lag1']

    df['rate_spread'] = df['market_rate'] - df['policy_rate']
    df['d_rate_spread'] = df['rate_spread'].diff()
    df['d_gap'] = df['gap'].diff()
    df['d_gap_trend'] = df['gap_trend'].diff()

    df_filtered = df[df['date'] >= start_date].copy().reset_index(drop=True)
    df_filtered['cpi_log_return'] = df_filtered['cpi_yoy'].diff()
    df_filtered['cpi_log_return_lag1'] = df_filtered['cpi_log_return'].shift(1)

    column_order = [
        'date', 'cpi_yoy', 'cpi_log_return', 'cpi_log_return_lag1',
        'rem_cpi_ratio',
        'd_gap_trend',
        'emae_index', 'cpi_yoy_raw', 'policy_rate_raw', 'market_rate_raw', 'rem_12_raw'
    ]

    final_columns = [col for col in column_order if col in df_filtered.columns]
    return df_filtered[final_columns]


def analyze_features(df, target='cpi_log_return'):
    print("="*80)
    print("FEATURE ANALYSIS")
    print("="*80)

    feature_cols = [col for col in df.columns if col not in ['date', target, 'cpi_yoy', 'cpi_yoy_raw',
                    'policy_rate_raw', 'market_rate_raw', 'rem_12_raw', 'emae_index']]

    df_clean = df.dropna(subset=feature_cols + [target])
    print(f"Samples: {len(df_clean)}\n")

    print("TARGET DISTRIBUTION")
    print("-"*80)
    target_data = df_clean[target]
    target_quantiles = np.percentile(target_data, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    print(f"{target}")
    print(f"  Mean: {target_data.mean():10.4f}  Std: {target_data.std():10.4f}")
    print(f"  Skew: {target_data.skew():10.4f}  Kurt: {target_data.kurtosis():10.4f}")
    print(f"  Quantiles: ", end="")
    for i, q in enumerate([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
        if i > 0:
            print(f"             ", end="")
        print(f"p{q:02d}={target_quantiles[i]:8.4f}")
    print()

    print("MARGINAL DISTRIBUTIONS")
    print("-"*80)
    for col in feature_cols:
        data = df_clean[col]
        quantiles = np.percentile(data, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        print(f"\n{col}")
        print(f"  Mean: {data.mean():10.4f}  Std: {data.std():10.4f}")
        print(f"  Skew: {data.skew():10.4f}  Kurt: {data.kurtosis():10.4f}")
        print(f"  Quantiles: ", end="")
        for i, q in enumerate([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
            if i > 0:
                print(f"             ", end="")
            print(f"p{q:02d}={quantiles[i]:8.4f}")

    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)

    pearson = df_clean[feature_cols].corr(method='pearson')
    spearman = df_clean[feature_cols].corr(method='spearman')

    print(f"\nPearson correlation matrix (shape: {pearson.shape})")
    print(pearson.round(3).to_string())

    print(f"\n\nSpearman correlation matrix (shape: {spearman.shape})")
    print(spearman.round(3).to_string())

    print("\n" + "="*80)
    print("HIGH CORRELATIONS (|r| > 0.8)")
    print("="*80)
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            p_corr = pearson.iloc[i, j]
            s_corr = spearman.iloc[i, j]
            if abs(p_corr) > 0.8 or abs(s_corr) > 0.8:
                high_corr_pairs.append((feature_cols[i], feature_cols[j], p_corr, s_corr))

    if high_corr_pairs:
        for f1, f2, p, s in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"{f1:25s} <-> {f2:25s}  Pearson: {p:7.4f}  Spearman: {s:7.4f}")
    else:
        print("No high correlations found")


def bootstrap_metrics(y_true, y_pred, n_bootstrap=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    n = len(y_true)

    r2_samples = []
    corr_samples = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        y_true_boot = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_pred_boot = y_pred.iloc[indices] if hasattr(y_pred, 'iloc') else y_pred[indices]

        r2_samples.append(r2_score(y_true_boot, y_pred_boot))
        corr_samples.append(pearsonr(y_true_boot, y_pred_boot)[0])

    r2_samples = np.array(r2_samples)
    corr_samples = np.array(corr_samples)

    return {
        'r2_mean': np.mean(r2_samples),
        'r2_ci_low': np.percentile(r2_samples, 2.5),
        'r2_ci_high': np.percentile(r2_samples, 97.5),
        'corr_mean': np.mean(corr_samples),
        'corr_ci_low': np.percentile(corr_samples, 2.5),
        'corr_ci_high': np.percentile(corr_samples, 97.5),
    }


def train_model(df, model_type='ridge', alpha=10.0, target='cpi_log_return', test_size=0.5, random_state=42,
                clip_y_percentiles=None, clip_residuals=False, clip_x_percentiles=None,
                transformer='standard', target_transformer=None, n_bootstrap=300, per_feature_transform=None,
                lasso_alphas=None):
    feature_cols = [col for col in df.columns if col not in ['date', target, 'cpi_yoy', 'cpi_yoy_raw',
                    'policy_rate_raw', 'market_rate_raw', 'rem_12_raw', 'emae_index']]

    df_clean = df.dropna(subset=feature_cols + [target]).copy()

    if clip_x_percentiles:
        for col in feature_cols:
            p_low, p_high = np.percentile(df_clean[col], clip_x_percentiles)
            df_clean[col] = df_clean[col].clip(p_low, p_high)

    X = df_clean[feature_cols]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if clip_y_percentiles:
        y_train_low, y_train_high = np.percentile(y_train, clip_y_percentiles)
        y_train_clipped = y_train.clip(y_train_low, y_train_high)
    else:
        y_train_clipped = y_train
        y_train_low, y_train_high = y_train.min(), y_train.max()

    if per_feature_transform:
        X_train_scaled = np.zeros_like(X_train)
        X_test_scaled = np.zeros_like(X_test)
        for i, feat in enumerate(feature_cols):
            trans_type = per_feature_transform.get(feat, 'standard')
            if trans_type == 'standard':
                scaler_feat = StandardScaler()
            elif trans_type == 'power':
                scaler_feat = PowerTransformer(method='yeo-johnson', standardize=True)
            elif trans_type == 'bayesian_quantile':
                scaler_feat = BayesianQuantileTransformer()
            elif trans_type == 'raw':
                scaler_feat = StandardScaler()
            else:
                raise ValueError(f"Unknown transformer: {trans_type}")
            X_train_scaled[:, i] = scaler_feat.fit_transform(X_train.iloc[:, i].values.reshape(-1, 1)).ravel()
            X_test_scaled[:, i] = scaler_feat.transform(X_test.iloc[:, i].values.reshape(-1, 1)).ravel()
    else:
        if transformer == 'standard':
            scaler = StandardScaler()
        elif transformer == 'power':
            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        elif transformer == 'quantile_normal':
            scaler = QuantileTransformer(output_distribution='normal', random_state=random_state)
        elif transformer == 'bayesian_quantile':
            scaler = BayesianQuantileTransformer()
        else:
            raise ValueError(f"Unknown transformer: {transformer}")

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    if target_transformer == 'bayesian_quantile':
        y_scaler = BayesianQuantileTransformer()
        y_train_clipped_transformed = y_scaler.fit_transform(y_train_clipped.values.reshape(-1, 1)).ravel()
    elif target_transformer == 'power':
        y_scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        y_train_clipped_transformed = y_scaler.fit_transform(y_train_clipped.values.reshape(-1, 1)).ravel()
    else:
        y_scaler = None
        y_train_clipped_transformed = y_train_clipped

    if model_type == 'ridge':
        model = Ridge(alpha=alpha)
    elif model_type == 'bayesian_ridge':
        model = BayesianRidge()
    elif model_type == 'lasso_cv':
        if lasso_alphas is None:
            lasso_alphas = np.logspace(-4, 1, 50)
        model = LassoCV(alphas=lasso_alphas, cv=5, random_state=random_state, max_iter=10000)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train_scaled, y_train_clipped_transformed)

    y_train_pred_raw = model.predict(X_train_scaled)
    y_test_pred_raw = model.predict(X_test_scaled)

    if y_scaler:
        y_train_pred_raw = y_scaler.inverse_transform(y_train_pred_raw.reshape(-1, 1)).ravel()
        y_test_pred_raw = y_scaler.inverse_transform(y_test_pred_raw.reshape(-1, 1)).ravel()

    if clip_residuals:
        train_residuals = y_train - y_train_pred_raw
        res_low, res_high = np.percentile(train_residuals, clip_y_percentiles if clip_y_percentiles else [5, 95])
        y_train_pred = y_train_pred_raw + train_residuals.clip(res_low, res_high)
        test_residuals_raw = y_test_pred_raw - y_train.mean()
        y_test_pred = y_test_pred_raw + test_residuals_raw.clip(res_low, res_high) - test_residuals_raw
    else:
        y_train_pred = y_train_pred_raw.clip(y_train_low, y_train_high)
        y_test_pred = y_test_pred_raw.clip(y_train_low, y_train_high)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_pearson = pearsonr(y_train, y_train_pred)[0]
    test_pearson = pearsonr(y_test, y_test_pred)[0]
    train_spearman = spearmanr(y_train, y_train_pred)[0]
    test_spearman = spearmanr(y_test, y_test_pred)[0]

    train_bootstrap = bootstrap_metrics(y_train, y_train_pred, n_bootstrap, random_state)
    test_bootstrap = bootstrap_metrics(y_test, y_test_pred, n_bootstrap, random_state)

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
    }

    if model_type == 'lasso_cv':
        result['lasso_alpha'] = model.alpha_
        result['lasso_coefs'] = dict(zip(feature_cols, model.coef_))

    return result


if __name__ == "__main__":
    df = reconstruct_model_data()
    console = Console()

    feature_cols = [col for col in df.columns if col not in ['date', 'cpi_log_return', 'cpi_yoy', 'cpi_yoy_raw',
                    'policy_rate_raw', 'market_rate_raw', 'rem_12_raw', 'emae_index']]

    df_clean = df.dropna(subset=feature_cols + ['cpi_log_return']).copy()
    y = df_clean['cpi_log_return']
    y_low, y_high = np.percentile(y, [10, 90])
    y_clipped = y.clip(y_low, y_high)

    table = Table(title="Feature Transformation Correlations (with 10-90 clipped target)", show_header=True, header_style="bold magenta")
    table.add_column("Feature", style="cyan", width=20)
    table.add_column("Raw", justify="right")
    table.add_column("BayesQuant", justify="right")
    table.add_column("Quant(norm)", justify="right")
    table.add_column("Quant(uni)", justify="right")
    table.add_column("Power", justify="right")
    table.add_column("Best", style="green", justify="right")

    for feat in feature_cols:
        X_raw = df_clean[feat].values.reshape(-1, 1)

        corr_raw = abs(pearsonr(X_raw.ravel(), y_clipped)[0])

        X_bayes = BayesianQuantileTransformer().fit_transform(X_raw).ravel()
        corr_bayes = abs(pearsonr(X_bayes, y_clipped)[0])

        X_qnorm = QuantileTransformer(output_distribution='normal').fit_transform(X_raw).ravel()
        corr_qnorm = abs(pearsonr(X_qnorm, y_clipped)[0])

        X_quni = QuantileTransformer(output_distribution='uniform').fit_transform(X_raw).ravel()
        corr_quni = abs(pearsonr(X_quni, y_clipped)[0])

        try:
            X_power = PowerTransformer(method='yeo-johnson').fit_transform(X_raw).ravel()
            corr_power = abs(pearsonr(X_power, y_clipped)[0])
        except:
            corr_power = 0.0

        correlations = {
            'Raw': corr_raw,
            'BayesQuant': corr_bayes,
            'Quant(norm)': corr_qnorm,
            'Quant(uni)': corr_quni,
            'Power': corr_power
        }

        best_transform = max(correlations, key=correlations.get)

        table.add_row(
            feat,
            f"{corr_raw:.4f}",
            f"{corr_bayes:.4f}",
            f"{corr_qnorm:.4f}",
            f"{corr_quni:.4f}",
            f"{corr_power:.4f}",
            best_transform
        )

    console.print(table)

    print("\n")
    print("="*80)
    print("MODEL SPECIFICATION")
    print("="*80)
    print("\nTarget: cpi_log_return = diff(log(cpi_yoy))")
    print("        Monthly CPI inflation log-return\n")

    print("Features:")
    feature_descriptions = {
        'cpi_log_return_lag1': ('lag(cpi_log_return)', 'StandardScaler'),
        'rem_cpi_ratio': ('log(rem_12) - log(cpi_yoy_lag1)', 'StandardScaler'),
        'd_gap_trend': ('diff(gap_trend)', 'StandardScaler'),
    }

    for feat, (desc, trans) in feature_descriptions.items():
        print(f"  {feat:20s}: {desc:45s} [{trans}]")

    print("\nModel: LassoCV with StandardScaler")
    print("Split: 50/50 train/test")
    print("="*80)
    print("\n")

    ablations = [
        ('Full model', {'model_type': 'lasso_cv', 'transformer': 'standard'}),
        ('→ BayesianRidge', {'model_type': 'bayesian_ridge', 'transformer': 'standard'}),
        ('→ X-clip 1%', {'model_type': 'lasso_cv', 'transformer': 'standard', 'clip_x_percentiles': (1, 99)}),
        ('→ X-clip 2%', {'model_type': 'lasso_cv', 'transformer': 'standard', 'clip_x_percentiles': (2, 98)}),
        ('→ X-clip 3%', {'model_type': 'lasso_cv', 'transformer': 'standard', 'clip_x_percentiles': (3, 97)}),
    ]

    results_list = []
    for name, params in ablations:
        results = train_model(df, **params)
        results_list.append((name, results))

    full_model_results = results_list[0][1]
    if 'lasso_alpha' in full_model_results:
        print(f"LassoCV selected alpha: {full_model_results['lasso_alpha']:.6f}\n")
        print("Feature coefficients:")
        for feat, coef in full_model_results['lasso_coefs'].items():
            if abs(coef) > 1e-10:
                print(f"  {feat:20s}: {coef:10.6f}")
            else:
                print(f"  {feat:20s}: {coef:10.6f}  (zeroed)")
        print("\n")

    table2 = Table(title="Ablation Study", show_header=True, header_style="bold magenta")
    table2.add_column("Configuration", style="cyan", width=20)
    table2.add_column("Train R² [95% CI]", justify="right", width=25)
    table2.add_column("Test R² [95% CI]", justify="right", width=25)
    table2.add_column("Test Pearson [95% CI]", justify="right", width=25)

    for name, res in results_list:
        train_r2_ci = f"{res['train_r2']:.3f} [{res['train_bootstrap']['r2_ci_low']:.3f}, {res['train_bootstrap']['r2_ci_high']:.3f}]"
        test_r2_ci = f"{res['test_r2']:.3f} [{res['test_bootstrap']['r2_ci_low']:.3f}, {res['test_bootstrap']['r2_ci_high']:.3f}]"
        test_pearson_ci = f"{res['test_pearson']:.3f} [{res['test_bootstrap']['corr_ci_low']:.3f}, {res['test_bootstrap']['corr_ci_high']:.3f}]"

        table2.add_row(
            name,
            train_r2_ci,
            test_r2_ci,
            test_pearson_ci,
        )

    console.print(table2)
