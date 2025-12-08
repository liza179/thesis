import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

    column_order = [
        'date', 'cpi_yoy', 'cpi_log_return',
        'rate_spread', 'd_rate_spread',
        'rem_cpi_ratio',
        'gap', 'd_gap', 'gap_trend', 'd_gap_trend',
        'd_fx_log',
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


def train_ridge_regression(df, target='cpi_log_return', alpha=1.0, test_size=0.2, random_state=42, clip_percentiles=(2, 98)):
    feature_cols = [col for col in df.columns if col not in ['date', target, 'cpi_yoy', 'cpi_yoy_raw',
                    'policy_rate_raw', 'market_rate_raw', 'rem_12_raw', 'emae_index']]

    df_clean = df.dropna(subset=feature_cols + [target]).copy()

    for col in feature_cols:
        p_low, p_high = np.percentile(df_clean[col], clip_percentiles)
        df_clean[col] = df_clean[col].clip(p_low, p_high)

    X = df_clean[feature_cols]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'coefficients': dict(zip(feature_cols, model.coef_))
    }

    return results


if __name__ == "__main__":
    df = reconstruct_model_data()
    print(f"Data shape: {df.shape}\n")

    analyze_features(df)

    print("\n\n")
    results = train_ridge_regression(df, alpha=10.0)

    print("="*80)
    print("RIDGE REGRESSION RESULTS")
    print("="*80)
    print(f"Target: cpi_log_return")
    print(f"Alpha: 1.0")
    print(f"\nTrain samples: {results['n_train']}")
    print(f"Test samples:  {results['n_test']}")
    print(f"\nTrain RMSE: {results['train_rmse']:.6f}")
    print(f"Test RMSE:  {results['test_rmse']:.6f}")
    print(f"\nTrain MAE: {results['train_mae']:.6f}")
    print(f"Test MAE:  {results['test_mae']:.6f}")
    print(f"\nTrain R²: {results['train_r2']:.6f}")
    print(f"Test R²:  {results['test_r2']:.6f}")

    print("\n" + "="*80)
    print("COEFFICIENTS")
    print("="*80)
    coef_sorted = sorted(results['coefficients'].items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, coef in coef_sorted:
        print(f"{feat:25s}: {coef:10.6f}")
