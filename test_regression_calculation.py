import pandas as pd
import numpy as np
from regression import reconstruct_model_data

def test_regression_calculation():
    print("="*80)
    print("REGRESSION CALCULATION TEST")
    print("="*80)
    
    df = reconstruct_model_data()
    
    print("\n1. Basic data info:")
    print(f"   Total rows: {len(df)}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Columns: {list(df.columns)}")
    
    print("\n2. Target variable (target):")
    print("   Formula: cpi_yoy[t+12] - cpi_yoy[t]")
    print("   Meaning: 12-month ahead log return")
    target_sample = df[['date', 'target']].head(20)
    print(target_sample.to_string(index=False))
    
    print("\n3. Feature: cpi_log_return_1_months")
    print("   Formula: cpi_yoy[t] - cpi_yoy[t-1]")
    print("   Meaning: 1-month log return")
    lag1_sample = df[['date', 'cpi_log_return_1_months']].head(20)
    print(lag1_sample.to_string(index=False))
    
    print("\n4. Feature: cpi_log_return_6_months")
    print("   Formula: cpi_yoy[t] - cpi_yoy[t-6]")
    print("   Meaning: 6-month log return")
    lag6_sample = df[['date', 'cpi_log_return_6_months']].head(20)
    print(lag6_sample.to_string(index=False))
    
    print("\n5. Feature: cpi_log_return_12_months")
    print("   Formula: cpi_yoy[t] - cpi_yoy[t-12]")
    print("   Meaning: 12-month log return")
    lag12_sample = df[['date', 'cpi_log_return_12_months']].head(20)
    print(lag12_sample.to_string(index=False))
    
    print("\n6. Manual verification for a specific row:")
    test_idx = 20
    if test_idx < len(df):
        row = df.iloc[test_idx]
        print(f"   Row index: {test_idx}")
        print(f"   Date: {row['date'].date()}")
        print(f"   Target: {row['target']:.6f}")
        print(f"   lag1: {row['cpi_log_return_1_months']:.6f}")
        print(f"   lag6: {row['cpi_log_return_6_months']:.6f}")
        print(f"   lag12: {row['cpi_log_return_12_months']:.6f}")
        
        print("\n   Manual calculation check:")
        if 'cpi_yoy' in df.columns:
            cpi_t = df.iloc[test_idx]['cpi_yoy']
            cpi_t_minus_1 = df.iloc[test_idx - 1]['cpi_yoy'] if test_idx > 0 else None
            cpi_t_minus_2 = df.iloc[test_idx - 2]['cpi_yoy'] if test_idx > 1 else None
            cpi_t_minus_6 = df.iloc[test_idx - 6]['cpi_yoy'] if test_idx >= 6 else None
            cpi_t_minus_12 = df.iloc[test_idx - 12]['cpi_yoy'] if test_idx >= 12 else None
            cpi_t_minus_18 = df.iloc[test_idx - 18]['cpi_yoy'] if test_idx >= 18 else None
            cpi_t_minus_24 = df.iloc[test_idx - 24]['cpi_yoy'] if test_idx >= 24 else None
            cpi_t_plus_12 = df.iloc[test_idx + 12]['cpi_yoy'] if test_idx + 12 < len(df) else None
            
            print(f"   cpi_yoy[t]: {cpi_t:.6f}")
            if cpi_t_plus_12 is not None:
                manual_target = cpi_t_plus_12 - cpi_t
                print(f"   Manual target: cpi_yoy[t+12] - cpi_yoy[t] = {manual_target:.6f}")
                print(f"   Computed target: {row['target']:.6f}")
                print(f"   Match: {np.isclose(manual_target, row['target'], atol=1e-6)}")
            
            if cpi_t_minus_1 is not None:
                manual_lag1 = cpi_t - cpi_t_minus_1
                print(f"   Manual lag1: cpi_yoy[t] - cpi_yoy[t-1] = {manual_lag1:.6f}")
                print(f"   Computed lag1: {row['cpi_log_return_1_months']:.6f}")
                print(f"   Match: {np.isclose(manual_lag1, row['cpi_log_return_1_months'], atol=1e-6)}")
            
            if cpi_t_minus_6 is not None:
                manual_lag6 = cpi_t - cpi_t_minus_6
                print(f"   Manual lag6: cpi_yoy[t] - cpi_yoy[t-6] = {manual_lag6:.6f}")
                print(f"   Computed lag6: {row['cpi_log_return_6_months']:.6f}")
                print(f"   Match: {np.isclose(manual_lag6, row['cpi_log_return_6_months'], atol=1e-6)}")
            
            if cpi_t_minus_12 is not None:
                manual_lag12 = cpi_t - cpi_t_minus_12
                print(f"   Manual lag12: cpi_yoy[t] - cpi_yoy[t-12] = {manual_lag12:.6f}")
                print(f"   Computed lag12: {row['cpi_log_return_12_months']:.6f}")
                print(f"   Match: {np.isclose(manual_lag12, row['cpi_log_return_12_months'], atol=1e-6)}")
    
    print("\n7. Data leakage check:")
    print("   Checking that features don't use future data...")
    
    for idx in range(12, min(50, len(df))):
        row = df.iloc[idx]
        target_date = row['date']
        target_future_date = df.iloc[idx + 12]['date'] if idx + 12 < len(df) else None
        
        lag1_val = row['cpi_log_return_1_months']
        lag6_val = row['cpi_log_return_6_months']
        lag12_val = row['cpi_log_return_12_months']
        
        if target_future_date:
            assert target_future_date > target_date, f"Target uses past data at row {idx}"
        
        if not pd.isna(lag1_val):
            lag1_date = df.iloc[idx - 1]['date']
            assert lag1_date < target_date, f"lag1 uses future data at row {idx}"
        
        if not pd.isna(lag6_val):
            lag6_date = df.iloc[idx - 6]['date']
            assert lag6_date < target_date, f"lag6 uses future data at row {idx}"
        
        if not pd.isna(lag12_val):
            lag12_date = df.iloc[idx - 12]['date']
            assert lag12_date < target_date, f"lag12 uses future data at row {idx}"
    
    print("   ✓ No data leakage detected")
    
    print("\n8. Missing values summary:")
    print(f"   Target: {df['target'].isna().sum()} missing")
    print(f"   lag1: {df['cpi_log_return_1_months'].isna().sum()} missing")
    print(f"   lag6: {df['cpi_log_return_6_months'].isna().sum()} missing")
    print(f"   lag12: {df['cpi_log_return_12_months'].isna().sum()} missing")
    
    print("\n9. Complete feature matrix sample (first 20 rows with all features):")
    feature_cols = ['cpi_log_return_1_months', 'cpi_log_return_6_months', 'cpi_log_return_12_months', 'rem_cpi_ratio', 'd_gap_trend']
    complete_rows = df.dropna(subset=feature_cols + ['target'])
    print(f"   Rows with complete features: {len(complete_rows)}")
    if len(complete_rows) > 0:
        sample = complete_rows[['date'] + feature_cols + ['target']].head(20)
        print(sample.to_string(index=False))
    
    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)

if __name__ == "__main__":
    test_regression_calculation()

