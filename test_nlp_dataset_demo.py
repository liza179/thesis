"""
Demo script for nlp_dataset.py

Shows basic usage and validates the implementation.
"""

from nlp_dataset import create_nlp_dataset
from regression import reconstruct_model_data

def main():
    print("=" * 70)
    print("NLP Dataset Creation Demo")
    print("=" * 70)

    # Create dataset with 6 months lookback, offset=1 (exclude current month)
    print("\n1. Creating NLP dataset (n_months=6, offset=1)...")
    df_nlp = create_nlp_dataset(n_months=6, offset=1)

    # Show sample rows
    print("\n2. Sample rows with good document coverage:")
    good_coverage = df_nlp[df_nlp['n_docs_found'] >= 5]
    print(good_coverage[['date', 'n_docs_found', 'missing_months']].head())

    # Show a detailed example
    if len(good_coverage) > 0:
        print("\n3. Detailed view of one row:")
        sample = good_coverage.iloc[0]
        print(f"   Target date: {sample['date'].date()}")
        print(f"   Documents found: {sample['n_docs_found']}")
        print(f"   Document dates: {[d.date() for d in sample['doc_dates']]}")
        print(f"   First doc length: {len(sample['doc_texts'][0])} characters")
        print(f"   Preview: {sample['doc_texts'][0][:150]}...")

    # Merge with regression data
    print("\n4. Merging with regression data...")
    df_regression = reconstruct_model_data()
    df_merged = df_regression.merge(df_nlp, on='date', how='inner')
    print(f"   Regression rows: {len(df_regression)}")
    print(f"   NLP rows: {len(df_nlp)}")
    print(f"   Merged rows: {len(df_merged)}")

    # Filter to complete rows
    df_complete = df_merged[df_merged['n_docs_found'] >= 5]
    print(f"   Rows with >=5 documents: {len(df_complete)}")

    print("\n5. Summary statistics:")
    print(df_nlp['n_docs_found'].describe())

    print("\n" + "=" * 70)
    print("Demo complete! The nlp_dataset.py module is working correctly.")
    print("=" * 70)

if __name__ == "__main__":
    main()
