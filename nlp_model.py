"""
Stage 1: NLP Model for Inflation Prediction

Trains a Ridge regression model that maps document embeddings to inflation targets.
The model output (nlp_pred) is then used as a feature in the final ensemble.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from nlp_dataset import create_nlp_dataset
from vector import get_model, encode
from regression import gaussian_rank_correlation, bootstrap_metrics


def embed_documents(df_nlp, model=None, verbose=True):
    """
    Generate embeddings for all documents in the dataset with caching.

    Args:
        df_nlp: DataFrame from create_nlp_dataset() with doc_texts and doc_dates columns
        model: BGE-M3 model (if None, will load)
        verbose: Show progress bar

    Returns:
        DataFrame with added 'doc_embeddings' column (List of 512-dim arrays)

    Note:
        Uses caching based on doc_dates to avoid re-embedding duplicate documents.
    """
    if model is None:
        print("Loading BGE-M3 model...")
        model = get_model(static=True)

    df_nlp = df_nlp.copy()

    # Build cache: {date: embedding}
    embedding_cache = {}

    # Collect all unique (date, text) pairs
    unique_docs = {}
    for _, row in df_nlp.iterrows():
        for date, text in zip(row['doc_dates'], row['doc_texts']):
            if date not in unique_docs:
                unique_docs[date] = text

    if verbose:
        print(f"Embedding {len(unique_docs)} unique documents (with caching)...")

    # Embed all unique documents
    iterator = tqdm(unique_docs.items(), desc="Embedding documents") if verbose else unique_docs.items()
    for date, text in iterator:
        emb = encode(text, model)
        embedding_cache[date] = emb

    # Build embeddings list for each row using cache
    doc_embeddings_list = []
    for _, row in df_nlp.iterrows():
        embeddings = []
        for date in row['doc_dates']:
            embeddings.append(embedding_cache[date])
        doc_embeddings_list.append(embeddings)

    df_nlp['doc_embeddings'] = doc_embeddings_list
    return df_nlp


def aggregate_embeddings(doc_embeddings, method='ema', ema_half_life=5):
    """
    Aggregate multiple document embeddings into a single vector.

    Args:
        doc_embeddings: List of embedding arrays (each 512-dim)
        method: 'ema' for exponential moving average
        ema_half_life: Number of documents for weight to decay by half

    Returns:
        Single aggregated embedding (512-dim array)

    EMA logic:
        - For 6 docs: oldest doc has weight = 0.5 × newest doc weight
        - decay_factor = 0.5^(1/5), so weight[0] = decay_factor^5 = 0.5
        - If fewer docs: renormalize weights to sum to 1
    """
    if len(doc_embeddings) == 0:
        # Return zero vector if no documents
        return np.zeros(512)

    if method == 'ema':
        n_docs = len(doc_embeddings)

        # Calculate decay factor: weight[0] = 0.5 * weight[n-1]
        # decay_factor^(n-1) = 0.5
        decay_factor = 0.5 ** (1 / ema_half_life)

        # Generate weights from oldest to newest
        weights = np.array([decay_factor ** (n_docs - 1 - i) for i in range(n_docs)])

        # Normalize to sum to 1
        weights = weights / weights.sum()

        # Weighted average
        embeddings_array = np.stack(doc_embeddings, axis=0)  # shape: (n_docs, 512)
        aggregated = np.average(embeddings_array, axis=0, weights=weights)

        return aggregated
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def train_nlp_model(
    df_nlp,
    df_regression,
    ridge_alpha=1.0,
    random_state=42,
    verbose=True
):
    """
    Train Stage 1 NLP model: embeddings → target.

    Args:
        df_nlp: DataFrame with 'doc_embeddings' column
        df_regression: DataFrame with 'target' column (from reconstruct_model_data)
        ridge_alpha: Ridge regularization strength
        random_state: For reproducible train/test split
        verbose: Print training info

    Returns:
        Dictionary with:
            - model: Trained Ridge model
            - train_pred: Predictions on train set
            - test_pred: Predictions on test set
            - train_indices: Train split indices
            - test_indices: Test split indices
            - metrics: R², Gaussian correlation, bootstrap CIs
    """
    # Merge datasets on date
    df_merged = df_nlp.merge(df_regression[['date', 'target']], on='date', how='inner')

    if verbose:
        print(f"\nMerged dataset: {len(df_merged)} rows")
        print(f"Documents coverage: {df_merged['n_docs_found'].mean():.2f} avg docs per row")

    # Aggregate embeddings for each row
    if verbose:
        print("Aggregating embeddings with EMA...")

    aggregated_embeddings = []
    for _, row in df_merged.iterrows():
        emb = aggregate_embeddings(row['doc_embeddings'], method='ema')
        aggregated_embeddings.append(emb)

    X = np.stack(aggregated_embeddings, axis=0)  # shape: (n_samples, 512)
    y = df_merged['target'].values

    if verbose:
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target shape: {y.shape}")

    # Use same 50/50 random split as regression.py
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * 0.5)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    if verbose:
        print(f"\nTrain/test split: {len(train_indices)}/{len(test_indices)} samples")
        print(f"Training Ridge model (alpha={ridge_alpha})...")

    # Train Ridge regression
    model = Ridge(alpha=ridge_alpha, random_state=random_state)
    model.fit(X_train, y_train)

    # Generate predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_gauss_corr = gaussian_rank_correlation(y_train, y_train_pred)
    test_gauss_corr = gaussian_rank_correlation(y_test, y_test_pred)

    # Bootstrap confidence intervals
    y_train_series = pd.Series(y_train)
    y_test_series = pd.Series(y_test)
    y_train_pred_series = pd.Series(y_train_pred)
    y_test_pred_series = pd.Series(y_test_pred)

    train_bootstrap = bootstrap_metrics(y_train_series, y_train_pred_series)
    test_bootstrap = bootstrap_metrics(y_test_series, y_test_pred_series)

    metrics = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_gauss_corr': train_gauss_corr,
        'test_gauss_corr': test_gauss_corr,
        'train_bootstrap': train_bootstrap,
        'test_bootstrap': test_bootstrap,
    }

    # Create full prediction array (preserving original order)
    full_predictions = np.zeros(len(X))
    full_predictions[train_indices] = y_train_pred
    full_predictions[test_indices] = y_test_pred

    return {
        'model': model,
        'full_predictions': full_predictions,
        'train_pred': y_train_pred,
        'test_pred': y_test_pred,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'metrics': metrics,
        'df_merged': df_merged,  # Includes dates for alignment
    }


def save_predictions_to_csv(
    nlp_results,
    df_regression,
    output_path='merged_data_with_nlp.csv'
):
    """
    Save regression data with NLP predictions to CSV.

    Args:
        nlp_results: Output from train_nlp_model()
        df_regression: Original regression DataFrame
        output_path: Output CSV path

    Returns:
        DataFrame that was saved
    """
    # Get dates and predictions from NLP model
    df_nlp_preds = nlp_results['df_merged'][['date']].copy()
    df_nlp_preds['nlp_pred'] = nlp_results['full_predictions']

    # Merge with regression data
    df_with_nlp = df_regression.merge(df_nlp_preds, on='date', how='left')

    # Fill NaN nlp_pred with 0 for rows without documents
    df_with_nlp['nlp_pred'] = df_with_nlp['nlp_pred'].fillna(0)

    # Save to CSV
    df_with_nlp.to_csv(output_path, index=False)
    print(f"\nSaved predictions to {output_path}")
    print(f"  Total rows: {len(df_with_nlp)}")
    print(f"  Rows with NLP predictions: {(df_with_nlp['nlp_pred'] != 0).sum()}")

    return df_with_nlp


def print_nlp_results(nlp_results):
    """Print formatted results table for NLP model."""
    console = Console()
    metrics = nlp_results['metrics']

    print("\n" + "="*80)
    print("STAGE 1: NLP MODEL RESULTS")
    print("="*80)
    print(f"\nModel: Ridge Regression (embeddings → target)")
    print(f"Input: 512-dim aggregated embeddings (EMA with half-life=5)")
    print(f"Train/Test split: 50/50 random (same as regression.py)")

    table = Table(show_header=True, header_style="bold magenta", title="\nPerformance Metrics")
    table.add_column("Split", style="cyan")
    table.add_column("R²", justify="right")
    table.add_column("Gaussian Rank Corr", justify="right")

    train_r2_ci = f"{metrics['train_r2']:.3f} [{metrics['train_bootstrap']['r2_ci_low']:.3f}, {metrics['train_bootstrap']['r2_ci_high']:.3f}]"
    test_r2_ci = f"{metrics['test_r2']:.3f} [{metrics['test_bootstrap']['r2_ci_low']:.3f}, {metrics['test_bootstrap']['r2_ci_high']:.3f}]"
    train_gauss_ci = f"{metrics['train_gauss_corr']:.3f} [{metrics['train_bootstrap']['gauss_corr_ci_low']:.3f}, {metrics['train_bootstrap']['gauss_corr_ci_high']:.3f}]"
    test_gauss_ci = f"{metrics['test_gauss_corr']:.3f} [{metrics['test_bootstrap']['gauss_corr_ci_low']:.3f}, {metrics['test_bootstrap']['gauss_corr_ci_high']:.3f}]"

    table.add_row("Train", train_r2_ci, train_gauss_ci)
    table.add_row("Test", test_r2_ci, test_gauss_ci)

    console.print(table)
    print("="*80)


if __name__ == "__main__":
    from regression import reconstruct_model_data

    print("="*80)
    print("STAGE 1: NLP MODEL TRAINING")
    print("="*80)

    # Load data
    print("\n1. Loading documents...")
    df_nlp = create_nlp_dataset(n_months=6, min_before=12, max_before=24)

    print("\n2. Loading regression data...")
    df_regression = reconstruct_model_data()

    # Generate embeddings
    print("\n3. Generating embeddings...")
    df_nlp = embed_documents(df_nlp, verbose=True)

    # Train NLP model
    print("\n4. Training NLP model...")
    nlp_results = train_nlp_model(
        df_nlp,
        df_regression,
        ridge_alpha=1.0,
        verbose=True
    )

    # Print results
    print_nlp_results(nlp_results)

    # Save predictions
    print("\n5. Saving predictions...")
    df_with_nlp = save_predictions_to_csv(
        nlp_results,
        df_regression,
        output_path='merged_data_with_nlp.csv'
    )

    print("\n" + "="*80)
    print("Stage 1 complete! Use 'merged_data_with_nlp.csv' in regression.py")
    print("="*80)
