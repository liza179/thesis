"""
Stage 1: NLP Model for Inflation Prediction

Trains a Ridge regression model that maps document embeddings to inflation targets.
The model output (nlp_pred) is then used as a feature in the final ensemble.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.cross_decomposition import PLSRegression, CCA
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from nlp_dataset import create_nlp_dataset
from vector import get_model, encode_with_cache
from regression import gaussian_rank_correlation, bootstrap_metrics


def embed_documents(df_nlp, model=None, cache_file='.embedding_cache.pkl', verbose=True):
    """
    Generate embeddings for all documents in the dataset with persistent caching.

    Args:
        df_nlp: DataFrame from create_nlp_dataset() with doc_texts and doc_dates columns
        model: BGE-M3 model (if None, will load)
        cache_file: Path to pickle file for persistent cache (None to disable)
        verbose: Show progress bar

    Returns:
        Tuple of (DataFrame with added 'doc_embeddings' column, embedding_cache dict mapping date -> embedding)

    Note:
        Uses persistent disk cache in vector.py to avoid re-embedding.
    """
    if model is None:
        print("Loading BGE-M3 model...")
        model = get_model(static=True)

    df_nlp = df_nlp.copy()

    # Collect all unique (date, text) pairs
    unique_docs = {}
    for _, row in df_nlp.iterrows():
        for date, text in zip(row['doc_dates'], row['doc_texts']):
            if date not in unique_docs:
                unique_docs[date] = text

    if verbose:
        print(f"Embedding {len(unique_docs)} unique documents (with caching)...")

    # Embed all unique documents (uses cache from vector.py)
    embedding_cache = {}
    iterator = tqdm(unique_docs.items(), desc="Processing documents") if verbose else unique_docs.items()
    for date, text in iterator:
        emb = encode_with_cache(text, model, cache_file=cache_file)
        embedding_cache[date] = emb

    # Build embeddings list for each row
    doc_embeddings_list = []
    for _, row in df_nlp.iterrows():
        embeddings = []
        for date in row['doc_dates']:
            embeddings.append(embedding_cache[date])
        doc_embeddings_list.append(embeddings)

    df_nlp['doc_embeddings'] = doc_embeddings_list
    return df_nlp, embedding_cache


def extract_embedding_pairs(embedding_cache, max_dt=6, verbose=False):
    """
    Extract (embedding[t], embedding[t+dt]) pairs from embedding cache in chronological order.
    Creates pairs for dt in 1...max_dt.
    
    Args:
        embedding_cache: Dict mapping date -> embedding (512-dim array)
        max_dt: Maximum time step difference (default 6)
        verbose: Print debug info
        
    Returns:
        X_pairs: Array of shape (n_pairs, 512) - embedding[t]
        Y_pairs: Array of shape (n_pairs, 512) - embedding[t+dt]
    """
    sorted_dates = sorted(embedding_cache.keys())
    
    if verbose:
        print(f"Debug extract_embedding_pairs: {len(sorted_dates)} dates in cache")
        if sorted_dates:
            print(f"Debug: Date range: {sorted_dates[0]} to {sorted_dates[-1]}")
    
    if len(sorted_dates) < max_dt + 1:
        if verbose:
            print(f"Debug: Not enough dates ({len(sorted_dates)}), need at least {max_dt + 1}, returning empty arrays")
        return np.array([]).reshape(0, 512), np.array([]).reshape(0, 512)
    
    X_pairs = []
    Y_pairs = []
    
    for dt in range(1, max_dt + 1):
        for i in range(len(sorted_dates) - dt):
            X_pairs.append(embedding_cache[sorted_dates[i]])
            Y_pairs.append(embedding_cache[sorted_dates[i + dt]])
    
    result_X = np.array(X_pairs)
    result_Y = np.array(Y_pairs)
    
    if verbose:
        print(f"Debug: Created {len(X_pairs)} pairs (dt=1..{max_dt}), shapes: X={result_X.shape}, Y={result_Y.shape}")
    
    return result_X, result_Y


def train_cca_projection(embedding_cache, n_components=50, max_dt=6, verbose=True):
    """
    Train CCA projection on (embedding[t], embedding[t+dt]) pairs from cache.
    Creates pairs for dt in 1...max_dt.
    
    Args:
        embedding_cache: Dict mapping date -> embedding
        n_components: Number of CCA components to use
        max_dt: Maximum time step difference (default 6)
        verbose: Print info
        
    Returns:
        CCA model fitted on the pairs
    """
    if verbose:
        print(f"Extracting embedding pairs for CCA from cache (chronological order, dt=1..{max_dt})...")
    
    X_pairs, Y_pairs = extract_embedding_pairs(embedding_cache, max_dt=max_dt, verbose=verbose)
    
    assert len(X_pairs) > 0, f"No embedding pairs found in cache (need at least 2 dates). Cache has {len(embedding_cache)} dates."
    assert X_pairs.ndim == 2, f"X_pairs should be 2D, got shape {X_pairs.shape}"
    assert Y_pairs.ndim == 2, f"Y_pairs should be 2D, got shape {Y_pairs.shape}"
    
    if verbose:
        print(f"Found {len(X_pairs)} embedding pairs")
        print(f"Training CCA with {n_components} components...")
    
    n_components = min(n_components, len(X_pairs), X_pairs.shape[1])
    cca = CCA(n_components=n_components, scale=False)
    cca.fit(X_pairs, Y_pairs)
    
    if verbose:
        print(f"CCA correlation: {cca.score(X_pairs, Y_pairs):.4f}")
    
    return cca


def aggregate_embeddings(doc_embeddings, method='concat', max_docs=None, projection=None):
    """
    Aggregate multiple document embeddings into a single vector.

    Args:
        doc_embeddings: List of embedding arrays (each 512-dim)
        method: 'concat' to concatenate all embeddings
        max_docs: Maximum number of documents (for padding). If None, uses length of doc_embeddings.
        projection: Optional CCA projection to apply before concatenation

    Returns:
        Concatenated embedding vector (max_docs * projected_dim or max_docs * 512-dim array, padded with zeros if needed)
    """
    if len(doc_embeddings) == 0:
        if max_docs is None:
            max_docs = 0
        if projection is not None:
            dim = projection.n_components
        else:
            dim = 512
        return np.zeros(max_docs * dim)

    if method == 'concat':
        if max_docs is None:
            max_docs = len(doc_embeddings)
        
        embeddings_array = np.stack(doc_embeddings, axis=0)  # shape: (n_docs, 512)
        
        if projection is not None:
            embeddings_array = projection.transform(embeddings_array)  # shape: (n_docs, n_components)
            dim = projection.n_components
        else:
            dim = 512
        
        concatenated = embeddings_array.flatten()  # shape: (n_docs * dim,)
        
        if len(doc_embeddings) < max_docs:
            padding = np.zeros((max_docs - len(doc_embeddings)) * dim)
            concatenated = np.concatenate([concatenated, padding])
        
        return concatenated
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def train_nlp_model(
    df_nlp,
    df_regression,
    embedding_cache=None,
    model_type='ridge',
    ridge_alpha=1.0,
    pls_n_components=1,
    use_cca=False,
    cca_components=50,
    verbose=True
):
    """
    Train Stage 1 NLP model: embeddings → target.

    Args:
        df_nlp: DataFrame with 'doc_embeddings' column
        df_regression: DataFrame with 'target' column (from reconstruct_model_data)
        embedding_cache: Dict mapping date -> embedding (required if use_cca=True)
        model_type: 'ridge', 'bayesian_ridge', or 'pls'
        ridge_alpha: Ridge regularization strength (only for model_type='ridge')
        pls_n_components: Number of components for PLSRegression (only for model_type='pls')
        use_cca: If True, apply CCA projection learned from (embedding[t] -> embedding[t+1]) pairs
        cca_components: Number of CCA components to use
        verbose: Print training info

    Returns:
        Dictionary with:
            - model: Trained model
            - train_pred: Predictions on train set
            - test_pred: Predictions on test set
            - train_indices: Train split indices
            - test_indices: Test split indices
            - metrics: R², Gaussian correlation, bootstrap CIs
            - model_type: Type of model used
    """
    # Merge datasets on date
    df_merged = df_nlp.merge(df_regression[['date', 'target']], on='date', how='inner')

    # Sort by date to ensure chronological order
    df_merged = df_merged.sort_values('date').reset_index(drop=True)

    if verbose:
        print(f"\nMerged dataset: {len(df_merged)} rows")
        print(f"Date range: {df_merged['date'].min().date()} to {df_merged['date'].max().date()}")
        print(f"Documents coverage: {df_merged['n_docs_found'].mean():.2f} avg docs per row")

    # Aggregate embeddings for each row (before CCA, to determine split)
    max_docs = max(len(row['doc_embeddings']) for _, row in df_merged.iterrows())
    if verbose:
        print(f"Maximum documents per row: {max_docs}")

    # Chronological 50/50 split (first half train, second half test)
    split_idx = int(len(df_merged) * 0.5)
    train_indices = np.arange(split_idx)
    test_indices = np.arange(split_idx, len(df_merged))
    
    df_train = df_merged.iloc[train_indices].copy()
    df_test = df_merged.iloc[test_indices].copy()

    # Train CCA projection if requested (only on training dates - chronologically)
    cca_projection = None
    if use_cca:
        if embedding_cache is None:
            raise ValueError("embedding_cache is required when use_cca=True")
        
        max_train_date = df_train['date'].max()
        train_dates_set = set(df_train['date'].values)
        
        if verbose:
            print(f"Debug: Total embedding_cache size: {len(embedding_cache)}")
            print(f"Debug: Training dates in df_train: {len(train_dates_set)}")
            print(f"Debug: Max train date: {max_train_date}")
            cache_dates = sorted(embedding_cache.keys())
            print(f"Debug: Cache date range: {cache_dates[0] if cache_dates else 'empty'} to {cache_dates[-1] if cache_dates else 'empty'}")
        
        train_cache = {date: emb for date, emb in embedding_cache.items() if date <= max_train_date}
        
        if verbose:
            print(f"Debug: Dates in train_cache (after filtering by max_train_date): {len(train_cache)}")
            train_cache_dates = sorted(train_cache.keys())
            if train_cache_dates:
                print(f"Debug: Train cache date range: {train_cache_dates[0]} to {train_cache_dates[-1]}")
        
        assert len(train_cache) >= 2, f"Not enough dates in training cache for CCA: {len(train_cache)} dates (need at least 2). Total cache: {len(embedding_cache)}, train dates: {len(train_dates_set)}, max_train_date: {max_train_date}"
        
        cca_projection = train_cca_projection(train_cache, n_components=cca_components, verbose=verbose)
        if verbose:
            print(f"Using CCA projection: {cca_components} components")

    # Aggregate embeddings for each row
    if verbose:
        if use_cca:
            print("Concatenating CCA-projected embeddings...")
        else:
            print("Concatenating embeddings...")

    aggregated_embeddings = []
    for _, row in df_merged.iterrows():
        emb = aggregate_embeddings(row['doc_embeddings'], method='concat', max_docs=max_docs, projection=cca_projection)
        aggregated_embeddings.append(emb)

    X = np.stack(aggregated_embeddings, axis=0)
    y = df_merged['target'].values

    if verbose:
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target shape: {y.shape}")

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    if verbose:
        train_dates = df_merged.iloc[train_indices]['date']
        test_dates = df_merged.iloc[test_indices]['date']
        print(f"\nChronological split: {len(train_indices)}/{len(test_indices)} samples")
        print(f"  Train period: {train_dates.min().date()} to {train_dates.max().date()}")
        print(f"  Test period: {test_dates.min().date()} to {test_dates.max().date()}")

        if model_type == 'bayesian_ridge':
            print(f"Training BayesianRidge model...")
        elif model_type == 'pls':
            print(f"Training PLSRegression model (n_components={pls_n_components})...")
        else:
            print(f"Training Ridge model (alpha={ridge_alpha})...")

    # Train model
    if model_type == 'bayesian_ridge':
        model = BayesianRidge(fit_intercept=False)
    elif model_type == 'ridge':
        model = Ridge(alpha=ridge_alpha, random_state=42, fit_intercept=False)
    elif model_type == 'pls':
        model = PLSRegression(n_components=pls_n_components, scale=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

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
        'model_type': model_type,
        'ridge_alpha': ridge_alpha if model_type == 'ridge' else None,
        'pls_n_components': pls_n_components if model_type == 'pls' else None,
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


def print_nlp_results(results_list):
    """Print formatted results table for NLP models."""
    console = Console()

    print("\n" + "="*80)
    print("STAGE 1: NLP MODEL RESULTS")
    print("="*80)
    print(f"\nInput: Concatenated embeddings (all documents per row)")
    print(f"Train/Test split: 50/50 chronological (first half train, second half test)")

    table = Table(show_header=True, header_style="bold magenta", title="\nPerformance Metrics")
    table.add_column("Model", style="cyan")
    table.add_column("Train R²", justify="right")
    table.add_column("Test R²", justify="right")
    table.add_column("Train Gauss Corr", justify="right")
    table.add_column("Test Gauss Corr", justify="right")

    for name, nlp_results in results_list:
        metrics = nlp_results['metrics']

        train_r2_ci = f"{metrics['train_r2']:.3f} [{metrics['train_bootstrap']['r2_ci_low']:.3f}, {metrics['train_bootstrap']['r2_ci_high']:.3f}]"
        test_r2_ci = f"{metrics['test_r2']:.3f} [{metrics['test_bootstrap']['r2_ci_low']:.3f}, {metrics['test_bootstrap']['r2_ci_high']:.3f}]"
        train_gauss_ci = f"{metrics['train_gauss_corr']:.3f} [{metrics['train_bootstrap']['gauss_corr_ci_low']:.3f}, {metrics['train_bootstrap']['gauss_corr_ci_high']:.3f}]"
        test_gauss_ci = f"{metrics['test_gauss_corr']:.3f} [{metrics['test_bootstrap']['gauss_corr_ci_low']:.3f}, {metrics['test_bootstrap']['gauss_corr_ci_high']:.3f}]"

        table.add_row(name, train_r2_ci, test_r2_ci, train_gauss_ci, test_gauss_ci)

    console.print(table)
    print("="*80)


if __name__ == "__main__":
    from regression import reconstruct_model_data

    print("="*80)
    print("STAGE 1: NLP MODEL TRAINING")
    print("="*80)

    # Load data
    print("\n1. Loading documents...")
    df_nlp = create_nlp_dataset(n_months=6, min_before=10, max_before=18)

    print("\n2. Loading regression data...")
    df_regression = reconstruct_model_data()

    # Generate embeddings (only once, reuse for all models)
    print("\n3. Generating embeddings...")
    df_nlp, embedding_cache = embed_documents(df_nlp, verbose=True)

    # Train multiple models
    print("\n4. Training NLP models...")

    models_to_test = [
        ('Ridge α=1e-2', {'model_type': 'ridge', 'ridge_alpha': 1e-2}),
        ('Ridge α=1e-1', {'model_type': 'ridge', 'ridge_alpha': 1e-1}),
        ('Ridge α=1', {'model_type': 'ridge', 'ridge_alpha': 1.0}),
        ('Ridge α=10', {'model_type': 'ridge', 'ridge_alpha': 10.0}),
        ('Ridge α=200', {'model_type': 'ridge', 'ridge_alpha': 200.0}),
        ('Ridge α=1e8', {'model_type': 'ridge', 'ridge_alpha': 1e8}),
        # ('BayesianRidge', {'model_type': 'bayesian_ridge'}),
        ('PLS k=1', {'model_type': 'pls', 'pls_n_components': 1}),
        ('PLS k=2', {'model_type': 'pls', 'pls_n_components': 2}),
        ('PLS k=3', {'model_type': 'pls', 'pls_n_components': 3}),
        ('Ridge+CCA k=1', {'model_type': 'ridge', 'ridge_alpha': 1e4, 'use_cca': True, 'cca_components': 1}),
        ('Ridge+CCA k=2', {'model_type': 'ridge', 'ridge_alpha': 1e4, 'use_cca': True, 'cca_components': 2}),
        ('Ridge+CCA k=3', {'model_type': 'ridge', 'ridge_alpha': 1e4, 'use_cca': True, 'cca_components': 3}),
        ('Ridge+CCA k=4', {'model_type': 'ridge', 'ridge_alpha': 1e4, 'use_cca': True, 'cca_components': 4}),
        ('Ridge+CCA k=5', {'model_type': 'ridge', 'ridge_alpha': 1e4, 'use_cca': True, 'cca_components': 5}),
        ('Ridge+CCA k=6', {'model_type': 'ridge', 'ridge_alpha': 1e4, 'use_cca': True, 'cca_components': 6}),
        ('PLS k=1+CCA k=50', {'model_type': 'pls', 'pls_n_components': 1, 'use_cca': True, 'cca_components': 1}),
        ('PLS k=2+CCA k=50', {'model_type': 'pls', 'pls_n_components': 1, 'use_cca': True, 'cca_components': 2}),
        ('PLS k=3+CCA k=50', {'model_type': 'pls', 'pls_n_components': 1, 'use_cca': True, 'cca_components': 3}),
    ]

    results_list = []
    for name, params in models_to_test:
        print(f"\nTraining {name}...")
        result = train_nlp_model(df_nlp, df_regression, embedding_cache=embedding_cache, verbose=False, **params)
        results_list.append((name, result))

    # Print comparison
    print_nlp_results(results_list)

    # Save predictions from best model (let's use Ridge α=200 by default)
    print("\n5. Saving predictions (using Ridge α=200)...")
    best_result = [r for n, r in results_list if n == 'Ridge α=200'][0]
    df_with_nlp = save_predictions_to_csv(
        best_result,
        df_regression,
        output_path='merged_data_with_nlp.csv'
    )

    print("\n" + "="*80)
    print("Stage 1 complete! Use 'merged_data_with_nlp.csv' in regression.py")
    print("="*80)
