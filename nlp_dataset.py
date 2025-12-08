from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import warnings
import re


def parse_document_filename(filename: str) -> Optional[datetime]:
    match = re.match(r'^(\d{2})_(\d{2})\.md$', filename)
    if not match:
        return None

    yy_str, mm_str = match.groups()
    yy, mm = int(yy_str), int(mm_str)

    if not (1 <= mm <= 12):
        warnings.warn(f"Invalid month in {filename}: {mm}")
        return None

    year = 2000 + yy

    try:
        if mm == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, mm + 1, 1)
        last_day = next_month - timedelta(days=1)
        return last_day
    except ValueError as e:
        warnings.warn(f"Invalid date in {filename}: {e}")
        return None


def load_documents(doc_dir: str = "translated") -> Dict[datetime, str]:
    doc_path = Path(doc_dir)

    if not doc_path.exists():
        raise FileNotFoundError(f"Document directory not found: {doc_dir}")

    if not doc_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {doc_dir}")

    doc_dict = {}
    skipped_empty = 0
    skipped_parse = 0

    for file_path in doc_path.glob("*.md"):
        filename = file_path.name

        doc_date = parse_document_filename(filename)
        if doc_date is None:
            skipped_parse += 1
            continue

        if file_path.stat().st_size == 0:
            skipped_empty += 1
            warnings.warn(f"Skipping empty file: {filename}")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            warnings.warn(f"Encoding error in {filename}, trying latin-1")
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()

        doc_dict[doc_date] = text

    print(f"Loaded {len(doc_dict)} documents from {doc_dir}")
    if skipped_empty > 0:
        print(f"  Skipped {skipped_empty} empty files")
    if skipped_parse > 0:
        print(f"  Skipped {skipped_parse} files (couldn't parse filename)")

    if len(doc_dict) > 0:
        print(f"  Date range: {min(doc_dict.keys()).date()} to {max(doc_dict.keys()).date()}")

    return doc_dict


def get_documents_for_date(
    target_date: datetime,
    n_months: int,
    min_before: int,
    max_before: int,
    doc_dict: Dict[datetime, str]
) -> Tuple[List[datetime], List[str]]:
    min_date = target_date - relativedelta(months=max_before)
    max_date = target_date - relativedelta(months=min_before)

    available_docs = []
    for doc_date, doc_text in doc_dict.items():
        if min_date <= doc_date <= max_date:
            available_docs.append((doc_date, doc_text))

    available_docs.sort(key=lambda x: x[0], reverse=True)
    selected = available_docs[:n_months]
    selected.sort(key=lambda x: x[0])

    doc_dates = [d[0] for d in selected]
    doc_texts = [d[1] for d in selected]

    return doc_dates, doc_texts


def create_nlp_dataset(
    csv_path: str = "merged_data.csv",
    doc_dir: str = "translated",
    n_months: int = 6,
    min_before: int = 1,
    max_before: int = 12,
    start_date: str = "2016-01-31"
) -> pd.DataFrame:
    print(f"Creating NLP dataset with {n_months} months lookback, min_before={min_before}, max_before={max_before}")
    print(f"Loading documents from {doc_dir}...")
    doc_dict = load_documents(doc_dir)

    if len(doc_dict) == 0:
        raise ValueError(f"No documents found in {doc_dir}")

    try:
        df = pd.read_csv(csv_path, parse_dates=['date'])
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {e}")

    if 'date' not in df.columns:
        raise ValueError("CSV must contain 'date' column")

    df_filtered = df[df['date'] >= start_date].copy().reset_index(drop=True)
    print(f"\nProcessing {len(df_filtered)} dates from {df_filtered['date'].min().date()} to {df_filtered['date'].max().date()}")

    results = []
    for _, row in df_filtered.iterrows():
        target_date = row['date']
        doc_dates, doc_texts = get_documents_for_date(target_date, n_months, min_before, max_before, doc_dict)

        results.append({
            'date': target_date,
            'n_docs_found': len(doc_dates),
            'doc_dates': doc_dates,
            'doc_texts': doc_texts,
            'missing_months': n_months - len(doc_dates)
        })

    df_result = pd.DataFrame(results)

    total_requested = len(df_result) * n_months
    total_found = df_result['n_docs_found'].sum()
    coverage_pct = 100 * total_found / total_requested if total_requested > 0 else 0

    print(f"\nDocument matching statistics:")
    print(f"  Total requested: {total_requested} documents")
    print(f"  Total found: {total_found} documents")
    print(f"  Coverage: {coverage_pct:.1f}%")
    print(f"  Rows with full coverage ({n_months} docs): {(df_result['missing_months'] == 0).sum()}")
    print(f"  Rows with partial coverage: {(df_result['missing_months'] > 0).sum()}")
    print(f"  Average docs per row: {df_result['n_docs_found'].mean():.2f}")

    if (df_result['n_docs_found'] == 0).sum() > 0:
        print(f"  WARNING: {(df_result['n_docs_found'] == 0).sum()} rows have no matched documents")

    return df_result
