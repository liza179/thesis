"""
NLP Dataset Creation Module

Creates a dataset linking monthly regression dates with historical monetary
policy documents from the translated/ directory.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import warnings
import re


def parse_document_filename(filename: str) -> Optional[datetime]:
    """
    Parse YY_MM.md filename into a datetime object (last day of month).

    Args:
        filename: Document filename in format "YY_MM.md" (e.g., "16_06.md")

    Returns:
        datetime object representing the last day of that month, or None if invalid

    Note:
        All YY values are interpreted as 2000+. This skips 99_*.md files (1999).

    Examples:
        "16_06.md" -> datetime(2016, 6, 30)
        "00_02.md" -> datetime(2000, 2, 29)  # leap year
    """
    # Match YY_MM.md pattern
    match = re.match(r'^(\d{2})_(\d{2})\.md$', filename)
    if not match:
        return None

    yy_str, mm_str = match.groups()
    yy, mm = int(yy_str), int(mm_str)

    # Validate month range
    if not (1 <= mm <= 12):
        warnings.warn(f"Invalid month in {filename}: {mm}")
        return None

    # Interpret year as 2000+
    year = 2000 + yy

    # Get last day of month: first day of next month - 1 day
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
    """
    Load all markdown documents from the translated directory.

    Args:
        doc_dir: Directory containing YY_MM.md files (default: "translated")

    Returns:
        Dictionary mapping datetime objects to document text content
    """
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

        # Parse filename to datetime
        doc_date = parse_document_filename(filename)
        if doc_date is None:
            skipped_parse += 1
            continue

        # Skip empty files
        if file_path.stat().st_size == 0:
            skipped_empty += 1
            warnings.warn(f"Skipping empty file: {filename}")
            continue

        # Read document text
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
    offset: int,
    doc_dict: Dict[datetime, str]
) -> Tuple[List[datetime], List[str]]:
    """
    Get N previous documents for a target date, with configurable offset.

    Args:
        target_date: The reference date (end of month)
        n_months: Number of documents to retrieve
        offset: Number of months to offset backward from target_date
                offset=0: Include target month
                offset=1: Start from 1 month before target
        doc_dict: Dictionary of {datetime: text} from load_documents()

    Returns:
        Tuple of (doc_dates, doc_texts):
            - doc_dates: List of datetime objects (chronologically ordered)
            - doc_texts: List of corresponding document texts
    """
    # Calculate effective reference date after offset
    effective_date = target_date - relativedelta(months=offset)

    # Generate list of N expected document dates going backwards
    expected_dates = []
    for i in range(n_months):
        candidate_date = effective_date - relativedelta(months=i)
        expected_dates.append(candidate_date)

    # Sort chronologically (oldest first)
    expected_dates.sort()

    # Match expected dates with available documents
    doc_dates = []
    doc_texts = []

    for expected in expected_dates:
        if expected in doc_dict:
            doc_dates.append(expected)
            doc_texts.append(doc_dict[expected])

    return doc_dates, doc_texts


def create_nlp_dataset(
    csv_path: str = "merged_data.csv",
    doc_dir: str = "translated",
    n_months: int = 6,
    offset: int = 1,
    start_date: str = "2016-01-31"
) -> pd.DataFrame:
    """
    Create dataset linking regression dates with historical documents.

    Args:
        csv_path: Path to merged_data.csv with regression data
        doc_dir: Directory containing translated markdown files
        n_months: Number of previous documents to retrieve per date
        offset: Month offset to avoid data leakage (0=include current, 1=exclude)
        start_date: Filter dates >= this value (matches regression.py default)

    Returns:
        DataFrame with columns:
            - date: Target date (datetime)
            - n_docs_found: Number of documents successfully matched (int)
            - doc_dates: List of matched document dates (List[datetime])
            - doc_texts: List of matched document texts (List[str])
            - missing_months: Number of requested docs not found (int)
    """
    # Load documents
    print(f"Loading documents from {doc_dir}...")
    doc_dict = load_documents(doc_dir)

    if len(doc_dict) == 0:
        raise ValueError(f"No documents found in {doc_dir}")

    # Load CSV
    try:
        df = pd.read_csv(csv_path, parse_dates=['date'])
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {e}")

    if 'date' not in df.columns:
        raise ValueError("CSV must contain 'date' column")

    # Filter by start_date
    df_filtered = df[df['date'] >= start_date].copy().reset_index(drop=True)
    print(f"\nProcessing {len(df_filtered)} dates from {df_filtered['date'].min().date()} to {df_filtered['date'].max().date()}")

    # Match documents for each date
    results = []
    for _, row in df_filtered.iterrows():
        target_date = row['date']
        doc_dates, doc_texts = get_documents_for_date(target_date, n_months, offset, doc_dict)

        results.append({
            'date': target_date,
            'n_docs_found': len(doc_dates),
            'doc_dates': doc_dates,
            'doc_texts': doc_texts,
            'missing_months': n_months - len(doc_dates)
        })

    df_result = pd.DataFrame(results)

    # Print coverage statistics
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
