"""Read a customer table out of a PDF so it can be scored like a CSV."""
import io

import pandas as pd
import pdfplumber

from src.features import CATEGORY_OPTIONS

NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def _squash(text: str) -> str:
    return "".join(text.split()).lower()


def _canonical_header(cell: str) -> str:
    # Column names never contain spaces, so any whitespace is a line wrap.
    return "".join((cell or "").split())


def read_pdf_table(source) -> pd.DataFrame:
    """source: a path, bytes, or file-like object. Multi-page tables are stitched together."""
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    header, rows = None, []
    with pdfplumber.open(source) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for raw in table:
                    if not any(c and c.strip() for c in raw):
                        continue
                    if header is None:
                        header = [_canonical_header(c) for c in raw]
                        continue
                    if [_canonical_header(c) for c in raw] == header:
                        continue  # header repeated on a new page
                    rows.append([" ".join((c or "").split()) for c in raw])
    if header is None:
        raise ValueError("No table found in the PDF. It needs a table with the customer columns.")

    df = pd.DataFrame(rows, columns=header)
    # Undo wraps inside values, e.g. "Month-to- month" -> "Month-to-month".
    for col, options in CATEGORY_OPTIONS.items():
        if col in df:
            lookup = {_squash(o): o for o in options}
            df[col] = df[col].map(lambda v: lookup.get(_squash(v), v))
    for col in NUMERIC_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col].str.replace(r"[$,\s]", "", regex=True), errors="coerce")
    return df.dropna(subset=[c for c in ("tenure", "MonthlyCharges") if c in df]).reset_index(drop=True)


def read_customer_file(name: str, data: bytes) -> pd.DataFrame:
    """Dispatch on file extension: .pdf or .csv."""
    if name.lower().endswith(".pdf"):
        return read_pdf_table(data)
    return pd.read_csv(io.BytesIO(data))
