from src.config import ROOT
from src.data import clean, load_raw
from src.features import RAW_FEATURES
from src.pdf_tables import read_pdf_table

SAMPLE = ROOT / "data" / "sample" / "customers_sample.pdf"


def test_sample_pdf_round_trips_exactly():
    pdf = read_pdf_table(SAMPLE)
    assert len(pdf) == 60
    assert set(RAW_FEATURES) <= set(pdf.columns)
    orig = clean(load_raw()).set_index("customerID").loc[pdf["customerID"], RAW_FEATURES]
    got = clean(pdf).set_index("customerID")[RAW_FEATURES]
    assert (orig.astype(str) == got.astype(str)).all().all()
