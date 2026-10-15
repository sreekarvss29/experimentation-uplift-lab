"""Load and prepare the Hillstrom MineThatData e-mail experiment (64,000 customers).

Customers were randomized into three equal groups: Mens E-Mail, Womens E-Mail, No E-Mail.
Outcomes (two weeks after the campaign): visit, conversion, spend.
Pre-treatment features: recency, history (past-year spend), mens, womens, zip_code, newbie, channel.
"""
from pathlib import Path
import urllib.request

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/W-Tran/uplift-modelling/master/data/hillstrom/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "hillstrom.csv"

PRE_FEATURES = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL = ["zip_code", "channel"]


def load(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    df = pd.read_csv(path)
    expected = {"recency", "history", "mens", "womens", "zip_code", "newbie",
                "channel", "segment", "visit", "conversion", "spend"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    return df


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-treatment features only (safe to use for CUPED/CUPAC and uplift models)."""
    X = df[PRE_FEATURES].astype(float).copy()
    dummies = pd.get_dummies(df[CATEGORICAL], drop_first=False, dtype=float)
    return pd.concat([X, dummies], axis=1)


def two_arm(df: pd.DataFrame, treatment: str = "Mens E-Mail", control: str = "No E-Mail") -> pd.DataFrame:
    """Subset to one treatment arm vs control, with a binary `treated` column."""
    out = df[df["segment"].isin([treatment, control])].copy()
    out["treated"] = (out["segment"] == treatment).astype(int)
    return out.reset_index(drop=True)
