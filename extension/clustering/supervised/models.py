"""Training-only residual regression using observable price and wording features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

FEATURES = ["price", "price2", "price3", "words", "characters", "numbers", "negation", "question"]
KINDS = ["ridge", "forest", "boosting"]
SCOPES = ["pooled", "family", "cluster"]


def validate_splits(frame: pd.DataFrame) -> None:
    for column, message in [("event_id", "Event leakage"), ("ticker", "Contract leakage")]:
        if frame.groupby(column).split.nunique().gt(1).any():
            raise ValueError(message)
    if not frame.split.isin(["train", "validation", "test"]).all():
        raise ValueError("Unknown split")


def combination_mask(contracts: pd.DataFrame) -> pd.Series:
    """Exclude known prefix combinations and explicit multi-award conjunctions.

    The inherited MVE prefix rule misses KXOSCARWINNERS combinations. The
    additional wording rule uses proposition structure, never outcomes/prices.
    It is conservative and is not a complete semantic combination detector.
    """
    return contracts.ticker.str.startswith(("KXMVE", "MVE")) | contracts.title.str.contains(
        r"\bwin [^?]*,[^?]* at the Oscars\b", case=False, regex=True, na=False
    )


def text_features(contracts: pd.DataFrame) -> pd.DataFrame:
    text = contracts.title.fillna("") + " " + contracts.yes_sub_title.fillna("")
    return pd.DataFrame(
        {
            "ticker": contracts.ticker,
            "words": text.str.split().str.len(),
            "characters": text.str.len(),
            "numbers": text.str.count(r"\d+(?:\.\d+)?"),
            "negation": text.str.lower().str.count(r"\b(?:no|not|never|without)\b"),
            "question": text.str.count(r"\?"),
        }
    )


def design(frame: pd.DataFrame) -> np.ndarray:
    p = frame.p.to_numpy(dtype=float) / 100
    return np.column_stack([p, p**2, p**3, frame[["words", "characters", "numbers", "negation", "question"]]])


class ResidualModel:
    """Fit outcome-minus-price, then return clipped conditional YES probabilities."""

    def __init__(self, kind: str, alpha: float = 10):
        self.kind = kind
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> ResidualModel:
        self.center = np.average(x, axis=0, weights=weights)
        self.scale = np.sqrt(np.average((x - self.center) ** 2, axis=0, weights=weights))
        self.scale[self.scale < 1e-10] = 1
        if self.kind == "ridge":
            self.model = Ridge(alpha=self.alpha)
        elif self.kind == "forest":
            self.model = RandomForestRegressor(
                n_estimators=60, max_depth=6, min_samples_leaf=20, bootstrap=False, random_state=340, n_jobs=2
            )
        elif self.kind == "boosting":
            self.model = HistGradientBoostingRegressor(
                max_iter=80,
                max_leaf_nodes=7,
                min_samples_leaf=30,
                l2_regularization=10,
                early_stopping=False,
                random_state=340,
            )
        else:
            raise ValueError(self.kind)
        self.model.fit((x - self.center) / self.scale, y - x[:, 0], sample_weight=weights / np.mean(weights))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x[:, 0] + self.model.predict((x - self.center) / self.scale), 0, 1)
