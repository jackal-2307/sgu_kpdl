"""
Credit Card Customer Segmentation Model
========================================
Module đóng gói preprocessing pipeline + KMeans clustering model.

Usage:
    from segmentation_model import SegmentationModel
    
    # Training
    model = SegmentationModel(k=4, random_state=42)
    model.fit(df_raw)
    model.cluster_names = {0: "VIP", 1: "Low Activity", ...}
    model.save("model.joblib")
    
    # Inference
    model = SegmentationModel.load("model.joblib")
    predictions = model.predict(new_data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


class CreditPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessing pipeline reproducing notebook transformations:
    - Drop ID column
    - KNN impute MINIMUM_PAYMENTS
    - Winsorize amount columns (1%-99% quantiles from train)
    - Feature engineering (ratios, INSTALLMENT_SHARE)
    - log1p transform on skewed amount columns
    - Feature selection (drop redundant features)
    - Standard scaling (fit on train, apply on inference)
    """
    
    def __init__(
        self,
        id_col: str = "CUST_ID",
        winsor_cols: Optional[List[str]] = None,
        log_cols: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.id_col = id_col
        self.random_state = random_state

        self.winsor_cols = winsor_cols or [
            "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
            "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS",
        ]
        self.log_cols = log_cols or [
            "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
            "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS",
        ]

        # Learned artifacts (populated during fit)
        self._knn_imputer: Optional[KNNImputer] = None
        self._winsor_bounds: Dict[str, tuple] = {}
        self._purchase_per_trx_p99: Optional[float] = None
        self._features_to_drop: List[str] = []
        self._feature_names_: List[str] = []
        self._scaler: Optional[StandardScaler] = None
        self._credit_limit_median_: Optional[float] = None

    @property
    def feature_names_(self) -> List[str]:
        """Return final feature names after preprocessing."""
        return list(self._feature_names_)

    def fit(self, X: pd.DataFrame, y=None):
        """Learn preprocessing parameters from training data."""
        df = self._prep_df(X, fit_mode=True)
        self._scaler = StandardScaler()
        self._scaler.fit(df.values)
        self._feature_names_ = df.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply learned preprocessing to new data."""
        if self._scaler is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        df = self._prep_df(X, fit_mode=False)

        # Align columns (robust to missing columns at inference)
        for c in self._feature_names_:
            if c not in df.columns:
                df[c] = 0.0
        df = df[self._feature_names_]

        return self._scaler.transform(df.values)

    # ----------------------------
    # Internal preprocessing logic
    # ----------------------------
    def _prep_df(self, X: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        """
        Core preprocessing steps matching notebook workflow.
        
        Args:
            X: Raw input DataFrame
            fit_mode: If True, learn parameters; if False, apply learned params
        
        Returns:
            Preprocessed DataFrame ready for scaling
        """
        df = X.copy()

        # Drop ID if present
        if self.id_col in df.columns:
            df = df.drop(columns=[self.id_col])

        # Ensure numeric types
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # --- KNN Impute MINIMUM_PAYMENTS ---
        impute_features = [
            c for c in ["BALANCE", "CREDIT_LIMIT", "PAYMENTS", "PURCHASES", "CASH_ADVANCE", "MINIMUM_PAYMENTS"]
            if c in df.columns
        ]
        if "MINIMUM_PAYMENTS" in df.columns and df["MINIMUM_PAYMENTS"].isna().any() and impute_features:
            if fit_mode:
                self._knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
                imp = self._knn_imputer.fit_transform(df[impute_features])
            else:
                if self._knn_imputer is None:
                    raise RuntimeError("KNNImputer missing. Fit preprocessor first.")
                imp = self._knn_imputer.transform(df[impute_features])
            
            imp_df = pd.DataFrame(imp, columns=impute_features, index=df.index)
            df["MINIMUM_PAYMENTS"] = imp_df["MINIMUM_PAYMENTS"]

        # CREDIT_LIMIT median fill (rare missing values)
        if "CREDIT_LIMIT" in df.columns and df["CREDIT_LIMIT"].isna().any():
            if fit_mode:
                self._credit_limit_median_ = float(df["CREDIT_LIMIT"].median())
            median_val = self._credit_limit_median_ if self._credit_limit_median_ is not None else float(df["CREDIT_LIMIT"].median())
            df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(median_val)

        # --- Winsorization (1%-99% quantiles from train) ---
        for col in self.winsor_cols:
            if col not in df.columns:
                continue
            if fit_mode:
                p1 = float(df[col].quantile(0.01))
                p99 = float(df[col].quantile(0.99))
                self._winsor_bounds[col] = (p1, p99)
            
            p1, p99 = self._winsor_bounds.get(col, (None, None))
            if p1 is not None and p99 is not None:
                df[col] = df[col].clip(lower=p1, upper=p99)

        # --- Feature Engineering ---
        EPS = 1e-9

        # CREDIT_UTILIZATION = BALANCE / CREDIT_LIMIT
        if "CREDIT_LIMIT" in df.columns and "BALANCE" in df.columns:
            df["CREDIT_UTILIZATION"] = np.where(
                df["CREDIT_LIMIT"] > 0,
                df["BALANCE"] / (df["CREDIT_LIMIT"] + EPS),
                0.0
            )
            df["CREDIT_UTILIZATION"] = np.clip(df["CREDIT_UTILIZATION"], 0, 1.5)

        # PAYMENT_RATIO = PAYMENTS / MINIMUM_PAYMENTS
        if "MINIMUM_PAYMENTS" in df.columns and "PAYMENTS" in df.columns:
            df["PAYMENT_RATIO"] = np.where(
                df["MINIMUM_PAYMENTS"] > 0,
                df["PAYMENTS"] / (df["MINIMUM_PAYMENTS"] + EPS),
                np.where(df["PAYMENTS"] > 0, 5.0, 0.0),
            )
            df["PAYMENT_RATIO"] = np.clip(df["PAYMENT_RATIO"], 0, 10)

        # CASH_RATIO = CASH_ADVANCE / CREDIT_LIMIT
        if "CREDIT_LIMIT" in df.columns and "CASH_ADVANCE" in df.columns:
            df["CASH_RATIO"] = np.where(
                df["CREDIT_LIMIT"] > 0,
                df["CASH_ADVANCE"] / (df["CREDIT_LIMIT"] + EPS),
                0.0
            )
            df["CASH_RATIO"] = np.clip(df["CASH_RATIO"], 0, 1.5)

        # PURCHASE_PER_TRX (with p99 cap from train)
        if "PURCHASES" in df.columns and "PURCHASES_TRX" in df.columns:
            pptrx = np.where(
                df["PURCHASES_TRX"] > 0,
                df["PURCHASES"] / (df["PURCHASES_TRX"] + EPS),
                0.0
            )
            if fit_mode:
                self._purchase_per_trx_p99 = float(pd.Series(pptrx).quantile(0.99))
            cap = self._purchase_per_trx_p99 if self._purchase_per_trx_p99 is not None else float(pd.Series(pptrx).quantile(0.99))
            df["PURCHASE_PER_TRX"] = np.clip(pptrx, 0, cap)

        # BALANCE_GROWTH_RATE (kept for consistency, will be dropped later)
        if all(c in df.columns for c in ["BALANCE", "PURCHASES", "CASH_ADVANCE", "PAYMENTS"]):
            bgr = np.zeros(len(df), dtype=float)
            mask = (df["BALANCE"].fillna(0) > 0).values
            if mask.any():
                bgr[mask] = (
                    (df.loc[mask, "PURCHASES"] + df.loc[mask, "CASH_ADVANCE"] - df.loc[mask, "PAYMENTS"]) 
                    / (df.loc[mask, "BALANCE"] + EPS)
                ).values
            df["BALANCE_GROWTH_RATE"] = np.clip(bgr, -2, 2)

        # INSTALLMENT_SHARE
        if "ONEOFF_PURCHASES" in df.columns and "INSTALLMENTS_PURCHASES" in df.columns:
            denom = df["ONEOFF_PURCHASES"] + df["INSTALLMENTS_PURCHASES"] + EPS
            df["INSTALLMENT_SHARE"] = (df["INSTALLMENTS_PURCHASES"] / denom).clip(0, 1)
        else:
            df["INSTALLMENT_SHARE"] = 0.0

        # --- Log1p transform on amount columns (after winsorization) ---
        for col in self.log_cols:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))

        # --- Feature Selection (drop redundant features) ---
        if fit_mode:
            features_to_drop: List[str] = []

            # Redundant purchases (total vs components)
            if "PURCHASES" in df.columns and all(c in df.columns for c in ["ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES"]):
                features_to_drop.append("PURCHASES")

            # Marketing-related redundant features
            marketing_redundant = [
                "PURCHASES_TRX",
                "CASH_ADVANCE_TRX",
                "ONEOFF_PURCHASES_FREQUENCY",
                "PURCHASES_INSTALLMENTS_FREQUENCY",
                "BALANCE_FREQUENCY",
                "BALANCE_GROWTH_RATE",
            ]
            features_to_drop.extend(marketing_redundant)

            # Drop raw features when ratio exists
            if "CREDIT_UTILIZATION" in df.columns and "BALANCE" in df.columns:
                features_to_drop.append("BALANCE")

            if "CASH_RATIO" in df.columns and "CASH_ADVANCE" in df.columns:
                features_to_drop.append("CASH_ADVANCE")

            if "PAYMENT_RATIO" in df.columns and "MINIMUM_PAYMENTS" in df.columns:
                features_to_drop.append("MINIMUM_PAYMENTS")

            # Keep only existing and unique features
            seen = set()
            self._features_to_drop = [
                c for c in features_to_drop 
                if (c in df.columns) and not (c in seen or seen.add(c))
            ]

        # Apply feature drop
        drop_cols = [c for c in self._features_to_drop if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # Final cleaning
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df = df.select_dtypes(include=[np.number]).astype(float)

        return df


@dataclass
class SegmentationModel:
    """
    Complete segmentation model with preprocessing + KMeans clustering.
    
    Attributes:
        k: Number of clusters
        random_state: Random seed for reproducibility
        n_init: Number of KMeans initializations
        cluster_names: Optional dict mapping cluster_id -> persona name
    """
    k: int = 4
    random_state: int = 42
    n_init: int = 50

    def __post_init__(self):
        self.preprocessor = CreditPreprocessor(random_state=self.random_state)
        self.kmeans = KMeans(
            n_clusters=self.k,
            init="k-means++",
            n_init=self.n_init,
            max_iter=300,
            random_state=self.random_state,
        )
        self.cluster_names: Optional[Dict[int, str]] = None

    def fit(self, df_raw: pd.DataFrame) -> "SegmentationModel":
        """
        Fit preprocessing pipeline and KMeans model on raw data.
        
        Args:
            df_raw: Raw DataFrame with original features
        
        Returns:
            self (fitted model)
        """
        X = self.preprocessor.fit_transform(df_raw)
        self.kmeans.fit(X)
        return self

    def predict(self, df_raw: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            df_raw: Raw DataFrame with same structure as training data
        
        Returns:
            Array of cluster labels (0 to k-1)
        """
        X = self.preprocessor.transform(df_raw)
        return self.kmeans.predict(X)

    def get_cluster_names(self) -> Dict[int, str]:
        """
        Return cluster names mapping.
        
        Returns:
            Dict mapping cluster_id -> persona name
        """
        if self.cluster_names is None:
            return {i: f"Cluster {i}" for i in range(self.k)}
        return self.cluster_names

    def save(self, path: str) -> None:
        """
        Save complete model to disk.
        
        Args:
            path: Path to save .joblib file
        """
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "SegmentationModel":
        """
        Load model from disk.
        
        Args:
            path: Path to .joblib file
        
        Returns:
            Loaded SegmentationModel instance
        """
        obj = joblib.load(path)
        if not isinstance(obj, SegmentationModel):
            raise TypeError(f"Loaded object is not SegmentationModel, got {type(obj)}")
        return obj
