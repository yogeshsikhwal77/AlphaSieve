
# src/model.py

import pandas as pd
import numpy as np
import xgboost as xgb
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score


class MLFilter:
    """
    Machine Learning filter to evaluate rule-based signals.

    Predicts the probability of a trade being profitable and
    filters out low-confidence signals.
    """

    def __init__(self, model_type="xgboost", random_state=42):
        self.model_type = model_type
        self.random_state = random_state

        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                random_state=self.random_state,
                eval_metric="logloss"
            )

        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state
            )

        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            )

        else:
            raise ValueError(
                "Unsupported model_type. Choose "
                "'xgboost', 'random_forest', or "
                "'logistic_regression'."
            )

    @staticmethod
    def create_target(
        df: pd.DataFrame,
        lookahead_days: int = 5,
        return_threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Creates a binary target variable.

        Target = 1 if the forward return over lookahead_days
        exceeds return_threshold; otherwise, Target = 0.
        """
        df = df.copy()

        # Calculate forward returns
        df["Forward_Return"] = (
            df["Close"]
            .pct_change(lookahead_days)
            .shift(-lookahead_days)
        )

        # Remove rows without a valid forward return
        df.dropna(subset=["Forward_Return"], inplace=True)

        # Define the binary target
        df["Target"] = (
            df["Forward_Return"] > return_threshold
        ).astype(int)

        return df

    def walk_forward_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> pd.Series:
        """
        Performs walk-forward validation to generate
        out-of-fold probability predictions.
        """
        n_samples = len(X)

        # Safety check for insufficient samples
        if n_samples < 2:
            print(
                f"Warning: Only {n_samples} valid samples available. "
                "Cannot train ML model."
            )
            return pd.Series(0.5, index=X.index)

        # Dynamically reduce the number of splits
        actual_splits = min(n_splits, n_samples - 1)

        tscv = TimeSeriesSplit(n_splits=actual_splits)

        oof_predictions = pd.Series(
            index=X.index,
            dtype=float
        )

        for train_index, test_index in tscv.split(X):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]

            y_train = y.iloc[train_index]

            # Train the model on historical data
            self.model.fit(X_train, y_train)

            # Predict probability of a profitable outcome
            probs = self.model.predict_proba(X_test)[:, 1]

            oof_predictions.iloc[test_index] = probs

        return oof_predictions

    def train_and_filter(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        signal_col: str,
        prob_threshold: float = 0.6
    ) -> pd.DataFrame:
        """
        Trains the selected model and filters rule-based signals.

        Buy signals (1) are accepted only when the ML probability
        meets prob_threshold. Sell signals (-1) pass through so
        the ML filter does not block risk-off exits.
        """

        # Ensure that the target variable exists
        if "Target" not in df.columns:
            df = self.create_target(df)

        # Remove rows with missing features or targets
        train_df = df.dropna(
            subset=feature_cols + ["Target"]
        ).copy()

        X = train_df[feature_cols]
        y = train_df["Target"]

        # Generate out-of-fold predictions
        print(
            f"Running walk-forward validation "
            f"using {self.model_type}..."
        )

        train_df["ML_Probability"] = (
            self.walk_forward_validation(X, y)
        )

        # Evaluate model performance
        eval_df = train_df.dropna(
            subset=["ML_Probability"]
        )

        if not eval_df.empty:
            preds = (
                eval_df["ML_Probability"] > 0.5
            ).astype(int)

            print("\n--- ML Filter Performance ---")

            print(
                f"Accuracy: "
                f"{accuracy_score(eval_df['Target'], preds):.4f}"
            )

            print(
                f"Precision: "
                f"{precision_score(eval_df['Target'], preds, zero_division=0):.4f}"
            )

        # Initialize filtered signals
        train_df["Filtered_Signal"] = 0

        # Accept Buy signals only above the probability threshold
        buy_condition = (
            (train_df[signal_col] == 1)
            & (
                train_df["ML_Probability"]
                >= prob_threshold
            )
        )

        train_df.loc[buy_condition, "Filtered_Signal"] = 1

        # Preserve Sell signals for risk-off exits
        sell_condition = train_df[signal_col] == -1

        train_df.loc[sell_condition, "Filtered_Signal"] = -1

        return train_df

    def get_shap_values(self, X: pd.DataFrame):
        """
        Generates SHAP values to explain feature contributions.

        Must be called after the model has been fitted.
        Returns the explainer and SHAP values.
        """

        if self.model_type == "xgboost":
            # Tree-based explanation for XGBoost
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)

        elif self.model_type == "random_forest":
            # Tree-based explanation for Random Forest
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)

            # Select the positive class for binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif getattr(shap_values, "ndim", 0) == 3:
                shap_values = shap_values[:, :, 1]

        elif self.model_type == "logistic_regression":
            # Linear explanation for Logistic Regression
            explainer = shap.LinearExplainer(
                self.model,
                X
            )

            shap_values = explainer.shap_values(X)

        else:
            raise ValueError(
                f"Unsupported model_type: {self.model_type}"
            )

        return explainer, shap_values
