import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# basic logging setup — not overthinking this
# if this gets more complex later, we can revisit
logger = logging.getLogger("iso_train")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# using a dataclass here mostly for readability;
# constants would also work, but this felt clearer at the time
@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int = 150
    contamination: float = 0.25
    seed: int = 42


# NOTE:
# These feature names are tightly coupled to the dataset schema.
FEATURE_COLUMNS = [
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Max",
    "Fwd Pkt Len Mean",
    "Bwd Pkt Len Max",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Init Bwd Win Byts",
]



def build_isolation_forest(cfg: ModelConfig):
    # wrapped in a function so it's easy to tweak params later
    # without hunting through the code
    return IsolationForest(
        n_estimators=cfg.n_estimators,
        contamination=cfg.contamination,
        random_state=cfg.seed,
        n_jobs=-1,  # might as well use all available cores
    )


class KaggleIsolationTrainer:
    def __init__(self, training_csv, model_path="isoforest_model.pkl", config=None):
        self.training_csv = Path(training_csv)
        self.model_path = Path(model_path)
        self.config = config or ModelConfig()

        # these get filled in as we go
        self.df = None
        self.X = None
        self.model = None

    def _ensure_dataframe(self):
        if self.df is None or self.df.empty:
            raise ValueError("training data not loaded (or it's empty)")
        return self.df

    def _ensure_features(self):
        if self.X is None:
            self.prepare_features()

        # defensive check — should never really trigger
        if self.X is None:
            raise RuntimeError("feature matrix is still missing somehow")

        return self.X

    @staticmethod
    def _check_expected_columns(df):
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns: {missing}")

    @staticmethod
    def _maybe_print_label_stats(df):
        # helpful when labels exist, harmless when they don't
        if "Label" in df.columns:
            logger.info("label distribution:")
            print(df["Label"].value_counts())

    def load_data(self):
        if not self.training_csv.exists():
            raise FileNotFoundError(str(self.training_csv))

        logger.info("loading training data from %s", self.training_csv)
        df = pd.read_csv(self.training_csv)

        if df.empty:
            raise ValueError("CSV loaded correctly but contains no rows")

        self._check_expected_columns(df)
        self._maybe_print_label_stats(df)

        self.df = df

    def prepare_features(self):
        df = self._ensure_dataframe()

        # pull out only the columns we care about
        X = df[FEATURE_COLUMNS].copy()

        # quick cleanup — this dataset loves infinities and NaNs
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)

        self.X = X

    def fit(self):
        X = self._ensure_features()

        logger.info("training using the following features:")
        for col in FEATURE_COLUMNS:
            print(f"  * {col}")

        logger.info("fitting Isolation Forest model")
        model = build_isolation_forest(self.config)
        model.fit(X)

        self.model = model

    def save_model(self):
        if self.model is None:
            raise ValueError("can't save model before training")

        with self.model_path.open("wb") as f:
            pickle.dump(self.model, f)

        logger.info("model saved to %s", self.model_path)

    def evaluate_on_training_data(self):
        if self.model is None:
            raise ValueError("model must be trained before evaluation")

        df = self._ensure_dataframe()
        X = self._ensure_features()

        logger.info("running model on training data (sanity check)")
        df["score"] = self.model.decision_function(X)
        df["prediction"] = self.model.predict(X)

        logger.info("prediction counts:")
        print(df["prediction"].value_counts())

    def run(self):
        # intentionally linear — easier to debug when something goes wrong
        self.load_data()

        # feature prep is lazy, but fitting expects it anyway
        self.fit()

        # save early so we don’t lose a good model if eval blows up
        self.save_model()

        # mostly a sanity check, not a real evaluation
        self.evaluate_on_training_data()

        logger.info("done — model trained and written to disk")


def parse_arguments(argv=None):
    # argparse boilerplate, yeah — but it works and it's readable
    parser = argparse.ArgumentParser(
        description=(
            "Train an Isolation Forest model on a CSV file. "
            "Assumes the dataset already has the expected feature columns."
        )
    )

    parser.add_argument(
        "--training-file",
        default="training_dataset.csv",
        help=(
            "CSV file used for training. Defaults to training_dataset.csv "
            "because that's what I kept naming it locally."
        ),
    )

    parser.add_argument(
        "--model-path",
        default="isoforest_model.pkl",
        help=(
            "Where to save the trained model. "
            "Will overwrite the file if it already exists."
        ),
    )

    # argv handling is mostly here to make testing less annoying
    if argv is not None:
        return parser.parse_args(list(argv))

    return parser.parse_args()


def main(argv=None):
    args = parse_arguments(argv)

    # keeping this explicit instead of clever
    trainer = KaggleIsolationTrainer(
        args.training_file,
        args.model_path,
    )

    try:
        trainer.run()
    except Exception as exc:
        # not doing anything fancy here on purpose
        logger.error("training failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
