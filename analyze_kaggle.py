import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# running this mostly on servers / notebooks without displays
matplotlib.use("Agg")

log = logging.getLogger("ddos_scan")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


FLOW_FEATURES = [
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


@dataclass
class DetectorConfig:
    model_path: Path = Path("isoforest_model.pkl")
    csv_path: Path = Path("unbalaced_20_80_dataset.csv")
    row_limit: int = 200000
    out_dir: Path = Path("output")


class DDoSAnomalyDetector:
    def __init__(self, config: DetectorConfig):
        self.config = config

        self.model = None
        self.data = None
        self.scaler = None
        self.features = None
        self.flagged_rows = None

        if not self.config.out_dir.exists():
            self.config.out_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        path = self.config.model_path
        if not path.exists():
            raise FileNotFoundError(f"could not find model file at {path}")

        log.info("loading model from %s", path)
        model = joblib.load(path)

        if not callable(getattr(model, "predict", None)):
            raise ValueError("loaded file is not something sklearn-like")

        self.model = model

    def load_csv(self):
        log.info("reading CSV: %s", self.config.csv_path)

        read_args = {"low_memory": False}
        if self.config.row_limit:
            read_args["nrows"] = self.config.row_limit

        df = pd.read_csv(self.config.csv_path, **read_args)

        if len(df) == 0:
            raise ValueError("CSV read succeeded but no rows were returned")

        log.info("rows loaded: %s", f"{len(df):,}")
        self.data = df

    def check_columns(self):
        if self.data is None:
            raise RuntimeError("dataframe missing — CSV probably never loaded")

        missing = []
        for col in FLOW_FEATURES:
            if col not in self.data.columns:
                missing.append(col)

        if missing:
            raise ValueError(f"dataset is missing required columns: {missing}")

        log.info("feature column check passed")

    def prepare_features(self):
        if self.data is None:
            raise RuntimeError("no data available for feature prep")

        raw = self.data[FLOW_FEATURES].copy()

        raw.replace([np.inf, -np.inf], np.nan, inplace=True)
        raw.fillna(0, inplace=True)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(raw)

        # FIX: Convert to DataFrame to remove sklearn warnings
        self.features = pd.DataFrame(scaled, columns=FLOW_FEATURES)

        self.scaler = scaler

    def detect_anomalies(self):
        if self.model is None:
            raise RuntimeError("model not loaded")
        if self.features is None:
            raise RuntimeError("features not prepared")
        if self.data is None:
            raise RuntimeError("no data to annotate")

        log.info("running anomaly detection")

        preds = self.model.predict(self.features)
        scores = self.model.decision_function(self.features)

        self.data["prediction"] = preds
        self.data["anomaly_score"] = scores
        self.data["prediction_label"] = [
            "ddos_anomaly" if v == -1 else "normal" for v in preds
        ]

        self.flagged_rows = self.data[self.data["prediction"] == -1]

        ratio = len(self.flagged_rows) / len(self.data)
        log.info(
            "flagged %s flows (%.1f%%)",
            f"{len(self.flagged_rows):,}",
            ratio * 100,
        )

    def save_outputs(self):
        if self.flagged_rows is None or len(self.flagged_rows) == 0:
            log.warning("no anomalies found — skipping CSV output")
            return

        out_csv = self.config.out_dir / "detected_anomalies.csv"
        self.flagged_rows.to_csv(out_csv, index=False)

        log.info("wrote anomaly CSV to %s", out_csv)

    def plot_scores(self):
        if self.data is None or "anomaly_score" not in self.data:
            log.warning("no anomaly scores available for plotting")
            return

        scores = self.data["anomaly_score"]

        plt.figure(figsize=(10, 5))
        plt.plot(scores, label="anomaly score")

        avg = scores.mean()
        plt.axhline(avg, linestyle="--", label="mean")

        plt.legend()
        plt.tight_layout()

        out_path = self.config.out_dir / "anomaly_score_plot.png"
        plt.savefig(out_path)
        plt.close()

        log.info("saved anomaly score plot")

    def generate_report(self):
        if self.flagged_rows is None:
            raise RuntimeError("no results available for report generation")

        sample = self.flagged_rows.head(20).to_html(index=False)

        html = f"""
<html>
  <head>
    <title>DDoS detection run</title>
    <style>
      body {{ font-family: sans-serif; margin: 32px; }}
      table {{ border-collapse: collapse; }}
      th, td {{ border: 1px solid #aaa; padding: 4px 6px; }}
    </style>
  </head>
  <body>
    <h1>DDoS anomaly detection</h1>
    <p>Total rows scanned: {len(self.data)}</p>
    <p>Anomalies detected: {len(self.flagged_rows)}</p>

    <h2>Example anomalies</h2>
    {sample}

    <h2>Anomaly score plot</h2>
    <img src="anomaly_score_plot.png" width="85%">
  </body>
</html>
"""

        out_html = self.config.out_dir / "ddos_kaggle_report.html"
        out_html.write_text(html, encoding="utf-8")

        log.info("HTML report generated")

    def run(self):
        self.load_model()
        self.load_csv()
        self.check_columns()
        self.prepare_features()
        self.detect_anomalies()
        self.save_outputs()
        self.plot_scores()
        self.generate_report()

        log.info("analysis finished")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Scan a CSV for DDoS-like anomalies")

    parser.add_argument("--model", default="isoforest_model.pkl")
    parser.add_argument("--input", default="unbalaced_20_80_dataset.csv")
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--out", default=".")

    args = parser.parse_args(argv)

    if args.max_rows == 0:
        args.max_rows = None

    return args


def main(argv=None):
    args = parse_arguments(argv)

    if args.model == args.input:
        raise SystemExit("model path and input CSV should not be the same file")

    config = DetectorConfig(
        model_path=Path(args.model),
        csv_path=Path(args.input),
        row_limit=args.max_rows,
        out_dir=Path(args.out).resolve(),
    )

    detector = DDoSAnomalyDetector(config)
    detector.run()


if __name__ == "__main__":
    main()
