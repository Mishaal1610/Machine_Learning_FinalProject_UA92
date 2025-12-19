import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


# basic logger, good enough for a data prep script
log = logging.getLogger("data_prep")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


@dataclass
class SamplingPlan:
    # numbers below are mostly empirical — tuned after a few trial runs
    target_benign: int = 50_000
    target_ddos: int = 20_000

    # chunking params to avoid blowing up memory on big CSVs
    chunk_size: int = 200_000
    start_row: int = 500_000
    max_rows: int = 5_000_000

    seed: int = 42


@dataclass
class TrainingDataCollector:
    source_csv: Path = Path("data.csv")
    output_csv: Path = Path("out.csv")
    plan: SamplingPlan = field(default_factory=SamplingPlan)

    # these get filled incrementally during the scan
    benign_parts: list = field(default_factory=list)
    ddos_parts: list = field(default_factory=list)

    final_df = None

    def scan(self):
        if not self.source_csv.exists():
            raise FileNotFoundError(str(self.source_csv))

        log.info("scanning source CSV: %s", self.source_csv)
        log.info(
            "target counts -> benign=%s | ddos=%s",
            self.plan.target_benign,
            self.plan.target_ddos,
        )

        benign_seen = 0
        ddos_seen = 0

        # walking the file in chunks instead of reading everything at once
        for offset in range(
            self.plan.start_row,
            self.plan.max_rows,
            self.plan.chunk_size,
        ):
            try:
                chunk = pd.read_csv(
                    self.source_csv,
                    skiprows=range(1, offset),  # ugly but effective
                    nrows=self.plan.chunk_size,
                    low_memory=False,
                )
            except Exception as err:
                log.warning("read failed around row %d: %s", offset, err)
                break

            if chunk.empty:
                log.info("no more data to read, stopping")
                break

            # normalizing label values — they’re not always consistent
            label_col = "Label"
            chunk[label_col] = chunk[label_col].astype(str).str.lower().str.strip()

            benign_chunk = chunk[chunk[label_col] == "benign"]
            ddos_chunk = chunk[chunk[label_col] == "ddos"]

            self.benign_parts.append(benign_chunk)
            self.ddos_parts.append(ddos_chunk)

            benign_seen += len(benign_chunk)
            ddos_seen += len(ddos_chunk)

            log.info(
                "progress -> benign=%d | ddos=%d",
                benign_seen,
                ddos_seen,
            )

            # stop early once we’ve got enough of both
            if (
                benign_seen >= self.plan.target_benign
                and ddos_seen >= self.plan.target_ddos
            ):
                log.info("enough samples collected, exiting scan loop")
                break

        if benign_seen < self.plan.target_benign or ddos_seen < self.plan.target_ddos:
            raise RuntimeError(
                "not enough samples collected — "
                "try increasing max_rows or lowering targets"
            )

    def build(self):
        # concatenating everything we gathered during the scan
        benign_df = pd.concat(self.benign_parts, ignore_index=True)
        ddos_df = pd.concat(self.ddos_parts, ignore_index=True)

        # sampling down to exact targets to keep things balanced
        benign_sample = benign_df.sample(
            self.plan.target_benign,
            random_state=self.plan.seed,
        )
        ddos_sample = ddos_df.sample(
            self.plan.target_ddos,
            random_state=self.plan.seed,
        )

        combined = pd.concat(
            [benign_sample, ddos_sample],
            ignore_index=True,
        )

        # shuffle once more so labels aren’t grouped
        self.final_df = combined.sample(
            frac=1.0, random_state=self.plan.seed
        ).reset_index(drop=True)

    def save(self):
        if self.final_df is None:
            raise RuntimeError("no dataset built yet — did you forget to call build()?")

        destination = self.output_csv
        destination.parent.mkdir(parents=True, exist_ok=True)

        self.final_df.to_csv(destination, index=False)
        log.info("saved dataset to %s", destination)

        # quick sanity check on label balance
        if "Label" in self.final_df.columns:
            log.info("final label distribution:")
            log.info(self.final_df["Label"].value_counts().to_string())

    def run(self):
        # simple, linear flow — easier to debug if something goes wrong
        self.scan()
        self.build()
        self.save()


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare a roughly balanced training dataset (no fancy tricks)."
    )

    parser.add_argument(
        "--source",
        default="unbalaced_20_80_dataset.csv",
        help="Path to the large source CSV file.",
    )
    parser.add_argument(
        "--output",
        default="training_dataset.csv",
        help="Where to write the sampled training dataset.",
    )
    parser.add_argument(
        "--target-benign",
        type=int,
        default=50_000,
        help="How many benign rows to keep.",
    )
    parser.add_argument(
        "--target-ddos",
        type=int,
        default=20_000,
        help="How many DDoS rows to keep.",
    )

    if argv is not None:
        return parser.parse_args(list(argv))

    return parser.parse_args()


def main(argv=None):
    args = parse_arguments(argv)

    plan = SamplingPlan(
        target_benign=args.target_benign,
        target_ddos=args.target_ddos,
    )

    collector = TrainingDataCollector(
        source_csv=Path(args.source),
        output_csv=Path(args.output),
        plan=plan,
    )

    collector.run()


if __name__ == "__main__":
    main()
