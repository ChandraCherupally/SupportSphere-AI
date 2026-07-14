"""
Dataset loader for SupportSphere AI evaluation.

Loads the ground-truth support tickets that will be used
for LangSmith evaluation.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd


DATASET_PATH = (
    Path(__file__).parent.parent
    / "support_tickets"
    / "sample_support_tickets_ground_truth.csv"
)


class EvaluationDataset:
    """Loads the evaluation dataset."""

    def __init__(self) -> None:
        self.df = pd.read_csv(DATASET_PATH)

    def load(self) -> list[dict]:
        """Return all evaluation examples."""

        examples = []

        for _, row in self.df.iterrows():

            examples.append(
                {
                    # ------------------------
                    # Input
                    # ------------------------
                    "issue": str(row["Issue"]).strip(),
                    "subject": str(row["Subject"]).strip(),
                    "company": str(row["Company"]).strip(),

                    # ------------------------
                    # Ground Truth
                    # ------------------------
                    "expected_request_type": str(
                        row["Request Type"]
                    ).strip(),

                    "expected_product_area": str(
                        row["Product Area"]
                    ).strip(),

                    "expected_status": str(
                        row["Status"]
                    ).strip(),

                    "expected_response": str(
                        row["Response"]
                    ).strip(),
                }
            )

        return examples


if __name__ == "__main__":
    dataset = EvaluationDataset()
    examples = dataset.load()
    print(f"Loaded {len(examples)} evaluation examples.")
    print(examples[0])