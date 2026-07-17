"""
Dataset loader for the SupportSphere AI evaluation framework.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple, Union

import pandas as pd

from evaluation.models import EvaluationSample

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Handles loading, validating, and converting ground truth support ticket CSV files.
    """

    def __init__(self, samples: List[EvaluationSample]) -> None:
        """
        Initialize the loader with a pre-validated list of evaluation samples.
        """
        self._samples = samples

    @property
    def samples(self) -> List[EvaluationSample]:
        """
        Get the list of loaded evaluation samples.
        """
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EvaluationSample:
        return self._samples[idx]

    def get_batches(self, batch_size: int) -> Generator[List[EvaluationSample], None, None]:
        """
        Yields batches of evaluation samples.

        Args:
            batch_size: Number of samples per batch.

        Yields:
            List of EvaluationSample objects.
        """
        for i in range(0, len(self._samples), batch_size):
            yield self._samples[i : i + batch_size]

    def train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[DatasetLoader, DatasetLoader]:
        """
        Splits the samples into train and test subsets (future-proofing).

        Args:
            test_size: Proportion of the dataset to include in the test split.
            random_state: Random state seed.

        Returns:
            Tuple of (train_loader, test_loader).
        """
        import random
        
        random.seed(random_state)
        shuffled = list(self._samples)
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * (1 - test_size))
        train_samples = shuffled[:split_idx]
        test_samples = shuffled[split_idx:]
        
        logger.info(
            f"Split dataset of size {len(self._samples)} into "
            f"train size {len(train_samples)} and test size {len(test_samples)}."
        )
        return DatasetLoader(train_samples), DatasetLoader(test_samples)

    @classmethod
    def from_csv(cls, filepath: Union[str, Path]) -> DatasetLoader:
        """
        Load support ticket records from a CSV file.

        Args:
            filepath: Path to the CSV file.

        Returns:
            An initialized DatasetLoader containing the validated samples.

        Raises:
            ValueError: If the file is missing or required columns cannot be parsed.
        """
        path = Path(filepath)
        if not path.exists():
            error_msg = f"Dataset file not found at: {path.absolute()}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Loading evaluation dataset from CSV: {path.name}")
        try:
            # Load the CSV, let pandas automatically handle malformed files and encodings
            df = pd.read_csv(path)
        except Exception as e:
            error_msg = f"Failed to parse CSV dataset: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        # Normalize column headers to lowercase and strip whitespaces
        column_map: Dict[str, str] = {str(col).lower().strip().replace(" ", "_"): str(col) for col in df.columns}
        
        # We check for a valid issue/query column
        issue_col = next((column_map[k] for k in ["issue", "query", "question"] if k in column_map), None)
        if not issue_col:
            error_msg = "Required 'issue' (or query/question) column is missing from the dataset schema."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Map optional/standard columns with fallbacks
        subject_col = next((column_map[k] for k in ["subject", "title"] if k in column_map), None)
        company_col = next((column_map[k] for k in ["company", "organization"] if k in column_map), None)
        response_col = next((column_map[k] for k in ["response", "expected_response", "ground_truth", "answer"] if k in column_map), None)
        req_type_col = next((column_map[k] for k in ["request_type", "expected_request_type", "type"] if k in column_map), None)
        prod_area_col = next((column_map[k] for k in ["product_area", "expected_product_area", "area"] if k in column_map), None)
        status_col = next((column_map[k] for k in ["status", "expected_status"] if k in column_map), None)

        samples: List[EvaluationSample] = []
        for idx, row in df.iterrows():
            issue = str(row[issue_col]).strip() if pd.notna(row[issue_col]) else ""
            if not issue:
                logger.warning(f"Row {idx} is missing issue description; skipping.")
                continue

            subject = str(row[subject_col]).strip() if subject_col and pd.notna(row[subject_col]) else ""
            company = str(row[company_col]).strip() if company_col and pd.notna(row[company_col]) else "None"
            # Explicitly match raw "None" strings or empty cells to "None"
            if company == "" or company.lower() == "nan" or company.lower() == "none":
                company = "None"
                
            expected_response = str(row[response_col]).strip() if response_col and pd.notna(row[response_col]) else ""
            expected_request_type = str(row[req_type_col]).strip() if req_type_col and pd.notna(row[req_type_col]) else "invalid"
            expected_product_area = str(row[prod_area_col]).strip() if prod_area_col and pd.notna(row[prod_area_col]) else ""
            expected_status = str(row[status_col]).strip() if status_col and pd.notna(row[status_col]) else "Replied"

            # Retain other columns as metadata dictionary
            metadata = {col: row[col] for col in df.columns if col not in [issue_col, subject_col, company_col, response_col, req_type_col, prod_area_col, status_col]}

            samples.append(
                EvaluationSample(
                    issue=issue,
                    subject=subject,
                    company=company,
                    expected_response=expected_response,
                    expected_request_type=expected_request_type,
                    expected_product_area=expected_product_area,
                    expected_status=expected_status,
                    metadata=metadata,
                )
            )

        logger.info(f"Successfully processed {len(samples)} valid evaluation samples from dataset.")
        return cls(samples)
