"""
Classification metrics module for Decision Engine evaluation.
Computes Request Type Accuracy, Product Area Accuracy, and Status Accuracy.
"""

from __future__ import annotations

from typing import Any, Dict, List


def compute_classification_metrics(success_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute Decision Engine classification accuracy metrics across successful prediction records.
    Only successful predictions with valid ground truth contribute to accuracy calculations.
    """
    if not success_results:
        return {
            "request_type_accuracy": 0.0,
            "product_area_accuracy": 0.0,
            "status_accuracy": 0.0,
        }

    req_valid = [r for r in success_results if r.get("norm_expected_request_type")]
    prod_valid = [r for r in success_results if r.get("norm_expected_product_area")]
    status_valid = [r for r in success_results if r.get("norm_expected_status")]

    req_acc = (sum(1 for r in req_valid if r.get("req_match", False)) / len(req_valid)) if req_valid else 1.0
    prod_acc = (sum(1 for r in prod_valid if r.get("prod_match", False)) / len(prod_valid)) if prod_valid else 1.0
    status_acc = (sum(1 for r in status_valid if r.get("status_match", False)) / len(status_valid)) if status_valid else 1.0

    return {
        "request_type_accuracy": round(req_acc, 4),
        "product_area_accuracy": round(prod_acc, 4),
        "status_accuracy": round(status_acc, 4),
    }
