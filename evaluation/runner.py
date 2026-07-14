"""
Entry point for SupportSphere evaluation.

Usage
-----
python -m evaluation.runner
"""

from __future__ import annotations
import sys
from pathlib import Path
import src.config as config
from evaluation.experiment import EvaluationExperiment

# Reconfigure stdout to use UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPORT_DIR = (Path(__file__).parent / "reports")

def get_report_file() -> Path:
    provider = config.LLM_PROVIDER.lower()
    model = config.LLM_MODEL.lower().replace("/", "_").replace("\\", "_")
    return REPORT_DIR / f"evaluation_results_{provider}_{model}.csv"

def print_summary(results):
    print("\n" + "=" * 80)
    print("SupportSphere Evaluation Report")
    print("=" * 80)
    
    # Print Table
    print(f"{'Ticket':<8}{'Request Type':<16}{'Product Area':<16}{'Status':<12}{'Correctness':<14}{'Faithfulness':<14}{'Pass':<6}")
    print("-" * 85)
    
    worst_score = float('inf')
    worst_ticket_idx = 1
    
    for idx, row in results.iterrows():
        t_id = idx + 1
        
        req_type_icon = "✅" if row["request_type_accuracy"] == 1 else "❌"
        prod_area_icon = "✅" if row["product_area_accuracy"] == 1 else "❌"
        status_icon = "✅" if row["status_accuracy"] == 1 else "❌"
        
        corr = row["answer_correctness"]
        faith = row["faithfulness"]
        
        # Pass logic
        is_pass = (
            row["request_type_accuracy"] == 1 and
            row["product_area_accuracy"] == 1 and
            row["status_accuracy"] == 1 and
            corr >= 0.8 and
            faith >= 0.8
        )
        pass_icon = "✅" if is_pass else "⚠️"
        
        print(f"{t_id:<8}{req_type_icon:<16}{prod_area_icon:<16}{status_icon:<12}{corr:<14.2f}{faith:<14.2f}{pass_icon:<6}")
        
        # Track worst ticket (prioritizing correctness)
        # Using a weighted score where correctness has a high impact
        score = (
            row["request_type_accuracy"] + 
            row["status_accuracy"] + 
            row["product_area_accuracy"]
        ) + corr * 1.5 + faith * 0.5
        
        if score < worst_score:
            worst_score = score
            worst_ticket_idx = t_id
            
    print("-" * 85)
    print("\nAccuracy\n")
    print(f"Request Type: {results['request_type_accuracy'].mean():.0%}\n")
    print(f"Product Area: {results['product_area_accuracy'].mean():.0%}\n")
    print(f"Status: {results['status_accuracy'].mean():.0%}\n")
    print(f"Average Faithfulness: {results['faithfulness'].mean():.2f}\n")
    print(f"Worst Ticket:\n#{worst_ticket_idx}")
    print("=" * 80)


def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True,)
    experiment = EvaluationExperiment()
    results = experiment.run()
    
    report_file = get_report_file()
    results.to_csv(report_file, index=False)
    print_summary(results)
    print(f"\nDetailed report saved to:\n{report_file}")


if __name__ == "__main__":
    main()
