import argparse
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(ROOT, "tests", "evaluation_reports")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.evolution_proof import (  # noqa: E402
    generate_prompt_evolution_cases,
    run_prompt_evolution_proof,
    write_jsonl,
    write_report,
)
from evoagent.ab_report import timestamped_run_directory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an auditable feedback-driven prompt evolution replay."
    )
    parser.add_argument("--dataset", default="")
    args = parser.parse_args()
    run_dir = timestamped_run_directory(
        REPORT_DIR, "prompt-evolution-feedback-gates"
    )
    dataset_path = args.dataset or os.path.join(
        run_dir, "prompt-evolution-cases.jsonl"
    )
    if not args.dataset:
        write_jsonl(generate_prompt_evolution_cases(), dataset_path)
    database_path = os.path.join(run_dir, "prompt-evolution-proof.db")
    if os.path.exists(database_path):
        raise SystemExit(
            "proof database already exists in the fresh evaluation run directory"
        )
    report = run_prompt_evolution_proof(dataset_path, database_path)
    paths = write_report(report, run_dir)
    print("decision:", report["evolution_run"]["decision"])
    print("run_id:", report["evolution_run"]["run_id"])
    print("run directory:", run_dir)
    print("json:", paths["json"])
    print("markdown:", paths["markdown"])


if __name__ == "__main__":
    main()
