"""Run the fixed 10-case self-reflection vs blind-challenger diagnostic."""
import argparse
import json
import os
import sys
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.adaptive import artifact_fingerprint
from evoagent.config import Settings
from evoagent.evaluation_harness import load_jsonl
from evoagent.multi_agent_ablation import (
    AB_BUDGET, canned_arm_reviewers, canned_domain_arm_reviewers,
    fairness_manifest, model_arm_reviewers, model_domain_arm_reviewers,
    render_markdown, run_paired_ablation,
)
from evoagent.reviewer import PrimaryReviewAgent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("offline", "real"), default="offline")
    parser.add_argument(
        "--experiment", choices=("challenger", "domains"), default="challenger",
        help="Isolate Challenger behavior or conditional Security/Reliability investigators",
    )
    parser.add_argument("--dataset", default=os.path.join(ROOT, "evaluation_data", "multi_agent_ablation_10.jsonl"))
    parser.add_argument("--output", default=os.path.join(ROOT, "reports", "multi_agent_ablation.json"))
    parser.add_argument("--markdown", default=os.path.join(ROOT, "reports", "multi_agent_ablation.md"))
    parser.add_argument("--checkpoint", default=os.path.join(ROOT, "reports", "multi_agent_ablation.checkpoint.json"))
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--case-ids", default="", help="Comma-separated diagnostic case IDs; use cases 3-6 for the mini gate")
    parser.add_argument("--resume-controls", default="", help="Reuse controls from an existing report when re-rendering a matching checkpoint")
    args = parser.parse_args()

    cases = load_jsonl(args.dataset)
    if args.case_ids:
        selected = {item.strip() for item in args.case_ids.split(",") if item.strip()}
        cases = [item for item in cases if item["id"] in selected]
        if {item["id"] for item in cases} != selected:
            raise ValueError("one or more --case-ids are not present in the dataset")
    if len(cases) not in {4, 10, 100}:
        raise ValueError("multi-agent ablation requires the 4-case mini gate, 10-case diagnostic, or 100-case controlled benchmark")
    ruleset_hash = artifact_fingerprint({
        "scanners": ["SecurityRuleReviewer:1", "ReliabilityRuleReviewer:1", "ContextRuleReviewer:1"]
    })
    if args.mode == "real":
        settings = Settings.from_env()
        config = settings.resolved_llm()
        if not config:
            raise ValueError("real mode requires a configured LLM provider")
        fallback = settings.resolved_llm_fallback()
        probe = PrimaryReviewAgent(
            config["base_url"], config["api_key"], config["model"],
            provider=config.get("provider", "custom"), extra_headers=config.get("headers") or {},
            disable_thinking=str(config.get("model", "")).lower().startswith("qwen3.7"),
        )
        builder = model_domain_arm_reviewers if args.experiment == "domains" else model_arm_reviewers
        builder_kwargs = {} if args.experiment == "domains" else {"include_conditional": True}
        reviewers = builder(
            config, AB_BUDGET, args.timeout, fallback,
            settings.llm_request_timeout_seconds, **builder_kwargs,
        )
        fallback_reviewers = (
            builder(
                fallback, AB_BUDGET, args.timeout,
                request_timeout_seconds=settings.llm_request_timeout_seconds,
                **builder_kwargs,
            )
            if fallback else None
        )
        model, prompt_hash = config["model"], probe.prompt_hash
    else:
        reviewers = (
            canned_domain_arm_reviewers(AB_BUDGET) if args.experiment == "domains"
            else canned_arm_reviewers(AB_BUDGET, include_conditional=True)
        )
        fallback_reviewers = None
        model, prompt_hash = "canned-v1", artifact_fingerprint({"prompt": "canned-primary-v1"})
    source_hash = artifact_fingerprint({
        path: Path(ROOT, path).read_text(encoding="utf-8")
        for path in ("evoagent/agents.py", "evoagent/adaptive.py", "evoagent/reviewer.py",
                     "evoagent/multi_agent_ablation.py")
    })
    controls = fairness_manifest(
        cases, model, prompt_hash, ruleset_hash, AB_BUDGET,
        fallback if args.mode == "real" else None,
        source_hash,
    )
    controls["experiment"] = args.experiment
    if args.resume_controls:
        with open(args.resume_controls, "r", encoding="utf-8") as handle:
            restored = dict(json.load(handle)["experiment"]["controls"])
        if restored != controls:
            raise ValueError("resume controls do not match the current dataset/source/config")
    def progress(value):
        print(json.dumps({"event": "case_complete", **value}, ensure_ascii=False), flush=True)
    report = run_paired_ablation(
        cases, reviewers, controls, offline=args.mode == "offline",
        checkpoint_path=args.checkpoint, progress=progress,
        fallback_reviewers=fallback_reviewers,
        baseline_name=("conditional_primary" if args.experiment == "domains" else "single_self_reflect"),
        candidate_name=("conditional_domain_agents" if args.experiment == "domains" else "conditional_adaptive"),
    )
    report["experiment"]["config_fingerprint"] = artifact_fingerprint(controls)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with open(args.markdown, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
    print(json.dumps({
        "status": report["comparison"]["diagnostic_gate"]["status"],
        "config_fingerprint": report["experiment"]["config_fingerprint"],
        "json": os.path.abspath(args.output), "markdown": os.path.abspath(args.markdown),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
