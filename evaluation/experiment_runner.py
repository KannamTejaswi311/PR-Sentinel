"""
PR-Sentinel: Central Experiment Runner
Executes reproducible experimental runs for E1-E6 and ablations (E6-A1 to E6-A4)
over the 100 benchmark PR diffs, saving immutable raw predictions to:
evaluation/results/frozen_protocol_v1/raw/{config_id}/{pr_id}.json
"""

import argparse
import csv
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import uuid

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sentinel.pipeline import PRSentinelPipeline
from sentinel.schemas import (
    ExecutionStatus,
    FindingCategory,
    FindingGranularity,
    ModelFinding,
    PipelineConfig,
    SeverityLevel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExperimentRunner")

DEFAULT_DIFFS_DIR = Path("evaluation/diffs")
DEFAULT_MANIFEST_PATH = Path("evaluation/diff_manifest.csv")
DEFAULT_RESULTS_DIR = Path("evaluation/results/frozen_protocol_v1/raw")

ALL_EXPERIMENT_CONFIGS = [
    "E1",       # Monolithic Prompting
    "E2",       # Static Analysis
    "E3",       # CodeBERT Classifier
    "E4",       # RAG + LLM Monolithic
    "E5",       # Multi-Agent Specialized Review
    "E6",       # Full PR-Sentinel System
    "E6_A1",    # Ablation: minus CodeBERT
    "E6_A2",    # Ablation: minus Static Analysis
    "E6_A3",    # Ablation: minus RAG
    "E6_A4",    # Ablation: minus Multi-Agent (Monolithic LLM)
]


def extract_model_findings(
    review_result: Dict[str, Any],
    pr_id: str,
    exp_id: str,
) -> List[ModelFinding]:
    """Convert pipeline review result findings into canonical ModelFinding objects."""
    findings: List[ModelFinding] = []
    raw_findings = review_result.get("summary", {}).get("findings", [])

    # If E3 only or if findings are in codebert_analysis:
    if not raw_findings and review_result.get("codebert_analysis", {}).get("findings"):
        raw_findings = review_result["codebert_analysis"]["findings"]

    # If static only (E2):
    if not raw_findings and review_result.get("static_findings"):
        raw_findings = review_result["static_findings"]

    for item in raw_findings:
        if isinstance(item, ModelFinding):
            if item.execution_status in (
                ExecutionStatus.TOOL_ERROR.value,
                ExecutionStatus.SKIPPED.value,
                ExecutionStatus.UNSUPPORTED_LANGUAGE.value,
            ) or item.rule_id == "LANG_UNSUPPORTED":
                continue
            findings.append(item)
            continue

        if item.get("execution_status") in (
            ExecutionStatus.TOOL_ERROR.value,
            ExecutionStatus.SKIPPED.value,
            ExecutionStatus.UNSUPPORTED_LANGUAGE.value,
        ) or item.get("rule_id") == "LANG_UNSUPPORTED":
            continue

        file_path = item.get("file") or item.get("file_path") or "unknown.py"
        line_start = item.get("line") or item.get("line_start") or 1
        line_end = item.get("line_end") or line_start

        raw_cat = str(item.get("category", "UNKNOWN")).upper()
        std_cat = FindingCategory.UNKNOWN.value
        for c in FindingCategory:
            if c.value in raw_cat or raw_cat in c.value:
                std_cat = c.value
                break

        granularity = item.get("granularity", FindingGranularity.STATEMENT.value)
        if exp_id == "E3":
            granularity = FindingGranularity.HUNK_SPAN.value

        findings.append(
            ModelFinding(
                pr_id=pr_id,
                source=item.get("source", exp_id),
                category=std_cat,
                file_path=file_path,
                line_start=int(line_start),
                line_end=int(line_end),
                evidence=item.get("evidence", ""),
                execution_status=ExecutionStatus.SUCCESS.value,
                finding_id=item.get("finding_id") or str(uuid.uuid4()),
                rule_id=item.get("title") or item.get("rule_id", ""),
                severity=item.get("severity", SeverityLevel.MEDIUM.value),
                description=item.get("description", ""),
                recommendation=item.get("suggested_fix") or item.get("recommendation", ""),
                granularity=granularity,
                metadata={"experiment_id": exp_id},
            )
        )

    return findings


class ExperimentRunner:
    """Orchestrates batch execution of frozen protocol experiments."""

    def __init__(
        self,
        diffs_dir: Path = DEFAULT_DIFFS_DIR,
        manifest_path: Path = DEFAULT_MANIFEST_PATH,
        results_dir: Path = DEFAULT_RESULTS_DIR,
    ):
        self.diffs_dir = Path(diffs_dir)
        self.manifest_path = Path(manifest_path)
        self.results_dir = Path(results_dir)

    def load_manifest(self) -> List[Dict[str, Any]]:
        """Load successful PR records from diff_manifest.csv."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        records = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status", "").upper() == "SUCCESS":
                    diff_file = row.get("diff_file", "")
                    pr_id = Path(diff_file).stem
                    row["pr_id"] = pr_id
                    records.append(row)
        return records

    def run_pr(
        self,
        pr_record: Dict[str, Any],
        exp_id: str,
        pipeline: PRSentinelPipeline,
    ) -> Dict[str, Any]:
        """Execute a single PR under a specific experiment configuration."""
        pr_id = pr_record["pr_id"]
        diff_file = Path(pr_record.get("diff_file", ""))
        if not diff_file.exists():
            diff_file = self.diffs_dir / f"{pr_id}.diff"

        if not diff_file.exists():
            return {
                "pr_id": pr_id,
                "experiment_id": exp_id,
                "status": "ERROR",
                "error": f"Diff file not found: {diff_file}",
                "findings": [],
                "duration_seconds": 0.0,
            }

        diff_content = diff_file.read_text(encoding="utf-8", errors="replace")
        repo_name = pr_record.get("repo", "")
        target_dir = None
        for candidate in [
            Path(f"evaluation/repos/{repo_name}_{pr_id}"),
            Path(f"evaluation/repos/{repo_name}_base"),
            Path(f"evaluation/repos/{repo_name}"),
        ]:
            if candidate.exists() and candidate.is_dir():
                target_dir = str(candidate.resolve())
                break

        pr_metadata = {
            "pr_id": pr_id,
            "repo": repo_name,
            "pr_number": pr_record.get("pr_number", 0),
            "title": f"PR {pr_id}",
            "target_repo_dir": target_dir,
            "commit_sha": pr_record.get("commit_sha", "HEAD"),
        }

        start_time = time.time()
        try:
            result = pipeline.execute_review_sync(
                diff_content=diff_content,
                filename=f"{pr_id}.diff",
                pr_metadata=pr_metadata,
                pr_id=pr_id,
            )
            duration = round(time.time() - start_time, 3)

            findings = extract_model_findings(result, pr_id=pr_id, exp_id=exp_id)
            findings_dicts = [asdict(f) for f in findings]

            return {
                "pr_id": pr_id,
                "experiment_id": exp_id,
                "status": "SUCCESS",
                "findings": findings_dicts,
                "findings_count": len(findings_dicts),
                "risk_assessment": result.get("risk_assessment", {}),
                "duration_seconds": duration,
                "execution_manifest": result.get("execution_manifest", {}),
            }
        except Exception as e:
            duration = round(time.time() - start_time, 3)
            logger.error(f"Error processing PR {pr_id} under {exp_id}: {str(e)}")
            return {
                "pr_id": pr_id,
                "experiment_id": exp_id,
                "status": "ERROR",
                "error": str(e),
                "findings": [],
                "findings_count": 0,
                "duration_seconds": duration,
            }

    def run_experiment(
        self,
        exp_id: str,
        pr_limit: Optional[int] = None,
        pr_ids: Optional[List[str]] = None,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """Execute a full or partial experimental run for a configuration."""
        logger.info(f"=== Starting Experiment Execution: {exp_id} (resume={resume}) ===")
        manifest_records = self.load_manifest()

        if pr_ids:
            target_ids = set(pr_ids)
            records = [r for r in manifest_records if r["pr_id"] in target_ids]
        else:
            records = manifest_records

        if pr_limit and pr_limit > 0:
            records = records[:pr_limit]

        logger.info(f"Configuration: {exp_id} | Total PRs to process: {len(records)}")

        # Configure pipeline for this specific experiment
        config = PipelineConfig.for_experiment(exp_id)
        pipeline = PRSentinelPipeline(config=config)

        # Ensure output directory exists
        out_dir = self.results_dir / exp_id
        out_dir.mkdir(parents=True, exist_ok=True)

        exp_manifest = {
            "experiment_id": exp_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": asdict(config),
            "total_prs_scheduled": len(records),
            "successful_prs": 0,
            "failed_prs": 0,
            "total_findings_generated": 0,
            "per_pr_results": {},
        }

        for idx, rec in enumerate(records, 1):
            pr_id = rec["pr_id"]
            pr_file = out_dir / f"{pr_id}.json"

            if resume and pr_file.exists():
                try:
                    with open(pr_file, "r", encoding="utf-8") as f:
                        cached_res = json.load(f)
                    if cached_res.get("status") == "SUCCESS":
                        findings_cnt = cached_res.get("findings_count", len(cached_res.get("findings", [])))
                        logger.info(
                            f"[{idx}/{len(records)}] Resuming {exp_id} on PR: {pr_id} "
                            f"(already SUCCESS, {findings_cnt} findings)"
                        )
                        exp_manifest["successful_prs"] += 1
                        exp_manifest["total_findings_generated"] += findings_cnt
                        exp_manifest["per_pr_results"][pr_id] = {
                            "status": "SUCCESS",
                            "findings_count": findings_cnt,
                            "duration_seconds": cached_res.get("duration_seconds", 0.0),
                            "error": None,
                        }
                        continue
                except Exception as e:
                    logger.warning(f"Could not load existing file for {pr_id}, re-running: {e}")

            logger.info(f"[{idx}/{len(records)}] Running {exp_id} on PR: {pr_id}")

            pr_res = self.run_pr(rec, exp_id=exp_id, pipeline=pipeline)

            # Save immutable raw PR prediction file
            with open(pr_file, "w", encoding="utf-8") as f:
                json.dump(pr_res, f, indent=2)

            if pr_res["status"] == "SUCCESS":
                exp_manifest["successful_prs"] += 1
                exp_manifest["total_findings_generated"] += pr_res["findings_count"]
            else:
                exp_manifest["failed_prs"] += 1

            exp_manifest["per_pr_results"][pr_id] = {
                "status": pr_res["status"],
                "findings_count": pr_res["findings_count"],
                "duration_seconds": pr_res["duration_seconds"],
                "error": pr_res.get("error"),
            }

            # Periodically update manifest on disk
            manifest_file = out_dir / "execution_manifest.json"
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(exp_manifest, f, indent=2)

        # Write final overall configuration manifest
        manifest_file = out_dir / "execution_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(exp_manifest, f, indent=2)

        logger.info(
            f"=== Finished {exp_id}: {exp_manifest['successful_prs']} OK, "
            f"{exp_manifest['failed_prs']} Failed, "
            f"{exp_manifest['total_findings_generated']} Findings ==="
        )
        return exp_manifest


def main():
    parser = argparse.ArgumentParser(description="PR-Sentinel Central Experiment Runner")
    parser.add_argument(
        "--experiment",
        "-e",
        choices=ALL_EXPERIMENT_CONFIGS + ["ALL"],
        default="E2",
        help="Experiment configuration to run (default: E2)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of PRs (useful for smoke tests)",
    )
    parser.add_argument(
        "--pr_id",
        type=str,
        default=None,
        help="Run for a specific single PR ID",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resuming from existing successful outputs (re-run all PRs)",
    )
    args = parser.parse_args()

    runner = ExperimentRunner()
    target_configs = ALL_EXPERIMENT_CONFIGS if args.experiment == "ALL" else [args.experiment]
    pr_ids = [args.pr_id] if args.pr_id else None
    resume = not args.no_resume

    for cfg in target_configs:
        runner.run_experiment(exp_id=cfg, pr_limit=args.limit, pr_ids=pr_ids, resume=resume)


if __name__ == "__main__":
    main()
