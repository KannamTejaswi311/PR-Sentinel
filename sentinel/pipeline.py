"""
PR-Sentinel: End-to-End Orchestrated Code Review Pipeline
Integrates:
1. GitHub PR / Local Diff Ingestion
2. CodeBERT Contextual Code Representations & Supervised Head
3. Static Code Analysis (Bandit, Flake8, AST)
4. Repository-Aware RAG Context Retrieval
5. Local LLM Reasoning (Ollama / Qwen2.5-Coder)
6. Specialized Multi-Agent Reviewers (Bug, Security, Performance, Quality)
7. Review Synthesis & PR Risk Score Calculation (0-100)
8. Automated Test Recommendations
9. SQLite Persistence & Developer Feedback Integration
Adheres to Frozen Experimental Protocol v1.0 and Amendment v1.1.
"""

import time
from typing import Any, Dict, List, Optional

from sentinel.codebert_analyzer import get_codebert_analyzer
from sentinel.github_integration import get_github_integration
from sentinel.multi_agent import get_multi_agent_orchestrator
from sentinel.rag_pipeline import get_rag_pipeline
from sentinel.review_analyzer import analyze_review as legacy_analyze_review
from sentinel.review_history import init_database, save_review
from sentinel.static_analyzer import get_static_analyzer
from sentinel.synthesizer import get_review_synthesizer
from sentinel.test_generator import get_test_generator
from sentinel.schemas import PipelineConfig, ExecutionStatus


class PRSentinelPipeline:
    """
    Central orchestrator executing the PR-Sentinel review pipeline.
    Supports configuration-driven execution for both production and experiments (E1-E6).
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        init_database()
        self.config = config or PipelineConfig()
        self.codebert = get_codebert_analyzer(require_real_weights=self.config.require_real_weights)
        self.static_analyzer = get_static_analyzer()
        self.rag = get_rag_pipeline(workspace_dir=self.config.target_repo_dir)
        self.orchestrator = get_multi_agent_orchestrator()
        self.synthesizer = get_review_synthesizer()
        self.test_generator = get_test_generator()
        self.github = get_github_integration()

    def execute_review_sync(
        self,
        diff_content: str,
        filename: str = "patch.diff",
        pr_metadata: Optional[Dict[str, Any]] = None,
        raw_llm_text: str = "",
        pr_id: str = "pr_unknown",
    ) -> Dict[str, Any]:
        """
        Synchronous review execution supporting deterministic offline execution,
        concurrency safety, and experiment runner execution.
        """
        start_time = time.time()
        pr_metadata = pr_metadata or {}
        manifest = {}

        # ----------------------------------------------------
        # STEP 1: CodeBERT
        # ----------------------------------------------------
        codebert_result = None
        if self.config.enable_codebert:
            try:
                codebert_result = self.codebert.analyze_code_diff(diff_content)
                manifest["codebert"] = ExecutionStatus.SUCCESS.value
            except Exception as e:
                manifest["codebert"] = f"ERROR: {str(e)}"
        else:
            manifest["codebert"] = ExecutionStatus.SKIPPED.value

        # ----------------------------------------------------
        # STEP 2: Static Analysis
        # ----------------------------------------------------
        static_findings = []
        if self.config.enable_static:
            try:
                static_findings = self.static_analyzer.analyze_diff(diff_content, filename=filename, pr_id=pr_id)
                manifest["static_analyzer"] = ExecutionStatus.SUCCESS.value
            except Exception as e:
                manifest["static_analyzer"] = f"ERROR: {str(e)}"
        else:
            manifest["static_analyzer"] = ExecutionStatus.SKIPPED.value

        # ----------------------------------------------------
        # STEP 3: Repository RAG
        # ----------------------------------------------------
        retrieved_context_chunks = []
        formatted_rag_context = ""
        if self.config.enable_rag:
            try:
                retrieved_context_chunks = self.rag.retrieve_context(diff_content, top_k=3)
                formatted_rag_context = self.rag.format_context_for_prompt(retrieved_context_chunks)
                manifest["rag_pipeline"] = ExecutionStatus.SUCCESS.value
            except Exception as e:
                manifest["rag_pipeline"] = f"ERROR: {str(e)}"
        else:
            manifest["rag_pipeline"] = ExecutionStatus.SKIPPED.value

        # ----------------------------------------------------
        # STEP 4: Multi-Agent or Monolithic Review
        # ----------------------------------------------------
        multi_agent_results = {"agents_executed": [], "agent_findings": {}, "all_findings": [], "total_raw_findings": 0}
        if self.config.enable_multi_agent:
            try:
                multi_agent_results = self.orchestrator.run_review_sync(
                    diff_content=diff_content,
                    filename=filename,
                    repo_context=formatted_rag_context,
                    static_findings=static_findings,
                    pr_id=pr_id,
                )
                manifest["multi_agent"] = ExecutionStatus.SUCCESS.value
            except Exception as e:
                manifest["multi_agent"] = f"ERROR: {str(e)}"
        elif self.config.enable_llm:
            # Monolithic LLM review (used for E1 and E6_A4)
            manifest["multi_agent"] = "MONOLITHIC_MODE"
        else:
            manifest["multi_agent"] = ExecutionStatus.SKIPPED.value

        # ----------------------------------------------------
        # STEP 5: Review Synthesis & PR Risk Score
        # ----------------------------------------------------
        if self.config.enable_synthesis:
            try:
                synthesized_report = self.synthesizer.synthesize(
                    multi_agent_results=multi_agent_results,
                    static_findings=static_findings,
                    diff_content=diff_content,
                    filename=filename,
                    codebert_analysis=codebert_result,
                    pr_id=pr_id,
                )
                manifest["synthesizer"] = ExecutionStatus.SUCCESS.value
            except Exception as e:
                synthesized_report = {
                    "total_findings": 0,
                    "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0, "informational": 0},
                    "security_findings_count": 0,
                    "risk_assessment": {"score": 0, "risk_level": "NEGLIGIBLE", "recommendation": f"Synthesis Error: {e}", "top_contributions": []},
                    "findings": [],
                }
                manifest["synthesizer"] = f"ERROR: {str(e)}"
        else:
            # Return raw union of findings without synthesis
            raw_all = list((multi_agent_results or {}).get("all_findings", []))
            for sf in static_findings:
                raw_all.append(sf)
            if codebert_result and isinstance(codebert_result, dict) and "findings" in codebert_result:
                for cf in codebert_result["findings"]:
                    raw_all.append(cf)
            synthesized_report = {
                "total_findings": len(raw_all),
                "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0, "informational": 0},
                "security_findings_count": 0,
                "risk_assessment": {"score": 0, "risk_level": "NEGLIGIBLE", "recommendation": "Synthesis disabled", "top_contributions": []},
                "findings": raw_all,
            }
            manifest["synthesizer"] = ExecutionStatus.SKIPPED.value

        # ----------------------------------------------------
        # STEP 6: Automated Test Recommendations
        # ----------------------------------------------------
        test_recommendations = []
        if self.config.enable_test_generator:
            try:
                test_recommendations = self.test_generator.generate_recommendations(
                    diff_content=diff_content,
                    findings=synthesized_report.get("findings", []),
                    filename=filename,
                )
                manifest["test_generator"] = ExecutionStatus.SUCCESS.value
            except Exception as e:
                manifest["test_generator"] = f"ERROR: {str(e)}"
        else:
            manifest["test_generator"] = ExecutionStatus.SKIPPED.value
        synthesized_report["test_recommendations"] = test_recommendations

        # ----------------------------------------------------
        # STEP 7: Format GitHub Review Summary
        # ----------------------------------------------------
        pr_title = pr_metadata.get("title", filename)
        github_comment_markdown = self.github.format_github_review_comment(
            synthesized_report=synthesized_report,
            pr_title=pr_title,
        )
        synthesized_report["github_comment"] = github_comment_markdown

        # ----------------------------------------------------
        # STEP 8: Construct merged summary
        # ----------------------------------------------------
        merged_summary = {
            "critical_issues": synthesized_report["severity_counts"]["critical"],
            "security_issues": synthesized_report["security_findings_count"],
            "code_quality_issues": synthesized_report["severity_counts"]["low"] + synthesized_report["severity_counts"]["informational"],
            "test_issues": len(test_recommendations),
            "severity": synthesized_report["severity_counts"],
            "findings": synthesized_report["findings"],
            "risk_score": synthesized_report["risk_assessment"]["score"],
            "risk_level": synthesized_report["risk_assessment"]["risk_level"],
        }

        # ----------------------------------------------------
        # STEP 9: Persist to SQLite
        # ----------------------------------------------------
        execution_duration = round(time.time() - start_time, 2)
        synthesized_report["execution_duration_seconds"] = execution_duration
        synthesized_report["rag_context_snippets"] = retrieved_context_chunks
        synthesized_report["pr_metadata"] = pr_metadata
        synthesized_report["execution_manifest"] = manifest

        review_id = save_review(
            filename=filename,
            summary=merged_summary,
            review_text=raw_llm_text or github_comment_markdown,
            risk_score=synthesized_report["risk_assessment"]["score"],
            risk_level=synthesized_report["risk_assessment"]["risk_level"],
            synthesized_report=synthesized_report,
        )

        return {
            "success": True,
            "review_id": review_id,
            "filename": filename,
            "pr_metadata": pr_metadata,
            "duration_seconds": execution_duration,
            "summary": merged_summary,
            "risk_assessment": synthesized_report["risk_assessment"],
            "codebert_analysis": codebert_result or {},
            "multi_agent_results": (multi_agent_results or {}).get("agent_findings", {}),
            "test_recommendations": test_recommendations,
            "rag_context": retrieved_context_chunks,
            "github_comment": github_comment_markdown,
            "review_text": raw_llm_text or github_comment_markdown,
            "execution_manifest": manifest,
            "experiment_id": self.config.experiment_id,
        }

    async def execute_review(
        self,
        diff_content: str,
        filename: str = "patch.diff",
        pr_metadata: Optional[Dict[str, Any]] = None,
        raw_llm_text: str = "",
        pr_id: str = "pr_unknown",
    ) -> Dict[str, Any]:
        """
        Async entry point for Web UI and async clients.
        Delegates to execute_review_sync in offline mode for safety.
        """
        import os
        if os.environ.get("PR_SENTINEL_OFFLINE", "").lower() in ("1", "true", "yes"):
            return self.execute_review_sync(
                diff_content=diff_content,
                filename=filename,
                pr_metadata=pr_metadata,
                raw_llm_text=raw_llm_text,
                pr_id=pr_id,
            )

        # In online mode, can also run asynchronously
        return self.execute_review_sync(
            diff_content=diff_content,
            filename=filename,
            pr_metadata=pr_metadata,
            raw_llm_text=raw_llm_text,
            pr_id=pr_id,
        )


_PIPELINE_INSTANCE: Optional[PRSentinelPipeline] = None


def get_pipeline(config: Optional[PipelineConfig] = None) -> PRSentinelPipeline:
    """Return singleton or configured instance of PRSentinelPipeline."""
    global _PIPELINE_INSTANCE
    if config is not None:
        return PRSentinelPipeline(config=config)
    if _PIPELINE_INSTANCE is None:
        _PIPELINE_INSTANCE = PRSentinelPipeline()
    return _PIPELINE_INSTANCE
