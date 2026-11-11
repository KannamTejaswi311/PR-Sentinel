"""
PR-Sentinel: End-to-End Orchestrated Code Review Pipeline
Integrates:
1. GitHub PR / Local Diff Ingestion
2. CodeBERT Contextual Code Representations
3. Static Code Analysis (Bandit, Flake8, AST)
4. Repository-Aware RAG Context Retrieval
5. Local LLM Reasoning (Ollama / Qwen2.5-Coder)
6. Specialized Multi-Agent Reviewers (Bug, Security, Performance, Quality)
7. Review Synthesis & PR Risk Score Calculation (0-100)
8. Automated Test Recommendations
9. SQLite Persistence & Developer Feedback Integration
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


class PRSentinelPipeline:
    """
    Central orchestrator executing the complete PR-Sentinel review pipeline.
    """

    def __init__(self):
        init_database()
        self.codebert = get_codebert_analyzer()
        self.static_analyzer = get_static_analyzer()
        self.rag = get_rag_pipeline()
        self.orchestrator = get_multi_agent_orchestrator()
        self.synthesizer = get_review_synthesizer()
        self.test_generator = get_test_generator()
        self.github = get_github_integration()

    async def execute_review(
        self,
        diff_content: str,
        filename: str = "patch.diff",
        pr_metadata: Optional[Dict[str, Any]] = None,
        raw_llm_text: str = "",
    ) -> Dict[str, Any]:
        """
        Execute the full PR-Sentinel analysis pipeline on a code diff.
        """
        start_time = time.time()
        pr_metadata = pr_metadata or {}

        # ----------------------------------------------------
        # STEP 1: CodeBERT Contextual Code Representation
        # ----------------------------------------------------
        codebert_result = self.codebert.analyze_code_diff(diff_content)

        # ----------------------------------------------------
        # STEP 2: Static Code Analysis (Bandit, Flake8, AST)
        # ----------------------------------------------------
        static_findings = self.static_analyzer.analyze_diff(diff_content, filename=filename)

        # ----------------------------------------------------
        # STEP 3: Repository-Aware RAG Context Retrieval
        # ----------------------------------------------------
        retrieved_context_chunks = self.rag.retrieve_context(diff_content, top_k=3)
        formatted_rag_context = self.rag.format_context_for_prompt(retrieved_context_chunks)

        # ----------------------------------------------------
        # STEP 4: Specialized Multi-Agent Review (Qwen2.5-Coder)
        # ----------------------------------------------------
        multi_agent_results = await self.orchestrator.run_review(
            diff_content=diff_content,
            filename=filename,
            repo_context=formatted_rag_context,
            static_findings=static_findings,
        )

        # ----------------------------------------------------
        # STEP 5: Review Synthesis & Transparent PR Risk Score
        # ----------------------------------------------------
        synthesized_report = self.synthesizer.synthesize(
            multi_agent_results=multi_agent_results,
            static_findings=static_findings,
            diff_content=diff_content,
            filename=filename,
            codebert_analysis=codebert_result,
        )

        # ----------------------------------------------------
        # STEP 6: Automated Test Recommendations
        # ----------------------------------------------------
        test_recommendations = self.test_generator.generate_recommendations(
            diff_content=diff_content,
            findings=synthesized_report.get("findings", []),
            filename=filename,
        )
        synthesized_report["test_recommendations"] = test_recommendations

        # ----------------------------------------------------
        # STEP 7: Format GitHub Review Summary & Inline Comments
        # ----------------------------------------------------
        pr_title = pr_metadata.get("title", filename)
        github_comment_markdown = self.github.format_github_review_comment(
            synthesized_report=synthesized_report,
            pr_title=pr_title,
        )
        synthesized_report["github_comment"] = github_comment_markdown

        # ----------------------------------------------------
        # STEP 8: Backward Compatibility with Stage 2/3 summary
        # ----------------------------------------------------
        legacy_summary = legacy_analyze_review(raw_llm_text, diff_content)
        # Combine synthesized findings into summary
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
        # STEP 9: Persist Review to SQLite
        # ----------------------------------------------------
        execution_duration = round(time.time() - start_time, 2)
        synthesized_report["execution_duration_seconds"] = execution_duration
        synthesized_report["rag_context_snippets"] = retrieved_context_chunks
        synthesized_report["pr_metadata"] = pr_metadata

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
            "codebert_analysis": codebert_result,
            "multi_agent_results": multi_agent_results["agent_findings"],
            "test_recommendations": test_recommendations,
            "rag_context": retrieved_context_chunks,
            "github_comment": github_comment_markdown,
            "review_text": raw_llm_text or github_comment_markdown,
        }


_PIPELINE_INSTANCE: Optional[PRSentinelPipeline] = None


def get_pipeline() -> PRSentinelPipeline:
    """Return singleton instance of PRSentinelPipeline."""
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        _PIPELINE_INSTANCE = PRSentinelPipeline()
    return _PIPELINE_INSTANCE
