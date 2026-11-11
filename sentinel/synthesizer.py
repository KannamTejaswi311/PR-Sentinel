"""
PR-Sentinel: Module 7 - Review Synthesizer and PR Risk Assessment
Normalizes, validates, and consolidates findings from LLM, static analysis,
and specialized multi-agent reviewers into a unified report with a transparent,
explainable Pull Request Risk Score (0-100).
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple


class ReviewSynthesizer:
    """
    Consolidates findings from static analysis, CodeBERT representations,
    and multi-agent reviews into a validated, deduplicated final review report.
    """

    SEVERITY_WEIGHTS = {
        "CRITICAL": 30.0,
        "HIGH": 15.0,
        "MEDIUM": 6.0,
        "LOW": 2.0,
        "INFORMATIONAL": 0.5,
    }

    SECURITY_MULTIPLIER = 1.5

    def synthesize(
        self,
        multi_agent_results: Dict[str, Any],
        static_findings: List[Dict[str, Any]],
        diff_content: str,
        filename: str = "patch.diff",
        codebert_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize all review inputs into a consolidated report.
        """
        # 1. Combine all raw findings
        combined: List[Dict[str, Any]] = []

        # From multi-agent orchestrator
        all_agent_findings = multi_agent_results.get("all_findings", [])
        combined.extend(all_agent_findings)

        # From static analysis
        for sf in static_findings:
            combined.append({
                "title": sf.get("title", "Static Analysis Issue"),
                "category": sf.get("source", "Static Analysis"),
                "severity": sf.get("severity", "MEDIUM").upper(),
                "file": sf.get("file", filename),
                "line": sf.get("line", 1),
                "description": sf.get("description", ""),
                "evidence": sf.get("evidence", ""),
                "suggested_fix": sf.get("recommendation", ""),
                "confidence": 0.95 if sf.get("confidence") == "HIGH" else 0.85,
            })

        # 2. Validate line numbers against actual diff content
        validated = self._validate_and_enrich_findings(combined, diff_content, filename)

        # 3. Deduplicate across sources
        unique_findings = self._deduplicate_findings(validated)

        # 4. Count findings by severity
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "informational": 0,
        }
        for f in unique_findings:
            sev = f.get("severity", "MEDIUM").lower()
            if sev in severity_counts:
                severity_counts[sev] += 1
            else:
                severity_counts["medium"] += 1

        # 5. Calculate transparent PR Risk Score
        risk_assessment = self.calculate_pr_risk_score(
            unique_findings,
            diff_content,
            codebert_analysis
        )

        # 6. Build consolidated summary
        security_count = sum(
            1 for f in unique_findings
            if "security" in f.get("category", "").lower()
            or "cwe" in f.get("title", "").lower()
            or f.get("severity") in ("CRITICAL", "HIGH") and "sql" in f.get("title", "").lower()
        )

        return {
            "total_findings": len(unique_findings),
            "severity_counts": severity_counts,
            "security_findings_count": security_count,
            "risk_assessment": risk_assessment,
            "findings": unique_findings,
            "codebert_summary": codebert_analysis.get("summary") if codebert_analysis else None,
        }

    def _validate_and_enrich_findings(
        self,
        findings: List[Dict[str, Any]],
        diff_content: str,
        default_file: str,
    ) -> List[Dict[str, Any]]:
        """Validate line numbers and extract supporting code evidence from diff."""
        lines = diff_content.splitlines()
        validated = []

        for f in findings:
            item = dict(f)
            # Ensure severity is valid
            sev = item.get("severity", "MEDIUM").upper()
            if sev not in self.SEVERITY_WEIGHTS:
                sev = "MEDIUM"
            item["severity"] = sev

            # Normalize file
            if not item.get("file") or item["file"] == "Unknown":
                item["file"] = default_file

            # If evidence is missing, attempt to extract line from diff
            line_no = item.get("line")
            if not item.get("evidence") and isinstance(line_no, int) and 1 <= line_no <= len(lines):
                item["evidence"] = lines[line_no - 1].strip()

            validated.append(item)

        return validated

    def _deduplicate_findings(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge identical or overlapping findings based on file, line, and semantic category.
        Preserves the highest severity and most detailed evidence.
        """
        clusters: Dict[Tuple[str, Any], Dict[str, Any]] = {}

        for f in findings:
            file_key = f.get("file", "")
            line_key = f.get("line", 1)
            # Simplify title for clustering (e.g. SQL Injection)
            title_lower = f.get("title", "").lower()
            topic = "other"
            for t in ["sql", "xss", "command", "password", "secret", "token", "zero", "unbounded", "bare"]:
                if t in title_lower:
                    topic = t
                    break

            cluster_key = (file_key, line_key, topic)

            if cluster_key not in clusters:
                clusters[cluster_key] = f
            else:
                existing = clusters[cluster_key]
                # Compare severity weights, keep higher
                curr_w = self.SEVERITY_WEIGHTS.get(f.get("severity", "MEDIUM"), 5.0)
                exist_w = self.SEVERITY_WEIGHTS.get(existing.get("severity", "MEDIUM"), 5.0)
                if curr_w > exist_w:
                    clusters[cluster_key] = f
                # Combine fix recommendations if existing is short
                if len(f.get("suggested_fix", "")) > len(existing.get("suggested_fix", "")):
                    clusters[cluster_key]["suggested_fix"] = f.get("suggested_fix")

        return list(clusters.values())

    def calculate_pr_risk_score(
        self,
        findings: List[Dict[str, Any]],
        diff_content: str,
        codebert_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate an explainable, deterministic Pull Request Risk Score (0 to 100).

        Formula:
        RawScore = Sum(Weight(severity) * (1.5 if is_security else 1.0))
        FileSpreadMultiplier = 1.0 + 0.1 * min(NumFiles - 1, 5)
        DiffSizeMultiplier = 1.0 + 0.05 * min(floor(DiffLines / 50), 4)
        FinalScore = min(100, round(RawScore * FileSpreadMultiplier * DiffSizeMultiplier))
        """
        raw_score = 0.0
        breakdown_items = []

        for f in findings:
            sev = f.get("severity", "MEDIUM").upper()
            base_w = self.SEVERITY_WEIGHTS.get(sev, 5.0)

            category = f.get("category", "").lower()
            title = f.get("title", "").lower()
            is_security = (
                "security" in category
                or "cwe" in title
                or any(w in title for w in ["sql", "injection", "token", "secret", "password", "xss"])
            )

            multiplier = self.SECURITY_MULTIPLIER if is_security else 1.0
            finding_impact = base_w * multiplier
            raw_score += finding_impact

            breakdown_items.append({
                "title": f.get("title"),
                "severity": sev,
                "is_security": is_security,
                "score_contribution": round(finding_impact, 1),
            })

        # Calculate affected files
        files_mentioned = {f.get("file", "unknown") for f in findings}
        num_files = max(1, len(files_mentioned))
        file_spread_mult = 1.0 + 0.1 * min(num_files - 1, 5)

        # Calculate diff line count
        diff_lines = len(diff_content.splitlines())
        diff_size_mult = 1.0 + 0.05 * min(diff_lines // 50, 4)

        # CodeBERT semantic deviation contribution if present
        codebert_penalty = 0.0
        if codebert_analysis:
            indicators = codebert_analysis.get("semantic_indicators", {})
            risk_dev = indicators.get("semantic_risk_deviation", 0.0)
            if risk_dev > 0.05:
                codebert_penalty = round(risk_dev * 20.0, 1)
                raw_score += codebert_penalty

        # Final calculated score clamped between 0 and 100
        calculated_score = min(100, round((raw_score * file_spread_mult * diff_size_mult)))

        # Determine risk level
        if calculated_score >= 75:
            risk_level = "CRITICAL RISK"
            recommendation = "Reject / Request Immediate Changes. Critical vulnerabilities or high-risk flaws detected."
        elif calculated_score >= 50:
            risk_level = "HIGH RISK"
            recommendation = "Block merge until high-severity security or logic issues are remediated."
        elif calculated_score >= 25:
            risk_level = "MEDIUM RISK"
            recommendation = "Review and address highlighted defects prior to deployment."
        elif calculated_score > 0:
            risk_level = "LOW RISK"
            recommendation = "Minor polish suggested. Low impact on stability and security."
        else:
            risk_level = "NEGLIGIBLE RISK"
            recommendation = "Clean code modification. Ready for merge."

        return {
            "score": calculated_score,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "formula_explanation": (
                "RiskScore = min(100, round(RawImpact * FileSpreadMultiplier * DiffSizeMultiplier)). "
                f"Weights: Critical=30, High=15, Med=6, Low=2 (x1.5 security multiplier)."
            ),
            "metrics": {
                "raw_impact_score": round(raw_score, 1),
                "file_spread_multiplier": round(file_spread_mult, 2),
                "diff_size_multiplier": round(diff_size_mult, 2),
                "codebert_penalty": codebert_penalty,
                "affected_files_count": num_files,
                "diff_line_count": diff_lines,
            },
            "top_contributions": sorted(breakdown_items, key=lambda x: x["score_contribution"], reverse=True)[:5],
        }


_SYNTHESIZER_INSTANCE: Optional[ReviewSynthesizer] = None


def get_review_synthesizer() -> ReviewSynthesizer:
    """Return singleton instance of ReviewSynthesizer."""
    global _SYNTHESIZER_INSTANCE
    if _SYNTHESIZER_INSTANCE is None:
        _SYNTHESIZER_INSTANCE = ReviewSynthesizer()
    return _SYNTHESIZER_INSTANCE
