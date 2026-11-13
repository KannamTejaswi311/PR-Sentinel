"""
PR-Sentinel: Module 7 - Review Synthesizer and PR Risk Assessment
Normalizes, validates, and consolidates findings from LLM, static analysis,
and specialized multi-agent reviewers into a unified report with a transparent,
explainable Pull Request Risk Score (0-100).
Adheres to Frozen Experimental Protocol v1.0 and Amendment v1.1.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from sentinel.schemas import (
    ModelFinding,
    FindingCategory,
    SeverityLevel,
    ExecutionStatus,
    FindingGranularity,
)


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
        pr_id: str = "pr_unknown",
    ) -> Dict[str, Any]:
        """
        Synthesize all review inputs into a consolidated report.
        """
        combined: List[Dict[str, Any]] = []

        # 1. From multi-agent orchestrator
        all_agent_findings = multi_agent_results.get("all_findings", [])
        for f in all_agent_findings:
            combined.append({
                "title": f.get("title", "Agent Finding"),
                "category": f.get("category", "Multi-Agent"),
                "source": f.get("source", "multi_agent"),
                "severity": f.get("severity", "MEDIUM").upper(),
                "file": f.get("file", filename),
                "line": max(1, int(f.get("line", 1))),
                "description": f.get("description", ""),
                "evidence": f.get("evidence", ""),
                "suggested_fix": f.get("suggested_fix", ""),
                "confidence": float(f.get("confidence", 0.8)),
                "rule_id": f.get("rule_id"),
            })

        # 2. From static analysis
        for sf in static_findings:
            src = sf.get("source", "Static Analysis")
            combined.append({
                "title": sf.get("title", "Static Analysis Issue"),
                "category": sf.get("category", sf.get("source", "Static Analysis")),
                "source": f"static_{src.lower()}",
                "severity": sf.get("severity", "MEDIUM").upper(),
                "file": sf.get("file", filename),
                "line": max(1, int(sf.get("line", 1))),
                "description": sf.get("description", ""),
                "evidence": sf.get("evidence", ""),
                "suggested_fix": sf.get("recommendation", ""),
                "confidence": 0.95 if sf.get("confidence") == "HIGH" else 0.85,
                "rule_id": sf.get("rule_id"),
                "execution_status": sf.get("execution_status", ExecutionStatus.SUCCESS.value),
            })

        # 3. From CodeBERT defect classifier
        if codebert_analysis and isinstance(codebert_analysis, dict) and "findings" in codebert_analysis:
            for cf in codebert_analysis["findings"]:
                combined.append({
                    "title": cf.get("title") or cf.get("rule_id", "CodeBERT Hunk Defect"),
                    "category": cf.get("category", "BUG"),
                    "source": cf.get("source", "codebert_head"),
                    "severity": cf.get("severity", "MEDIUM").upper(),
                    "file": cf.get("file_path") or cf.get("file", filename),
                    "line": max(1, int(cf.get("line_start", cf.get("line", 1)))),
                    "line_end": int(cf.get("line_end", cf.get("line", 1))),
                    "description": cf.get("description", ""),
                    "evidence": cf.get("evidence", ""),
                    "suggested_fix": cf.get("recommendation") or cf.get("suggested_fix", ""),
                    "confidence": float(cf.get("confidence", 0.85)),
                    "rule_id": cf.get("rule_id", "CODEBERT_DEFECT"),
                    "granularity": cf.get("granularity", FindingGranularity.HUNK_SPAN.value),
                    "execution_status": cf.get("execution_status", ExecutionStatus.SUCCESS.value),
                })

        # 4. Validate line numbers against diff content
        validated = self._validate_and_enrich_findings(combined, diff_content, filename)

        # 5. Deduplicate across sources
        unique_findings = self._deduplicate_findings(validated)

        # 5. Count findings by severity
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

        # 6. Calculate transparent PR Risk Score
        risk_assessment = self.calculate_pr_risk_score(
            unique_findings,
            diff_content,
            codebert_analysis
        )

        # 7. Count security findings
        security_count = sum(
            1 for f in unique_findings
            if "security" in f.get("category", "").lower()
            or "cwe" in f.get("title", "").lower()
            or (f.get("severity") in ("CRITICAL", "HIGH") and "sql" in f.get("title", "").lower())
        )

        return {
            "total_findings": len(unique_findings),
            "severity_counts": severity_counts,
            "security_findings_count": security_count,
            "risk_assessment": risk_assessment,
            "findings": unique_findings,
            "codebert_summary": codebert_analysis.get("summary", "") if codebert_analysis else "",
        }

    def _validate_and_enrich_findings(
        self,
        findings: List[Dict[str, Any]],
        diff_content: str,
        default_filename: str
    ) -> List[Dict[str, Any]]:
        """Validate finding fields, normalize severity, and enrich evidence."""
        lines = diff_content.splitlines()
        validated = []

        for f in findings:
            item = dict(f)

            # Ensure valid severity
            sev = item.get("severity", "MEDIUM").upper()
            if sev not in self.SEVERITY_WEIGHTS:
                sev = "MEDIUM"
            item["severity"] = sev

            # Ensure filename
            if not item.get("file"):
                item["file"] = default_filename

            # If evidence is missing, pull from diff lines if line index exists
            line_no = item.get("line", 1)
            if not item.get("evidence") and 1 <= line_no <= len(lines):
                item["evidence"] = lines[line_no - 1].strip()

            validated.append(item)

        return validated

    def _deduplicate_findings(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate findings from different analyzers while preserving distinct valid issues.
        Clusters by (file, line, topic_keyword).
        """
        clusters: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}

        for f in findings:
            file_name = f.get("file", "")
            line = f.get("line", 1)
            title = f.get("title", "").lower()

            topic = "other"
            for kw in [
                "sql", "xss", "command", "password", "secret", "token",
                "zero", "unbounded", "bare", "syntax", "pep8", "naming",
                "complexity", "import", "type", "memory", "leak"
            ]:
                if kw in title:
                    topic = kw
                    break

            key = (file_name, line, topic)
            clusters.setdefault(key, []).append(f)

        deduped = []
        for key, group in clusters.items():
            # Pick highest severity finding in group
            best = max(
                group,
                key=lambda x: (
                    self.SEVERITY_WEIGHTS.get(x.get("severity", "LOW"), 1.0),
                    len(x.get("suggested_fix", "") or x.get("recommendation", "") or "")
                )
            )
            # Combine sources if multiple tools flagged the same issue
            sources = list(dict.fromkeys(g.get("source", "") for g in group if g.get("source")))
            if len(sources) > 1:
                best["all_sources"] = sources
            deduped.append(best)

        return deduped

    def calculate_pr_risk_score(
        self,
        findings: List[Dict[str, Any]],
        diff_content: str,
        codebert_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Transparent PR Risk Score (0-100).
        Formula:
          BaseScore = sum(weight(sev) * (1.5 if security else 1.0))
          ScaledScore = BaseScore * FileSpreadMultiplier * DiffSizeMultiplier
          FinalScore = min(100, round(ScaledScore + CodeBERTPenalty))
        """
        raw_score = 0.0
        contributions = []

        for f in findings:
            sev = f.get("severity", "MEDIUM").upper()
            base_weight = self.SEVERITY_WEIGHTS.get(sev, 2.0)

            is_security = (
                "security" in f.get("category", "").lower()
                or "cwe" in f.get("title", "").lower()
                or (sev in ("CRITICAL", "HIGH") and any(k in f.get("title", "").lower() for k in ("sql", "injection", "secret", "command")))
            )

            multiplier = self.SECURITY_MULTIPLIER if is_security else 1.0
            point_val = base_weight * multiplier
            raw_score += point_val

            contributions.append({
                "title": f.get("title", ""),
                "severity": sev,
                "is_security": is_security,
                "points_added": round(point_val, 1),
            })

        # Diff surface area multipliers
        diff_lines = len(diff_content.splitlines()) if diff_content else 1
        num_files = len(re.findall(r"^\+\+\+\s+b/(.+)$", diff_content, re.MULTILINE)) or 1

        file_spread_mult = 1.0 + (0.1 * min(num_files - 1, 5))
        diff_size_mult = 1.0 + (0.05 * min(diff_lines // 50, 4))

        scaled_score = raw_score * file_spread_mult * diff_size_mult

        # CodeBERT additive penalty (non-compounded per protocol audit)
        codebert_penalty = 0.0
        if codebert_analysis:
            indicators = codebert_analysis.get("semantic_indicators", {})
            deviation = indicators.get("semantic_risk_deviation", 0.0)
            if deviation > 0.05:
                codebert_penalty = min(15.0, deviation * 30.0)

        final_score = min(100, round(scaled_score + codebert_penalty))

        # Determine qualitative risk level
        if final_score >= 75:
            risk_level = "CRITICAL RISK"
            recommendation = "DO NOT MERGE: PR contains critical security or logic defects. Extensive remediation required."
        elif final_score >= 50:
            risk_level = "HIGH RISK"
            recommendation = "CHANGES REQUESTED: High-priority defects detected. Address flagged vulnerabilities prior to merge."
        elif final_score >= 25:
            risk_level = "MEDIUM RISK"
            recommendation = "COMMENT / REVISE: Moderate code quality or logic concerns identified. Address before release."
        elif final_score > 0:
            risk_level = "LOW RISK"
            recommendation = "APPROVE WITH SUGGESTIONS: Minor style or non-critical observations."
        else:
            risk_level = "NEGLIGIBLE"
            recommendation = "APPROVE: No significant defects, security concerns, or architectural flaws identified."

        return {
            "score": final_score,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "metrics": {
                "raw_finding_score": round(raw_score, 1),
                "file_spread_multiplier": round(file_spread_mult, 2),
                "diff_size_multiplier": round(diff_size_mult, 2),
                "codebert_penalty_added": round(codebert_penalty, 1),
                "total_diff_lines": diff_lines,
                "modified_files_count": num_files,
            },
            "formula_explanation": (
                f"PR Risk Score = min(100, round(RawScore({raw_score:.1f}) * "
                f"FileSpread({file_spread_mult:.2f}) * DiffSize({diff_size_mult:.2f}) + "
                f"CodeBERTPenalty({codebert_penalty:.1f}))) = {final_score}"
            ),
            "top_contributions": sorted(contributions, key=lambda x: x["points_added"], reverse=True)[:5],
        }


_SYNTHESIZER_INSTANCE: Optional[ReviewSynthesizer] = None


def get_review_synthesizer() -> ReviewSynthesizer:
    """Return singleton instance of ReviewSynthesizer."""
    global _SYNTHESIZER_INSTANCE
    if _SYNTHESIZER_INSTANCE is None:
        _SYNTHESIZER_INSTANCE = ReviewSynthesizer()
    return _SYNTHESIZER_INSTANCE
