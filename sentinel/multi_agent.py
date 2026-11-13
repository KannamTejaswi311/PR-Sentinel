"""
PR-Sentinel: Module 6 - Specialized Multi-Agent Code Review System
Implements four specialized review agents:
1. Bug Detection Agent (Logical bugs, condition errors, runtime exceptions)
2. Security Review Agent (Vulnerabilities, injection, credentials, auth)
3. Performance Review Agent (Inefficient loops, memory, redundant computations)
4. Code Quality Agent (Maintainability, readability, PEP8, naming, duplication)
Orchestrates agent execution using Qwen2.5-Coder via Ollama with robust offline fallback,
provenance preservation, and canonical ModelFinding schema validation.
Adheres to Frozen Experimental Protocol v1.0 and Amendment v1.1.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx

from sentinel.schemas import (
    ModelFinding,
    FindingCategory,
    SeverityLevel,
    ExecutionStatus,
    FindingGranularity,
)


class BaseReviewAgent:
    """Base class for specialized code review agents."""

    def __init__(self, name: str, role: str, focus_areas: List[str], category: str):
        self.name = name
        self.role = role
        self.focus_areas = focus_areas
        self.category = category

    def build_agent_prompt(
        self,
        diff_content: str,
        filename: str,
        repo_context: str,
        static_findings: List[Dict[str, Any]],
    ) -> str:
        """Construct specialized prompt for this agent."""
        focus_str = ", ".join(self.focus_areas)
        static_str = (
            json.dumps(static_findings, indent=2)
            if static_findings
            else "None detected"
        )

        return f"""You are the {self.name} in the PR-Sentinel automated code review system.
Your specific role: {self.role}
Your focus areas: {focus_str}

Analyze the following code change strictly through your specialty.
Target File: {filename}

Static Analysis Evidence:
{static_str}

Repository Context:
{repo_context}

Code Diff:
```diff
{diff_content}
```

Instructions:
Respond ONLY with a JSON array of findings matching this exact schema:
[
  {{
    "title": "Clear concise finding title",
    "category": "{self.category}",
    "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "INFORMATIONAL",
    "file": "{filename}",
    "line": <line_number_or_1>,
    "description": "Detailed explanation of the problem",
    "evidence": "Exact code snippet or condition causing the issue",
    "suggested_fix": "Concrete code correction or recommendation",
    "confidence": 0.85
  }}
]

If no issues in your focus area are found, return an empty JSON array: [].
Do not include any conversational preamble or markdown outside the JSON block.
"""

    def analyze_heuristically(
        self,
        diff_content: str,
        filename: str,
        static_findings: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Specialized heuristic and rule-based analysis for this agent."""
        return []


class BugDetectionAgent(BaseReviewAgent):
    """Detects logical flaws, incorrect conditions, and potential runtime exceptions."""

    def __init__(self):
        super().__init__(
            name="Bug Detection Agent",
            role="Detect logical flaws, condition errors, edge-case mishandling, and runtime bugs",
            focus_areas=[
                "Logical errors",
                "Off-by-one errors",
                "Null/None pointer dereference",
                "Unhandled exceptions",
                "Incorrect return values or state mutations",
                "Division by zero without guard",
            ],
            category=FindingCategory.BUG.value,
        )

    def analyze_heuristically(
        self,
        diff_content: str,
        filename: str,
        static_findings: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        static_findings = static_findings or []
        """
        Documented Agent Heuristic Rules:
        - BUG-01: Division by zero literal (/ 0)
        - BUG-02: Unguarded variable division
        - BUG-03: Mutable default argument in function definition
        - BUG-04: Subscripting fetchone() without None check
        """
        findings = []

        for idx, line in enumerate(diff_content.splitlines(), 1):
            if line.startswith("+") and not line.startswith("+++"):
                code = line[1:].strip()

                # Rule BUG-01: Division by zero literal
                if re.search(r"/\s*0(?:\.0*)?(?!\d)", code):
                    findings.append({
                        "title": "Division by Zero Error",
                        "category": self.category,
                        "severity": "CRITICAL",
                        "file": filename,
                        "line": idx,
                        "description": "Division operation with a zero denominator will cause ZeroDivisionError.",
                        "evidence": code,
                        "suggested_fix": "Add a pre-condition guard check ensuring the denominator is non-zero.",
                        "confidence": 0.95,
                    })

                # Rule BUG-02: Unguarded division by variable
                elif re.search(r"/\s*[A-Za-z_][A-Za-z0-9_]*", code) and not any(k in code for k in ("!= 0", "> 0", "if")):
                    # Exclude comment lines or URL strings
                    if not code.startswith("#") and "://" not in code:
                        findings.append({
                            "title": "Potential Zero-Division by Variable",
                            "category": self.category,
                            "severity": "HIGH",
                            "file": filename,
                            "line": idx,
                            "description": "Division by variable without apparent non-zero validation.",
                            "evidence": code,
                            "suggested_fix": "Ensure variable is validated != 0 before division.",
                            "confidence": 0.70,
                        })

                # Rule BUG-03: Mutable default arguments (def foo(bar=[]))
                if re.search(r"def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(.*=\s*(\[\]|\{\})", code):
                    findings.append({
                        "title": "Mutable Default Argument Bug",
                        "category": self.category,
                        "severity": "MEDIUM",
                        "file": filename,
                        "line": idx,
                        "description": "Mutable default argument retains state across multiple function calls.",
                        "evidence": code,
                        "suggested_fix": "Use None as the default value and initialize the mutable object inside the function body.",
                        "confidence": 0.90,
                    })

                # Rule BUG-04: Subscripting fetchone() directly without None check
                if re.search(r"\.fetchone\(\)\[", code):
                    findings.append({
                        "title": "Potential TypeError on fetchone() None result",
                        "category": self.category,
                        "severity": "HIGH",
                        "file": filename,
                        "line": idx,
                        "description": "Subscripting cursor.fetchone() directly will raise TypeError if query returns no rows.",
                        "evidence": code,
                        "suggested_fix": "Store fetchone() in a variable, verify it is not None, then access columns.",
                        "confidence": 0.85,
                    })

        # Deduplicate
        seen = set()
        deduped = []
        for f in findings:
            key = (f["title"], f["line"])
            if key not in seen:
                seen.add(key)
                deduped.append(f)
        return deduped


class SecurityReviewAgent(BaseReviewAgent):
    """Detects security vulnerabilities, injection flaws, secrets, and auth bypasses."""

    def __init__(self):
        super().__init__(
            name="Security Review Agent",
            role="Detect security vulnerabilities (OWASP Top 10, CWE), credential leaks, and injection vectors",
            focus_areas=[
                "SQL Injection (CWE-89)",
                "Command Injection (CWE-78)",
                "Cross-Site Scripting XSS (CWE-79)",
                "Hardcoded Credentials & API Keys (CWE-798)",
                "Path Traversal (CWE-22)",
                "Insecure Deserialization",
                "Authentication and Authorization Bypass",
            ],
            category=FindingCategory.SECURITY.value,
        )

    def analyze_heuristically(
        self,
        diff_content: str,
        filename: str,
        static_findings: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Documented Agent Heuristic Rules:
        - Ingests high-confidence static findings from Bandit / AST
        - SEC-01: Raw SQL string concatenation
        - SEC-02: Unsafe subprocess or os.system execution
        """
        static_findings = static_findings or []
        findings = []

        # Promote static findings into agent findings
        for sf in static_findings:
            if sf.get("severity") in ("CRITICAL", "HIGH", "MEDIUM"):
                findings.append({
                    "title": sf.get("title", "Static Security Finding"),
                    "category": self.category,
                    "severity": sf.get("severity", "HIGH"),
                    "file": sf.get("file", filename),
                    "line": sf.get("line", 1),
                    "description": sf.get("description", ""),
                    "evidence": sf.get("evidence", ""),
                    "suggested_fix": sf.get("recommendation", "Apply security patch."),
                    "confidence": 0.95,
                })

        for idx, line in enumerate(diff_content.splitlines(), 1):
            if line.startswith("+") and not line.startswith("+++"):
                code = line[1:].strip()

                # Rule SEC-01: Direct SQL concatenation
                if any(k in code.upper() for k in ("SELECT ", "INSERT ", "UPDATE ", "DELETE ")) and ("+" in code or "f\"" in code or "f'" in code):
                    findings.append({
                        "title": "SQL Injection Risk (Direct Concatenation)",
                        "category": self.category,
                        "severity": "HIGH",
                        "file": filename,
                        "line": idx,
                        "description": "SQL statement dynamically built via string concatenation or f-string interpolation.",
                        "evidence": code,
                        "suggested_fix": "Use parameterized queries with placeholder syntax.",
                        "confidence": 0.90,
                    })

                # Rule SEC-02: Command injection
                if any(k in code for k in ("os.system(", "subprocess.Popen(", "subprocess.call(")):
                    if "+" in code or "shell=True" in code or "f\"" in code or "f'" in code:
                        findings.append({
                            "title": "Command Injection Vulnerability",
                            "category": self.category,
                            "severity": "CRITICAL",
                            "file": filename,
                            "line": idx,
                            "description": "System command execution with dynamic arguments or shell=True.",
                            "evidence": code,
                            "suggested_fix": "Pass command arguments as a list with shell=False.",
                            "confidence": 0.95,
                        })

        seen = set()
        deduped = []
        for f in findings:
            key = (f["title"], f["line"])
            if key not in seen:
                seen.add(key)
                deduped.append(f)
        return deduped


class PerformanceReviewAgent(BaseReviewAgent):
    """Identifies algorithmic inefficiencies, memory leaks, and redundant operations."""

    def __init__(self):
        super().__init__(
            name="Performance Review Agent",
            role="Identify performance bottlenecks, quadratic complexities, memory bloat, and redundant queries",
            focus_areas=[
                "Quadratic or exponential loops (O(N^2))",
                "N+1 database query patterns",
                "Unbounded data fetches (missing LIMIT/pagination)",
                "Memory leaks and unclosed file/socket descriptors",
                "Redundant object creation or serialization",
            ],
            category=FindingCategory.PERFORMANCE.value,
        )

    def analyze_heuristically(
        self,
        diff_content: str,
        filename: str,
        static_findings: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Documented Agent Heuristic Rules:
        - PERF-01: Unbounded database query without LIMIT
        - PERF-02: Database connection opened without context manager
        """
        static_findings = static_findings or []
        findings = []

        for idx, line in enumerate(diff_content.splitlines(), 1):
            if line.startswith("+") and not line.startswith("+++"):
                code = line[1:].strip()

                # Rule PERF-01: Unbounded SELECT
                if "SELECT " in code.upper() and "FROM " in code.upper() and "LIMIT " not in code.upper():
                    findings.append({
                        "title": "Unbounded Database Query (Missing LIMIT)",
                        "category": self.category,
                        "severity": "LOW",
                        "file": filename,
                        "line": idx,
                        "description": "SQL query lacks pagination or LIMIT clause, risking memory exhaustion.",
                        "evidence": code,
                        "suggested_fix": "Add a LIMIT clause or implement cursor pagination.",
                        "confidence": 0.70,
                    })

                # Rule PERF-02: Connection leak
                if "connect(" in code and "with " not in code:
                    findings.append({
                        "title": "Unmanaged Connection Resource",
                        "category": self.category,
                        "severity": "LOW",
                        "file": filename,
                        "line": idx,
                        "description": "Connection opened without context manager ('with' statement) may leak sockets/handles.",
                        "evidence": code,
                        "suggested_fix": "Wrap database connection in a 'with' context manager.",
                        "confidence": 0.75,
                    })

        seen = set()
        deduped = []
        for f in findings:
            key = (f["title"], f["line"])
            if key not in seen:
                seen.add(key)
                deduped.append(f)
        return deduped


class CodeQualityAgent(BaseReviewAgent):
    """Examines readability, PEP8, naming conventions, complexity, and maintainability."""

    def __init__(self):
        super().__init__(
            name="Code Quality Agent",
            role="Ensure clean code practices, PEP8 compliance, naming standards, and architectural clarity",
            focus_areas=[
                "PEP 8 violations",
                "Dead / Unused variables and imports",
                "Excessive function length or cognitive complexity",
                "Misleading or non-standard identifier naming",
                "Missing docstrings and type hints",
            ],
            category=FindingCategory.CODE_QUALITY.value,
        )

    def analyze_heuristically(
        self,
        diff_content: str,
        filename: str,
        static_findings: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Documented Agent Heuristic Rules:
        - Ingests Flake8 style violations
        - QUAL-01: Single-letter identifier assignment
        """
        static_findings = static_findings or []
        findings = []

        # Check static analysis style findings (Flake8)
        for sf in static_findings:
            if sf.get("source") == "Flake8" or "style" in sf.get("title", "").lower():
                findings.append({
                    "title": sf.get("title", "Code Style Finding"),
                    "category": self.category,
                    "severity": sf.get("severity", "LOW"),
                    "file": sf.get("file", filename),
                    "line": sf.get("line", 1),
                    "description": sf.get("description", ""),
                    "evidence": sf.get("evidence", ""),
                    "suggested_fix": sf.get("recommendation", "Refactor according to code standards."),
                    "confidence": 0.90,
                })

        for idx, line in enumerate(diff_content.splitlines(), 1):
            if line.startswith("+") and not line.startswith("+++"):
                code = line[1:].strip()

                # Rule QUAL-01: Single letter variable names (except conventional i, j, x, y, e, k, v)
                if re.search(r"^\s*[a-z]\s*=\s*", code) and not re.search(r"^\s*[ijxyekv_]\s*=\s*", code):
                    findings.append({
                        "title": "Non-Descriptive Identifier Name",
                        "category": self.category,
                        "severity": "INFORMATIONAL",
                        "file": filename,
                        "line": idx,
                        "description": "Single-letter variable names reduce readability and maintainability.",
                        "evidence": code,
                        "suggested_fix": "Use descriptive naming reflecting the entity's domain purpose.",
                        "confidence": 0.75,
                    })

                # Rule QUAL-02: Bare except clause
                if re.search(r"except\s*:", code):
                    findings.append({
                        "title": "Bare Exception Clause",
                        "category": self.category,
                        "severity": "MEDIUM",
                        "file": filename,
                        "line": idx,
                        "description": "Bare 'except:' catches SystemExit and KeyboardInterrupt, concealing critical execution faults.",
                        "evidence": code,
                        "suggested_fix": "Catch specific exceptions such as 'except Exception:' or domain-specific exceptions.",
                        "confidence": 0.90,
                    })

        seen = set()
        deduped = []
        for f in findings:
            key = (f["title"], f["line"])
            if key not in seen:
                seen.add(key)
                deduped.append(f)
        return deduped


class MultiAgentOrchestrator:
    """
    Coordinates the four specialized review agents, dispatches analysis,
    and aggregates agent-specific findings deterministically.
    Supports monolithic review for E1 and E6-A4.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434/api/generate",
        model: str = "qwen2.5-coder:3b-instruct",
        timeout: float = 30.0,
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.timeout = timeout
        self._ollama_reachable = None
        self._last_checked_ollama = 0.0
        self.agents = [
            BugDetectionAgent(),
            SecurityReviewAgent(),
            PerformanceReviewAgent(),
            CodeQualityAgent(),
        ]

    async def _check_ollama(self) -> bool:
        """Check if Ollama server is responsive (cached for 30s) using fast non-blocking probe."""
        import socket

        if os.environ.get("PR_SENTINEL_OFFLINE", "").lower() in ("1", "true", "yes"):
            return False

        now = time.time()
        if self._ollama_reachable is not None and (now - self._last_checked_ollama) < 30.0:
            return self._ollama_reachable

        # Fast socket probe directly to IPv4 to prevent Windows IPv6 SYN timeout hang
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.15)
        try:
            s.connect(("127.0.0.1", 11434))
            s.close()
            self._ollama_reachable = True
        except Exception:
            self._ollama_reachable = False
        finally:
            try:
                s.close()
            except Exception:
                pass

        self._last_checked_ollama = now
        return self._ollama_reachable

    def run_review_sync(
        self,
        diff_content: str,
        filename: str = "modified_code.py",
        repo_context: str = "",
        static_findings: Optional[List[Dict[str, Any]]] = None,
        pr_id: str = "pr_unknown",
    ) -> Dict[str, Any]:
        """
        Synchronous deterministic execution of the four review agents.
        Invokes Ollama model when online, with heuristic rule analyzer fallback.
        """
        static_findings = static_findings or []
        is_offline = os.environ.get("PR_SENTINEL_OFFLINE", "").lower() in ("1", "true", "yes")

        ollama_online = False
        if not is_offline:
            try:
                with httpx.Client(timeout=3.0) as client:
                    r = client.get("http://localhost:11434/api/tags")
                    ollama_online = (r.status_code == 200)
            except Exception:
                ollama_online = False

        results: Dict[str, Any] = {
            "agents_executed": [a.name for a in self.agents],
            "agent_findings": {},
            "agent_execution_status": {},
            "total_raw_findings": 0,
            "llm_used": False,
        }

        all_findings = []
        for agent in self.agents:
            agent_findings = []
            heuristic_findings = agent.analyze_heuristically(diff_content, filename, static_findings)
            agent_findings.extend(heuristic_findings)

            exec_status = ExecutionStatus.SUCCESS.value
            err_msg = None

            if ollama_online:
                llm_res, status, err = self._call_agent_llm_sync(agent, diff_content, filename, repo_context, static_findings)
                exec_status = status
                err_msg = err
                if llm_res:
                    results["llm_used"] = True
                    agent_findings.extend(llm_res)
            elif is_offline:
                exec_status = ExecutionStatus.SUCCESS.value
            else:
                exec_status = ExecutionStatus.SKIPPED.value
                err_msg = "Ollama service offline"

            deduped = self._deduplicate(agent_findings)
            results["agent_execution_status"][agent.name] = {
                "status": exec_status,
                "error": err_msg,
            }
            results["agent_findings"][agent.name] = {
                "role": agent.role,
                "focus_areas": agent.focus_areas,
                "findings_count": len(deduped),
                "findings": deduped,
            }
            all_findings.extend(deduped)

        results["total_raw_findings"] = len(all_findings)
        results["all_findings"] = all_findings
        return results

    async def run_review(
        self,
        diff_content: str,
        filename: str = "modified_code.py",
        repo_context: str = "",
        static_findings: Optional[List[Dict[str, Any]]] = None,
        pr_id: str = "pr_unknown",
    ) -> Dict[str, Any]:
        """
        Run all 4 agents deterministically, collecting their findings.
        Attempts LLM generation with Qwen2.5-Coder and seamlessly merges with agent heuristics.
        """
        if os.environ.get("PR_SENTINEL_OFFLINE", "").lower() in ("1", "true", "yes"):
            return self.run_review_sync(diff_content, filename, repo_context, static_findings, pr_id)
        static_findings = static_findings or []
        ollama_online = await self._check_ollama()
        results: Dict[str, Any] = {
            "agents_executed": [a.name for a in self.agents],
            "agent_findings": {},
            "agent_execution_status": {},
            "total_raw_findings": 0,
            "llm_used": False,
        }

        all_findings = []

        # Single-threaded deterministic execution per Protocol refinement 7
        for agent in self.agents:
            agent_findings = []
            exec_status = ExecutionStatus.SUCCESS.value
            err_msg = None

            # 1. Run agent-specific heuristic analysis
            heuristic_findings = agent.analyze_heuristically(diff_content, filename, static_findings)
            agent_findings.extend(heuristic_findings)

            # 2. Attempt LLM call if Ollama is accessible
            if ollama_online:
                llm_res, status, err = await self._call_agent_llm(agent, diff_content, filename, repo_context, static_findings)
                exec_status = status
                err_msg = err
                if llm_res:
                    results["llm_used"] = True
                    agent_findings.extend(llm_res)
            else:
                exec_status = ExecutionStatus.SKIPPED.value if not os.environ.get("PR_SENTINEL_OFFLINE") else ExecutionStatus.FALLBACK_USED.value

            results["agent_execution_status"][agent.name] = {
                "status": exec_status,
                "error": err_msg
            }

            # Deduplicate per agent
            deduped = self._deduplicate(agent_findings)
            results["agent_findings"][agent.name] = {
                "role": agent.role,
                "focus_areas": agent.focus_areas,
                "findings_count": len(deduped),
                "findings": deduped,
            }
            all_findings.extend(deduped)

        results["total_raw_findings"] = len(all_findings)
        results["all_findings"] = all_findings
        return results

    async def run_monolithic_review(
        self,
        diff_content: str,
        filename: str = "modified_code.py",
        repo_context: str = "",
        pr_id: str = "pr_unknown",
    ) -> List[ModelFinding]:
        """
        Single-prompt review for Configuration E1 (Standalone LLM) and E6-A4 (E6 w/o Multi-Agent).
        """
        ollama_online = await self._check_ollama()
        if not ollama_online:
            return [ModelFinding(
                pr_id=pr_id,
                source="llm_monolithic",
                category=FindingCategory.BUG.value,
                file_path=filename,
                line_start=1,
                evidence="",
                description="Ollama server offline or unreachable.",
                execution_status=ExecutionStatus.TOOL_ERROR.value,
            )]

        prompt = f"""You are an expert automated code reviewer.
Review the following pull request code diff and report any bugs, security vulnerabilities, or critical flaws.
Target File: {filename}

Repository Context:
{repo_context}

Code Diff:
```diff
{diff_content}
```

Instructions:
Respond ONLY with a JSON array of findings matching this exact schema:
[
  {{
    "title": "Clear concise finding title",
    "category": "BUG" | "SECURITY" | "PERFORMANCE" | "CODE_QUALITY",
    "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "INFORMATIONAL",
    "file": "{filename}",
    "line": <line_number_or_1>,
    "description": "Detailed explanation of the problem",
    "evidence": "Exact code snippet",
    "suggested_fix": "Concrete code correction",
    "confidence": 0.85
  }}
]
If no issues are found, return []."""

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    self.ollama_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data.get("response", "").strip()
                    parsed_dicts = self._parse_json_findings(raw_text, "Monolithic LLM", filename)
                    model_findings = []
                    for d in parsed_dicts:
                        model_findings.append(ModelFinding(
                            pr_id=pr_id,
                            source="llm_monolithic",
                            category=d.get("category", FindingCategory.BUG.value),
                            file_path=d.get("file", filename),
                            line_start=int(d.get("line", 1)),
                            evidence=d.get("evidence", ""),
                            severity=d.get("severity", SeverityLevel.MEDIUM.value),
                            confidence=float(d.get("confidence", 0.8)),
                            description=d.get("description", ""),
                            recommendation=d.get("suggested_fix", ""),
                            execution_status=ExecutionStatus.SUCCESS.value
                        ))
                    return model_findings
                else:
                    return [ModelFinding(
                        pr_id=pr_id,
                        source="llm_monolithic",
                        category=FindingCategory.BUG.value,
                        file_path=filename,
                        line_start=1,
                        evidence="",
                        description=f"Ollama returned HTTP status {resp.status_code}",
                        execution_status=ExecutionStatus.LLM_ERROR.value,
                    )]
        except Exception as e:
            return [ModelFinding(
                pr_id=pr_id,
                source="llm_monolithic",
                category=FindingCategory.BUG.value,
                file_path=filename,
                line_start=1,
                evidence="",
                description=f"Ollama invocation failed: {str(e)}",
                execution_status=ExecutionStatus.TIMEOUT.value if isinstance(e, httpx.TimeoutException) else ExecutionStatus.LLM_ERROR.value,
            )]

    def run_monolithic_review_sync(
        self,
        diff_content: str,
        filename: str = "modified_code.py",
        repo_context: str = "",
        pr_id: str = "pr_unknown",
    ) -> Dict[str, Any]:
        """
        Synchronous single-prompt review for Configuration E1, E4, and E6-A4.
        Returns dictionary with keys:
            "findings": List[ModelFinding]
            "execution_status": str
            "error": Optional[str]
        """
        if os.environ.get("PR_SENTINEL_OFFLINE", "").lower() in ("1", "true", "yes"):
            finding = ModelFinding(
                pr_id=pr_id,
                source="llm_monolithic",
                category=FindingCategory.UNKNOWN.value,
                file_path=filename,
                line_start=1,
                evidence="",
                description="Ollama offline or execution bypassed via PR_SENTINEL_OFFLINE.",
                execution_status=ExecutionStatus.SKIPPED.value,
            )
            return {
                "findings": [finding],
                "execution_status": ExecutionStatus.SKIPPED.value,
                "error": "Offline mode active",
            }

        prompt = f"""You are an expert automated code reviewer.
Review the following pull request code diff and report any bugs, security vulnerabilities, or critical flaws.
Target File: {filename}

Repository Context:
{repo_context}

Code Diff:
```diff
{diff_content}
```

Instructions:
Respond ONLY with a JSON array of findings matching this exact schema:
[
  {{
    "title": "Clear concise finding title",
    "category": "BUG" | "SECURITY" | "PERFORMANCE" | "CODE_QUALITY",
    "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "INFORMATIONAL",
    "file": "{filename}",
    "line": <line_number_or_1>,
    "description": "Detailed explanation of the problem",
    "evidence": "Exact code snippet",
    "suggested_fix": "Concrete code correction",
    "confidence": 0.85
  }}
]
If no issues are found, return []."""

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    self.ollama_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data.get("response", "").strip()
                    parsed = self._parse_json_findings(raw_text, "Monolithic Reviewer", filename)
                    model_findings = []
                    for d in parsed:
                        model_findings.append(ModelFinding(
                            pr_id=pr_id,
                            source="llm_monolithic",
                            category=d.get("category", FindingCategory.BUG.value),
                            file_path=d.get("file", filename),
                            line_start=int(d.get("line", 1)),
                            evidence=d.get("evidence", ""),
                            severity=d.get("severity", SeverityLevel.MEDIUM.value),
                            confidence=float(d.get("confidence", 0.85)),
                            description=d.get("description", ""),
                            recommendation=d.get("suggested_fix", ""),
                            execution_status=ExecutionStatus.SUCCESS.value
                        ))
                    return {
                        "findings": model_findings,
                        "execution_status": ExecutionStatus.SUCCESS.value,
                        "error": None,
                    }
                else:
                    return {
                        "findings": [ModelFinding(
                            pr_id=pr_id,
                            source="llm_monolithic",
                            category=FindingCategory.UNKNOWN.value,
                            file_path=filename,
                            line_start=1,
                            evidence="",
                            description=f"Ollama returned HTTP status {resp.status_code}",
                            execution_status=ExecutionStatus.LLM_ERROR.value,
                        )],
                        "execution_status": ExecutionStatus.LLM_ERROR.value,
                        "error": f"HTTP {resp.status_code}",
                    }
        except httpx.TimeoutException:
            return {
                "findings": [ModelFinding(
                    pr_id=pr_id,
                    source="llm_monolithic",
                    category=FindingCategory.UNKNOWN.value,
                    file_path=filename,
                    line_start=1,
                    evidence="",
                    description=f"Ollama timed out after {self.timeout}s",
                    execution_status=ExecutionStatus.TIMEOUT.value,
                )],
                "execution_status": ExecutionStatus.TIMEOUT.value,
                "error": f"Timeout after {self.timeout}s",
            }
        except Exception as e:
            return {
                "findings": [ModelFinding(
                    pr_id=pr_id,
                    source="llm_monolithic",
                    category=FindingCategory.UNKNOWN.value,
                    file_path=filename,
                    line_start=1,
                    evidence="",
                    description=f"Ollama invocation failed: {str(e)}",
                    execution_status=ExecutionStatus.TOOL_ERROR.value,
                )],
                "execution_status": ExecutionStatus.TOOL_ERROR.value,
                "error": str(e),
            }

    async def _call_agent_llm(
        self,
        agent: BaseReviewAgent,
        diff_content: str,
        filename: str,
        repo_context: str,
        static_findings: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
        """Call Ollama for agent-specific reasoning, capturing errors explicitly."""
        prompt = agent.build_agent_prompt(diff_content, filename, repo_context, static_findings)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    self.ollama_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data.get("response", "").strip()
                    findings = self._parse_json_findings(raw_text, agent.name, filename)
                    return findings, ExecutionStatus.SUCCESS.value, None
                else:
                    return [], ExecutionStatus.LLM_ERROR.value, f"HTTP {resp.status_code}"
        except httpx.TimeoutException:
            return [], ExecutionStatus.TIMEOUT.value, f"Timed out after {self.timeout}s"
        except Exception as exc:
            return [], ExecutionStatus.LLM_ERROR.value, str(exc)

    def _call_agent_llm_sync(
        self,
        agent: BaseReviewAgent,
        diff_content: str,
        filename: str,
        repo_context: str,
        static_findings: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
        """Call Ollama synchronously for agent-specific reasoning, capturing errors explicitly."""
        prompt = agent.build_agent_prompt(diff_content, filename, repo_context, static_findings)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    self.ollama_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data.get("response", "").strip()
                    findings = self._parse_json_findings(raw_text, agent.name, filename)
                    return findings, ExecutionStatus.SUCCESS.value, None
                else:
                    return [], ExecutionStatus.LLM_ERROR.value, f"HTTP {resp.status_code}"
        except httpx.TimeoutException:
            return [], ExecutionStatus.TIMEOUT.value, f"Timed out after {self.timeout}s"
        except Exception as exc:
            return [], ExecutionStatus.LLM_ERROR.value, str(exc)

    def _parse_json_findings(self, text: str, agent_name: str, default_file: str) -> List[Dict[str, Any]]:
        """Safely extract JSON array of findings from LLM output without greedy blowout."""
        if not text:
            return []
        
        # Clean markdown code blocks if any
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return self._normalize_findings(parsed, agent_name, default_file)
            elif isinstance(parsed, dict) and "findings" in parsed:
                return self._normalize_findings(parsed["findings"], agent_name, default_file)
        except Exception:
            pass

        # Non-greedy search for JSON array block
        match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return self._normalize_findings(parsed, agent_name, default_file)
            except Exception:
                pass

        return []

    def _normalize_findings(self, items: List[Any], agent_name: str, default_file: str) -> List[Dict[str, Any]]:
        normalized = []
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized.append({
                "title": item.get("title", "Review Finding"),
                "category": item.get("category", agent_name),
                "severity": item.get("severity", "MEDIUM").upper(),
                "file": item.get("file", default_file),
                "line": max(1, int(item.get("line", 1))),
                "description": item.get("description", ""),
                "evidence": item.get("evidence", ""),
                "suggested_fix": item.get("suggested_fix", ""),
                "confidence": float(item.get("confidence", 0.8)),
            })
        return normalized

    def _deduplicate(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique = []
        for f in findings:
            key = (f.get("title"), f.get("line"), f.get("file"))
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique


_ORCHESTRATOR_INSTANCE: Optional[MultiAgentOrchestrator] = None


def get_multi_agent_orchestrator() -> MultiAgentOrchestrator:
    """Return singleton instance of MultiAgentOrchestrator."""
    global _ORCHESTRATOR_INSTANCE
    if _ORCHESTRATOR_INSTANCE is None:
        _ORCHESTRATOR_INSTANCE = MultiAgentOrchestrator()
    return _ORCHESTRATOR_INSTANCE
