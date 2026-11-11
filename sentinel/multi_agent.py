"""
PR-Sentinel: Module 6 - Specialized Multi-Agent Code Review System
Implements four specialized review agents:
1. Bug Detection Agent (Logical bugs, condition errors, runtime exceptions)
2. Security Review Agent (Vulnerabilities, injection, credentials, auth)
3. Performance Review Agent (Inefficient loops, memory, redundant computations)
4. Code Quality Agent (Maintainability, readability, PEP8, naming, duplication)
Orchestrates agent execution using Qwen2.5-Coder via Ollama with robust offline fallback.
"""

import ast
import json
import re
from typing import Any, Dict, List, Optional
import httpx


class BaseReviewAgent:
    """Base class for specialized code review agents."""

    def __init__(self, name: str, role: str, focus_areas: List[str]):
        self.name = name
        self.role = role
        self.focus_areas = focus_areas

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
    "category": "{self.name}",
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
        static_findings: List[Dict[str, Any]]
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
                "Incorrect boolean conditions",
                "Division by zero",
            ],
        )

    def analyze_heuristically(
        self,
        diff_content: str,
        filename: str,
        static_findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        findings = []
        added_lines = []
        for idx, line in enumerate(diff_content.splitlines(), 1):
            if line.startswith("+") and not line.startswith("+++"):
                code = line[1:].strip()
                added_lines.append((idx, code))

                # Check for possible division by zero
                if re.search(r"/\s*0\b", code):
                    findings.append({
                        "title": "Definite ZeroDivisionError (Division by Literal Zero)",
                        "category": self.name,
                        "severity": "CRITICAL",
                        "file": filename,
                        "line": idx,
                        "description": "Division by literal zero will raise a ZeroDivisionError at runtime.",
                        "evidence": code,
                        "suggested_fix": "Fix divisor expression to avoid division by zero.",
                        "confidence": 1.0,
                    })
                elif re.search(r"/\s*[A-Za-z_][A-Za-z0-9_]*", code) and not re.search(r"if\s+.*!=\s*0", code):
                    match = re.search(r"/\s*([A-Za-z_][A-Za-z0-9_]*)", code)
                    var_name = match.group(1) if match else "variable"
                    if var_name not in ("len", "2", "10", "100"):
                        findings.append({
                            "title": f"Potential ZeroDivisionError on variable '{var_name}'",
                            "category": self.name,
                            "severity": "HIGH",
                            "file": filename,
                            "line": idx,
                            "description": f"Division by variable '{var_name}' without zero check can cause runtime crash.",
                            "evidence": code,
                            "suggested_fix": f"Add zero-check guard before division: if {var_name} != 0:",
                            "confidence": 0.85,
                        })

                # Check for mutable default argument in Python
                if re.search(r"def\s+\w+\(.*=\s*(\[\]|\{\})\)", code):
                    findings.append({
                        "title": "Mutable Default Argument in Function Definition",
                        "category": self.name,
                        "severity": "MEDIUM",
                        "file": filename,
                        "line": idx,
                        "description": "Default parameter values are evaluated once at function definition, causing shared state bugs.",
                        "evidence": code,
                        "suggested_fix": "Use 'arg=None' as default and initialize inside function: if arg is None: arg = []",
                        "confidence": 0.95,
                    })

                # Check for missing return or NoneType dereference
                if re.search(r"\.fetchone\(\)\[", code):
                    findings.append({
                        "title": "Unchecked Fetchone Subscripting",
                        "category": self.name,
                        "severity": "HIGH",
                        "file": filename,
                        "line": idx,
                        "description": "Accessing subscript on .fetchone() directly will raise TypeError if query returns no rows.",
                        "evidence": code,
                        "suggested_fix": "Store row in a variable first: row = result.fetchone(); return row[0] if row else None",
                        "confidence": 0.90,
                    })

        return findings


class SecurityReviewAgent(BaseReviewAgent):
    """Examines security vulnerabilities, injection flaws, hardcoded credentials, and auth."""

    def __init__(self):
        super().__init__(
            name="Security Review Agent",
            role="Identify vulnerabilities, CWE/OWASP risks, injection vectors, and hardcoded secrets",
            focus_areas=[
                "SQL Injection",
                "Command Injection",
                "Cross-Site Scripting (XSS)",
                "Hardcoded credentials and API keys",
                "Path Traversal",
                "Authentication & Authorization bypass",
                "Insecure Deserialization",
            ],
        )

    def analyze_heuristically(
        self,
        diff_content: str,
        filename: str,
        static_findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        findings = []

        # Incorporate static findings from Bandit / AST
        for sf in static_findings:
            if sf.get("severity") in ("CRITICAL", "HIGH", "MEDIUM"):
                findings.append({
                    "title": sf.get("title", "Security Finding"),
                    "category": self.name,
                    "severity": sf.get("severity", "HIGH"),
                    "file": sf.get("file", filename),
                    "line": sf.get("line", 1),
                    "description": sf.get("description", ""),
                    "evidence": sf.get("evidence", ""),
                    "suggested_fix": sf.get("recommendation", "Remediate security issue."),
                    "confidence": 0.95,
                })

        for idx, line in enumerate(diff_content.splitlines(), 1):
            if line.startswith("+") and not line.startswith("+++"):
                code = line[1:].strip()

                # SQL Injection check
                if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE)\b", code, re.I) and ("+" in code or "f\"" in code or "f'" in code):
                    findings.append({
                        "title": "CWE-89: SQL Injection Vector",
                        "category": self.name,
                        "severity": "HIGH",
                        "file": filename,
                        "line": idx,
                        "description": "Dynamic SQL construction using untrusted string interpolation or concatenation.",
                        "evidence": code,
                        "suggested_fix": "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE name = %s', (name,))",
                        "confidence": 0.98,
                    })

                # Command Injection
                if re.search(r"\b(os\.system|subprocess\.Popen|subprocess\.call)\s*\(", code) and ("+" in code or "shell=True" in code or "f\"" in code):
                    findings.append({
                        "title": "CWE-78: Command Injection Risk",
                        "category": self.name,
                        "severity": "CRITICAL",
                        "file": filename,
                        "line": idx,
                        "description": "Shell command executed with dynamic concatenation or shell=True.",
                        "evidence": code,
                        "suggested_fix": "Pass command arguments as a list with shell=False: subprocess.run(['cmd', arg], check=True)",
                        "confidence": 0.95,
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


class PerformanceReviewAgent(BaseReviewAgent):
    """Examines algorithmic complexity, unnecessary computations, and resource efficiency."""

    def __init__(self):
        super().__init__(
            name="Performance Review Agent",
            role="Identify performance bottlenecks, redundant iterations, memory inefficiencies, and unindexed queries",
            focus_areas=[
                "Quadratic time complexity (O(N^2))",
                "Repeated database queries inside loops (N+1 query problem)",
                "Unbounded data fetches (missing pagination/LIMIT)",
                "Memory leaks and unclosed file/connection handles",
                "Redundant object serialization",
            ],
        )

    def analyze_heuristically(
        self,
        diff_content: str,
        filename: str,
        static_findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        findings = []

        # Check for unindexed / unbounded SELECT * without LIMIT
        for idx, line in enumerate(diff_content.splitlines(), 1):
            if line.startswith("+") and not line.startswith("+++"):
                code = line[1:].strip()

                if re.search(r"SELECT\s+\*\s+FROM\s+\w+", code, re.I) and not re.search(r"LIMIT\s+\d+", code, re.I):
                    findings.append({
                        "title": "Unbounded Database Query Without LIMIT Clause",
                        "category": self.name,
                        "severity": "LOW",
                        "file": filename,
                        "line": idx,
                        "description": "Selecting all columns without pagination or LIMIT can degrade query latency on large datasets.",
                        "evidence": code,
                        "suggested_fix": "Specify required columns explicitly and apply pagination using LIMIT and OFFSET.",
                        "confidence": 0.80,
                    })

                # Check for connection opened without context manager (with)
                if re.search(r"connect\(", code) and not re.search(r"with\s+", code):
                    findings.append({
                        "title": "Resource Connection Without Context Manager",
                        "category": self.name,
                        "severity": "LOW",
                        "file": filename,
                        "line": idx,
                        "description": "Database or socket connections should use a context manager ('with') to prevent connection leaks.",
                        "evidence": code,
                        "suggested_fix": "Use context manager: with sqlite3.connect(...) as connection:",
                        "confidence": 0.85,
                    })

        return findings


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
        )

    def analyze_heuristically(
        self,
        diff_content: str,
        filename: str,
        static_findings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        findings = []

        # Check static analysis style findings (Flake8)
        for sf in static_findings:
            if sf.get("source") == "Flake8" or "style" in sf.get("title", "").lower():
                findings.append({
                    "title": sf.get("title", "Code Style Finding"),
                    "category": self.name,
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

                # Single letter variable names
                if re.search(r"^\s*[a-z]\s*=\s*", code) and not re.search(r"^\s*[ijxye]\s*=\s*", code):
                    findings.append({
                        "title": "Non-Descriptive Identifier Name",
                        "category": self.name,
                        "severity": "INFORMATIONAL",
                        "file": filename,
                        "line": idx,
                        "description": "Single-letter variable names reduce readability and maintainability.",
                        "evidence": code,
                        "suggested_fix": "Use descriptive naming reflecting the entity's domain purpose.",
                        "confidence": 0.75,
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


class MultiAgentOrchestrator:
    """
    Coordinates the four specialized review agents, dispatches analysis,
    and aggregates agent-specific findings.
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
        """Check if Ollama server is responsive (cached for 30s)."""
        import os
        import time

        if os.environ.get("PR_SENTINEL_OFFLINE", "").lower() in ("1", "true", "yes"):
            return False

        now = time.time()
        if self._ollama_reachable is not None and (now - self._last_checked_ollama) < 30.0:
            return self._ollama_reachable

        try:
            probe_url = self.ollama_url.replace("/api/generate", "/api/tags")
            async with httpx.AsyncClient(timeout=0.8) as client:
                resp = await client.get(probe_url)
                self._ollama_reachable = (resp.status_code == 200)
        except Exception:
            self._ollama_reachable = False

        self._last_checked_ollama = now
        return self._ollama_reachable

    async def run_review(
        self,
        diff_content: str,
        filename: str = "modified_code.py",
        repo_context: str = "",
        static_findings: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run all 4 agents sequentially or in parallel, collecting their findings.
        Attempts LLM generation with Qwen2.5-Coder and seamlessly merges with agent heuristics.
        """
        static_findings = static_findings or []
        ollama_online = await self._check_ollama()
        results: Dict[str, Any] = {
            "agents_executed": [a.name for a in self.agents],
            "agent_findings": {},
            "total_raw_findings": 0,
            "llm_used": False,
        }

        all_findings = []

        for agent in self.agents:
            agent_findings = []
            # 1. Run agent-specific heuristic / static analysis
            heuristic_findings = agent.analyze_heuristically(diff_content, filename, static_findings)
            agent_findings.extend(heuristic_findings)

            # 2. Attempt LLM call if Ollama is accessible
            if ollama_online:
                llm_findings = await self._call_agent_llm(agent, diff_content, filename, repo_context, static_findings)
                if llm_findings:
                    results["llm_used"] = True
                    agent_findings.extend(llm_findings)

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

    async def _call_agent_llm(
        self,
        agent: BaseReviewAgent,
        diff_content: str,
        filename: str,
        repo_context: str,
        static_findings: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Call Ollama for agent-specific reasoning."""
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
                    return self._parse_json_findings(raw_text, agent.name, filename)
        except Exception:
            pass
        return []

    def _parse_json_findings(self, text: str, agent_name: str, default_file: str) -> List[Dict[str, Any]]:
        """Safely extract JSON array of findings from LLM output."""
        if not text:
            return []
        try:
            # Direct parse
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return self._normalize_findings(parsed, agent_name, default_file)
            elif isinstance(parsed, dict) and "findings" in parsed:
                return self._normalize_findings(parsed["findings"], agent_name, default_file)
        except Exception:
            pass

        # Regex search for JSON array block
        match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
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
                "line": item.get("line", 1),
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
