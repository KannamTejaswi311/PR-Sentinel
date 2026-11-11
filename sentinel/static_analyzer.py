"""
PR-Sentinel: Module 3 - Static Code Analysis Engine
Integrates Bandit, Flake8, and Python AST analysis to detect deterministic
security vulnerabilities, code style violations, and logical flaws.
"""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


class StaticAnalyzer:
    """
    Orchestrates static analysis using Bandit, Flake8, and AST checks,
    normalizing findings into a unified schema for multi-agent review and synthesis.
    """

    def __init__(self, python_exe: Optional[str] = None):
        self.python_exe = python_exe or sys.executable

    def analyze_diff(self, diff_content: str, filename: str = "patch.py") -> List[Dict[str, Any]]:
        """
        Analyze code diff by reconstructing target file content and running
        deterministic static analysis tools.
        """
        findings: List[Dict[str, Any]] = []

        # 1. Extract added / modified code blocks from diff
        extracted_code, target_file, line_offset = self._extract_code_from_diff(diff_content, filename)

        if not extracted_code.strip():
            return findings

        # 2. Run AST-based structural code analysis
        ast_findings = self._run_ast_analysis(extracted_code, target_file, line_offset)
        findings.extend(ast_findings)

        # 3. Create a temporary file to run Bandit and Flake8
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=Path(target_file).suffix or ".py",
            delete=False,
            encoding="utf-8"
        ) as tmp:
            tmp.write(extracted_code)
            tmp_path = tmp.name

        try:
            # 4. Run Bandit security analysis
            bandit_findings = self._run_bandit(tmp_path, target_file, line_offset)
            findings.extend(bandit_findings)

            # 5. Run Flake8 code quality & style analysis
            flake8_findings = self._run_flake8(tmp_path, target_file, line_offset)
            findings.extend(flake8_findings)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        # 6. Deduplicate findings
        return self._deduplicate_findings(findings)

    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Run static analysis on an existing repository file."""
        findings: List[Dict[str, Any]] = []
        path = Path(file_path)
        if not path.is_file():
            return findings

        try:
            code = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return findings

        findings.extend(self._run_ast_analysis(code, file_path, line_offset=0))
        findings.extend(self._run_bandit(file_path, file_path, line_offset=0))
        findings.extend(self._run_flake8(file_path, file_path, line_offset=0))

        return self._deduplicate_findings(findings)

    def _extract_code_from_diff(self, diff_content: str, default_filename: str) -> tuple[str, str, int]:
        """
        Extract code additions from unified diff and identify target filename.
        """
        target_file = default_filename
        # Check diff headers for filename
        file_match = re.search(r"^\+\+\+\s+b/(.+)$", diff_content, re.MULTILINE)
        if file_match:
            target_file = file_match.group(1).strip()
        else:
            git_match = re.search(r"^diff --git a/.+ b/(.+)$", diff_content, re.MULTILINE)
            if git_match:
                target_file = git_match.group(1).strip()

        # Parse hunks to get added code and starting line number
        lines = diff_content.splitlines()
        first_line = 1
        code_lines = []

        hunk_started = False
        for line in lines:
            hunk_match = re.search(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if hunk_match:
                if not hunk_started:
                    first_line = int(hunk_match.group(1))
                    hunk_started = True
                continue

            if hunk_started:
                if line.startswith("+") and not line.startswith("+++"):
                    code_lines.append(line[1:])
                elif line.startswith(" "):
                    code_lines.append(line[1:])
                elif not line.startswith("-"):
                    code_lines.append(line)

        # Fallback if no diff headers were matched
        if not code_lines:
            for line in lines:
                if line.startswith("+") and not line.startswith("+++"):
                    code_lines.append(line[1:])
                elif not line.startswith("-") and not line.startswith("@@"):
                    code_lines.append(line)

        return "\n".join(code_lines), target_file, first_line - 1

    def _run_bandit(self, temp_file_path: str, original_file: str, line_offset: int) -> List[Dict[str, Any]]:
        """Run Bandit security analysis via subprocess."""
        findings = []
        try:
            # Check if bandit is executable in python env
            cmd = [self.python_exe, "-m", "bandit", "-f", "json", "-q", temp_file_path]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
                check=False
            )
            output = result.stdout.strip()
            if not output:
                return findings

            data = json.loads(output)
            results = data.get("results", [])

            for item in results:
                raw_line = item.get("line_number", 1)
                adjusted_line = max(1, raw_line + line_offset)
                severity = item.get("issue_severity", "MEDIUM").upper()

                findings.append({
                    "source": "Bandit",
                    "rule_id": item.get("test_id", "B000"),
                    "title": item.get("test_name", "Security Vulnerability"),
                    "description": item.get("issue_text", "Detected security concern"),
                    "severity": severity,
                    "confidence": item.get("issue_confidence", "HIGH").upper(),
                    "file": original_file,
                    "line": adjusted_line,
                    "evidence": item.get("code", "").strip(),
                    "recommendation": item.get("more_info", "Review security best practices."),
                })
        except Exception:
            pass
        return findings

    def _run_flake8(self, temp_file_path: str, original_file: str, line_offset: int) -> List[Dict[str, Any]]:
        """Run Flake8 linting and code quality analysis."""
        findings = []
        try:
            cmd = [
                self.python_exe,
                "-m",
                "flake8",
                "--format=%(code)s|%(row)d|%(col)d|%(text)s",
                "--max-line-length=120",
                temp_file_path
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
                check=False
            )
            output = result.stdout.strip()
            if not output:
                return findings

            for line in output.splitlines():
                parts = line.split("|", 3)
                if len(parts) == 4:
                    code, row, col, text = parts
                    row_num = max(1, int(row) + line_offset)
                    # Classify severity based on Flake8 error codes
                    # F: Pyflakes (errors, unused vars), E/W: PEP8, C: Complexity
                    if code.startswith("F821") or code.startswith("F822") or code.startswith("F831"):
                        severity = "HIGH"
                        category = "Undefined Variable / Runtime Error"
                    elif code.startswith("F"):
                        severity = "MEDIUM"
                        category = "Code Quality / Logic Issue"
                    elif code.startswith("E") or code.startswith("W"):
                        severity = "LOW"
                        category = "Code Style / PEP8"
                    elif code.startswith("C"):
                        severity = "MEDIUM"
                        category = "Cyclomatic Complexity"
                    else:
                        severity = "INFORMATIONAL"
                        category = "Code Quality"

                    findings.append({
                        "source": "Flake8",
                        "rule_id": code,
                        "title": f"Linting {code}: {category}",
                        "description": text.strip(),
                        "severity": severity,
                        "confidence": "HIGH",
                        "file": original_file,
                        "line": row_num,
                        "evidence": f"Column {col}",
                        "recommendation": "Fix style or linting issue according to PEP8 guidelines.",
                    })
        except Exception:
            pass
        return findings

    def _run_ast_analysis(self, code: str, filename: str, line_offset: int) -> List[Dict[str, Any]]:
        """
        Deterministic AST-level inspection for security vulnerabilities and code smells.
        """
        findings = []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            findings.append({
                "source": "AST_Parser",
                "rule_id": "SYNTAX_ERR",
                "title": "Syntax Error",
                "description": f"Code contains syntax error: {e.msg}",
                "severity": "CRITICAL",
                "confidence": "HIGH",
                "file": filename,
                "line": max(1, (e.lineno or 1) + line_offset),
                "evidence": e.text.strip() if e.text else "",
                "recommendation": "Correct syntax error before deployment.",
            })
            return findings

        # Walk AST
        for node in ast.walk(tree):
            line_no = max(1, getattr(node, "lineno", 1) + line_offset)

            # 1. Check for dangerous builtin calls: eval, exec
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in ("eval", "exec"):
                    findings.append({
                        "source": "AST_Security",
                        "rule_id": "AST_DANGEROUS_CALL",
                        "title": f"Unsafe Dynamic Execution ({func_name})",
                        "description": f"Use of {func_name}() can allow arbitrary remote code execution.",
                        "severity": "CRITICAL",
                        "confidence": "HIGH",
                        "file": filename,
                        "line": line_no,
                        "evidence": f"Call to '{func_name}'",
                        "recommendation": "Avoid eval() and exec(). Use ast.literal_eval() or safe parsers.",
                    })

                # 2. Check for sqlite3/DB execute with string concatenation or f-strings
                if func_name == "execute" and node.args:
                    first_arg = node.args[0]
                    # Check string concatenation: query + username
                    if isinstance(first_arg, ast.BinOp) and isinstance(first_arg.op, ast.Add):
                        findings.append({
                            "source": "AST_Security",
                            "rule_id": "AST_SQL_CONCAT",
                            "title": "SQL Injection Vector (String Concatenation)",
                            "description": "SQL statement constructed by concatenating variables directly into execute().",
                            "severity": "HIGH",
                            "confidence": "HIGH",
                            "file": filename,
                            "line": line_no,
                            "evidence": "execute(query + ...)",
                            "recommendation": "Use parameterized queries or prepared statements (e.g. execute(query, (params,)))",
                        })
                    # Check f-strings
                    elif isinstance(first_arg, ast.JoinedStr):
                        findings.append({
                            "source": "AST_Security",
                            "rule_id": "AST_SQL_FSTRING",
                            "title": "SQL Injection Vector (Formatted String)",
                            "description": "SQL statement constructed using f-string interpolation inside execute().",
                            "severity": "HIGH",
                            "confidence": "HIGH",
                            "file": filename,
                            "line": line_no,
                            "evidence": "execute(f'...{param}...')",
                            "recommendation": "Use parameterized queries with placeholders instead of string formatting.",
                        })

            # 3. Check for bare except:
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                findings.append({
                    "source": "AST_Quality",
                    "rule_id": "AST_BARE_EXCEPT",
                    "title": "Bare Exception Handler",
                    "description": "Using bare 'except:' catches SystemExit, KeyboardInterrupt, and masks critical bugs.",
                    "severity": "MEDIUM",
                    "confidence": "HIGH",
                    "file": filename,
                    "line": line_no,
                    "evidence": "except:",
                    "recommendation": "Catch specific exceptions (e.g., except ValueError:, except Exception:).",
                })

            # 4. Check for hardcoded secrets in variable assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name_lower = target.id.lower()
                        if any(k in name_lower for k in ("password", "secret", "api_key", "token", "auth_key")):
                            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                val = node.value.value
                                if len(val) >= 6 and val not in ("default", "change_me", "none", "secret", ""):
                                    findings.append({
                                        "source": "AST_Security",
                                        "rule_id": "AST_HARDCODED_SECRET",
                                        "title": f"Hardcoded Secret ({target.id})",
                                        "description": f"Variable '{target.id}' contains a plaintext hardcoded secret.",
                                        "severity": "HIGH",
                                        "confidence": "HIGH",
                                        "file": filename,
                                        "line": line_no,
                                        "evidence": f"{target.id} = '***'",
                                        "recommendation": "Extract credentials to environment variables or secret managers.",
                                    })

        return findings

    def _deduplicate_findings(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate findings based on file, line, and rule or title."""
        seen = set()
        unique = []
        for f in findings:
            key = (f.get("file"), f.get("line"), f.get("rule_id"), f.get("title"))
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique


_STATIC_ANALYZER_INSTANCE: Optional[StaticAnalyzer] = None


def get_static_analyzer() -> StaticAnalyzer:
    """Return singleton instance of StaticAnalyzer."""
    global _STATIC_ANALYZER_INSTANCE
    if _STATIC_ANALYZER_INSTANCE is None:
        _STATIC_ANALYZER_INSTANCE = StaticAnalyzer()
    return _STATIC_ANALYZER_INSTANCE
