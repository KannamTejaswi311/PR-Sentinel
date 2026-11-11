"""
PR-Sentinel: Module 3 - Static Code Analysis Engine
Integrates Bandit, Flake8, and Python AST analysis to detect deterministic
security vulnerabilities, code style violations, and logical flaws.
Adheres to Frozen Experimental Protocol v1.0 and Amendment v1.1.
"""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sentinel.schemas import (
    ModelFinding,
    FindingCategory,
    SeverityLevel,
    ExecutionStatus,
    FindingGranularity,
)


class StaticAnalyzer:
    """
    Orchestrates static analysis using Bandit, Flake8, and AST checks across
    all files and hunks in a multi-file unified diff.
    """

    PYTHON_EXTENSIONS = {".py"}
    SUPPORTED_EXTENSIONS = {".py"}

    def __init__(self, python_exe: Optional[str] = None):
        self.python_exe = python_exe or sys.executable

    def analyze_diff(self, diff_content: str, filename: str = "patch.py", pr_id: str = "pr_unknown") -> List[Dict[str, Any]]:
        """
        Analyze code diff by reconstructing target file content across all modified files
        and hunks, running deterministic static analysis tools.
        """
        findings: List[Dict[str, Any]] = []
        if not diff_content or not diff_content.strip():
            return findings

        # Split multi-file diff into per-file chunks
        file_diffs = self._split_multi_file_diff(diff_content, default_filename=filename)

        for file_path, file_diff_content in file_diffs.items():
            ext = Path(file_path).suffix.lower()

            # Handle unsupported languages explicitly per protocol
            if ext and ext not in self.PYTHON_EXTENSIONS:
                findings.append({
                    "source": "StaticAnalyzer",
                    "rule_id": "LANG_UNSUPPORTED",
                    "title": f"Unsupported Language ({ext})",
                    "description": f"Static analysis currently supports Python. File '{file_path}' has unsupported extension '{ext}'.",
                    "severity": "INFORMATIONAL",
                    "confidence": "HIGH",
                    "file": file_path,
                    "line": 1,
                    "evidence": "",
                    "recommendation": f"Add external static analyzer for '{ext}' (e.g. ESLint, SwiftLint, Clippy).",
                    "category": FindingCategory.CONVENTION.value,
                    "execution_status": ExecutionStatus.UNSUPPORTED_LANGUAGE.value,
                    "granularity": FindingGranularity.FILE.value,
                })
                continue

            # Extract added code with hunk-accurate line mapping
            extracted_code, line_map = self._extract_code_with_line_map(file_diff_content, file_path)

            if not extracted_code.strip():
                continue

            # 1. Run AST-based structural code analysis
            ast_findings = self._run_ast_analysis(extracted_code, file_path, line_map)
            findings.extend(ast_findings)

            # 2. Create temporary file for Bandit and Flake8
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                encoding="utf-8"
            ) as tmp:
                tmp.write(extracted_code)
                tmp_path = tmp.name

            try:
                # 3. Run Bandit security analysis
                bandit_findings = self._run_bandit(tmp_path, file_path, line_map)
                findings.extend(bandit_findings)

                # 4. Run Flake8 code quality & style analysis
                flake8_findings = self._run_flake8(tmp_path, file_path, line_map)
                findings.extend(flake8_findings)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        return self._deduplicate_findings(findings)

    def analyze_diff_to_findings(self, diff_content: str, pr_id: str = "pr_unknown", filename: str = "patch.py") -> List[ModelFinding]:
        """Convert findings to standard ModelFinding instances."""
        raw_findings = self.analyze_diff(diff_content, filename=filename, pr_id=pr_id)
        model_findings: List[ModelFinding] = []

        for f in raw_findings:
            src = f.get("source", "static_analyzer")
            if "bandit" in src.lower():
                source_tag = "static_bandit"
            elif "flake8" in src.lower():
                source_tag = "static_flake8"
            else:
                source_tag = "static_ast"

            cat = f.get("category", FindingCategory.CODE_QUALITY.value)
            if "security" in f.get("title", "").lower() or "sql" in f.get("title", "").lower() or "secret" in f.get("title", "").lower():
                cat = FindingCategory.SECURITY.value
            elif "syntax" in f.get("title", "").lower() or "runtime" in f.get("title", "").lower():
                cat = FindingCategory.BUG.value

            model_findings.append(ModelFinding(
                pr_id=pr_id,
                source=source_tag,
                category=cat,
                file_path=f.get("file", filename),
                line_start=f.get("line", 1),
                line_end=f.get("line", 1),
                evidence=f.get("evidence", ""),
                rule_id=f.get("rule_id"),
                severity=f.get("severity", SeverityLevel.MEDIUM.value),
                description=f.get("description", ""),
                recommendation=f.get("recommendation", ""),
                execution_status=f.get("execution_status", ExecutionStatus.SUCCESS.value),
                granularity=f.get("granularity", FindingGranularity.STATEMENT.value),
            ))

        return model_findings

    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Run static analysis on an existing repository file."""
        findings: List[Dict[str, Any]] = []
        path = Path(file_path)
        if not path.is_file():
            return findings

        ext = path.suffix.lower()
        if ext not in self.PYTHON_EXTENSIONS:
            findings.append({
                "source": "StaticAnalyzer",
                "rule_id": "LANG_UNSUPPORTED",
                "title": f"Unsupported Language ({ext})",
                "description": f"File '{file_path}' has unsupported extension '{ext}'.",
                "severity": "INFORMATIONAL",
                "confidence": "HIGH",
                "file": file_path,
                "line": 1,
                "evidence": "",
                "recommendation": "Use language-specific linter.",
                "execution_status": ExecutionStatus.UNSUPPORTED_LANGUAGE.value,
            })
            return findings

        try:
            code = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            findings.append({
                "source": "StaticAnalyzer",
                "rule_id": "READ_ERROR",
                "title": "File Read Error",
                "description": str(e),
                "severity": "HIGH",
                "confidence": "HIGH",
                "file": file_path,
                "line": 1,
                "evidence": "",
                "recommendation": "Verify file permissions.",
                "execution_status": ExecutionStatus.TOOL_ERROR.value,
            })
            return findings

        # Identity line map for full files
        line_map = {i: i for i in range(1, len(code.splitlines()) + 2)}
        findings.extend(self._run_ast_analysis(code, file_path, line_map))
        findings.extend(self._run_bandit(file_path, file_path, line_map))
        findings.extend(self._run_flake8(file_path, file_path, line_map))

        return self._deduplicate_findings(findings)

    def _split_multi_file_diff(self, diff_content: str, default_filename: str) -> Dict[str, str]:
        """Split a unified diff containing multiple files into a mapping of file_path -> file_diff."""
        file_diffs: Dict[str, List[str]] = {}
        current_file = default_filename
        current_lines: List[str] = []

        diff_file_pattern = re.compile(r"^diff --git a/(.+?)\s+b/(.+)$")
        plus_file_pattern = re.compile(r"^\+\+\+\s+(?:b/)?(.+)$")

        for line in diff_content.splitlines():
            m_diff = diff_file_pattern.match(line)
            m_plus = plus_file_pattern.match(line)

            if m_diff:
                if current_lines:
                    file_diffs.setdefault(current_file, []).extend(current_lines)
                    current_lines = []
                current_file = m_diff.group(2).strip()
                current_lines.append(line)
            elif m_plus and not line.startswith("+++ /dev/null"):
                # If diff --git was not present, catch +++ b/
                candidate = m_plus.group(1).strip()
                if candidate != current_file and not any(current_file in k for k in file_diffs):
                    current_file = candidate
                current_lines.append(line)
            else:
                current_lines.append(line)

        if current_lines:
            file_diffs.setdefault(current_file, []).extend(current_lines)

        return {f: "\n".join(lines) for f, lines in file_diffs.items() if lines}

    def _extract_code_with_line_map(self, file_diff: str, file_path: str) -> Tuple[str, Dict[int, int]]:
        """
        Extract code additions from unified diff hunks and construct an accurate
        line_map mapping 1-based temp-file line numbers to real target file line numbers.
        """
        hunk_pattern = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
        extracted_lines: List[str] = []
        line_map: Dict[int, int] = {}  # temp_line_no -> real_target_line_no

        current_real_line = 1
        in_hunk = False
        temp_line_counter = 1

        for raw_line in file_diff.splitlines():
            hunk_match = hunk_pattern.match(raw_line)
            if hunk_match:
                current_real_line = int(hunk_match.group(1))
                in_hunk = True
                continue

            if in_hunk:
                if raw_line.startswith("+") and not raw_line.startswith("+++"):
                    extracted_lines.append(raw_line[1:])
                    line_map[temp_line_counter] = current_real_line
                    temp_line_counter += 1
                    current_real_line += 1
                elif raw_line.startswith(" "):
                    extracted_lines.append(raw_line[1:])
                    line_map[temp_line_counter] = current_real_line
                    temp_line_counter += 1
                    current_real_line += 1
                elif raw_line.startswith("-"):
                    # Deleted line: does not advance real target file line count
                    pass

        # Fallback if diff lacks hunk markers (plain code additions)
        if not extracted_lines:
            for raw_line in file_diff.splitlines():
                if raw_line.startswith("+") and not raw_line.startswith("+++"):
                    extracted_lines.append(raw_line[1:])
                    line_map[temp_line_counter] = temp_line_counter
                    temp_line_counter += 1

        return "\n".join(extracted_lines), line_map

    def _run_bandit(self, temp_file_path: str, original_file: str, line_map: Dict[int, int]) -> List[Dict[str, Any]]:
        """Run Bandit security analysis via subprocess, mapping line numbers accurately."""
        findings = []
        try:
            cmd = [self.python_exe, "-m", "bandit", "-f", "json", "-q", temp_file_path]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=20,
                check=False
            )
            output = result.stdout.strip()
            if not output:
                return findings

            data = json.loads(output)
            results = data.get("results", [])

            for item in results:
                raw_line = item.get("line_number", 1)
                real_line = line_map.get(raw_line, raw_line)
                severity = item.get("issue_severity", "MEDIUM").upper()

                findings.append({
                    "source": "Bandit",
                    "rule_id": item.get("test_id", "B000"),
                    "title": item.get("test_name", "Security Vulnerability"),
                    "description": item.get("issue_text", "Detected security concern"),
                    "severity": severity,
                    "confidence": item.get("issue_confidence", "HIGH").upper(),
                    "file": original_file,
                    "line": real_line,
                    "evidence": item.get("code", "").strip(),
                    "recommendation": item.get("more_info", "Review security best practices."),
                    "category": FindingCategory.SECURITY.value,
                    "execution_status": ExecutionStatus.SUCCESS.value,
                    "granularity": FindingGranularity.STATEMENT.value,
                })
        except subprocess.TimeoutExpired:
            findings.append({
                "source": "Bandit",
                "rule_id": "BANDIT_TIMEOUT",
                "title": "Bandit Timeout",
                "description": "Bandit security check timed out after 20 seconds.",
                "severity": "LOW",
                "confidence": "HIGH",
                "file": original_file,
                "line": 1,
                "evidence": "",
                "recommendation": "Analyze smaller diff chunks.",
                "category": FindingCategory.SECURITY.value,
                "execution_status": ExecutionStatus.TIMEOUT.value,
            })
        except Exception as exc:
            findings.append({
                "source": "Bandit",
                "rule_id": "BANDIT_ERROR",
                "title": "Bandit Tool Error",
                "description": f"Bandit execution failed: {exc}",
                "severity": "LOW",
                "confidence": "HIGH",
                "file": original_file,
                "line": 1,
                "evidence": "",
                "recommendation": "Verify bandit is installed in python environment.",
                "category": FindingCategory.SECURITY.value,
                "execution_status": ExecutionStatus.TOOL_ERROR.value,
            })
        return findings

    def _run_flake8(self, temp_file_path: str, original_file: str, line_map: Dict[int, int]) -> List[Dict[str, Any]]:
        """Run Flake8 linting and code quality analysis, mapping line numbers accurately."""
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
                timeout=20,
                check=False
            )
            output = result.stdout.strip()
            if not output:
                return findings

            for line in output.splitlines():
                parts = line.split("|", 3)
                if len(parts) == 4:
                    code, row, col, text = parts
                    temp_row = int(row)
                    real_row = line_map.get(temp_row, temp_row)

                    if code.startswith("F821") or code.startswith("F822") or code.startswith("F831"):
                        severity = "HIGH"
                        category = "Undefined Variable / Runtime Error"
                        std_cat = FindingCategory.BUG.value
                    elif code.startswith("F"):
                        severity = "MEDIUM"
                        category = "Code Quality / Logic Issue"
                        std_cat = FindingCategory.BUG.value
                    elif code.startswith("E") or code.startswith("W"):
                        severity = "LOW"
                        category = "Code Style / PEP8"
                        std_cat = FindingCategory.CODE_QUALITY.value
                    elif code.startswith("C"):
                        severity = "MEDIUM"
                        category = "Cyclomatic Complexity"
                        std_cat = FindingCategory.PERFORMANCE.value
                    else:
                        severity = "INFORMATIONAL"
                        category = "Code Quality"
                        std_cat = FindingCategory.CODE_QUALITY.value

                    findings.append({
                        "source": "Flake8",
                        "rule_id": code,
                        "title": f"Linting {code}: {category}",
                        "description": text.strip(),
                        "severity": severity,
                        "confidence": "HIGH",
                        "file": original_file,
                        "line": real_row,
                        "evidence": f"Column {col}",
                        "recommendation": "Fix style or linting issue according to PEP8 guidelines.",
                        "category": std_cat,
                        "execution_status": ExecutionStatus.SUCCESS.value,
                        "granularity": FindingGranularity.STATEMENT.value,
                    })
        except subprocess.TimeoutExpired:
            findings.append({
                "source": "Flake8",
                "rule_id": "FLAKE8_TIMEOUT",
                "title": "Flake8 Timeout",
                "description": "Flake8 execution timed out after 20 seconds.",
                "severity": "LOW",
                "confidence": "HIGH",
                "file": original_file,
                "line": 1,
                "evidence": "",
                "recommendation": "Check diff size.",
                "category": FindingCategory.CODE_QUALITY.value,
                "execution_status": ExecutionStatus.TIMEOUT.value,
            })
        except Exception as exc:
            findings.append({
                "source": "Flake8",
                "rule_id": "FLAKE8_ERROR",
                "title": "Flake8 Tool Error",
                "description": f"Flake8 execution failed: {exc}",
                "severity": "LOW",
                "confidence": "HIGH",
                "file": original_file,
                "line": 1,
                "evidence": "",
                "recommendation": "Verify flake8 is installed in python environment.",
                "category": FindingCategory.CODE_QUALITY.value,
                "execution_status": ExecutionStatus.TOOL_ERROR.value,
            })
        return findings

    def _run_ast_analysis(self, code: str, filename: str, line_map: Dict[int, int]) -> List[Dict[str, Any]]:
        """Deterministic AST-level inspection for security vulnerabilities and code smells."""
        findings = []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raw_line = e.lineno or 1
            real_line = line_map.get(raw_line, raw_line)
            findings.append({
                "source": "AST_Parser",
                "rule_id": "SYNTAX_ERR",
                "title": "Syntax Error",
                "description": f"Code contains syntax error: {e.msg}",
                "severity": "CRITICAL",
                "confidence": "HIGH",
                "file": filename,
                "line": real_line,
                "evidence": e.text.strip() if e.text else "",
                "recommendation": "Correct syntax error before deployment.",
                "category": FindingCategory.BUG.value,
                "execution_status": ExecutionStatus.SUCCESS.value,
                "granularity": FindingGranularity.STATEMENT.value,
            })
            return findings

        # Walk AST
        for node in ast.walk(tree):
            raw_line = getattr(node, "lineno", 1)
            real_line = line_map.get(raw_line, raw_line)

            # 1. Dangerous builtin calls: eval, exec
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
                        "line": real_line,
                        "evidence": f"Call to '{func_name}'",
                        "recommendation": "Avoid eval() and exec(). Use ast.literal_eval() or safe parsers.",
                        "category": FindingCategory.SECURITY.value,
                        "execution_status": ExecutionStatus.SUCCESS.value,
                        "granularity": FindingGranularity.STATEMENT.value,
                    })

                # 2. SQL injection via execute(query + var) or execute(f"...")
                if func_name == "execute" and node.args:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.BinOp) and isinstance(first_arg.op, ast.Add):
                        findings.append({
                            "source": "AST_Security",
                            "rule_id": "AST_SQL_CONCAT",
                            "title": "SQL Injection Vector (String Concatenation)",
                            "description": "SQL statement constructed by concatenating variables directly into execute().",
                            "severity": "HIGH",
                            "confidence": "HIGH",
                            "file": filename,
                            "line": real_line,
                            "evidence": "execute(query + ...)",
                            "recommendation": "Use parameterized queries or prepared statements (e.g. execute(query, (params,)))",
                            "category": FindingCategory.SECURITY.value,
                            "execution_status": ExecutionStatus.SUCCESS.value,
                            "granularity": FindingGranularity.STATEMENT.value,
                        })
                    elif isinstance(first_arg, ast.JoinedStr):
                        findings.append({
                            "source": "AST_Security",
                            "rule_id": "AST_SQL_FSTRING",
                            "title": "SQL Injection Vector (Formatted String)",
                            "description": "SQL statement constructed using f-string interpolation inside execute().",
                            "severity": "HIGH",
                            "confidence": "HIGH",
                            "file": filename,
                            "line": real_line,
                            "evidence": "execute(f'...{param}...')",
                            "recommendation": "Use parameterized queries with placeholders instead of string formatting.",
                            "category": FindingCategory.SECURITY.value,
                            "execution_status": ExecutionStatus.SUCCESS.value,
                            "granularity": FindingGranularity.STATEMENT.value,
                        })

            # 3. Bare except:
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                findings.append({
                    "source": "AST_Quality",
                    "rule_id": "AST_BARE_EXCEPT",
                    "title": "Bare Exception Handler",
                    "description": "Using bare 'except:' catches SystemExit, KeyboardInterrupt, and masks critical bugs.",
                    "severity": "MEDIUM",
                    "confidence": "HIGH",
                    "file": filename,
                    "line": real_line,
                    "evidence": "except:",
                    "recommendation": "Catch specific exceptions (e.g., except ValueError:, except Exception:).",
                    "category": FindingCategory.BUG.value,
                    "execution_status": ExecutionStatus.SUCCESS.value,
                    "granularity": FindingGranularity.STATEMENT.value,
                })

            # 4. Hardcoded secrets
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
                                        "line": real_line,
                                        "evidence": f"{target.id} = '***'",
                                        "recommendation": "Extract credentials to environment variables or secret managers.",
                                        "category": FindingCategory.SECURITY.value,
                                        "execution_status": ExecutionStatus.SUCCESS.value,
                                        "granularity": FindingGranularity.STATEMENT.value,
                                    })

        return findings

    def _deduplicate_findings(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate findings based on file, line, rule_id, and title."""
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
