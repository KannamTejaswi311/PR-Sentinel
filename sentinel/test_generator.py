"""
PR-Sentinel: Module 8 - Automated Test Recommendations Engine
Generates comprehensive unit, boundary, negative, and security test scenarios
based on synthesized code review findings and modified source code.
"""

import ast
import re
from typing import Any, Dict, List, Optional


class AutomatedTestGenerator:
    """
    Generates actionable, structured automated test recommendations
    for modified functions and detected code vulnerabilities.
    """

    def generate_recommendations(
        self,
        diff_content: str,
        findings: List[Dict[str, Any]],
        filename: str = "example.py"
    ) -> List[Dict[str, Any]]:
        """
        Generate targeted test scenarios covering security, boundary conditions,
        functional requirements, and negative inputs.
        """
        recommendations = []

        # 1. Identify functions modified in diff
        modified_functions = self._extract_modified_functions(diff_content)

        # 2. Generate security test cases based on detected vulnerabilities
        for f in findings:
            title_lower = f.get("title", "").lower()
            target_func = self._guess_function_for_finding(f, modified_functions)

            if "sql" in title_lower or "injection" in title_lower:
                recommendations.append({
                    "test_type": "Security / Negative Test",
                    "target_component": target_func,
                    "scenario": "SQL Injection Resistance & Parameterization Verification",
                    "input_conditions": "username=\"admin' OR '1'='1--\"",
                    "expected_result": "Safe execution via query parameterization; returns None or empty match without SQL syntax error",
                    "rationale": "Validates that user input cannot alter SQL AST structure or bypass authentication checks (CWE-89).",
                    "sample_test_code": (
                        f"def test_{target_func}_sql_injection_resilience():\n"
                        f"    malicious_payload = \"admin' OR '1'='1--\"\n"
                        f"    result = {target_func}(malicious_payload)\n"
                        f"    assert result is None or result.get('username') == malicious_payload"
                    ),
                    "status": "Recommended (Not Executed)",
                })

            elif "command" in title_lower:
                recommendations.append({
                    "test_type": "Security Test",
                    "target_component": target_func,
                    "scenario": "Shell Command Injection Prevention",
                    "input_conditions": "cmd_arg=\"; echo 'INJECTED' ;\"",
                    "expected_result": "Command execution fails or argument is escaped; shell metacharacter is treated as literal data",
                    "rationale": "Prevents arbitrary command execution and privilege escalation (CWE-78).",
                    "sample_test_code": (
                        f"def test_{target_func}_command_injection_prevention():\n"
                        f"    payload = '; id;'\n"
                        f"    with pytest.raises(Exception):\n"
                        f"        {target_func}(payload)"
                    ),
                    "status": "Recommended (Not Executed)",
                })

            elif "secret" in title_lower or "credential" in title_lower or "password" in title_lower:
                recommendations.append({
                    "test_type": "Secret Scanning Test",
                    "target_component": target_func,
                    "scenario": "Environment Variable Secret Loading Verification",
                    "input_conditions": "Environment variable set via os.environ['DB_PASSWORD']='test_pw'",
                    "expected_result": "Credentials load cleanly from environment; no plaintext secrets present in codebase",
                    "rationale": "Ensures adherence to 12-factor configuration and eliminates credential leakage.",
                    "sample_test_code": (
                        f"def test_{target_func}_loads_from_env(monkeypatch):\n"
                        f"    monkeypatch.setenv('SECRET_KEY', 'test_key_123')\n"
                        f"    res = {target_func}()\n"
                        f"    assert res is not None"
                    ),
                    "status": "Recommended (Not Executed)",
                })

            elif "zero" in title_lower or "division" in title_lower:
                recommendations.append({
                    "test_type": "Boundary / Edge-Case Test",
                    "target_component": target_func,
                    "scenario": "Division by Zero Boundary Guard",
                    "input_conditions": "divisor=0",
                    "expected_result": "Function handles divisor=0 gracefully by raising ValueError or returning a default value",
                    "rationale": "Prevents uncaught ZeroDivisionError crashes in production.",
                    "sample_test_code": (
                        f"def test_{target_func}_zero_divisor():\n"
                        f"    with pytest.raises(ValueError):\n"
                        f"        {target_func}(divisor=0)"
                    ),
                    "status": "Recommended (Not Executed)",
                })

        # 3. Generate baseline unit & boundary tests for all modified functions
        for func_name in modified_functions:
            # Boundary test for None/Empty
            recommendations.append({
                "test_type": "Boundary Condition Test",
                "target_component": func_name,
                "scenario": f"Null / Empty Input Handling for {func_name}()",
                "input_conditions": "input=None, empty string '', or empty collection []",
                "expected_result": "Function returns sensible default (e.g. None or []) without throwing UnboundLocalError or TypeError",
                "rationale": "Ensures robust input validation on boundary and missing data.",
                "sample_test_code": (
                    f"def test_{func_name}_empty_input():\n"
                    f"    result = {func_name}('')\n"
                    f"    assert result is None or result == []"
                ),
                "status": "Recommended (Not Executed)",
            })

            # Happy path unit test
            recommendations.append({
                "test_type": "Functional Unit Test",
                "target_component": func_name,
                "scenario": f"Standard Happy Path Execution for {func_name}()",
                "input_conditions": "Valid canonical test arguments",
                "expected_result": "Function produces correct deterministic output and maintains expected return type",
                "rationale": "Verifies primary business logic requirements.",
                "sample_test_code": (
                    f"def test_{func_name}_happy_path():\n"
                    f"    result = {func_name}('valid_test_input')\n"
                    f"    assert result is not None"
                ),
                "status": "Recommended (Not Executed)",
            })

        # Deduplicate recommendations by scenario and target
        seen = set()
        unique = []
        for r in recommendations:
            key = (r["target_component"], r["scenario"])
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique

    def _extract_modified_functions(self, diff_content: str) -> List[str]:
        """Parse diff to find names of functions defined or touched."""
        funcs = []
        for line in diff_content.splitlines():
            # Match def function_name(...)
            m = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
            if m:
                funcs.append(m.group(1))

        if not funcs:
            funcs.append("main_handler")
        return list(dict.fromkeys(funcs))

    def _guess_function_for_finding(self, finding: Dict[str, Any], funcs: List[str]) -> str:
        """Associate finding with the most relevant modified function."""
        evidence = finding.get("evidence", "")
        for func in funcs:
            if func in evidence:
                return func
        return funcs[0] if funcs else "handler"


_TEST_GENERATOR_INSTANCE: Optional[AutomatedTestGenerator] = None


def get_test_generator() -> AutomatedTestGenerator:
    """Return singleton instance of AutomatedTestGenerator."""
    global _TEST_GENERATOR_INSTANCE
    if _TEST_GENERATOR_INSTANCE is None:
        _TEST_GENERATOR_INSTANCE = AutomatedTestGenerator()
    return _TEST_GENERATOR_INSTANCE
