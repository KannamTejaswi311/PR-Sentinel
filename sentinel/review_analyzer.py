import re


# ============================================================
# SECURITY CATEGORY PATTERNS
# ============================================================

SECURITY_PATTERNS = {
    "SQL Injection": [
        r"sql\s+injection",
        r"sql\s+inject",
    ],

    "Cross-Site Scripting (XSS)": [
        r"cross[- ]site\s+scripting",
        r"\bxss\b",
    ],

    "Command Injection": [
        r"command\s+injection",
    ],

    "Hardcoded Credentials": [
        r"hardcoded\s+(password|credential|credentials|secret)",
    ],

    "API Key Exposure": [
        r"api\s*key\s*exposure",
        r"exposed\s+api\s*key",
    ],

    "Access Token Exposure": [
        r"access\s+token\s+exposure",
        r"exposed\s+access\s+token",
    ],

    "Sensitive Information Exposure": [
        r"sensitive\s+information\s+exposure",
        r"sensitive\s+data\s+exposure",
    ],

    "Authentication Vulnerability": [
        r"authentication\s+vulnerab",
        r"authentication\s+bypass",
        r"weak\s+authentication",
        r"missing\s+authentication",
    ],

    "Authorization Vulnerability": [
        r"authorization\s+vulnerab",
        r"authorization\s+bypass",
        r"broken\s+authorization",
        r"missing\s+authorization",
        r"privilege\s+escalation",
    ],

    "Unsafe File Access": [
        r"unsafe\s+file\s+access",
        r"path\s+traversal",
    ],

    "Weak Input Validation": [
        r"weak\s+input\s+validation",
        r"insufficient\s+input\s+validation",
        r"missing\s+input\s+validation",
        r"improper\s+input\s+validation",
    ],
}


# ============================================================
# DEFAULT SECURITY SEVERITY
# ============================================================

DEFAULT_SECURITY_SEVERITY = {
    "SQL Injection": "HIGH",
    "Cross-Site Scripting (XSS)": "HIGH",
    "Command Injection": "CRITICAL",
    "Hardcoded Credentials": "HIGH",
    "API Key Exposure": "HIGH",
    "Access Token Exposure": "CRITICAL",
    "Sensitive Information Exposure": "MEDIUM",
    "Authentication Vulnerability": "HIGH",
    "Authorization Vulnerability": "HIGH",
    "Unsafe File Access": "HIGH",
    "Weak Input Validation": "MEDIUM",
}


# ============================================================
# RECOMMENDATIONS
# ============================================================

RECOMMENDATIONS = {
    "SQL Injection": (
        "Use parameterized queries or prepared statements instead of "
        "concatenating or interpolating user-controlled input into SQL."
    ),

    "Cross-Site Scripting (XSS)": (
        "Validate and sanitize untrusted input and use context-aware "
        "output encoding. Avoid inserting untrusted data directly into HTML."
    ),

    "Command Injection": (
        "Avoid executing shell commands with untrusted input. Use safe "
        "APIs and validate input using a strict allowlist."
    ),

    "Hardcoded Credentials": (
        "Remove credentials from source code and store them securely using "
        "environment variables or a secrets-management system."
    ),

    "API Key Exposure": (
        "Remove API keys from source code, rotate exposed keys, and store "
        "them securely using environment variables or a secrets manager."
    ),

    "Access Token Exposure": (
        "Remove access tokens from source code, rotate exposed tokens, "
        "and store secrets securely."
    ),

    "Sensitive Information Exposure": (
        "Avoid exposing sensitive information in source code, logs, API "
        "responses, or error messages."
    ),

    "Authentication Vulnerability": (
        "Strengthen authentication controls and ensure protected resources "
        "require proper authentication."
    ),

    "Authorization Vulnerability": (
        "Implement proper authorization checks for protected resources "
        "and enforce least-privilege access."
    ),

    "Unsafe File Access": (
        "Validate and normalize file paths and prevent user-controlled "
        "paths from accessing files outside the permitted directory."
    ),

    "Weak Input Validation": (
        "Validate untrusted input using strict allowlists and reject "
        "unexpected or malformed values."
    ),
}


# ============================================================
# TEST RECOMMENDATIONS
# ============================================================

TEST_RECOMMENDATIONS = {
    "SQL Injection": (
        "Add tests using malicious SQL payloads and verify that "
        "parameterized queries prevent unauthorized database operations."
    ),

    "Cross-Site Scripting (XSS)": (
        "Add tests containing common XSS payloads and verify that "
        "output is properly escaped or sanitized."
    ),

    "Command Injection": (
        "Add tests with shell metacharacters and malicious command "
        "payloads to verify that arbitrary commands cannot execute."
    ),

    "Hardcoded Credentials": (
        "Add automated secret-scanning tests and verify that credentials "
        "are loaded from secure configuration rather than source code."
    ),

    "API Key Exposure": (
        "Add secret-scanning checks and verify that API keys are not "
        "present in source files, logs, or responses."
    ),

    "Access Token Exposure": (
        "Add secret-scanning tests and verify that access tokens are "
        "never stored or returned from application source code."
    ),

    "Sensitive Information Exposure": (
        "Test error responses, logs, and API responses to ensure that "
        "sensitive information is not exposed."
    ),

    "Authentication Vulnerability": (
        "Add tests for unauthenticated access, invalid credentials, "
        "expired sessions, and authentication bypass attempts."
    ),

    "Authorization Vulnerability": (
        "Add tests verifying that users cannot access resources or "
        "operations outside their assigned permissions."
    ),

    "Unsafe File Access": (
        "Add path traversal tests using payloads such as ../ and verify "
        "that files outside the permitted directory cannot be accessed."
    ),

    "Weak Input Validation": (
        "Add boundary, malformed-input, and malicious-input tests to "
        "verify that invalid input is rejected."
    ),
}


# ============================================================
# DETECT SEVERITY
# ============================================================

def detect_severity(text, category=None):
    """
    Detect an explicit severity from the AI review.

    If no explicit severity is available, use the default
    severity assigned to the security category.
    """

    if not text:
        text = ""

    # Prefer severity appearing close to the category.
    if category:
        category_match = re.search(
            re.escape(category),
            text,
            re.IGNORECASE
        )

        if category_match:
            nearby_text = text[
                max(0, category_match.start() - 100):
                category_match.end() + 300
            ]

            severity_match = re.search(
                r"\b(CRITICAL|HIGH|MEDIUM|LOW)\b",
                nearby_text,
                re.IGNORECASE
            )

            if severity_match:
                return severity_match.group(1).upper()

    # General severity search.
    patterns = [
        r"severity\s*[:\-]\s*(CRITICAL|HIGH|MEDIUM|LOW)",
        r"\b(CRITICAL|HIGH|MEDIUM|LOW)\s+severity\b",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).upper()

    return DEFAULT_SECURITY_SEVERITY.get(
        category,
        "MEDIUM"
    )


# ============================================================
# EXTRACT FILE FROM AI REVIEW
# ============================================================

def extract_file(text):
    patterns = [
        r"file\s*[:\-]\s*([^\s,]+)",
        r"filename\s*[:\-]\s*([^\s,]+)",
        r"\b([A-Za-z0-9_.\-/\\]+\.(?:py|js|jsx|ts|tsx|java|cpp|c|h|go|rs|php|rb))\b",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip("`'\"")

    return "Unknown"


# ============================================================
# EXTRACT LINE FROM AI REVIEW
# ============================================================

def extract_line(text):
    patterns = [
        r"line\s*(?:number)?\s*[:\-]\s*(\d+)",
        r"\bline\s+(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return int(match.group(1))

    return "Unknown"


# ============================================================
# CREATE SECURITY FINDING
# ============================================================

def create_security_finding(
    category,
    description="Security vulnerability detected.",
    severity=None,
    file="Unknown",
    line="Unknown",
):
    if severity is None:
        severity = DEFAULT_SECURITY_SEVERITY.get(
            category,
            "MEDIUM"
        )

    return {
        "category": category,
        "severity": severity.upper(),
        "file": file,
        "line": line,
        "description": description,
        "recommendation": RECOMMENDATIONS.get(
            category,
            "Review and fix the identified security issue."
        ),
        "test_recommendation": TEST_RECOMMENDATIONS.get(
            category,
            "Add automated tests covering the identified security issue."
        ),
    }


# ============================================================
# DIFF FILE EXTRACTION
# ============================================================

def extract_diff_file(diff_content):
    match = re.search(
        r"^\+\+\+\s+b/(.+)$",
        diff_content,
        re.MULTILINE
    )

    if match:
        return match.group(1).strip()

    match = re.search(
        r"^diff --git a/.+ b/(.+)$",
        diff_content,
        re.MULTILINE
    )

    if match:
        return match.group(1).strip()

    return "Unknown"


# ============================================================
# DIFF LINE NUMBER
# ============================================================

def get_added_line_number(diff_content, target_index):
    """
    Calculate the approximate new-file line number of an
    added line in a Git diff.
    """

    lines = diff_content.splitlines()

    new_line = None

    for index, line in enumerate(lines):

        if index >= target_index:
            break

        hunk_match = re.search(
            r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@",
            line
        )

        if hunk_match:
            new_line = int(hunk_match.group(1))
            continue

        if new_line is None:
            continue

        if line.startswith("+++"):
            continue

        if line.startswith("+"):
            new_line += 1

        elif line.startswith("-"):
            pass

        else:
            new_line += 1

    if new_line is not None:
        return new_line

    return "Unknown"


# ============================================================
# DIRECT SQL INJECTION DETECTION
# ============================================================

def detect_sql_injection_from_diff(diff_content):
    """
    Detect SQL injection using the actual changed code.

    This is intentionally evidence-based. Merely mentioning
    SQL injection in an AI review does not create a finding.
    """

    findings = []

    if not diff_content:
        return findings

    lines = diff_content.splitlines()

    file_name = extract_diff_file(diff_content)

    for index, line in enumerate(lines):

        # Only inspect added lines.
        if not line.startswith("+"):
            continue

        if line.startswith("+++"):
            continue

        code = line[1:].strip()

        # ----------------------------------------------------
        # SQL keyword
        # ----------------------------------------------------

        has_sql = bool(
            re.search(
                r"\b(SELECT|INSERT|UPDATE|DELETE)\b",
                code,
                re.IGNORECASE
            )
        )

        # ----------------------------------------------------
        # Dynamic string construction
        # ----------------------------------------------------

        has_fstring_input = bool(
            re.search(
                r"f[\"'].*\{[^}]+\}",
                code,
                re.IGNORECASE
            )
        )

        has_string_concatenation = bool(
            re.search(
                r"[\"']\s*\+\s*[A-Za-z_][A-Za-z0-9_]*",
                code
            )
            or
            re.search(
                r"[A-Za-z_][A-Za-z0-9_]*\s*\+\s*[\"']",
                code
            )
        )

        has_percent_formatting = bool(
            re.search(
                r"[\"'].*%[srdf].*[%\(]",
                code,
                re.IGNORECASE
            )
        )

        # ----------------------------------------------------
        # execute() call
        # ----------------------------------------------------

        has_execute = bool(
            re.search(
                r"\.execute\s*\(",
                code,
                re.IGNORECASE
            )
        )

        # ----------------------------------------------------
        # Detect vulnerability
        # ----------------------------------------------------

        vulnerable = False

        if has_sql and (
            has_fstring_input
            or has_string_concatenation
            or has_percent_formatting
        ):
            vulnerable = True

        if has_execute and (
            has_fstring_input
            or has_string_concatenation
            or has_percent_formatting
        ):
            vulnerable = True

        if not vulnerable:
            continue

        line_number = get_added_line_number(
            diff_content,
            index
        )

        findings.append(
            create_security_finding(
                category="SQL Injection",
                severity="HIGH",
                file=file_name,
                line=line_number,
                description=(
                    "User-controlled or dynamically constructed input "
                    "is incorporated directly into an SQL statement. "
                    "An attacker may be able to alter the intended SQL "
                    "query."
                ),
            )
        )

    return findings


# ============================================================
# DIRECT XSS DETECTION
# ============================================================

def detect_xss_from_diff(diff_content):
    findings = []

    if not diff_content:
        return findings

    lines = diff_content.splitlines()
    file_name = extract_diff_file(diff_content)

    for index, line in enumerate(lines):

        if not line.startswith("+") or line.startswith("+++"):
            continue

        code = line[1:].strip()

        dangerous = (
            re.search(r"innerHTML\s*=", code, re.IGNORECASE)
            or re.search(
                r"document\.write\s*\(",
                code,
                re.IGNORECASE
            )
            or re.search(
                r"dangerouslySetInnerHTML",
                code,
                re.IGNORECASE
            )
        )

        if dangerous:
            findings.append(
                create_security_finding(
                    category="Cross-Site Scripting (XSS)",
                    severity="HIGH",
                    file=file_name,
                    line=get_added_line_number(
                        diff_content,
                        index
                    ),
                    description=(
                        "Untrusted data may be inserted directly into "
                        "HTML without sufficient output encoding or "
                        "sanitization."
                    ),
                )
            )

    return findings


# ============================================================
# DIRECT COMMAND INJECTION DETECTION
# ============================================================

def detect_command_injection_from_diff(diff_content):
    findings = []

    if not diff_content:
        return findings

    lines = diff_content.splitlines()
    file_name = extract_diff_file(diff_content)

    for index, line in enumerate(lines):

        if not line.startswith("+") or line.startswith("+++"):
            continue

        code = line[1:].strip()

        shell_call = (
            re.search(r"os\.system\s*\(", code)
            or re.search(r"subprocess\.(run|call|Popen)\s*\(", code)
        )

        shell_true = re.search(
            r"shell\s*=\s*True",
            code,
            re.IGNORECASE
        )

        if shell_call or shell_true:

            findings.append(
                create_security_finding(
                    category="Command Injection",
                    severity="CRITICAL",
                    file=file_name,
                    line=get_added_line_number(
                        diff_content,
                        index
                    ),
                    description=(
                        "A shell command is executed in the changed code. "
                        "If attacker-controlled input reaches this command, "
                        "arbitrary command execution may be possible."
                    ),
                )
            )

    return findings


# ============================================================
# DIRECT HARDCODED CREDENTIAL DETECTION
# ============================================================

def detect_hardcoded_credentials_from_diff(diff_content):
    findings = []

    if not diff_content:
        return findings

    lines = diff_content.splitlines()
    file_name = extract_diff_file(diff_content)

    for index, line in enumerate(lines):

        if not line.startswith("+") or line.startswith("+++"):
            continue

        code = line[1:].strip()

        # Strong evidence: password/secret assignment with a
        # non-empty literal value.
        match = re.search(
            r"\b(password|passwd|pwd|secret)\s*=\s*"
            r"[\"'][^\"']{3,}[\"']",
            code,
            re.IGNORECASE
        )

        if not match:
            continue

        findings.append(
            create_security_finding(
                category="Hardcoded Credentials",
                severity="HIGH",
                file=file_name,
                line=get_added_line_number(
                    diff_content,
                    index
                ),
                description=(
                    "A credential-like value is hardcoded directly "
                    "in the source code."
                ),
            )
        )

    return findings


# ============================================================
# DIRECT API KEY / TOKEN DETECTION
# ============================================================

def detect_secret_exposure_from_diff(diff_content):
    findings = []

    if not diff_content:
        return findings

    lines = diff_content.splitlines()
    file_name = extract_diff_file(diff_content)

    for index, line in enumerate(lines):

        if not line.startswith("+") or line.startswith("+++"):
            continue

        code = line[1:].strip()

        # API key assignment
        api_key = re.search(
            r"\b(api[_-]?key|apikey)\s*=\s*"
            r"[\"'][^\"']{8,}[\"']",
            code,
            re.IGNORECASE
        )

        # Token assignment
        access_token = re.search(
            r"\b(access[_-]?token|token)\s*=\s*"
            r"[\"'][A-Za-z0-9._\-]{8,}[\"']",
            code,
            re.IGNORECASE
        )

        if api_key:

            findings.append(
                create_security_finding(
                    category="API Key Exposure",
                    severity="HIGH",
                    file=file_name,
                    line=get_added_line_number(
                        diff_content,
                        index
                    ),
                    description=(
                        "An API key appears to be stored directly "
                        "in source code."
                    ),
                )
            )

        elif access_token:

            findings.append(
                create_security_finding(
                    category="Access Token Exposure",
                    severity="CRITICAL",
                    file=file_name,
                    line=get_added_line_number(
                        diff_content,
                        index
                    ),
                    description=(
                        "An access token appears to be stored directly "
                        "in source code."
                    ),
                )
            )

    return findings


# ============================================================
# DIRECT PATH TRAVERSAL DETECTION
# ============================================================

def detect_path_traversal_from_diff(diff_content):
    findings = []

    if not diff_content:
        return findings

    lines = diff_content.splitlines()
    file_name = extract_diff_file(diff_content)

    for index, line in enumerate(lines):

        if not line.startswith("+") or line.startswith("+++"):
            continue

        code = line[1:].strip()

        if "../" in code:

            findings.append(
                create_security_finding(
                    category="Unsafe File Access",
                    severity="HIGH",
                    file=file_name,
                    line=get_added_line_number(
                        diff_content,
                        index
                    ),
                    description=(
                        "The changed code contains a path traversal "
                        "sequence that may allow access outside the "
                        "intended directory."
                    ),
                )
            )

    return findings


# ============================================================
# REMOVE DUPLICATE FINDINGS
# ============================================================

def remove_duplicate_findings(findings):
    """
    Merge findings with the same category and file.

    If one finding has a known line number and another does not,
    retain the known line number.
    """

    unique = []

    for finding in findings:

        category = finding.get("category")
        file = finding.get("file")

        duplicate = None

        for existing in unique:

            if (
                existing.get("category") == category
                and existing.get("file") == file
            ):
                duplicate = existing
                break

        if duplicate:

            if (
                duplicate.get("line") == "Unknown"
                and finding.get("line") != "Unknown"
            ):
                duplicate["line"] = finding.get("line")

            if (
                len(
                    str(finding.get("description", ""))
                )
                >
                len(
                    str(duplicate.get("description", ""))
                )
            ):
                duplicate["description"] = finding.get(
                    "description"
                )

        else:
            unique.append(finding)

    return unique


# ============================================================
# MAIN ANALYZER
# ============================================================

def analyze_review(review, diff_content=""):
    """
    Analyze the AI review and the actual Git diff.

    Stage 2:
        - Critical issues
        - Security issues
        - Code quality issues
        - Test issues

    Stage 3:
        - Security category
        - Severity
        - File
        - Line
        - Recommendation
        - Test recommendation

    IMPORTANT:
    Security findings are based primarily on evidence in the
    actual changed code, rather than merely security terms
    mentioned by the LLM.
    """

    if not review:
        review = ""

    review_lower = review.lower()

    # ========================================================
    # STAGE 2 - CRITICAL ISSUES
    # ========================================================

    critical_issues = len(
        re.findall(
            r"\bcritical\b",
            review_lower
        )
    )

    # ========================================================
    # STAGE 2 - CODE QUALITY
    # ========================================================

    code_quality_keywords = [
        "code quality",
        "maintainability",
        "readability",
        "refactor",
        "code smell",
        "duplication",
        "complexity",
        "clean code",
    ]

    code_quality_issues = sum(
        1
        for keyword in code_quality_keywords
        if keyword in review_lower
    )

    # ========================================================
    # STAGE 2 - TEST ISSUES
    # ========================================================

    test_keywords = [
        "missing test",
        "missing tests",
        "add tests",
        "test recommendation",
        "unit test",
        "integration test",
        "test coverage",
        "insufficient test",
    ]

    test_issues = sum(
        1
        for keyword in test_keywords
        if keyword in review_lower
    )

    # ========================================================
    # STAGE 3 - DIRECT SECURITY ANALYSIS
    # ========================================================

    findings = []

    # SQL Injection
    findings.extend(
        detect_sql_injection_from_diff(
            diff_content
        )
    )

    # XSS
    findings.extend(
        detect_xss_from_diff(
            diff_content
        )
    )

    # Command Injection
    findings.extend(
        detect_command_injection_from_diff(
            diff_content
        )
    )

    # Hardcoded credentials
    findings.extend(
        detect_hardcoded_credentials_from_diff(
            diff_content
        )
    )

    # API keys and access tokens
    findings.extend(
        detect_secret_exposure_from_diff(
            diff_content
        )
    )

    # Path traversal
    findings.extend(
        detect_path_traversal_from_diff(
            diff_content
        )
    )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    findings = remove_duplicate_findings(
        findings
    )

    # ========================================================
    # SECURITY ISSUE COUNT
    # ========================================================

    security_issues = len(findings)

    # ========================================================
    # SEVERITY COUNTERS
    # ========================================================

    severity = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
    }

    for finding in findings:

        finding_severity = (
            finding.get(
                "severity",
                "MEDIUM"
            ).lower()
        )

        if finding_severity in severity:
            severity[finding_severity] += 1

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "critical_issues": critical_issues,
        "security_issues": security_issues,
        "code_quality_issues": code_quality_issues,
        "test_issues": test_issues,

        "severity": severity,

        "findings": findings,
    }