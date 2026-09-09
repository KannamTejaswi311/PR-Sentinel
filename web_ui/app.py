from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import os
import tempfile

from pr_agent.agent.pr_agent import PRAgent
from pr_agent.config_loader import get_settings


# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI(
    title="AI Code Review Assistant"
)


# =========================================================
# Static Files
# =========================================================

app.mount(
    "/static",
    StaticFiles(directory="web_ui/static"),
    name="static"
)


# =========================================================
# HTML Templates
# =========================================================

templates = Jinja2Templates(
    directory="web_ui/templates"
)


# =========================================================
# Home Page
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )


# =========================================================
# Review Analysis
# =========================================================

def analyze_review(review):
    """
    Analyze PR-Agent's generated review and calculate
    dashboard counters.

    The counters are based on the actual AI-generated
    review text.
    """

    critical_issues = 0
    security_issues = 0
    code_quality_issues = 0
    test_issues = 0

    # Convert review to lowercase so matching is
    # case-insensitive.
    review_lower = review.lower()

    # -----------------------------------------------------
    # Security Issues
    # -----------------------------------------------------

    # PR-Agent may explicitly state that no security
    # concerns were found.
    no_security_concerns = (
        "no security concerns identified" in review_lower
        or "no security concerns" in review_lower
    )

    security_patterns = [
        "security concerns",
        "security concern",
        "security issue",
        "security vulnerability",

        # Injection vulnerabilities
        "sql injection",
        "xss",
        "cross-site scripting",
        "command injection",

        # Authentication / authorization
        "authentication vulnerability",
        "authorization vulnerability",

        # Credentials and secrets
        "hard-coded credentials",
        "hardcoded credentials",
        "hard-coded password",
        "hardcoded password",
        "hard-coded secret",
        "hardcoded secret",
        "sensitive information exposure",
        "sensitive information",
        "credential exposure",

        # Other sensitive values
        "api key",
        "api keys",
        "access token",
        "access tokens",
        "secret key",
        "secret keys"
    ]

    # Only count a security issue when the AI has
    # actually reported a positive security finding.
    if not no_security_concerns:

        for pattern in security_patterns:

            if pattern in review_lower:

                security_issues = 1
                break

    # -----------------------------------------------------
    # Critical Issues
    # -----------------------------------------------------

    critical_patterns = [
        "critical issue",
        "critical vulnerability",
        "severity: critical",
        "critical security"
    ]

    for pattern in critical_patterns:

        if pattern in review_lower:

            critical_issues = 1
            break

    # -----------------------------------------------------
    # Test Issues
    # -----------------------------------------------------

    if (
        "no relevant tests" in review_lower
        or "tests are missing" in review_lower
        or "missing tests" in review_lower
        or "test coverage" in review_lower
    ):

        test_issues = 1

    # -----------------------------------------------------
    # Code Quality Issues
    # -----------------------------------------------------

    quality_patterns = [
        "code quality",
        "code smell",
        "maintainability",
        "performance issue",
        "bug",
        "issue:",
        "problem:",
        "should be changed",
        "should be fixed",
        "improvement"
    ]

    for pattern in quality_patterns:

        if pattern in review_lower:

            code_quality_issues = 1
            break

    # -----------------------------------------------------
    # Return Dashboard Summary
    # -----------------------------------------------------

    return {
        "critical_issues": critical_issues,
        "security_issues": security_issues,
        "code_quality_issues": code_quality_issues,
        "test_issues": test_issues
    }


# =========================================================
# Upload and Review Diff
# =========================================================

@app.post("/upload")
async def upload_diff(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Validate File
    # -----------------------------------------------------

    if not file.filename.lower().endswith(
        (".diff", ".patch")
    ):

        return {
            "message": "Please upload a .diff or .patch file."
        }

    content = await file.read()

    diff_path = None
    output_path = None

    try:

        # -------------------------------------------------
        # Save Uploaded Diff
        # -------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".diff"
        ) as diff_file:

            diff_file.write(content)
            diff_path = diff_file.name

        # -------------------------------------------------
        # Read Diff as UTF-8
        # -------------------------------------------------

        with open(
            diff_path,
            "r",
            encoding="utf-8"
        ) as diff_file:

            diff_content = diff_file.read()

        # -------------------------------------------------
        # Check Empty Diff
        # -------------------------------------------------

        if not diff_content.strip():

            return {
                "message": "The uploaded diff file is empty."
            }

        # -------------------------------------------------
        # Create Temporary Review Output
        # -------------------------------------------------

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".md"
        )

        output_path = output_file.name

        output_file.close()

        # -------------------------------------------------
        # Configure PR-Agent
        # -------------------------------------------------

        settings = get_settings()

        # Use Plain Diff provider because the web interface
        # receives a .diff/.patch file directly.
        settings.set(
            "config.git_provider",
            "plain-diff"
        )

        # Give PR-Agent the uploaded diff.
        settings.set(
            "plain_diff.content",
            diff_content
        )

        # Tell PlainDiffGitProvider where to write
        # the generated AI review.
        settings.set(
            "plain_diff.output_path",
            output_path
        )

        # -------------------------------------------------
        # Run PR-Agent
        # -------------------------------------------------

        agent = PRAgent()

        success = await agent.handle_request(
            "local_diff",
            ["review"]
        )

        # -------------------------------------------------
        # Read AI Review
        # -------------------------------------------------

        review = None

        if os.path.exists(output_path):

            with open(
                output_path,
                "r",
                encoding="utf-8"
            ) as review_file:

                review = review_file.read()

        # -------------------------------------------------
        # Analyze Successful Review
        # -------------------------------------------------

        if review and review.strip():

            summary = analyze_review(review)

            return {
                "message": "AI code review completed successfully.",
                "filename": file.filename,
                "review": review,
                "summary": summary
            }

        # -------------------------------------------------
        # PR-Agent Completed but No Output
        # -------------------------------------------------

        if success:

            return {
                "message": (
                    "PR-Agent completed, but no review "
                    "output was generated."
                ),
                "error": "No review output was found."
            }

        # -------------------------------------------------
        # PR-Agent Failed
        # -------------------------------------------------

        return {
            "message": "PR-Agent review failed.",
            "error": (
                "The review engine returned an "
                "unsuccessful result."
            )
        }

    # =====================================================
    # Exception Handling
    # =====================================================

    except Exception as error:

        return {
            "message": "An error occurred during the review.",
            "error": str(error)
        }

    # =====================================================
    # Cleanup Temporary Files
    # =====================================================

    finally:

        if (
            diff_path
            and os.path.exists(diff_path)
        ):

            os.remove(diff_path)

        if (
            output_path
            and os.path.exists(output_path)
        ):

            os.remove(output_path)