from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sentinel.review_analyzer import analyze_review

import httpx


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Code Review Assistant"
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="web_ui/static"),
    name="static"
)


# ============================================================
# TEMPLATES
# ============================================================

templates = Jinja2Templates(
    directory="web_ui/templates"
)


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

OLLAMA_MODEL = "qwen2.5-coder:3b-instruct"


# ============================================================
# HOME PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request
        }
    )


# ============================================================
# UPLOAD + CODE REVIEW
# ============================================================

@app.post("/upload")
async def upload_diff(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        return {
            "message": "No file was selected."
        }


    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    if not file.filename.lower().endswith(
        (".diff", ".patch")
    ):

        return {
            "message": (
                "Please upload a .diff or .patch file."
            )
        }


    # --------------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------------

    content = await file.read()


    # --------------------------------------------------------
    # Decode UTF-8
    # --------------------------------------------------------

    try:

        diff_content = content.decode("utf-8")

    except UnicodeDecodeError:

        return {
            "message": (
                "The uploaded diff must be UTF-8 encoded."
            ),

            "error": (
                "The file could not be decoded as UTF-8. "
                "Please save the diff file using UTF-8 encoding."
            )
        }


    # --------------------------------------------------------
    # Empty file check
    # --------------------------------------------------------

    if not diff_content.strip():

        return {
            "message": (
                "The uploaded diff file is empty."
            )
        }


    # ========================================================
    # AI REVIEW PROMPT
    # ========================================================

    prompt = f"""
You are an expert software engineer and security-focused
code reviewer.

Perform a detailed review of the following Git diff.

Analyze the ACTUAL CHANGED CODE carefully.

Do not say "No security concerns identified" if the changed
code contains evidence of a security vulnerability.

Focus on:

1. Bugs
2. Security vulnerabilities
3. SQL injection
4. Cross-Site Scripting (XSS)
5. Command injection
6. Hardcoded credentials
7. API key exposure
8. Access token exposure
9. Authentication vulnerabilities
10. Authorization vulnerabilities
11. Unsafe file access
12. Path traversal
13. Weak input validation
14. Code quality problems
15. Performance problems
16. Missing or insufficient tests

For every important security issue, provide:

Category:
Severity:
File:
Line:
Description:
Recommendation:
Test Recommendation:

Use ONLY these severity levels:

CRITICAL
HIGH
MEDIUM
LOW

For security findings, use this exact format:

SECURITY FINDING
Category: <category>
Severity: <CRITICAL/HIGH/MEDIUM/LOW>
File: <filename>
Line: <line number if known>
Description: <description>
Recommendation: <recommendation>
Test Recommendation: <test recommendation>
END SECURITY FINDING

If multiple security issues exist, create multiple
SECURITY FINDING blocks.

If there are no security vulnerabilities, write:

NO SECURITY FINDINGS

Important:

- Inspect the changed code rather than relying only on comments.
- SQL queries constructed using user-controlled strings should
  be considered for SQL injection.
- Directly concatenating or interpolating user input into SQL
  queries is dangerous.
- Recommend parameterized queries for SQL injection.
- Do not ignore a vulnerability merely because the code comment
  says it is intentional.
- Do not report a vulnerability without evidence.

Also provide a normal readable code review after the
structured findings.

Here is the Git diff:

```diff
{diff_content}
```

Also provide a normal readable code review after the
structured findings.
"""

    # ========================================================
    # CALL OLLAMA
    # ========================================================

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                }
            )

        if response.status_code != 200:
            return {
                "message": "Ollama review failed.",
                "error": response.text
            }

        ollama_data = response.json()
        review = ollama_data.get("response", "")

        if not review.strip():
            return {
                "message": "Ollama returned an empty review.",
                "error": (
                    "Qwen2.5-Coder did not return any review content."
                )
            }

        summary = analyze_review(review, diff_content)

        return {
            "message": "AI code review completed successfully.",
            "filename": file.filename,
            "review": review,
            "summary": summary
        }

    except httpx.ConnectError:
        return {
            "message": "Unable to connect to Ollama.",
            "error": (
                "Ollama is not running or cannot be reached. "
                "Please make sure Ollama is running on "
                "http://localhost:11434."
            )
        }

    except httpx.TimeoutException:
        return {
            "message": "The AI review timed out.",
            "error": "Qwen2.5-Coder took too long to complete the code review."
        }

    except ValueError:
        return {
            "message": "Invalid response received from Ollama.",
            "error": "Ollama returned a response that could not be parsed."
        }

    except Exception as error:
        return {
            "message": "An error occurred during the review.",
            "error": str(error)
        }