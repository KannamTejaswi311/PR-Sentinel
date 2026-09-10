from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import httpx

from sentinel.review_analyzer import analyze_review
from sentinel.review_history import (
    get_all_reviews,
    get_review,
    init_database,
    save_review,
)


app = FastAPI(title="AI Code Review Assistant")

init_database()

app.mount(
    "/static",
    StaticFiles(directory="web_ui/static"),
    name="static",
)

templates = Jinja2Templates(directory="web_ui/templates")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:3b-instruct"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/history")
async def review_history():
    try:
        return {"success": True, "history": get_all_reviews()}
    except Exception:
        return {
            "success": False,
            "message": "Unable to load review history.",
        }


@app.get("/history/{review_id}")
async def review_history_detail(review_id: int):
    try:
        review = get_review(review_id)
        if review is None:
            return {"success": False, "message": "Review not found."}
        return {"success": True, "review": review}
    except Exception:
        return {
            "success": False,
            "message": "Unable to load review.",
        }


@app.post("/upload")
async def upload_diff(file: UploadFile = File(...)):
    if not file.filename:
        return {"message": "No file was selected."}

    if not file.filename.lower().endswith((".diff", ".patch")):
        return {"message": "Please upload a .diff or .patch file."}

    content = await file.read()

    try:
        diff_content = content.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "message": "The uploaded diff must be UTF-8 encoded.",
            "error": (
                "The file could not be decoded as UTF-8. "
                "Please save the diff file using UTF-8 encoding."
            ),
        }

    if not diff_content.strip():
        return {"message": "The uploaded diff file is empty."}

    prompt = f"""
You are an expert software engineer and security-focused code reviewer.

Perform a detailed review of the following Git diff. Analyze the actual
changed code carefully. Focus on bugs, security vulnerabilities, SQL injection,
XSS, command injection, hardcoded credentials, exposed keys or tokens,
authentication, authorization, unsafe file access, path traversal, input
validation, code quality, performance, and missing tests.

For security findings, use this format:
SECURITY FINDING
Category: <category>
Severity: <CRITICAL/HIGH/MEDIUM/LOW>
File: <filename>
Line: <line number if known>
Description: <description>
Recommendation: <recommendation>
Test Recommendation: <test recommendation>
END SECURITY FINDING

If there are no security vulnerabilities, write: NO SECURITY FINDINGS.
Do not report vulnerabilities without evidence in the changed code.

Here is the Git diff:

```diff
{diff_content}
```

Also provide a normal readable code review after the structured findings.
"""

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                },
            )

        if response.status_code != 200:
            return {
                "message": "Ollama review failed.",
                "error": response.text,
            }

        ollama_data = response.json()
        review = ollama_data.get("response", "")

        if not review.strip():
            return {
                "message": "Ollama returned an empty review.",
                "error": "Qwen2.5-Coder did not return any review content.",
            }

        summary = analyze_review(review, diff_content)
        review_id = save_review(
            filename=file.filename,
            summary=summary,
            review_text=review,
        )

        return {
            "message": "AI code review completed successfully.",
            "filename": file.filename,
            "review_id": review_id,
            "review": review,
            "summary": summary,
        }

    except httpx.ConnectError:
        return {
            "message": "Unable to connect to Ollama.",
            "error": (
                "Ollama is not running or cannot be reached. "
                "Please make sure Ollama is running on "
                "http://localhost:11434."
            ),
        }
    except httpx.TimeoutException:
        return {
            "message": "The AI review timed out.",
            "error": "Qwen2.5-Coder took too long to complete the code review.",
        }
    except ValueError:
        return {
            "message": "Invalid response received from Ollama.",
            "error": "Ollama returned a response that could not be parsed.",
        }
    except Exception as error:
        return {
            "message": "An error occurred during the review.",
            "error": str(error),
        }
