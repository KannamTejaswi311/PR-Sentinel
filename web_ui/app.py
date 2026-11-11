"""
PR-Sentinel: FastAPI Application and Web Interface Server
Integrates GitHub PR & Local Diff Ingestion, CodeBERT, Static Analysis,
Repository-Aware RAG, Multi-Agent Review, Review Synthesis, Risk Scoring,
Automated Test Recommendations, and Developer Feedback.
"""

from typing import Any, Dict, Optional
import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from sentinel.codebert_analyzer import get_codebert_analyzer
from sentinel.datasets_info import get_controlled_benchmarks, get_dataset_registry
from sentinel.github_integration import get_github_integration
from sentinel.pipeline import get_pipeline
from sentinel.rag_pipeline import get_rag_pipeline
from sentinel.review_history import (
    get_all_reviews,
    get_review,
    init_database,
    save_feedback,
)


app = FastAPI(
    title="PR-Sentinel — AI-Powered Code Review System",
    description="Intelligent Pull Request reviewer combining CodeBERT, Static Analysis, Local LLMs, and RAG.",
    version="2.0.0",
)

init_database()

app.mount(
    "/static",
    StaticFiles(directory="web_ui/static"),
    name="static",
)

templates = Jinja2Templates(directory="web_ui/templates")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5-coder:3b-instruct"


class GitHubPRRequest(BaseModel):
    pr_url: str


class FeedbackRequest(BaseModel):
    review_id: int
    finding_index: int
    feedback_type: str
    comment: Optional[str] = ""


# ============================================================
# WEB PAGES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ============================================================
# REVIEW HISTORY ENDPOINTS
# ============================================================

@app.get("/history")
async def review_history():
    try:
        return {"success": True, "history": get_all_reviews()}
    except Exception as exc:
        return {
            "success": False,
            "message": "Unable to load review history.",
            "error": str(exc),
        }


@app.get("/history/{review_id}")
async def review_history_detail(review_id: int):
    try:
        review = get_review(review_id)
        if review is None:
            return {"success": False, "message": "Review not found."}
        return {"success": True, "review": review}
    except Exception as exc:
        return {
            "success": False,
            "message": "Unable to load review.",
            "error": str(exc),
        }


# ============================================================
# LOCAL DIFF UPLOAD & ANALYSIS ENDPOINT
# ============================================================

@app.post("/upload")
async def upload_diff(file: UploadFile = File(...)):
    if not file.filename:
        return JSONResponse(status_code=400, content={"message": "No file was selected."})

    if not file.filename.lower().endswith((".diff", ".patch")):
        return JSONResponse(status_code=400, content={"message": "Please upload a .diff or .patch file."})

    content = await file.read()

    try:
        diff_content = content.decode("utf-8")
    except UnicodeDecodeError:
        return JSONResponse(
            status_code=400,
            content={
                "message": "The uploaded diff must be UTF-8 encoded.",
                "error": "The file could not be decoded as UTF-8. Please save the diff file using UTF-8 encoding.",
            },
        )

    if not diff_content.strip():
        return JSONResponse(status_code=400, content={"message": "The uploaded diff file is empty."})

    # Optional attempt to call Ollama directly for whole-code review text
    raw_llm_text = ""
    prompt = f"""You are an expert software engineer and security-focused code reviewer.
Perform a detailed review of the following Git diff:
```diff
{diff_content}
```
"""
    import socket
    ollama_ready = False
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=0.1):
            ollama_ready = True
    except Exception:
        ollama_ready = False

    if ollama_ready:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    OLLAMA_URL,
                    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                )
                if resp.status_code == 200:
                    raw_llm_text = resp.json().get("response", "")
        except Exception:
            pass

    # Execute full PR-Sentinel Pipeline (CodeBERT, Static Analysis, RAG, Multi-Agent, Synthesis, Risk)
    pipeline = get_pipeline()
    result = await pipeline.execute_review(
        diff_content=diff_content,
        filename=file.filename,
        pr_metadata={"title": f"Local Diff: {file.filename}", "source": "local_upload"},
        raw_llm_text=raw_llm_text,
    )

    return {
        "message": "PR-Sentinel code review completed successfully.",
        "filename": file.filename,
        "review_id": result["review_id"],
        "review": result["review_text"],
        "summary": result["summary"],
        "risk_assessment": result["risk_assessment"],
        "codebert_analysis": result["codebert_analysis"],
        "multi_agent_results": result["multi_agent_results"],
        "test_recommendations": result["test_recommendations"],
        "rag_context": result["rag_context"],
        "github_comment": result["github_comment"],
    }


# ============================================================
# GITHUB PULL REQUEST INGESTION & REVIEW ENDPOINT
# ============================================================

@app.post("/review/github")
async def review_github_pr(payload: GitHubPRRequest):
    if not payload.pr_url or "github.com" not in payload.pr_url:
        return JSONResponse(
            status_code=400,
            content={"message": "Invalid GitHub PR URL. Format: https://github.com/owner/repo/pull/123"},
        )

    github = get_github_integration()
    try:
        pr_data = await github.fetch_pr_data(payload.pr_url)
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"message": f"Failed to fetch PR from GitHub: {str(exc)}"},
        )

    diff_content = pr_data.get("diff_content", "")
    if not diff_content.strip():
        return JSONResponse(
            status_code=400,
            content={"message": "No code changes found in the specified Pull Request."},
        )

    pipeline = get_pipeline()
    filename = f"{pr_data.get('owner')}_{pr_data.get('repo')}_pr{pr_data.get('pr_number')}.diff"

    result = await pipeline.execute_review(
        diff_content=diff_content,
        filename=filename,
        pr_metadata=pr_data,
        raw_llm_text="",
    )

    return {
        "message": f"GitHub PR #{pr_data.get('pr_number')} reviewed successfully.",
        "filename": filename,
        "pr_metadata": pr_data,
        "review_id": result["review_id"],
        "review": result["review_text"],
        "summary": result["summary"],
        "risk_assessment": result["risk_assessment"],
        "codebert_analysis": result["codebert_analysis"],
        "multi_agent_results": result["multi_agent_results"],
        "test_recommendations": result["test_recommendations"],
        "rag_context": result["rag_context"],
        "github_comment": result["github_comment"],
    }


# ============================================================
# DEVELOPER FEEDBACK ENDPOINT
# ============================================================

@app.post("/feedback")
async def submit_feedback(payload: FeedbackRequest):
    valid_types = {"useful", "not_useful", "false_positive", "correct_finding"}
    if payload.feedback_type not in valid_types:
        return JSONResponse(
            status_code=400,
            content={"message": f"Invalid feedback type. Must be one of {valid_types}"},
        )

    fb_id = save_feedback(
        review_id=payload.review_id,
        finding_index=payload.finding_index,
        feedback_type=payload.feedback_type,
        comment=payload.comment or "",
    )

    return {
        "success": True,
        "feedback_id": fb_id,
        "message": "Feedback recorded successfully. Stored for ongoing model calibration.",
    }


# ============================================================
# RAG & REPOSITORY MANAGEMENT ENDPOINTS
# ============================================================

@app.get("/rag/status")
async def rag_status():
    rag = get_rag_pipeline()
    return rag.get_stats()


@app.post("/rag/index")
async def rag_reindex():
    rag = get_rag_pipeline()
    stats = rag.index_repository(force_reindex=True)
    return {
        "success": True,
        "message": "Repository indexed into RAG vector store successfully.",
        "stats": stats,
    }


# ============================================================
# CODEBERT STATUS & BENCHMARKS ENDPOINTS
# ============================================================

@app.get("/codebert/status")
async def codebert_status():
    codebert = get_codebert_analyzer()
    return codebert.status


@app.get("/benchmarks")
async def get_benchmarks():
    return {
        "datasets": get_dataset_registry(),
        "controlled_benchmarks": get_controlled_benchmarks(),
    }
