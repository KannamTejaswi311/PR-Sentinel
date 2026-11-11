"""
PR-Sentinel: FastAPI Endpoint Verification Tests
Tests all HTTP routes exposed by web_ui.app:
1. GET / (Dashboard HTML)
2. GET /history (Review history list)
3. GET /rag/status (RAG index status)
4. GET /codebert/status (CodeBERT representation engine status)
5. GET /benchmarks (Google-discovered datasets & benchmark suite)
6. POST /upload (Local diff review pipeline execution)
7. POST /review/github (GitHub PR simulation & review)
8. POST /feedback (Developer calibration feedback)
"""

import os
os.environ["PR_SENTINEL_OFFLINE"] = "1"

import pytest
import httpx
from web_ui.app import app


@pytest.mark.asyncio
async def test_home_page():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "PR-Sentinel" in resp.text
        assert "Qwen2.5-Coder" in resp.text


@pytest.mark.asyncio
async def test_history_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert isinstance(data["history"], list)


@pytest.mark.asyncio
async def test_rag_status_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/rag/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_chunks" in data
        assert "indexed_files_count" in data


@pytest.mark.asyncio
async def test_codebert_status_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/codebert/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "microsoft/codebert-base"
        assert data["vector_dimension"] == 768


@pytest.mark.asyncio
async def test_benchmarks_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/benchmarks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["datasets"]) >= 3
        assert len(data["controlled_benchmarks"]) >= 5


@pytest.mark.asyncio
async def test_upload_diff_endpoint():
    transport = httpx.ASGITransport(app=app)
    diff_content = b"""diff --git a/test.py b/test.py
--- /dev/null
+++ b/test.py
@@ -0,0 +1,4 @@
+import sqlite3
+def query(u):
+    return sqlite3.connect("db").execute("SELECT * FROM users WHERE name = '" + u + "'")
+"""
    files = {"file": ("test_upload.diff", diff_content, "text/plain")}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/upload", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert "review_id" in data
        assert "risk_assessment" in data
        assert data["risk_assessment"]["score"] > 0
        assert len(data["test_recommendations"]) > 0

        # Now test feedback on this review
        rev_id = data["review_id"]
        fb_resp = await client.post(
            "/feedback",
            json={
                "review_id": rev_id,
                "finding_index": 0,
                "feedback_type": "useful",
                "comment": "Accurate SQL injection flag"
            }
        )
        assert fb_resp.status_code == 200
        assert fb_resp.json()["success"] is True


@pytest.mark.asyncio
async def test_github_review_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/review/github",
            json={"pr_url": "https://github.com/octocat/Spoon-Knife/pull/100"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "review_id" in data
        assert "github_comment" in data
