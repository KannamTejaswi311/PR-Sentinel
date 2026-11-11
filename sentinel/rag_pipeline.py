"""
PR-Sentinel: Module 5 - Repository-Aware RAG Pipeline
Ingests target repository source code, documentation, and configuration files;
chunks code with AST metadata; generates vector representations using CodeBERT;
and retrieves relevant repository context for LLM prompt injection.
Adheres to Frozen Experimental Protocol v1.0 and Amendment v1.1.
"""

import ast
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from sentinel.codebert_analyzer import get_codebert_analyzer


class RepositoryRAGPipeline:
    """
    Repository-aware Retrieval-Augmented Generation (RAG) system.
    Indexes target codebase into chunked semantic representations, stores metadata,
    and provides similarity-based context retrieval for code reviews.
    """

    INDEX_FILE = "rag_vector_index.json"

    IGNORE_DIRS = {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        "pr_agent.egg-info",
        ".egg-info",
        "dist",
        "build",
        ".pytest_cache",
        ".idea",
        ".vscode",
        "site",
    }

    ALLOWED_EXTENSIONS = {
        ".py",
        ".md",
        ".toml",
        ".yaml",
        ".yml",
        ".sql",
        ".js",
        ".ts",
        ".json",
    }

    def __init__(self, workspace_dir: Optional[str] = None, repo_id: str = "default"):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path(__file__).resolve().parent.parent
        self.repo_id = repo_id
        
        # Isolate indexes in a dedicated storage folder if not indexing workspace root directly
        index_dir = self.workspace_dir
        self.index_path = index_dir / self.INDEX_FILE
        self.chunks: List[Dict[str, Any]] = []
        self.indexed_files_count: int = 0
        self.last_indexed_time: Optional[str] = None
        self.codebert = get_codebert_analyzer()
        self._load_index()

    def _load_index(self) -> bool:
        """Load persistent vector index from disk if present."""
        if not self.index_path.exists():
            return False

        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.chunks = data.get("chunks", [])
                self.indexed_files_count = data.get("indexed_files_count", 0)
                self.last_indexed_time = data.get("last_indexed_time")
                return True
        except Exception:
            return False

    def _save_index(self):
        """Persist vector index to disk."""
        data = {
            "repo_id": self.repo_id,
            "workspace_dir": str(self.workspace_dir),
            "indexed_files_count": self.indexed_files_count,
            "total_chunks": len(self.chunks),
            "last_indexed_time": self.last_indexed_time,
            "chunks": self.chunks,
        }
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def get_stats(self) -> Dict[str, Any]:
        """Return current RAG indexing statistics."""
        return {
            "is_indexed": len(self.chunks) > 0,
            "indexed_files_count": self.indexed_files_count,
            "total_chunks": len(self.chunks),
            "last_indexed_time": self.last_indexed_time or "Never",
            "index_storage_file": str(self.index_path.name),
            "workspace_dir": str(self.workspace_dir),
        }

    def index_repository(self, force_reindex: bool = False, max_files: int = 150) -> Dict[str, Any]:
        """
        Scan repository files, parse AST / text blocks, extract CodeBERT embeddings,
        and build persistent index.
        """
        if self.chunks and not force_reindex:
            return self.get_stats()

        processed_files = 0
        all_chunks: List[Dict[str, Any]] = []

        for root, dirs, files in os.walk(self.workspace_dir):
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS and not d.startswith(".")]

            for file in files:
                if processed_files >= max_files:
                    break

                ext = Path(file).suffix.lower()
                if ext not in self.ALLOWED_EXTENSIONS:
                    continue

                file_path = Path(root) / file
                rel_path = str(file_path.relative_to(self.workspace_dir)).replace("\\", "/")

                if "review_history.db" in rel_path or "vector_index" in rel_path:
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    file_chunks = self._chunk_file(rel_path, content, ext)
                    all_chunks.extend(file_chunks)
                    processed_files += 1
                except Exception:
                    pass

        # Generate embeddings for each chunk
        for idx, chunk in enumerate(all_chunks):
            chunk["chunk_id"] = f"{self.repo_id}_chunk_{idx+1:04d}"
            embedding_info = self.codebert.extract_code_representation(chunk["content"])
            chunk["embedding"] = embedding_info["vector"]

        self.chunks = all_chunks
        self.indexed_files_count = processed_files
        self.last_indexed_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save_index()

        return self.get_stats()

    def _chunk_file(self, rel_path: str, content: str, ext: str) -> List[Dict[str, Any]]:
        """Chunk file using AST for Python or line blocks for other formats."""
        chunks: List[Dict[str, Any]] = []

        if ext == ".py":
            try:
                tree = ast.parse(content)
                lines = content.splitlines()

                docstring = ast.get_docstring(tree)
                if docstring:
                    chunks.append({
                        "file_path": rel_path,
                        "type": "module_doc",
                        "name": rel_path,
                        "line_start": 1,
                        "line_end": min(25, len(lines)),
                        "content": f"# {rel_path} Overview\n{docstring}",
                    })

                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        start_line = node.lineno
                        end_line = getattr(node, "end_lineno", start_line + 30)
                        chunk_code = "\n".join(lines[start_line - 1 : end_line])
                        chunks.append({
                            "file_path": rel_path,
                            "type": "function",
                            "name": node.name,
                            "line_start": start_line,
                            "line_end": end_line,
                            "content": chunk_code,
                        })
                    elif isinstance(node, ast.ClassDef):
                        start_line = node.lineno
                        end_line = getattr(node, "end_lineno", start_line + 40)
                        chunk_code = "\n".join(lines[start_line - 1 : min(end_line, start_line + 30)])
                        chunks.append({
                            "file_path": rel_path,
                            "type": "class",
                            "name": node.name,
                            "line_start": start_line,
                            "line_end": end_line,
                            "content": chunk_code,
                        })
            except Exception:
                pass

        if not chunks:
            lines = content.splitlines()
            step = 35
            overlap = 5
            for i in range(0, max(1, len(lines)), step):
                chunk_lines = lines[i : i + step + overlap]
                chunk_text = "\n".join(chunk_lines).strip()
                if chunk_text:
                    chunks.append({
                        "file_path": rel_path,
                        "type": "block",
                        "name": f"{rel_path}:{i+1}",
                        "line_start": i + 1,
                        "line_end": min(i + step + overlap, len(lines)),
                        "content": chunk_text,
                    })

        return chunks

    def retrieve_context(
        self,
        query_or_diff: str,
        top_k: int = 4,
        exclude_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most semantically relevant code or documentation chunks
        for a given diff or code query using normalized cosine similarity in CodeBERT vector space.
        """
        if not self.chunks:
            self.index_repository(force_reindex=False)

        if not self.chunks:
            return []

        query_rep = self.codebert.extract_code_representation(query_or_diff)
        query_vec = query_rep["vector"]
        norm_q = math.sqrt(sum(a * a for a in query_vec))

        scored_chunks: List[Tuple[float, Dict[str, Any]]] = []

        for chunk in self.chunks:
            if exclude_file and chunk["file_path"] == exclude_file:
                continue

            chunk_vec = chunk.get("embedding", [])
            if not chunk_vec or len(chunk_vec) != len(query_vec):
                continue

            norm_c = math.sqrt(sum(b * b for b in chunk_vec))
            if norm_q < 1e-9 or norm_c < 1e-9:
                score = 0.0
            else:
                dot = sum(a * b for a, b in zip(query_vec, chunk_vec))
                cos_sim = dot / (norm_q * norm_c)
                score = max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))

            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        results = []
        for rank, (score, chunk) in enumerate(scored_chunks[:top_k], 1):
            results.append({
                "chunk_id": chunk.get("chunk_id", f"chunk_{rank}"),
                "file_path": chunk["file_path"],
                "type": chunk.get("type", "code"),
                "name": chunk.get("name", ""),
                "line_start": chunk.get("line_start", 1),
                "line_end": chunk.get("line_end", 1),
                "similarity_score": round(score, 4),
                "rank": rank,
                "content_snippet": chunk["content"][:400],
            })

        return results

    def format_context_for_prompt(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved repository context into markdown for LLM injection."""
        if not retrieved_chunks:
            return "No additional repository context retrieved."

        lines = ["### Relevant Repository Context (Retrieved via RAG):"]
        for i, c in enumerate(retrieved_chunks, 1):
            lines.append(
                f"\n**[Context {i}] {c['file_path']} ({c['name']}) "
                f"[Lines {c['line_start']}-{c['line_end']}] - Similarity: {c['similarity_score'] * 100:.1f}%**"
            )
            lines.append("```python")
            lines.append(c["content_snippet"])
            lines.append("```")

        return "\n".join(lines)

    @staticmethod
    def calculate_dfor_at_k(retrieved_chunks: List[Dict[str, Any]], diff_file_paths: Set[str], k: int = 3) -> float:
        """
        Calculates Diff File Overlap Ratio (DFOR@K) per Protocol Section 6.1.
        DFOR@K = |{retrieved files} ∩ {diff files}| / min(K, |diff files|)
        """
        if not diff_file_paths or k <= 0:
            return 0.0
        retrieved_files = {c["file_path"] for c in retrieved_chunks[:k]}
        intersection = retrieved_files.intersection(diff_file_paths)
        denom = min(k, len(diff_file_paths))
        return len(intersection) / float(denom) if denom > 0 else 0.0

    @staticmethod
    def calculate_mrr_and_ndcg(
        retrieved_chunks: List[Dict[str, Any]],
        relevant_chunk_ids: Set[str],
        k: int = 5
    ) -> Tuple[float, float]:
        """
        Calculates MRR and nDCG@K based on frozen relevance judgments (Protocol Section 6.2).
        """
        if not retrieved_chunks or not relevant_chunk_ids:
            return 0.0, 0.0

        # Calculate MRR
        mrr = 0.0
        for idx, chunk in enumerate(retrieved_chunks[:k]):
            cid = chunk.get("chunk_id", "")
            if cid in relevant_chunk_ids or chunk.get("file_path") in relevant_chunk_ids:
                mrr = 1.0 / (idx + 1)
                break

        # Calculate DCG@K
        dcg = 0.0
        for idx, chunk in enumerate(retrieved_chunks[:k]):
            cid = chunk.get("chunk_id", "")
            rel = 1.0 if (cid in relevant_chunk_ids or chunk.get("file_path") in relevant_chunk_ids) else 0.0
            dcg += rel / math.log2(idx + 2)

        # Ideal DCG@K
        num_rel = min(k, len(relevant_chunk_ids))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(num_rel))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        return round(mrr, 4), round(ndcg, 4)


_RAG_PIPELINE_INSTANCE: Optional[RepositoryRAGPipeline] = None


def get_rag_pipeline(workspace_dir: Optional[str] = None, repo_id: str = "default") -> RepositoryRAGPipeline:
    """Return singleton instance of RepositoryRAGPipeline."""
    global _RAG_PIPELINE_INSTANCE
    if _RAG_PIPELINE_INSTANCE is None or workspace_dir is not None:
        _RAG_PIPELINE_INSTANCE = RepositoryRAGPipeline(workspace_dir=workspace_dir, repo_id=repo_id)
    return _RAG_PIPELINE_INSTANCE
