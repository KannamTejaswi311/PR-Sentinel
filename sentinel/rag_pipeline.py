"""
PR-Sentinel: Module 5 - Repository-Aware RAG Pipeline
Ingests repository source code, documentation, and configuration files;
chunks code with AST metadata; generates vector representations;
and retrieves relevant repository context for LLM prompt injection.
"""

import ast
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sentinel.codebert_analyzer import get_codebert_analyzer


class RepositoryRAGPipeline:
    """
    Repository-aware Retrieval-Augmented Generation (RAG) system.
    Indexes codebase into chunked semantic representations, stores metadata,
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

    def __init__(self, workspace_dir: Optional[str] = None):
        self.workspace_dir = Path(workspace_dir or Path(__file__).resolve().parent.parent)
        self.index_path = self.workspace_dir / self.INDEX_FILE
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
        }

    def index_repository(self, force_reindex: bool = False, max_files: int = 150) -> Dict[str, Any]:
        """
        Scan and index the repository:
        1. Discover relevant source files
        2. Filter out binary & vendor directories
        3. Parse AST for Python functions and classes
        4. Generate CodeBERT embeddings
        5. Persist vector store
        """
        if self.chunks and not force_reindex:
            return self.get_stats()

        discovered_chunks = []
        processed_files = 0

        for root, dirs, files in os.walk(self.workspace_dir):
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS and not d.startswith(".")]

            for file_name in files:
                ext = Path(file_name).suffix.lower()
                if ext not in self.ALLOWED_EXTENSIONS:
                    continue

                full_path = Path(root) / file_name
                try:
                    rel_path = full_path.relative_to(self.workspace_dir).as_posix()
                except ValueError:
                    rel_path = str(full_path)

                # Skip large test artifacts or databases
                if "review_history.db" in rel_path or "vector_index" in rel_path:
                    continue

                try:
                    content = full_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue

                # File-level chunking
                file_chunks = self._chunk_file(rel_path, content, ext)
                discovered_chunks.extend(file_chunks)
                processed_files += 1

                if processed_files >= max_files:
                    break

            if processed_files >= max_files:
                break

        # Compute embeddings for all chunks
        for chunk in discovered_chunks:
            embedding_info = self.codebert.extract_code_representation(chunk["content"])
            chunk["embedding"] = embedding_info["vector"]

        self.chunks = discovered_chunks
        self.indexed_files_count = processed_files
        self.last_indexed_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save_index()

        return self.get_stats()

    def _chunk_file(self, rel_path: str, content: str, ext: str) -> List[Dict[str, Any]]:
        """
        Chunk code into semantic units. Uses Python AST for .py files,
        and section headers for Markdown/configuration files.
        """
        chunks = []

        if ext == ".py":
            try:
                tree = ast.parse(content)
                lines = content.splitlines()

                # Extract module-level docstring if present
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
                        # Store class signature and docstring/head
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

        # Fallback for non-Python or files without AST definitions: block chunking
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
        for a given diff or code query using cosine similarity in CodeBERT vector space.
        """
        if not self.chunks:
            self.index_repository(force_reindex=False)

        if not self.chunks:
            return []

        # Generate query representation
        query_rep = self.codebert.extract_code_representation(query_or_diff)
        query_vec = query_rep["vector"]

        scored_chunks: List[Tuple[float, Dict[str, Any]]] = []

        for chunk in self.chunks:
            if exclude_file and chunk["file_path"] == exclude_file:
                continue

            chunk_vec = chunk.get("embedding", [])
            if not chunk_vec or len(chunk_vec) != len(query_vec):
                continue

            # Cosine similarity
            dot = sum(a * b for a, b in zip(query_vec, chunk_vec))
            score = max(0.0, min(1.0, (dot + 1.0) / 2.0))

            scored_chunks.append((score, chunk))

        # Sort descending by similarity score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, chunk in scored_chunks[:top_k]:
            results.append({
                "file_path": chunk["file_path"],
                "type": chunk.get("type", "code"),
                "name": chunk.get("name", ""),
                "line_start": chunk.get("line_start", 1),
                "line_end": chunk.get("line_end", 1),
                "similarity_score": round(score, 4),
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


_RAG_PIPELINE_INSTANCE: Optional[RepositoryRAGPipeline] = None


def get_rag_pipeline() -> RepositoryRAGPipeline:
    """Return singleton instance of RepositoryRAGPipeline."""
    global _RAG_PIPELINE_INSTANCE
    if _RAG_PIPELINE_INSTANCE is None:
        _RAG_PIPELINE_INSTANCE = RepositoryRAGPipeline()
    return _RAG_PIPELINE_INSTANCE
