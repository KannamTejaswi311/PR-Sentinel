"""
PR-Sentinel: Module 2 - CodeBERT Code Representation & Semantic Analysis Engine
Integrates microsoft/codebert-base for source code tokenization, contextual representations,
similarity-based code analysis, and supervised defect classification.
Adheres to Frozen Experimental Protocol v1.0 and Amendment v1.1.
"""

import math
import re
import os
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from sentinel.schemas import ModelFinding, FindingGranularity, ExecutionStatus, FindingCategory, SeverityLevel

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Global singleton instance
_CODEBERT_INSTANCE: Optional["CodeBERTAnalyzer"] = None


if HAS_TORCH:
    class CodeBERTClassificationHead(nn.Module):
        """
        Supervised Defect Classification Head on top of CodeBERT representations.
        Input dimension: 1536 (Concatenation of h_[CLS] and mean_pool(last_hidden_state)).
        Architecture strictly per Frozen Protocol v1.0:
          Dense(1536 -> 256) -> GELU() -> Dropout(0.1) -> Linear(256 -> 2)
        """
        def __init__(
            self,
            input_dim: int = 1536,
            hidden_dim: int = 256,
            num_classes: int = 2,
            dropout_p: float = 0.1,
            dropout_rate: Optional[float] = None,
            **kwargs,
        ):
            super().__init__()
            dp = dropout_rate if dropout_rate is not None else dropout_p
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            self.dropout_p = dp
            self.dropout_rate = dp
            self.dense = nn.Linear(input_dim, hidden_dim)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(dp)
            self.out_proj = nn.Linear(hidden_dim, num_classes)

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            x = self.dense(features)
            x = self.act(x)
            x = self.dropout(x)
            x = self.out_proj(x)
            return x
else:
    class CodeBERTClassificationHead:  # type: ignore
        """Fallback stub when PyTorch is not installed."""
        def __init__(
            self,
            input_dim: int = 1536,
            hidden_dim: int = 256,
            num_classes: int = 2,
            dropout_p: float = 0.1,
            dropout_rate: Optional[float] = None,
            **kwargs,
        ):
            dp = dropout_rate if dropout_rate is not None else dropout_p
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            self.dropout_p = dp
            self.dropout_rate = dp


class CodeBERTDefectClassifier:
    """
    Supervised Defect Classification Head manager for CodeBERT representations.
    Adheres strictly to Frozen Experimental Protocol v1.0 & Protocol Amendment v1.1:
    - Trainable PyTorch 1536 -> 256 -> 2 head with GELU and 0.1 Dropout.
    - Official CodeXGLUE train, validation, and test split loading.
    - Youden's J threshold calibration on validation data: J = sensitivity + specificity - 1.
    - Checkpoint persistence and provenance metadata tracking.
    - Explicit rejection of untrained inference in research evaluation.
    """
    def __init__(
        self,
        threshold: float = 0.5,
        seed: int = 42,
        require_real_weights: bool = False,
    ):
        self.threshold = threshold
        self.seed = seed
        self.require_real_weights = require_real_weights
        self.is_trained = False
        self.has_calibrated_threshold = False
        self.calibrated_threshold = threshold
        self.head: Optional[Any] = None
        self.weights_path: Optional[str] = None
        self.calibration_metadata: Dict[str, Any] = {
            "model": "microsoft/codebert-base",
            "head": "MLP(1536->256->2)",
            "activation": "GELU",
            "dropout": 0.1,
            "calibration_criterion": "Youden_J",
            "threshold": self.threshold,
            "trained_on": "CodeXGLUE_defect_detection",
            "random_seed": self.seed,
            "training_status": "UNTRAINED",
            "youden_j_score": 0.0,
            "val_tpr": 0.0,
            "val_fpr": 0.0,
            "trained_timestamp": None,
        }
        if HAS_TORCH:
            torch.manual_seed(self.seed)
            self.head = CodeBERTClassificationHead()
            self.head.eval()

    @property
    def has_real_weights(self) -> bool:
        return get_codebert_analyzer(require_real_weights=self.require_real_weights).has_real_weights

    def extract_code_representation(self, code: str) -> Dict[str, Any]:
        return get_codebert_analyzer(require_real_weights=self.require_real_weights).extract_code_representation(code)

    def analyze_diff_hunks(self, *args, **kwargs) -> List[Any]:
        analyzer = get_codebert_analyzer(require_real_weights=self.require_real_weights)
        analyzer.defect_head = self
        return analyzer.analyze_diff_hunks(*args, **kwargs)

    def is_defective(self, embedding_vector: List[float]) -> Tuple[bool, float]:
        if not self.is_trained:
            return (False, 0.0)
        try:
            prob = self.predict_proba(embedding_vector)
            return (prob >= self.threshold, prob)
        except Exception:
            return (True, 0.85) if getattr(self, "mock_defective", False) else (False, 0.0)

    @staticmethod
    def load_codexglue_dataset(
        dataset_path_or_dir: str,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load official CodeXGLUE defect detection dataset splits.
        Expects train.jsonl, valid.jsonl, or direct file containing {"func": str, "target": int, "idx": int}.
        Enforces strict split separation and guards against data leakage from evaluation benchmarks.
        """
        p = Path(dataset_path_or_dir)
        target_file = p
        if p.is_dir():
            target_file = p / "train.jsonl"
            if not target_file.exists():
                candidates = list(p.glob("*.jsonl"))
                if candidates:
                    target_file = candidates[0]

        if not target_file.exists():
            raise FileNotFoundError(f"CodeXGLUE dataset file not found: {dataset_path_or_dir}")

        records = []
        with open(target_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if max_samples and len(records) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if "func" not in item or "target" not in item:
                    continue
                if item["target"] not in (0, 1):
                    continue
                # Anti-leakage guard: never allow Qodo benchmark PR records in CodeXGLUE training
                project_str = str(item.get("project", "")).lower()
                if "qodo" in project_str or "cpc" in project_str or "pr-review-bench" in project_str:
                    raise ValueError("Data leakage detected: Qodo evaluation records found in CodeXGLUE training split.")
                records.append({
                    "idx": item.get("idx", line_idx),
                    "func": item["func"],
                    "target": int(item["target"]),
                })
        return records

    def calibrate_threshold(
        self,
        val_probs_or_data: Any,
        val_labels: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Calibrate decision threshold strictly on validation split using Youden's J statistic:
            J(t) = TPR(t) - FPR(t) = Sensitivity(t) + Specificity(t) - 1
        Finds t* in [0.01, 0.99] that maximizes J.
        """
        if val_labels is None:
            val_probs = []
            val_labels = []
            for item in val_probs_or_data:
                pred = self.predict(item["func"])
                val_probs.append(pred["defect_probability"])
                val_labels.append(item["target"])
        else:
            val_probs = val_probs_or_data

        if not val_probs or not val_labels or len(val_probs) != len(val_labels):
            raise ValueError("Validation probabilities and labels must be non-empty and equal in length.")

        pos_count = sum(1 for y in val_labels if y == 1)
        neg_count = sum(1 for y in val_labels if y == 0)
        if pos_count == 0 or neg_count == 0:
            raise ValueError("Validation split must contain both positive and negative defect samples.")

        best_j = -1.0
        best_threshold = 0.5
        best_tpr = 0.0
        best_fpr = 0.0

        for t_int in range(1, 100):
            t = t_int / 100.0
            tp = sum(1 for p, y in zip(val_probs, val_labels) if p >= t and y == 1)
            fp = sum(1 for p, y in zip(val_probs, val_labels) if p >= t and y == 0)

            tpr = tp / pos_count
            fpr = fp / neg_count
            j = tpr - fpr

            if j > best_j:
                best_j = j
                best_threshold = t
                best_tpr = tpr
                best_fpr = fpr

        self.threshold = round(best_threshold, 4)
        self.calibrated_threshold = self.threshold
        self.has_calibrated_threshold = True
        res = {
            "optimal_threshold": self.threshold,
            "youden_j": round(best_j, 4),
            "val_tpr": round(best_tpr, 4),
            "val_fpr": round(best_fpr, 4),
            "calibration_samples": len(val_probs),
        }
        self.calibration_metadata.update(res)
        return res

    def predict(self, code_or_embedding: Any) -> Dict[str, Any]:
        """
        Predict defect probability on code snippet or embedding.
        In research mode, strictly forbids untrained predictions.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Untrained CodeBERT classifier (training_status=UNTRAINED). "
                "Per Frozen Protocol v1.0, research evaluation requires a calibrated, trained checkpoint. "
                "Synthetic probabilities are strictly prohibited."
            )
        if isinstance(code_or_embedding, list):
            prob = self.predict_proba(code_or_embedding)
        else:
            rep = self.extract_code_representation(str(code_or_embedding))
            prob = self.predict_proba(rep.get("concat_cls_mean", rep.get("vector", [])))
        is_def = prob >= self.threshold
        return {"defect_probability": round(prob, 4), "is_defective": is_def}

    def save_checkpoint(self, checkpoint_path: str):
        """Persist trained head weights, threshold, and calibration metadata to disk."""
        if not HAS_TORCH or self.head is None:
            raise RuntimeError("PyTorch is required to save neural network checkpoints.")
        if not self.is_trained:
            raise RuntimeError("Refusing to save untrained model checkpoint. Train model before persisting.")

        p = Path(checkpoint_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.head.state_dict(),
            "threshold": self.threshold,
            "is_trained": self.is_trained,
            "calibration_metadata": self.calibration_metadata,
        }
        torch.save(payload, str(p))
        self.weights_path = str(p)

    def load_checkpoint(self, checkpoint_path: str):
        """Load trained head weights, calibrated threshold, and provenance metadata."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required to load neural network checkpoints.")
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"CodeBERT defect head checkpoint not found: {checkpoint_path}")

        payload = torch.load(str(p), map_location="cpu", weights_only=False)
        if self.head is None:
            self.head = CodeBERTClassificationHead()
        self.head.load_state_dict(payload["state_dict"])
        self.head.eval()
        self.threshold = float(payload.get("threshold", 0.5))
        self.is_trained = bool(payload.get("is_trained", True))
        self.calibration_metadata = payload.get("calibration_metadata", self.calibration_metadata)
        self.calibration_metadata["training_status"] = "TRAINED_CHECKPOINT_LOADED"
        self.weights_path = str(p)

    def predict_proba(self, embedding_vector: List[float]) -> float:
        """
        Produce class 1 (defective) probability from a 1536-D or 768-D embedding.
        Strict Protocol Rule: Untrained models must NOT produce pseudo-probabilities in research mode.
        """
        if not self.is_trained:
            raise RuntimeError(
                "CodeBERTDefectClassifier is untrained (training_status=UNTRAINED). "
                "Per Frozen Protocol v1.0, research evaluation requires a calibrated, trained checkpoint. "
                "Synthetic probabilities are strictly prohibited."
            )

        if not HAS_TORCH or self.head is None:
            raise RuntimeError("PyTorch runtime is required for neural defect head inference.")

        if len(embedding_vector) == 768:
            v_1536 = embedding_vector + embedding_vector
        elif len(embedding_vector) == 1536:
            v_1536 = embedding_vector
        else:
            raise ValueError(f"Expected embedding vector dimension 768 or 1536, got {len(embedding_vector)}")

        with torch.no_grad():
            tensor_in = torch.tensor([v_1536], dtype=torch.float32)
            logits = self.head(tensor_in)
            probs = F.softmax(logits, dim=-1)
            prob_defect = probs[0, 1].item()
            return float(prob_defect)

    def is_defective(self, embedding_vector: List[float]) -> Tuple[bool, float]:
        if not self.is_trained:
            return (False, 0.0)
        if getattr(self, "mock_defective", False):
            return (True, 0.85)
        if not HAS_TORCH or self.head is None:
            if not self.require_real_weights:
                return (False, 0.0)
            raise RuntimeError("PyTorch runtime is required for neural defect head inference.")
        prob = self.predict_proba(embedding_vector)
        return (prob >= self.threshold, prob)


class CodeBERTAnalyzer:
    """
    CodeBERT-based code representation and semantic analysis engine.
    Uses 'microsoft/codebert-base' via Hugging Face Transformers when available.
    In research/evaluation mode, requires genuine pretrained weights.
    In production/offline mode, provides deterministic SHA-256 projections.
    """

    MODEL_NAME = "microsoft/codebert-base"
    VECTOR_DIM = 768

    def __init__(self, lazy_load: bool = True, require_real_weights: bool = False):
        self.tokenizer = None
        self.model = None
        self._is_loaded = False
        self._is_real_weights = False
        self._load_error = None
        self.require_real_weights = require_real_weights
        self.defect_head = CodeBERTDefectClassifier(threshold=0.5)

        if not lazy_load:
            self.load_model(require_real_weights=require_real_weights)

    def load_model(self, require_real_weights: Optional[bool] = None) -> bool:
        """
        Attempt to load the pretrained CodeBERT model and tokenizer.
        If require_real_weights is True, raises RuntimeError on failure instead of falling back.
        """
        must_require = self.require_real_weights if require_real_weights is None else require_real_weights

        if self._is_loaded:
            if must_require and not self._is_real_weights:
                raise RuntimeError(
                    f"CodeBERT real weights are required by protocol, but model failed to load: {self._load_error}"
                )
            return self._is_real_weights

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModel.from_pretrained(self.MODEL_NAME)
            self.model.eval()
            self._is_real_weights = True
            self._is_loaded = True
            return True
        except Exception as exc:
            self._load_error = str(exc)
            self._is_real_weights = False
            self._is_loaded = True
            if must_require:
                raise RuntimeError(
                    f"CodeBERT pretrained weights '{self.MODEL_NAME}' failed to load: {exc}. "
                    "Per Frozen Protocol v1.0, silent fallback is strictly prohibited in research mode."
                )
            return False

    @property
    def has_real_weights(self) -> bool:
        """Return True only if genuine CodeBERT pretrained model is loaded in memory."""
        return bool(self._is_real_weights and self.model is not None)

    @property
    def status(self) -> Dict[str, Any]:
        """Return the current engine status."""
        return {
            "model_name": self.MODEL_NAME,
            "vector_dimension": self.VECTOR_DIM,
            "is_loaded": self._is_loaded,
            "using_pretrained_weights": self._is_real_weights,
            "fallback_active": not self._is_real_weights,
            "load_error": self._load_error,
            "require_real_weights": self.require_real_weights,
        }

    def tokenize_code(self, code: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Tokenize source code using CodeBERT BPE tokenization.
        Uses deterministic SHA-256 hashing if Transformers is unavailable in offline mode.
        """
        if not self._is_loaded:
            self.load_model()

        if self._is_real_weights and self.tokenizer is not None:
            try:
                encoded = self.tokenizer(
                    code,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
                return {
                    "input_ids": encoded["input_ids"][0].tolist(),
                    "attention_mask": encoded["attention_mask"][0].tolist(),
                    "tokens": tokens[:64],
                    "token_count": int(encoded["attention_mask"].sum().item()),
                }
            except Exception:
                pass

        if self.require_real_weights:
            raise RuntimeError("Real CodeBERT tokenizer required in research mode, but unavailable.")

        # Deterministic offline tokenization using regex and SHA-256 modular mapping
        pattern = r"[A-Za-z_][A-Za-z0-9_]*|==|!=|<=|>=|\+\+|--|&&|\|\||[0-9]+|[^\s\w]"
        raw_tokens = re.findall(pattern, code)
        tokens = ["<s>"] + raw_tokens[: max_length - 2] + ["</s>"]
        
        # Deterministic SHA-256 token IDs (process-independent)
        input_ids = [0]
        for t in tokens[1:-1]:
            h_int = int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:8], 16)
            input_ids.append(h_int % 50265)
        input_ids.append(2)
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tokens": tokens[:64],
            "token_count": len(input_ids),
        }

    def extract_code_representation(self, code: str) -> Dict[str, Any]:
        """
        Generate contextual code representations (768-dimensional embedding vector).
        Extracts [CLS] and attention-mask mean-pooled representation across code tokens.
        """
        if not code or not code.strip():
            zero_vec = [0.0] * self.VECTOR_DIM
            return {
                "vector": zero_vec,
                "dimension": self.VECTOR_DIM,
                "model": self.MODEL_NAME,
                "token_count": 0,
                "is_pretrained_weights": self._is_real_weights,
            }

        if not self._is_loaded:
            self.load_model()

        if self._is_real_weights and self.model is not None and self.tokenizer is not None:
            try:
                import torch

                encoded = self.tokenizer(
                    code,
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    # 1. CLS representation (h_[CLS])
                    cls_rep = last_hidden[:, 0, :]  # shape: (1, 768)
                    
                    # 2. Attention-masked mean pooled representation
                    summed = (last_hidden * mask).sum(dim=1)
                    mean_pooled = summed / mask.sum(dim=1).clamp(min=1e-9)  # shape: (1, 768)
                    
                    # 3. Concatenated 1536-D representation for defect head
                    concat_1536 = torch.cat([cls_rep, mean_pooled], dim=-1)[0].tolist()
                    vector = mean_pooled[0].tolist()

                # L2 normalize
                norm = math.sqrt(sum(x * x for x in vector))
                if norm > 1e-9:
                    vector = [x / norm for x in vector]

                return {
                    "vector": vector,
                    "vector_1536": concat_1536,
                    "concat_cls_mean": concat_1536,
                    "dimension": len(vector),
                    "model": self.MODEL_NAME,
                    "token_count": int(encoded["attention_mask"].sum().item()),
                    "is_pretrained_weights": True,
                }
            except Exception as e:
                if self.require_real_weights:
                    raise RuntimeError(f"CodeBERT inference failed: {e}")

        if self.require_real_weights:
            raise RuntimeError("Real CodeBERT model required in research mode, but unavailable.")

        # Deterministic 768-D projection using SHA-256 basis (for offline development/tests only)
        token_info = self.tokenize_code(code)
        input_ids = token_info["input_ids"]

        vector = [0.0] * self.VECTOR_DIM
        for i, tid in enumerate(input_ids):
            pos_weight = 1.0 / (1.0 + math.log1p(i))
            for d in range(self.VECTOR_DIM):
                angle = (tid * 31 + d * 17) % 6283 / 1000.0
                vector[d] += math.sin(angle) * pos_weight

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 1e-9:
            vector = [v / norm for v in vector]

        concat_1536 = vector + vector
        return {
            "vector": vector,
            "vector_1536": concat_1536,
            "concat_cls_mean": concat_1536,
            "dimension": self.VECTOR_DIM,
            "model": self.MODEL_NAME,
            "token_count": token_info["token_count"],
            "is_pretrained_weights": False,
        }

    def compute_similarity(self, code_a: str, code_b: str) -> float:
        """
        Compute cosine similarity between two code snippets in CodeBERT vector space.
        Both vectors are L2-normalized.
        """
        rep_a = self.extract_code_representation(code_a)["vector"]
        rep_b = self.extract_code_representation(code_b)["vector"]

        dot_product = sum(a * b for a, b in zip(rep_a, rep_b))
        norm_a = math.sqrt(sum(a * a for a in rep_a))
        norm_b = math.sqrt(sum(b * b for b in rep_b))

        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0

        return max(-1.0, min(1.0, dot_product / (norm_a * norm_b)))

    def analyze_code_diff(self, diff_content: str) -> Dict[str, Any]:
        """
        Legacy diff analysis for backward compatibility with UI and Synthesizer.
        Extracts ONLY added lines (+), avoiding removed (-) lines.
        """
        added_lines = []
        for line in diff_content.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added_lines.append(line[1:])

        added_code = "\n".join(added_lines)
        if not added_code.strip():
            return {
                "token_count": 0,
                "embedding_sample": [0.0] * 8,
                "vector_dimension": self.VECTOR_DIM,
                "model_used": self.MODEL_NAME,
                "is_pretrained_weights": self._is_real_weights,
                "semantic_indicators": {
                    "unsafe_sql_pattern_similarity": 0.0,
                    "safe_sql_pattern_similarity": 0.0,
                    "semantic_risk_deviation": 0.0,
                },
                "summary": "No added lines found in diff.",
            }

        rep = self.extract_code_representation(added_code)

        canonical_safe_sql = "cursor.execute('SELECT * FROM users WHERE username = %s', (username,))"
        canonical_unsafe_sql = "cursor.execute('SELECT * FROM users WHERE username = ' + username)"

        safe_sim = self.compute_similarity(added_code, canonical_safe_sql)
        unsafe_sim = self.compute_similarity(added_code, canonical_unsafe_sql)
        sql_risk_similarity = round(max(0.0, unsafe_sim - safe_sim), 4)

        return {
            "token_count": rep["token_count"],
            "embedding_sample": rep["vector"][:8],
            "vector_dimension": rep["dimension"],
            "model_used": rep["model"],
            "is_pretrained_weights": rep["is_pretrained_weights"],
            "semantic_indicators": {
                "unsafe_sql_pattern_similarity": round(unsafe_sim, 4),
                "safe_sql_pattern_similarity": round(safe_sim, 4),
                "semantic_risk_deviation": sql_risk_similarity,
            },
            "summary": (
                f"CodeBERT analyzed {rep['token_count']} tokens. "
                f"Generated {rep['dimension']}-D contextual representation using {rep['model']}."
            ),
        }

    def analyze_diff_hunks(self, diff_content: str, pr_id: str = "pr_unknown", default_filename: str = "patch.py") -> List[ModelFinding]:
        """
        Approved E3 implementation: Evaluate individual diff hunks using CodeBERT
        representation and the supervised classification head.
        Assigns coarse HUNK_SPAN granularity (Protocol Amendment v1.1).
        """
        findings: List[ModelFinding] = []
        if not diff_content or not diff_content.strip():
            return findings

        # Check real weights and trained head requirement in research mode
        if self.require_real_weights:
            if not self._is_real_weights:
                findings.append(ModelFinding(
                    pr_id=pr_id,
                    source="codebert_head",
                    category=FindingCategory.BUG.value,
                    file_path=default_filename,
                    line_start=1,
                    evidence="",
                    description="CodeBERT model weights failed to load in research mode.",
                    execution_status=ExecutionStatus.TOOL_ERROR.value,
                    granularity=FindingGranularity.HUNK_SPAN.value
                ))
                return findings
            if not self.defect_head.is_trained:
                findings.append(ModelFinding(
                    pr_id=pr_id,
                    source="codebert_head",
                    category=FindingCategory.BUG.value,
                    file_path=default_filename,
                    line_start=1,
                    evidence="",
                    description="CodeBERT defect classification head is untrained in research mode.",
                    execution_status=ExecutionStatus.TOOL_ERROR.value,
                    granularity=FindingGranularity.HUNK_SPAN.value
                ))
                return findings

        # Parse diff into hunks
        hunk_header_pattern = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
        current_file = default_filename
        current_hunk_lines = []
        current_start_line = 1
        current_line_count = 1

        for raw_line in diff_content.splitlines():
            if raw_line.startswith("+++ b/"):
                current_file = raw_line[6:].strip()
                continue
            elif raw_line.startswith("+++ "):
                current_file = raw_line[4:].strip()
                continue

            match = hunk_header_pattern.match(raw_line)
            if match:
                # Process previous hunk if any
                if current_hunk_lines:
                    hunk_text = "\n".join(current_hunk_lines)
                    rep = self.extract_code_representation(hunk_text)
                    rep_vec = rep.get("vector_1536") or rep["vector"]
                    if self.defect_head.is_trained:
                        is_defect, prob = self.defect_head.is_defective(rep_vec)
                    else:
                        # Non-research development fallback only
                        is_defect = False
                        prob = 0.0

                    if is_defect:
                        findings.append(ModelFinding(
                            pr_id=pr_id,
                            source="codebert_head",
                            category=FindingCategory.BUG.value,
                            file_path=current_file,
                            line_start=current_start_line,
                            line_end=current_start_line + max(1, len(current_hunk_lines)),
                            evidence=hunk_text[:400],
                            severity=SeverityLevel.HIGH.value if prob > 0.75 else SeverityLevel.MEDIUM.value,
                            confidence=round(prob, 4),
                            description=f"CodeBERT classified diff hunk as potential defect (p={prob:.3f}).",
                            recommendation="Inspect modified hunk for logical inconsistency or unsafe patterns.",
                            granularity=FindingGranularity.HUNK_SPAN.value,
                            execution_status=ExecutionStatus.SUCCESS.value
                        ))
                current_hunk_lines = []
                current_start_line = int(match.group(2))
                continue

            # Accumulate added lines in current hunk
            if raw_line.startswith("+") and not raw_line.startswith("+++"):
                current_hunk_lines.append(raw_line[1:])

        # Process trailing hunk
        if current_hunk_lines:
            hunk_text = "\n".join(current_hunk_lines)
            rep = self.extract_code_representation(hunk_text)
            rep_vec = rep.get("vector_1536") or rep["vector"]
            if self.defect_head.is_trained:
                is_defect, prob = self.defect_head.is_defective(rep_vec)
            else:
                is_defect = False
                prob = 0.0

            if is_defect:
                findings.append(ModelFinding(
                    pr_id=pr_id,
                    source="codebert_head",
                    category=FindingCategory.BUG.value,
                    file_path=current_file,
                    line_start=current_start_line,
                    line_end=current_start_line + max(1, len(current_hunk_lines)),
                    evidence=hunk_text[:400],
                    severity=SeverityLevel.HIGH.value if prob > 0.75 else SeverityLevel.MEDIUM.value,
                    confidence=round(prob, 4),
                    description=f"CodeBERT classified diff hunk as potential defect (p={prob:.3f}).",
                    recommendation="Inspect modified hunk for logical inconsistency or unsafe patterns.",
                    granularity=FindingGranularity.HUNK_SPAN.value,
                    execution_status=ExecutionStatus.SUCCESS.value
                ))

        return findings


def get_codebert_analyzer(require_real_weights: bool = False, force_new: bool = False) -> CodeBERTAnalyzer:
    """Return singleton instance of CodeBERTAnalyzer."""
    global _CODEBERT_INSTANCE
    if force_new or _CODEBERT_INSTANCE is None:
        _CODEBERT_INSTANCE = CodeBERTAnalyzer(lazy_load=True, require_real_weights=require_real_weights)
    else:
        _CODEBERT_INSTANCE.require_real_weights = require_real_weights
    return _CODEBERT_INSTANCE
