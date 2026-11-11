"""
PR-Sentinel: Canonical Schemas and Data Models
Defines standard schemas for findings, ground-truth records, and pipeline configuration
following Frozen Experimental Protocol v1.0 and Amendments v1.1.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid

class FindingCategory(str, Enum):
    BUG = "BUG"
    SECURITY = "SECURITY"
    PERFORMANCE = "PERFORMANCE"
    CODE_QUALITY = "CODE_QUALITY"
    CONVENTION = "CONVENTION"
    UNKNOWN = "UNKNOWN"

class SeverityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"

class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
    TOOL_ERROR = "TOOL_ERROR"
    LLM_ERROR = "LLM_ERROR"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"
    FALLBACK_USED = "FALLBACK_USED"

class FindingGranularity(str, Enum):
    STATEMENT = "STATEMENT"
    HUNK_SPAN = "HUNK_SPAN"
    FILE = "FILE"

@dataclass
class ModelFinding:
    pr_id: str
    source: str  # e.g., "static_bandit", "static_flake8", "static_ast", "codebert_head", "llm_bug", "llm_security", etc.
    category: str  # BUG, SECURITY, PERFORMANCE, CODE_QUALITY, etc.
    file_path: str
    line_start: int
    evidence: str
    execution_status: str = ExecutionStatus.SUCCESS.value
    finding_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    line_end: Optional[int] = None
    rule_id: Optional[str] = None
    severity: str = SeverityLevel.MEDIUM.value
    confidence: Optional[float] = None
    description: str = ""
    recommendation: str = ""
    granularity: str = FindingGranularity.STATEMENT.value
    related_context_chunks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class GroundTruthIssue:
    issue_id: str
    repo: str
    pr_number: int
    pr_url: str
    cohort: str  # COHORT_1_CONFIRMED, COHORT_2_CANDIDATE, COHORT_3_UNRESOLVED
    verifiable: bool
    file_path: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    problematic_code: Optional[str] = None
    rule_name: Optional[str] = None
    standardized_category: str = FindingCategory.UNKNOWN.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PipelineConfig:
    experiment_id: str = "production"
    enable_codebert: bool = True
    enable_static: bool = True
    enable_rag: bool = True
    enable_llm: bool = True
    enable_multi_agent: bool = True
    enable_synthesis: bool = True
    enable_test_generator: bool = True
    require_real_weights: bool = False  # Set True in research/eval mode
    target_repo_dir: Optional[str] = None  # Mandatory for isolated RAG
    llm_model: str = "qwen2.5-coder:3b-instruct"
    ollama_url: str = "http://localhost:11434/api/generate"
    timeout_seconds: float = 30.0

    @classmethod
    def for_experiment(cls, exp_id: str, target_repo_dir: Optional[str] = None) -> "PipelineConfig":
        """Pre-configured experiment configurations adhering to Frozen Protocol v1.0."""
        exp = exp_id.upper()
        if exp == "E1":  # Standalone LLM
            return cls(
                experiment_id="E1",
                enable_codebert=False,
                enable_static=False,
                enable_rag=False,
                enable_llm=True,
                enable_multi_agent=False,  # Single-prompt review
                enable_synthesis=False,
                enable_test_generator=False,
                require_real_weights=False,
                target_repo_dir=None
            )
        elif exp == "E2":  # Static Analysis Only
            return cls(
                experiment_id="E2",
                enable_codebert=False,
                enable_static=True,
                enable_rag=False,
                enable_llm=False,
                enable_multi_agent=False,
                enable_synthesis=False,
                enable_test_generator=False,
                require_real_weights=False,
                target_repo_dir=None
            )
        elif exp == "E3":  # CodeBERT Only
            return cls(
                experiment_id="E3",
                enable_codebert=True,
                enable_static=False,
                enable_rag=False,
                enable_llm=False,
                enable_multi_agent=False,
                enable_synthesis=False,
                enable_test_generator=False,
                require_real_weights=True,  # STRICT requirement
                target_repo_dir=None
            )
        elif exp == "E4":  # LLM + RAG
            return cls(
                experiment_id="E4",
                enable_codebert=False,
                enable_static=False,
                enable_rag=True,
                enable_llm=True,
                enable_multi_agent=False,
                enable_synthesis=False,
                enable_test_generator=False,
                require_real_weights=False,
                target_repo_dir=target_repo_dir
            )
        elif exp == "E5":  # LLM + Multi-Agent
            return cls(
                experiment_id="E5",
                enable_codebert=False,
                enable_static=False,
                enable_rag=False,
                enable_llm=True,
                enable_multi_agent=True,
                enable_synthesis=True,
                enable_test_generator=False,
                require_real_weights=False,
                target_repo_dir=None
            )
        elif exp == "E6":  # Full PR-Sentinel
            return cls(
                experiment_id="E6",
                enable_codebert=True,
                enable_static=True,
                enable_rag=True,
                enable_llm=True,
                enable_multi_agent=True,
                enable_synthesis=True,
                enable_test_generator=True,
                require_real_weights=True,
                target_repo_dir=target_repo_dir
            )
        # Ablations
        elif exp == "E6_A1":  # E6 w/o CodeBERT
            return cls(
                experiment_id="E6_A1",
                enable_codebert=False,
                enable_static=True,
                enable_rag=True,
                enable_llm=True,
                enable_multi_agent=True,
                enable_synthesis=True,
                enable_test_generator=True,
                require_real_weights=False,
                target_repo_dir=target_repo_dir
            )
        elif exp == "E6_A2":  # E6 w/o Static Analysis
            return cls(
                experiment_id="E6_A2",
                enable_codebert=True,
                enable_static=False,
                enable_rag=True,
                enable_llm=True,
                enable_multi_agent=True,
                enable_synthesis=True,
                enable_test_generator=True,
                require_real_weights=True,
                target_repo_dir=target_repo_dir
            )
        elif exp == "E6_A3":  # E6 w/o RAG
            return cls(
                experiment_id="E6_A3",
                enable_codebert=True,
                enable_static=True,
                enable_rag=False,
                enable_llm=True,
                enable_multi_agent=True,
                enable_synthesis=True,
                enable_test_generator=True,
                require_real_weights=True,
                target_repo_dir=None
            )
        elif exp == "E6_A4":  # E6 w/o Multi-Agent (Monolithic prompt)
            return cls(
                experiment_id="E6_A4",
                enable_codebert=True,
                enable_static=True,
                enable_rag=True,
                enable_llm=True,
                enable_multi_agent=False,
                enable_synthesis=True,
                enable_test_generator=True,
                require_real_weights=True,
                target_repo_dir=target_repo_dir
            )
        else:
            return cls(experiment_id=exp_id)
