"""ox.cognition — 100× upgrade: hierarchical memory, metacognition, MCTS planning,
differential diagnosis, dynamic tool synthesis, DAG execution, transactions,
security, failure autopsy, preference learning, workspace state, explainability,
speculative execution, self-modification, anti-hallucination guards, dependency
awareness, semantic AST queries, git-native operations, database client,
progressive disclosure, clarification engineering, and OpenAPI client generation.
"""

from .diagnosis import DifferentialEngine
from .explanation import DecisionLedger, DecisionRecord
from .integration import CognitiveLayer
from .learning import FailureAutopsy, PreferenceLearner, SkillExtractor
from .memory import MemoryStore
from .metacognition import ConfidenceTracker, UncertaintyGate, combine_independent
from .planner import BacktrackingExecutor, MCTSPlanner, Plan, Step
from .security import InjectionSanitizer, Redactor
from .state import WorkspaceState
from .tools import DAGExecutor, EscalationChain, ToolSynthesizer, map_reduce
from .transactions import FileTransaction, StaleStateError
from .vectors import cosine, embed
from .speculation import SpeculativeEngine, SpeculativeBranch, BranchResult
from .selftuning import SelfTuner, ContextCompressor, ToolPatternOptimiser, BehaviouralAdapter
from .parameter_guard import ParameterGuard, SchemaAdapter, ToolSchema, SchemaField, ValidationReport, ValidationIssue
from .dependencies import DependencyModel, SemanticASTQueries, Dependency, Conflict
from .git_ops import GitOps, DatabaseClient, QueryResult, MergeConflict, BranchInfo, GitDiffHunk
from .disclosure import ProgressiveDisclosure, ClarificationEngine, ClarificationQuestion, OpenAPIGenerator

__all__ = [
    # memory & state
    "MemoryStore", "WorkspaceState",
    # metacognition
    "ConfidenceTracker", "UncertaintyGate", "combine_independent",
    # planning
    "MCTSPlanner", "BacktrackingExecutor", "Plan", "Step",
    # diagnosis & reasoning
    "DifferentialEngine",
    # tool ecosystem
    "ToolSynthesizer", "DAGExecutor", "EscalationChain", "map_reduce",
    # transactions
    "FileTransaction", "StaleStateError",
    # security
    "Redactor", "InjectionSanitizer",
    # learning
    "FailureAutopsy", "PreferenceLearner", "SkillExtractor",
    # explainability
    "DecisionLedger", "DecisionRecord",
    # speculation & self-modification
    "SpeculativeEngine", "SpeculativeBranch", "BranchResult",
    "SelfTuner", "ContextCompressor", "ToolPatternOptimiser", "BehaviouralAdapter",
    # anti-hallucination & schema flexibility
    "ParameterGuard", "SchemaAdapter", "ToolSchema", "SchemaField", "ValidationReport", "ValidationIssue",
    # dependency & AST semantics
    "DependencyModel", "SemanticASTQueries", "Dependency", "Conflict",
    # git & DB
    "GitOps", "DatabaseClient", "QueryResult", "MergeConflict", "BranchInfo", "GitDiffHunk",
    # disclosure & integration
    "ProgressiveDisclosure", "ClarificationEngine", "ClarificationQuestion", "OpenAPIGenerator",
    "CognitiveLayer",
    # vectors
    "embed", "cosine",
]
