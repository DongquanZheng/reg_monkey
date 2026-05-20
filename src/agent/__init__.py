"""Minimal scaffolding for the future Reg Monkey Analysis Planner."""

from .analysis_planner import generate_analysis_plan
from .narrative import NarrativeInput, NarrativeResult, generate_narrative, narrative_markdown, narrative_to_dict
from .schemas import AnalysisPlan, ModelRecommendation, PlannerWarning, VariableCandidate
from .workflow import GuidedWorkflowResult, WorkflowComparison, WorkflowModelResult, WorkflowRunConfig, build_workflow_config, run_guided_workflow, workflow_to_dict

__all__ = [
    "AnalysisPlan",
    "GuidedWorkflowResult",
    "ModelRecommendation",
    "NarrativeInput",
    "NarrativeResult",
    "PlannerWarning",
    "VariableCandidate",
    "WorkflowComparison",
    "WorkflowModelResult",
    "WorkflowRunConfig",
    "build_workflow_config",
    "generate_analysis_plan",
    "generate_narrative",
    "narrative_markdown",
    "narrative_to_dict",
    "run_guided_workflow",
    "workflow_to_dict",
]
