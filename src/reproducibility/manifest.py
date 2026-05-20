from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.models.execution import ModelRunResult
from src.reproducibility.serializers import to_jsonable


EXPORT_SCHEMA_VERSION = "1.0"
REG_MONKEY_RELEASE_LABEL = "v2.0-reproducibility-pack-foundation"


def build_reproducibility_manifest(
    result: ModelRunResult,
    explanation_mode: str = "rule_based",
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    timestamp = generated_at or datetime.now(timezone.utc)
    spec = result.spec
    fit_metrics = dict(result.fit_metrics or {})
    advanced_outputs = dict(result.advanced_outputs or {})
    did_summary = advanced_outputs.get("did_summary") if isinstance(advanced_outputs.get("did_summary"), dict) else {}
    iv_summary = advanced_outputs.get("iv_summary") if isinstance(advanced_outputs.get("iv_summary"), dict) else {}
    psm_summary = advanced_outputs.get("psm_summary") if isinstance(advanced_outputs.get("psm_summary"), dict) else {}
    psm_balance_overview = (
        advanced_outputs.get("psm_balance_overview") if isinstance(advanced_outputs.get("psm_balance_overview"), dict) else {}
    )
    main_explanatory_variables = list(spec.main_independent_variables)
    if result.model_id == "did":
        main_explanatory_variables = list(fit_metrics.get("main_independent_variables") or main_explanatory_variables)
    if result.model_id == "iv_2sls":
        main_explanatory_variables = list(fit_metrics.get("main_independent_variables") or main_explanatory_variables)
    predictors = list(main_explanatory_variables) + list(spec.numeric_control_variables)

    return to_jsonable(
        {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "reg_monkey_release_label": REG_MONKEY_RELEASE_LABEL,
            "generated_at": timestamp.isoformat(),
            "model_id": result.model_id,
            "model_type": fit_metrics.get("model_type", result.model_id),
            "dependent_variable": spec.dependent_variable,
            "main_explanatory_variables": list(main_explanatory_variables),
            "treatment_variable": spec.treatment_variable,
            "post_variable": spec.post_variable,
            "group_variable": spec.group_variable,
            "cluster_variable": spec.cluster_variable,
            "endogenous_variable": spec.endogenous_variable,
            "instrument_variable": spec.instrument_variable,
            "instruments": list(spec.instruments),
            "exogenous_controls": list(spec.exogenous_controls),
            "iv_term": fit_metrics.get("iv_term") or iv_summary.get("iv_term"),
            "iv_estimate": fit_metrics.get("iv_estimate") if fit_metrics.get("iv_estimate") is not None else iv_summary.get("iv_estimate"),
            "first_stage_f_statistic": fit_metrics.get("first_stage_f_statistic")
            if fit_metrics.get("first_stage_f_statistic") is not None
            else iv_summary.get("first_stage_f_statistic"),
            "first_stage_r_squared": fit_metrics.get("first_stage_r_squared")
            if fit_metrics.get("first_stage_r_squared") is not None
            else iv_summary.get("first_stage_r_squared"),
            "iv_assumption_notes": iv_summary.get("assumption_notes", []) if result.model_id == "iv_2sls" else [],
            "did_term": fit_metrics.get("did_term") or did_summary.get("did_term"),
            "did_estimate": fit_metrics.get("did_estimate") if fit_metrics.get("did_estimate") is not None else did_summary.get("did_estimate"),
            "did_cell_counts": did_summary.get("cell_counts", []),
            "did_assumption_notes": (
                [
                    "parallel_trends_required_not_tested",
                    "conditional_did_estimate_no_automatic_causal_claim",
                ]
                if result.model_id == "did"
                else []
            ),
            "matching_covariates": list(spec.matching_covariates),
            "outcome_variable": psm_summary.get("outcome_variable") if result.model_id == "psm" else None,
            "psm_estimand": spec.psm_estimand or psm_summary.get("estimand"),
            "matching_method": spec.matching_method or psm_summary.get("matching_method"),
            "replacement_matching": psm_summary.get("replacement_matching") if result.model_id == "psm" else None,
            "caliper": spec.caliper if spec.caliper is not None else psm_summary.get("caliper"),
            "treated_count": psm_summary.get("treated_count"),
            "control_count": psm_summary.get("control_count"),
            "matched_treated_count": psm_summary.get("matched_treated_count"),
            "matched_control_count": psm_summary.get("matched_control_count"),
            "unmatched_treated_count": psm_summary.get("unmatched_treated_count"),
            "att_estimate": psm_summary.get("att_estimate"),
            "balance_summary_available": bool(advanced_outputs.get("balance_summary")),
            "psm_balance_summary_available": bool(advanced_outputs.get("balance_summary")),
            "max_absolute_smd_before": psm_balance_overview.get("max_absolute_smd_before"),
            "max_absolute_smd_after": psm_balance_overview.get("max_absolute_smd_after"),
            "mean_absolute_smd_before": psm_balance_overview.get("mean_absolute_smd_before"),
            "mean_absolute_smd_after": psm_balance_overview.get("mean_absolute_smd_after"),
            "covariates_improved_count": psm_balance_overview.get("covariates_improved_count"),
            "covariates_worsened_count": psm_balance_overview.get("covariates_worsened_count"),
            "psm_assumption_notes": psm_summary.get("assumption_notes", []) if result.model_id == "psm" else [],
            "controls": list(spec.numeric_control_variables),
            "numeric_controls": list(spec.numeric_control_variables),
            "categorical_controls": list(spec.categorical_control_variables),
            "entity_id": spec.entity_id,
            "time_id": spec.time_id,
            "time_variable": spec.time_id,
            "entity_effects": spec.entity_effects,
            "time_effects": spec.time_effects,
            "fixed_effects": {
                "entity": spec.entity_effects,
                "time": spec.time_effects,
            },
            "standard_errors": spec.standard_errors,
            "observations_used": result.sample_info.get("final_rows") or fit_metrics.get("n_obs"),
            "predictor_count": len([item for item in predictors if item]),
            "fit_metrics_available": sorted(str(key) for key in fit_metrics.keys()),
            "diagnostics_count": len(result.structured_diagnostics),
            "warnings_count": len(result.warnings),
            "advanced_outputs_available": sorted(str(key) for key in advanced_outputs.keys()),
            "explanation_mode": explanation_mode,
            "deterministic_execution_note": "Statistical outputs come from deterministic ModelSpec -> registry -> runner -> ModelRunResult execution.",
            "llm_safety_note": "LLM or mock explanation, if selected, cannot alter coefficients, p-values, diagnostics, warnings, fit metrics, or sample size.",
        }
    )
