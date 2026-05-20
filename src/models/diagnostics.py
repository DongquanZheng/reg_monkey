from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


class DiagnosticSeverity:
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    CONSTRAINT = "constraint"


class DiagnosticCode:
    VALIDATION_ERROR = "validation_error"
    MODEL_EXECUTION_FAILED = "model_execution_failed"
    MISSING_DATA_DROPPED = "missing_data_dropped"
    SMALL_SAMPLE = "small_sample"
    BINARY_OUTCOME_CHECK = "binary_outcome_check"
    BINARY_CLASS_IMBALANCE = "binary_class_imbalance"
    CONSTANT_VARIABLE = "constant_variable"
    HIGH_MULTICOLLINEARITY = "high_multicollinearity"
    SERIOUS_MULTICOLLINEARITY = "serious_multicollinearity"
    PERFECT_SEPARATION_RISK = "perfect_separation_risk"
    PANEL_STRUCTURE = "panel_structure"
    PANEL_FE_SPECIFICATION = "panel_fe_specification"
    WITHIN_VARIATION = "within_variation"
    FIXED_EFFECT_INTERPRETATION = "fixed_effect_interpretation"
    ASSOCIATION_NOT_CAUSATION = "association_not_causation"
    STANDARD_ERROR_NOTE = "standard_error_note"
    LEGACY_WARNING = "legacy_warning"
    ROBUST_STANDARD_ERRORS_USED = "robust_standard_errors_used"
    MODEL_WIDE_TEST_UNAVAILABLE = "model_wide_test_unavailable"
    HETEROSKEDASTICITY_DETECTED = "heteroskedasticity_detected"
    NO_STRONG_HETEROSKEDASTICITY_EVIDENCE = "no_strong_heteroskedasticity_evidence"
    ODDS_RATIOS_AVAILABLE = "odds_ratios_available"
    MARGINAL_EFFECTS_AVAILABLE = "marginal_effects_available"
    MARGINAL_EFFECTS_FAILED = "marginal_effects_failed"
    DID_PARALLEL_TRENDS_ASSUMPTION = "did_parallel_trends_assumption"
    DID_SPEC_MISSING_FIELD = "did_spec_missing_field"
    DID_SPEC_MISSING_COLUMN = "did_spec_missing_column"
    DID_CELL_COVERAGE = "did_cell_coverage"
    DID_SPARSE_CELL_SUPPORT = "did_sparse_cell_support"
    DID_LIMITED_VARIATION = "did_limited_variation"
    DID_INVALID_INDICATOR = "did_invalid_indicator"
    DID_INSUFFICIENT_SAMPLE = "did_insufficient_sample"
    DID_COLLINEAR_DESIGN = "did_collinear_design"
    DID_UNSUPPORTED_STAGGERED_ADOPTION = "did_unsupported_staggered_adoption"
    IV_IDENTIFICATION_ASSUMPTION = "iv_identification_assumption"
    IV_FIRST_STAGE_STRENGTH = "iv_first_stage_strength"
    IV_FIRST_STAGE_UNAVAILABLE = "iv_first_stage_unavailable"
    IV_LOW_FIRST_STAGE_R2 = "iv_low_first_stage_r2"
    IV_INSTRUMENT_COUNT = "iv_instrument_count"
    IV_CONTROLS_INCLUDED = "iv_controls_included"
    IV_SINGLE_ENDOGENOUS_LIMITATION = "iv_single_endogenous_limitation"
    PSM_IDENTIFICATION_ASSUMPTION_OBSERVED_COVARIATES = "psm_identification_assumption_observed_covariates"
    PSM_NO_UNOBSERVED_CONFOUNDING_CAUTION = "psm_no_unobserved_confounding_caution"
    PSM_COMMON_SUPPORT_SUMMARY = "psm_common_support_summary"
    PSM_POOR_OVERLAP_WARNING = "psm_poor_overlap_warning"
    PSM_SPARSE_CONTROL_POOL = "psm_sparse_control_pool"
    PSM_BALANCE_NOT_IMPROVED = "psm_balance_not_improved"
    PSM_SMALL_MATCHED_SAMPLE = "psm_small_matched_sample"
    PSM_CALIPER_DROPS_TREATED_UNITS = "psm_caliper_drops_treated_units"
    PSM_BALANCE_SUMMARY_AVAILABLE = "psm_balance_summary_available"
    PSM_HIGH_RESIDUAL_IMBALANCE = "psm_high_residual_imbalance"
    PSM_REPLACEMENT_MATCHING_NOTICE = "psm_replacement_matching_notice"
    PSM_ATT_ONLY_NOTICE = "psm_att_only_notice"
    PSM_PROPENSITY_ESTIMATION_FAILED = "psm_propensity_estimation_failed"
    PSM_MATCHING_SUPPORT_FAILED = "psm_matching_support_failed"


@dataclass(frozen=True)
class ModelDiagnostic:
    code: str
    severity: str
    title: str
    message: str
    affected_variables: list[str] = field(default_factory=list)
    recommendation: str = ""
    show_in_ui: bool = True
    show_in_report: bool = True
    llm_instruction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnostic_to_dict(diagnostic: ModelDiagnostic) -> dict[str, Any]:
    return diagnostic.to_dict()


def diagnostics_to_dict(diagnostics: list[ModelDiagnostic]) -> list[dict[str, Any]]:
    return [diagnostic.to_dict() for diagnostic in diagnostics]


def blocking_validation_diagnostics(errors: list[str], model_id: str = "") -> list[ModelDiagnostic]:
    if model_id == "panel_fe":
        return [_panel_validation_diagnostic(error) for error in errors]
    if model_id == "did":
        return [_did_validation_diagnostic(error) for error in errors]
    if model_id == "iv_2sls":
        return [_iv_validation_diagnostic(error) for error in errors]
    if model_id == "psm":
        return [_psm_validation_diagnostic(error) for error in errors]
    return [
        ModelDiagnostic(
            code=DiagnosticCode.VALIDATION_ERROR,
            severity=DiagnosticSeverity.ERROR,
            title="Model specification is not runnable",
            message=error,
            recommendation="Review the selected variables and model options before running the model.",
            llm_instruction="Treat this as a blocking error. Do not interpret model results because no model was estimated.",
        )
        for error in errors
    ]


def failed_run_diagnostic(error: str, model_id: str = "") -> ModelDiagnostic:
    if model_id == "panel_fe":
        return _panel_failure_diagnostic(error)
    if model_id == "psm":
        return _psm_failure_diagnostic(error)
    return ModelDiagnostic(
        code=DiagnosticCode.MODEL_EXECUTION_FAILED,
        severity=DiagnosticSeverity.ERROR,
        title="Model execution failed",
        message=error,
        recommendation="Review the model sample, selected variables, and model-specific requirements.",
        llm_instruction="Treat this as a failed deterministic model run. Explain the failure and do not infer coefficient results.",
    )


def _panel_validation_diagnostic(error: str) -> ModelDiagnostic:
    lower = error.lower()
    title = "Panel FE specification is not runnable"
    recommendation = "Review the panel identifiers, selected variables, fixed effects, and standard error setting."
    if "entity id" in lower:
        title = "Missing or invalid entity ID"
        recommendation = "Select a valid entity ID before running Panel Fixed Effects."
    elif "time id" in lower or "time period" in lower:
        title = "Missing or invalid time variable"
        recommendation = "Select a valid time variable with more than one period."
    elif "duplicate entity-time" in lower:
        title = "Duplicate entity-time observations"
        recommendation = "Ensure each entity-time pair appears once before running Panel Fixed Effects."
    elif "too few observations" in lower:
        title = "Insufficient Panel FE estimation sample"
        recommendation = "Check missing values, entity ID, time variable, or use OLS as a baseline first."
    elif "constant-only" in lower or "absorbed" in lower:
        title = "Variables may be absorbed by fixed effects"
        recommendation = "Remove variables that have no usable variation after fixed effects are applied."
    elif "standard errors" in lower:
        title = "Unsupported Panel FE standard error setting"
        recommendation = "Use conventional, robust, or cluster-by-entity standard errors."
    return ModelDiagnostic(
        code=DiagnosticCode.PANEL_STRUCTURE if "entity" in lower or "time" in lower or "duplicate" in lower else DiagnosticCode.VALIDATION_ERROR,
        severity=DiagnosticSeverity.ERROR,
        title=title,
        message=error,
        recommendation=recommendation,
        llm_instruction="Treat this as a blocking Panel FE validation issue. Do not interpret model results because no Panel FE model was estimated.",
    )


def _panel_failure_diagnostic(error: str) -> ModelDiagnostic:
    lower = error.lower()
    title = "Panel FE estimation failed"
    recommendation = "Review the panel sample, selected variables, entity/time identifiers, and fixed-effect specification."
    if "too few observations" in lower:
        title = "Insufficient Panel FE estimation sample"
        recommendation = "Check missing values, entity ID, time variable, or use OLS as a baseline first."
    elif "absorbed" in lower:
        title = "Variables may be absorbed by fixed effects"
        recommendation = "Review whether selected variables have enough variation after fixed effects are applied."
    elif "insufficient independent variation" in lower or "rank" in lower or "collinear" in lower:
        title = "Insufficient variation after fixed effects"
        recommendation = "Simplify the specification or choose variables with enough within-entity variation."
    return ModelDiagnostic(
        code=DiagnosticCode.MODEL_EXECUTION_FAILED,
        severity=DiagnosticSeverity.ERROR,
        title=title,
        message=error,
        recommendation=recommendation,
        llm_instruction="Explain the Panel FE failure in user-facing language and keep technical details separate.",
    )


def _did_validation_diagnostic(error: str) -> ModelDiagnostic:
    lower = error.lower()
    code = DiagnosticCode.VALIDATION_ERROR
    title = "DID specification is not runnable"
    recommendation = "Review the outcome, treatment, post-period, controls, and standard error setting."
    if "missing required field" in lower:
        code = DiagnosticCode.DID_SPEC_MISSING_FIELD
        title = "DID specification is incomplete"
        recommendation = "Select valid outcome, treatment, and post-period variables before running DID."
    elif "not in the dataset" in lower:
        code = DiagnosticCode.DID_SPEC_MISSING_COLUMN
        title = "DID specification references a missing column"
        recommendation = "Select DID variables that exist in the current dataset."
    elif "binary-like" in lower or "binary or numeric indicator" in lower:
        code = DiagnosticCode.DID_INVALID_INDICATOR
        title = "Invalid DID indicator"
        recommendation = "Use 0/1 indicators or two-level categorical variables for treatment and post-period fields."
    elif "not enough observations" in lower or "no complete observations remain" in lower:
        code = DiagnosticCode.DID_INSUFFICIENT_SAMPLE
        title = "Insufficient DID estimation sample"
        recommendation = "Check missing values, selected controls, and DID cell support before estimating the model."
    elif "staggered adoption" in lower or "time-varying treatment" in lower:
        code = DiagnosticCode.DID_UNSUPPORTED_STAGGERED_ADOPTION
        title = "Unsupported DID treatment timing"
        recommendation = "Use the minimal DID runner only for a stable treated/control and pre/post structure."
    elif "collinear" in lower or "absorbed" in lower:
        code = DiagnosticCode.DID_COLLINEAR_DESIGN
        title = "Collinear DID design"
        recommendation = "Remove absorbed controls or revise the DID specification before running the model."
    elif "variation" in lower or "interaction" in lower:
        code = DiagnosticCode.DID_LIMITED_VARIATION
        title = "Limited DID variation"
        recommendation = "Check whether treatment, post-period, and treatment-by-post cells contain enough variation."
    elif "cluster" in lower:
        title = "Invalid DID cluster setting"
        recommendation = "Select a valid cluster variable with at least two clusters, or use non-clustered standard errors."
    return ModelDiagnostic(
        code=code,
        severity=DiagnosticSeverity.ERROR,
        title=title,
        message=error,
        recommendation=recommendation,
        llm_instruction="Treat this as a blocking DID validation issue. Do not interpret DID results because no DID model was estimated.",
    )


def _iv_validation_diagnostic(error: str) -> ModelDiagnostic:
    lower = error.lower()
    title = "IV/2SLS specification is not runnable"
    recommendation = "Review the outcome, endogenous variable, instruments, controls, and sample support."
    if "missing required field" in lower or "requires at least one instrument" in lower:
        title = "IV/2SLS specification is incomplete"
        recommendation = "Select outcome, endogenous, and instrument variables before running IV/2SLS."
    elif "not in the dataset" in lower:
        title = "IV/2SLS specification references a missing column"
        recommendation = "Select IV/2SLS variables that exist in the current dataset."
    elif "cannot also be used" in lower or "must not duplicate" in lower or "distinct" in lower:
        title = "IV/2SLS variable roles conflict"
        recommendation = "Use distinct variables for outcome, endogenous regressor, instruments, and controls."
    elif "numeric" in lower:
        title = "IV/2SLS variables must be numeric"
        recommendation = "Use numeric variables or preprocess variables before running minimal IV/2SLS."
    elif "not enough observations" in lower or "no complete observations" in lower:
        title = "Insufficient IV/2SLS estimation sample"
        recommendation = "Check missing values and simplify the IV/2SLS specification."
    elif "collinear" in lower or "rank" in lower:
        title = "Collinear IV/2SLS design"
        recommendation = "Remove collinear controls or revise the instrument/control set."
    return ModelDiagnostic(
        code=DiagnosticCode.VALIDATION_ERROR,
        severity=DiagnosticSeverity.ERROR,
        title=title,
        message=error,
        recommendation=recommendation,
        llm_instruction="Treat this as a blocking IV/2SLS validation issue. Do not interpret IV results because no IV model was estimated.",
    )


def _psm_validation_diagnostic(error: str) -> ModelDiagnostic:
    lower = error.lower()
    title = "PSM specification is not runnable"
    recommendation = "Review the outcome, treatment, matching covariates, estimand, matching method, and sample support."
    if "missing required field" in lower or "matching covariate" in lower:
        title = "PSM specification is incomplete"
        recommendation = "Select outcome, treatment, and matching covariates before running PSM."
    elif "not in the dataset" in lower:
        title = "PSM specification references a missing column"
        recommendation = "Select PSM variables that exist in the current dataset."
    elif "cannot also" in lower or "must not duplicate" in lower or "duplicate" in lower:
        title = "PSM variable roles conflict"
        recommendation = "Use distinct variables for outcome, treatment, and matching covariates."
    elif "binary" in lower:
        title = "Invalid PSM treatment indicator"
        recommendation = "Use a binary 0/1 treatment variable for minimal PSM."
    elif "numeric" in lower:
        title = "PSM variables must be numeric"
        recommendation = "Use numeric outcome, treatment, and matching covariates or preprocess variables before running minimal PSM."
    elif "no complete observations" in lower or "not enough observations" in lower:
        title = "Insufficient PSM estimation sample"
        recommendation = "Check missing values and simplify the matching specification."
    elif "collinear" in lower or "rank" in lower:
        title = "Collinear PSM propensity-score design"
        recommendation = "Remove collinear matching covariates before running PSM."
    elif "control observation" in lower:
        title = "Insufficient control observations"
        recommendation = "PSM nearest-neighbor ATT requires at least one usable control observation."
    elif "treated observation" in lower:
        title = "Insufficient treated observations"
        recommendation = "PSM ATT requires at least one usable treated observation."
    return ModelDiagnostic(
        code=DiagnosticCode.VALIDATION_ERROR,
        severity=DiagnosticSeverity.ERROR,
        title=title,
        message=error,
        recommendation=recommendation,
        llm_instruction="Treat this as a blocking PSM validation issue. Do not interpret PSM results because no PSM model was estimated.",
    )


def _psm_failure_diagnostic(error: str) -> ModelDiagnostic:
    lower = error.lower()
    if (
        "caliper eliminated all treated observations" in lower
        or "no valid control matches remain" in lower
        or "no valid matches" in lower
        or "no matched" in lower
    ):
        return ModelDiagnostic(
            code=DiagnosticCode.PSM_MATCHING_SUPPORT_FAILED,
            severity=DiagnosticSeverity.ERROR,
            title="PSM matching support failed",
            message="No valid matches were formed under the current caliper/common-support condition, so ATT cannot be estimated reliably.",
            recommendation="Review propensity-score overlap, simplify matching covariates, or relax the caliper before rerunning PSM.",
            llm_instruction="Treat this as a failed PSM run. Do not report ATT; explain that the matching specification left no valid treated-control comparison.",
        )
    if "propensity score" in lower or "overlap" in lower or "support" in lower:
        return ModelDiagnostic(
            code=DiagnosticCode.PSM_PROPENSITY_ESTIMATION_FAILED,
            severity=DiagnosticSeverity.ERROR,
            title="PSM propensity-score model could not be estimated reliably",
            message=error,
            recommendation="Review treatment/control overlap, simplify matching covariates, or remove collinear covariates before interpreting PSM.",
            llm_instruction="Treat this as a failed PSM run. Do not report ATT; explain that propensity estimation was unstable and observed-covariate overlap should be reviewed.",
        )
    return ModelDiagnostic(
        code=DiagnosticCode.MODEL_EXECUTION_FAILED,
        severity=DiagnosticSeverity.ERROR,
        title="PSM estimation failed",
        message=error,
        recommendation="Review treatment/control support, selected covariates, caliper, and sample size before rerunning PSM.",
        llm_instruction="Treat this as a failed PSM run. Do not infer ATT because matching did not complete.",
    )


def diagnose_missingness(input_rows: int, final_rows: int) -> list[ModelDiagnostic]:
    dropped = input_rows - final_rows
    if input_rows <= 0 or dropped <= 0:
        return []
    dropped_pct = dropped / input_rows * 100
    severity = DiagnosticSeverity.WARNING if dropped_pct > 30 else DiagnosticSeverity.INFO
    return [
        ModelDiagnostic(
            code=DiagnosticCode.MISSING_DATA_DROPPED,
            severity=severity,
            title="Missing data dropped from estimation sample",
            message=f"Model estimation used {final_rows} of {input_rows} rows after dropping missing values ({dropped_pct:.1f}% dropped).",
            recommendation="Check whether missingness is systematic before interpreting results.",
            llm_instruction="Mention that the model sample differs from the confirmed dataset when explaining limitations.",
        )
    ]


def diagnose_sample_size(n_obs: int, predictor_count: int = 0, model_id: str = "") -> list[ModelDiagnostic]:
    diagnostics: list[ModelDiagnostic] = []
    if n_obs and n_obs < 30:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.SMALL_SAMPLE,
                severity=DiagnosticSeverity.WARNING,
                title="Small estimation sample",
                message=f"The estimation sample has {n_obs} observations, so estimates may be unstable.",
                recommendation="Use more data or simplify the model if possible.",
                llm_instruction="State that the result is sample-size sensitive.",
            )
        )
    if predictor_count and n_obs and n_obs <= predictor_count * 10:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.SMALL_SAMPLE,
                severity=DiagnosticSeverity.WARNING,
                title="Many predictors relative to observations",
                message=f"The model uses {predictor_count} predictor(s) with {n_obs} observations.",
                recommendation="Consider a simpler specification or more observations.",
                llm_instruction="Warn that coefficient estimates may be unstable due to model complexity relative to sample size.",
            )
        )
    return diagnostics


def diagnose_binary_outcome(df: pd.DataFrame, y_col: str) -> list[ModelDiagnostic]:
    if not y_col or y_col not in df.columns:
        return []
    values = df[y_col].dropna()
    unique_count = int(values.nunique(dropna=True))
    is_binary = unique_count == 2
    severity = DiagnosticSeverity.INFO if is_binary else DiagnosticSeverity.ERROR
    message = (
        f"Dependent variable '{y_col}' has two non-missing outcome classes."
        if is_binary
        else f"Dependent variable '{y_col}' has {unique_count} non-missing outcome class(es), but Logit/Probit require exactly two."
    )
    diagnostics = [
        ModelDiagnostic(
            code=DiagnosticCode.BINARY_OUTCOME_CHECK,
            severity=severity,
            title="Binary outcome check",
            message=message,
            affected_variables=[y_col],
            recommendation="Use Logit/Probit only for binary outcomes.",
            llm_instruction="For binary models, explain coefficient signs as probability tendency, not percentage-point changes.",
        )
    ]
    if is_binary and len(values) > 0:
        counts = values.value_counts(dropna=True)
        minority_count = int(counts.min())
        minority_share = minority_count / len(values)
        if minority_count < 5 or minority_share < 0.10:
            diagnostics.append(
                ModelDiagnostic(
                    code=DiagnosticCode.BINARY_CLASS_IMBALANCE,
                    severity=DiagnosticSeverity.WARNING,
                    title="Binary outcome class imbalance",
                    message=f"Binary outcome '{y_col}' is imbalanced: the minority class has {minority_count} observation(s) ({minority_share * 100:.1f}%).",
                    affected_variables=[y_col],
                    recommendation="Check whether the rare outcome class has enough observations for stable Logit/Probit estimates.",
                    llm_instruction="Mention class imbalance as a limitation for binary model interpretation.",
                )
            )
    return diagnostics


def diagnose_constant_variables(df: pd.DataFrame, variables: list[str]) -> list[ModelDiagnostic]:
    constant = [variable for variable in variables if variable in df.columns and df[variable].nunique(dropna=True) <= 1]
    near_constant: list[str] = []
    for variable in variables:
        if variable not in df.columns or variable in constant:
            continue
        values = df[variable].dropna()
        if values.empty:
            continue
        top_share = values.value_counts(dropna=True, normalize=True).max()
        if top_share >= 0.98:
            near_constant.append(variable)

    diagnostics: list[ModelDiagnostic] = []
    if constant:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.CONSTANT_VARIABLE,
                severity=DiagnosticSeverity.WARNING,
                title="Constant variables",
                message="These selected variables have no variation in the estimation sample: " + ", ".join(constant) + ".",
                affected_variables=constant,
                recommendation="Remove constant variables from the model specification.",
                llm_instruction="Do not interpret coefficients for variables without meaningful variation.",
            )
        )
    if near_constant:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.CONSTANT_VARIABLE,
                severity=DiagnosticSeverity.WARNING,
                title="Near-constant variables",
                message="These selected variables have very limited variation in the estimation sample: " + ", ".join(near_constant) + ".",
                affected_variables=near_constant,
                recommendation="Check whether near-constant variables are meaningful predictors.",
                llm_instruction="Mention near-constant predictors as an estimate stability and interpretation limitation.",
            )
        )
    return diagnostics


def diagnose_multicollinearity(vif_df: pd.DataFrame | None) -> list[ModelDiagnostic]:
    if vif_df is None or vif_df.empty or "VIF" not in vif_df.columns:
        return []
    frame = vif_df.copy()
    frame["VIF"] = pd.to_numeric(frame["VIF"], errors="coerce")
    serious = frame[frame["VIF"] > 10]
    high = frame[(frame["VIF"] > 5) & (frame["VIF"] <= 10)]
    diagnostics: list[ModelDiagnostic] = []
    if not high.empty:
        variables = high["variable"].astype(str).tolist()
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.HIGH_MULTICOLLINEARITY,
                severity=DiagnosticSeverity.WARNING,
                title="High multicollinearity",
                message="VIF exceeds 5 for: " + ", ".join(variables) + ".",
                affected_variables=variables,
                recommendation="Check correlations or simplify highly related controls.",
                llm_instruction="Mention that coefficient precision and interpretation may be affected by multicollinearity.",
            )
        )
    if not serious.empty:
        variables = serious["variable"].astype(str).tolist()
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.SERIOUS_MULTICOLLINEARITY,
                severity=DiagnosticSeverity.WARNING,
                title="Serious multicollinearity",
                message="VIF exceeds 10 for: " + ", ".join(variables) + ".",
                affected_variables=variables,
                recommendation="Interpret coefficients cautiously and consider removing highly collinear variables.",
                llm_instruction="Flag this as a serious interpretation risk for coefficient-level findings.",
            )
        )
    return diagnostics


def diagnose_robust_standard_errors(model_id: str, summary: dict[str, Any]) -> list[ModelDiagnostic]:
    if model_id != "ols" or not summary.get("robust_standard_errors"):
        return []
    cov_type = str(summary.get("robust_cov_type") or summary.get("standard_errors") or "HC3")
    return [
        ModelDiagnostic(
            code=DiagnosticCode.ROBUST_STANDARD_ERRORS_USED,
            severity=DiagnosticSeverity.INFO,
            title="Robust standard errors used",
            message=f"{cov_type} robust standard errors are used to reduce sensitivity to heteroskedasticity.",
            recommendation="Interpret statistical significance using the reported robust standard errors, while still reviewing model diagnostics.",
            show_in_ui=False,
            show_in_report=True,
            llm_instruction="Mention robust standard errors as an uncertainty adjustment, not as a fix for all model misspecification.",
        )
    ]


def diagnose_heteroskedasticity(test_result: dict[str, Any] | None) -> list[ModelDiagnostic]:
    if not test_result:
        return []
    p_value = test_result.get("p_value")
    try:
        p = float(p_value)
    except (TypeError, ValueError):
        return []
    if pd.isna(p):
        return []
    statistic = test_result.get("statistic")
    statistic_text = f"{float(statistic):.3f}" if statistic is not None and not pd.isna(statistic) else "N/A"
    p_text = "<0.001" if p < 0.001 else f"{p:.3f}"
    if p < 0.05:
        return [
            ModelDiagnostic(
                code=DiagnosticCode.HETEROSKEDASTICITY_DETECTED,
                severity=DiagnosticSeverity.WARNING,
                title="Heteroskedasticity detected",
                message=f"Breusch-Pagan test suggests heteroskedasticity (statistic {statistic_text}, p = {p_text}).",
                recommendation="Consider robust standard errors and interpret conventional standard errors cautiously.",
                llm_instruction="Mention heteroskedasticity once as a limitation for OLS uncertainty estimates.",
            )
        ]
    return [
        ModelDiagnostic(
            code=DiagnosticCode.NO_STRONG_HETEROSKEDASTICITY_EVIDENCE,
            severity=DiagnosticSeverity.INFO,
            title="No strong heteroskedasticity evidence",
            message=f"Breusch-Pagan test does not show strong evidence of heteroskedasticity (statistic {statistic_text}, p = {p_text}).",
            recommendation="Continue reviewing other OLS diagnostics before interpreting results.",
            show_in_ui=False,
            show_in_report=True,
            llm_instruction="Do not overstate this as proof of homoskedasticity; describe it as no strong evidence in this diagnostic.",
        )
    ]


def diagnose_advanced_outputs(model_id: str, advanced_outputs: dict[str, Any] | None) -> list[ModelDiagnostic]:
    if not advanced_outputs:
        return []
    diagnostics: list[ModelDiagnostic] = []
    if model_id == "logit" and isinstance(advanced_outputs.get("odds_ratio_table"), pd.DataFrame):
        table = advanced_outputs.get("odds_ratio_table")
        if table is not None and not table.empty:
            diagnostics.append(
                ModelDiagnostic(
                    code=DiagnosticCode.ODDS_RATIOS_AVAILABLE,
                    severity=DiagnosticSeverity.INFO,
                    title="Odds ratios available",
                    message="Logit odds ratios are available as a structured advanced output.",
                    recommendation="Use odds ratios as an odds-scale interpretation, not as probability percentage-point changes.",
                    show_in_ui=False,
                    show_in_report=True,
                    llm_instruction="When interpreting Logit, odds ratios above 1 indicate higher odds and below 1 indicate lower odds, holding the model specification constant.",
                )
            )
    marginal_table = advanced_outputs.get("marginal_effects_table")
    if model_id in {"logit", "probit"} and isinstance(marginal_table, pd.DataFrame) and not marginal_table.empty:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.MARGINAL_EFFECTS_AVAILABLE,
                severity=DiagnosticSeverity.INFO,
                title="Marginal effects available",
                message="Average marginal effects are available as a structured advanced output.",
                recommendation="Use marginal effects for probability-scale interpretation while avoiding causal language.",
                show_in_ui=False,
                show_in_report=True,
                llm_instruction="Prefer marginal effects for probability-scale binary model interpretation when available.",
            )
        )
    failure = advanced_outputs.get("marginal_effects_error")
    if model_id in {"logit", "probit"} and failure:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.MARGINAL_EFFECTS_FAILED,
                severity=DiagnosticSeverity.WARNING,
                title="Marginal effects unavailable",
                message=f"Marginal effects could not be computed: {failure}",
                recommendation="Interpret coefficients on their model scale and review model stability diagnostics.",
                llm_instruction="Explain that marginal effects are unavailable and fall back to coefficient-scale interpretation.",
            )
        )
    return diagnostics


def diagnose_separation_risk(df: pd.DataFrame, y_col: str, predictors: list[str]) -> list[ModelDiagnostic]:
    if not y_col or y_col not in df.columns or not predictors:
        return []
    y = df[y_col].dropna()
    if y.nunique(dropna=True) != 2:
        return []
    risk_variables: list[str] = []
    working = df[[y_col] + [col for col in predictors if col in df.columns]].dropna()
    for variable in predictors:
        if variable not in working.columns:
            continue
        grouped = working.groupby(variable, dropna=True)[y_col].nunique(dropna=True)
        if not grouped.empty and (grouped <= 1).all() and working[variable].nunique(dropna=True) <= max(20, len(working) // 2):
            risk_variables.append(variable)
    if not risk_variables:
        return []
    return [
        ModelDiagnostic(
            code=DiagnosticCode.PERFECT_SEPARATION_RISK,
            severity=DiagnosticSeverity.WARNING,
            title="Possible perfect or quasi-complete separation",
            message="Some predictors may strongly separate the binary outcome: " + ", ".join(risk_variables) + ".",
            affected_variables=risk_variables,
            recommendation="Review cross-tabs and consider a simpler binary model if estimation is unstable.",
            llm_instruction="Warn that Logit/Probit coefficients may be unstable or non-estimable under separation.",
        )
    ]


def diagnose_panel_structure(structure: dict[str, Any] | None) -> list[ModelDiagnostic]:
    if not structure:
        return []
    diagnostics: list[ModelDiagnostic] = []
    observations = int(structure.get("observations") or 0)
    entities = int(structure.get("entities") or 0)
    periods = int(structure.get("time_periods") or 0)
    if observations and entities and periods:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.PANEL_STRUCTURE,
                severity=DiagnosticSeverity.INFO,
                title="Panel structure summary",
                message=f"Panel sample includes {observations} observations, {entities} entities, and {periods} time period(s).",
                recommendation="Use this summary to confirm the entity-time structure before interpreting Panel FE results.",
                show_in_ui=False,
                show_in_report=True,
                llm_instruction="Use the entity and time-period counts when summarizing Panel FE sample structure.",
            )
        )
    if structure.get("entities", 0) <= 1 or structure.get("time_periods", 0) <= 1:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.PANEL_STRUCTURE,
                severity=DiagnosticSeverity.ERROR,
                title="Insufficient panel structure",
                message="Panel Fixed Effects requires more than one entity and more than one time period.",
                recommendation="Check the entity ID and time variable.",
                llm_instruction="Treat this as a blocking panel-data issue if the model was not estimated.",
            )
        )
    if structure.get("duplicate_entity_time_rows", 0) > 0:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.PANEL_STRUCTURE,
                severity=DiagnosticSeverity.ERROR,
                title="Duplicate entity-time observations",
                message="Panel data contains duplicate entity-time rows.",
                recommendation="Ensure each entity-time combination appears once before running Panel FE.",
                llm_instruction="Do not proceed with Panel FE interpretation when duplicate entity-time rows block estimation.",
            )
        )
    if not structure.get("balanced_panel", True):
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.PANEL_STRUCTURE,
                severity=DiagnosticSeverity.INFO,
                title="Unbalanced panel",
                message="Some entities are missing observations in some time periods.",
                recommendation="Interpret Panel FE as an unbalanced-panel estimate.",
                llm_instruction="Mention that the panel is unbalanced when describing the sample.",
            )
        )
    if structure.get("singleton_entities", 0) > 0:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.PANEL_STRUCTURE,
                severity=DiagnosticSeverity.WARNING,
                title="Singleton entities",
                message=f"{structure.get('singleton_entities')} entity/entities have only one observation.",
                recommendation="Check whether singleton entities should remain in the panel sample.",
                llm_instruction="Explain that singleton entities add limited within-entity information.",
            )
        )
    return diagnostics


def diagnose_panel_fe_specification(summary: dict[str, Any] | None) -> list[ModelDiagnostic]:
    if not summary:
        return []
    if str(summary.get("model_type") or "") != "panel_fe":
        return []
    fe_parts = []
    if summary.get("entity_effects"):
        fe_parts.append("entity fixed effects")
    if summary.get("time_effects"):
        fe_parts.append("time fixed effects")
    fe_text = " and ".join(fe_parts) if fe_parts else "no fixed effects"
    se = str(summary.get("standard_errors") or "cluster_entity")
    diagnostics = [
        ModelDiagnostic(
            code=DiagnosticCode.PANEL_FE_SPECIFICATION,
            severity=DiagnosticSeverity.INFO,
            title="Panel FE specification",
            message=f"Panel FE uses {fe_text}; standard errors: {se}.",
            affected_variables=[
                str(summary.get("entity_id") or ""),
                str(summary.get("time_id") or ""),
            ],
            recommendation="Interpret coefficients according to the selected fixed effects and standard error setting.",
            show_in_ui=False,
            show_in_report=True,
            llm_instruction="Mention fixed effects and standard error setting when explaining Panel FE results.",
        )
    ]
    if se == "cluster_entity":
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.STANDARD_ERROR_NOTE,
                severity=DiagnosticSeverity.INFO,
                title="Clustered standard errors",
                message="Panel FE standard errors are clustered by entity.",
                affected_variables=[str(summary.get("entity_id") or "")],
                recommendation="Interpret p-values using the entity-clustered standard errors.",
                show_in_ui=False,
                show_in_report=True,
                llm_instruction="State that uncertainty is computed with entity-clustered standard errors.",
            )
        )
    return diagnostics


def diagnose_within_variation(variation: dict[str, Any] | None, entity_effects: bool, time_effects: bool) -> list[ModelDiagnostic]:
    if not variation:
        return []
    diagnostics: list[ModelDiagnostic] = []
    no_within = list(variation.get("no_within_entity_variation") or [])
    low_within = list(variation.get("low_within_entity_variation") or [])
    time_only = list(variation.get("time_only_variation") or [])
    if entity_effects and no_within:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.WITHIN_VARIATION,
                severity=DiagnosticSeverity.WARNING,
                title="No within-entity variation",
                message="These variables may be absorbed by entity fixed effects: " + ", ".join(no_within) + ".",
                affected_variables=no_within,
                recommendation="Do not interpret variables that have no within-entity variation under entity fixed effects.",
                llm_instruction="Explain that FE estimates depend on within-unit variation.",
            )
        )
    if entity_effects and low_within:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.WITHIN_VARIATION,
                severity=DiagnosticSeverity.WARNING,
                title="Limited within-entity variation",
                message="These variables have little within-entity variation: " + ", ".join(low_within) + ".",
                affected_variables=low_within,
                recommendation="Interpret these coefficients cautiously.",
                llm_instruction="Flag limited within variation as an interpretation constraint.",
            )
        )
    if time_effects and time_only:
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.WITHIN_VARIATION,
                severity=DiagnosticSeverity.WARNING,
                title="Time-only variation",
                message="These variables may be absorbed by time fixed effects: " + ", ".join(time_only) + ".",
                affected_variables=time_only,
                recommendation="Review whether these variables can be estimated with time fixed effects.",
                llm_instruction="Explain that variables varying only by time may be absorbed by time FE.",
            )
        )
    return diagnostics


def diagnose_interpretation_constraints(model_id: str, summary: dict[str, Any], spec_standard_errors: str = "") -> list[ModelDiagnostic]:
    diagnostics = [
        ModelDiagnostic(
            code=DiagnosticCode.ASSOCIATION_NOT_CAUSATION,
            severity=DiagnosticSeverity.CONSTRAINT,
            title="Association, not automatic causation",
            message="Regression results describe statistical associations and do not automatically establish causal effects.",
            recommendation="Avoid causal language unless the research design supports causal identification.",
            show_in_ui=False,
            show_in_report=True,
            llm_instruction="Never present a coefficient as causal proof.",
        )
    ]
    if model_id == "panel_fe":
        diagnostics.append(
            ModelDiagnostic(
                code=DiagnosticCode.FIXED_EFFECT_INTERPRETATION,
                severity=DiagnosticSeverity.CONSTRAINT,
                title="Fixed effects interpretation",
                message="Panel FE coefficients should be interpreted as within-entity relationships over time.",
                recommendation="Emphasize within-entity variation rather than cross-sectional differences.",
                show_in_ui=True,
                show_in_report=True,
                llm_instruction="Describe Panel FE results as within the same entity over time.",
            )
        )
    se = str(summary.get("standard_errors") or spec_standard_errors or ("hc3" if summary.get("robust_standard_errors") else "conventional"))
    diagnostics.append(
        ModelDiagnostic(
            code=DiagnosticCode.STANDARD_ERROR_NOTE,
            severity=DiagnosticSeverity.INFO,
            title="Standard error setting",
            message=_standard_error_message(se),
            recommendation="Interpret statistical significance using the reported standard error setting.",
            show_in_ui=False,
            show_in_report=True,
            llm_instruction="Mention the standard error setting when explaining p-values and uncertainty.",
        )
    )
    return diagnostics


def diagnostics_from_legacy_warnings(warnings: list[str]) -> list[ModelDiagnostic]:
    diagnostics: list[ModelDiagnostic] = []
    for warning in warnings:
        text = str(warning)
        lower = text.lower()
        if "high multicollinearity" in lower:
            code = DiagnosticCode.HIGH_MULTICOLLINEARITY
            title = "High multicollinearity"
        elif "serious multicollinearity" in lower:
            code = DiagnosticCode.SERIOUS_MULTICOLLINEARITY
            title = "Serious multicollinearity"
        elif "small sample" in lower or "sample size is below" in lower:
            code = DiagnosticCode.SMALL_SAMPLE
            title = "Small estimation sample"
        elif "perfect separation" in lower or "separation" in lower:
            code = DiagnosticCode.PERFECT_SEPARATION_RISK
            title = "Possible separation"
        elif "missing values" in lower:
            code = DiagnosticCode.MISSING_DATA_DROPPED
            title = "Missing data dropped"
        elif "marginal effects could not be computed" in lower:
            code = DiagnosticCode.MARGINAL_EFFECTS_FAILED
            title = "Marginal effects unavailable"
        elif "model-wide f test could not be computed" in lower:
            code = DiagnosticCode.MODEL_WIDE_TEST_UNAVAILABLE
            title = "Model-wide F test unavailable"
        else:
            code = DiagnosticCode.LEGACY_WARNING
            title = "Model diagnostic warning"
        diagnostics.append(
            ModelDiagnostic(
                code=code,
                severity=DiagnosticSeverity.WARNING,
                title=title,
                message=text,
                recommendation="Review this diagnostic before interpreting the model.",
                llm_instruction="Include this warning when summarizing model limitations.",
            )
        )
    return diagnostics


def diagnostic_messages(
    diagnostics: list[ModelDiagnostic],
    severities: set[str] | None = None,
    ui_only: bool = False,
    report_only: bool = False,
) -> list[str]:
    messages: list[str] = []
    for diagnostic in diagnostics:
        if severities is not None and diagnostic.severity not in severities:
            continue
        if ui_only and not diagnostic.show_in_ui:
            continue
        if report_only and not diagnostic.show_in_report:
            continue
        messages.append(diagnostic.message)
    return list(dict.fromkeys(messages))


def dedupe_diagnostics(diagnostics: list[ModelDiagnostic]) -> list[ModelDiagnostic]:
    deduped: list[ModelDiagnostic] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for diagnostic in diagnostics:
        key = (diagnostic.code, diagnostic.message, tuple(diagnostic.affected_variables))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(diagnostic)
    return deduped


def _standard_error_message(standard_errors: str) -> str:
    se = standard_errors.lower()
    if se in {"2sls_unadjusted", "iv2sls_unadjusted"}:
        return "IV/2SLS standard errors use an unadjusted 2SLS covariance estimator."
    if se in {"cluster_entity", "clustered"}:
        return "Standard errors are clustered by entity."
    if se in {"robust", "hc3"}:
        return "Robust standard errors are used."
    return "Conventional standard errors are used."
