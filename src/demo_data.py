from __future__ import annotations

from dataclasses import dataclass, asdict
from io import BytesIO

import pandas as pd


@dataclass(frozen=True)
class DemoDataset:
    dataset_id: str
    model_id: str
    display_name_key: str
    description_key: str
    filename: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


DEMO_DATASETS: tuple[DemoDataset, ...] = (
    DemoDataset("sample_ols", "ols", "demo_dataset_ols_name", "demo_dataset_ols_description", "sample_ols.csv"),
    DemoDataset("sample_logit", "logit", "demo_dataset_logit_name", "demo_dataset_logit_description", "sample_logit.csv"),
    DemoDataset("sample_panel_fe", "panel_fe", "demo_dataset_panel_fe_name", "demo_dataset_panel_fe_description", "sample_panel_fe.csv"),
    DemoDataset("sample_did", "did", "demo_dataset_did_name", "demo_dataset_did_description", "sample_did.csv"),
    DemoDataset("sample_iv_2sls", "iv_2sls", "demo_dataset_iv_name", "demo_dataset_iv_description", "sample_iv_2sls.csv"),
    DemoDataset("sample_psm", "psm", "demo_dataset_psm_name", "demo_dataset_psm_description", "sample_psm.csv"),
)


def list_demo_datasets() -> list[DemoDataset]:
    return list(DEMO_DATASETS)


def get_demo_dataset(dataset_id: str) -> DemoDataset:
    for dataset in DEMO_DATASETS:
        if dataset.dataset_id == dataset_id:
            return dataset
    raise ValueError(f"Unknown demo dataset: {dataset_id}")


def load_demo_dataset(dataset_id: str) -> pd.DataFrame:
    builders = {
        "sample_ols": _ols_sample,
        "sample_logit": _logit_sample,
        "sample_panel_fe": _panel_fe_sample,
        "sample_did": _did_sample,
        "sample_iv_2sls": _iv_sample,
        "sample_psm": _psm_sample,
    }
    if dataset_id not in builders:
        raise ValueError(f"Unknown demo dataset: {dataset_id}")
    return builders[dataset_id]().copy(deep=True)


def demo_dataset_to_csv_bytes(dataset_id: str) -> bytes:
    return load_demo_dataset(dataset_id).to_csv(index=False).encode("utf-8")


def demo_dataset_upload_cache(dataset_id: str) -> dict[str, object]:
    dataset = get_demo_dataset(dataset_id)
    data = demo_dataset_to_csv_bytes(dataset_id)
    return {
        "name": dataset.filename,
        "type": "text/csv",
        "data": data,
        "source_metadata": {
            "kind": "demo",
            "dataset_id": dataset.dataset_id,
            "model_id": dataset.model_id,
            "filename": dataset.filename,
            "synthetic": True,
        },
    }


def demo_dataset_file(dataset_id: str) -> BytesIO:
    cache = demo_dataset_upload_cache(dataset_id)
    restored = BytesIO(cache["data"])
    restored.name = str(cache["name"])
    restored.type = str(cache["type"])
    restored.size = len(cache["data"])
    return restored


def _ols_sample() -> pd.DataFrame:
    rows = []
    industries = ["manufacturing", "services", "retail"]
    regions = ["north", "east", "south", "west"]
    for index in range(18):
        digital_index = round(0.2 + index * 0.18, 2)
        firm_age = 3 + index % 6
        training_hours = 8 + (index % 5) * 2
        revenue_growth = round(1.4 + 0.72 * digital_index + 0.04 * training_hours - 0.03 * firm_age + (index % 3) * 0.05, 3)
        rows.append(
            {
                "revenue_growth": revenue_growth,
                "digital_index": digital_index,
                "firm_age": firm_age,
                "training_hours": training_hours,
                "industry": industries[index % len(industries)],
                "region": regions[index % len(regions)],
            }
        )
    return pd.DataFrame(rows)


def _logit_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "adopted": [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
            "digital_index": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
            "leverage": [0.62, 0.58, 0.55, 0.60, 0.52, 0.48, 0.50, 0.45, 0.47, 0.42, 0.44, 0.40, 0.38, 0.43, 0.36, 0.34, 0.41, 0.32, 0.30, 0.39, 0.31, 0.37, 0.29, 0.33],
            "support_score": [1.0, 1.2, 1.1, 1.6, 1.4, 1.8, 1.5, 2.0, 1.7, 2.1, 1.9, 2.3, 2.4, 2.0, 2.6, 2.7, 2.2, 2.8, 3.0, 2.5, 3.1, 2.9, 3.2, 3.4],
            "industry": ["manufacturing", "manufacturing", "services", "services", "retail", "retail"] * 4,
        }
    )


def _panel_fe_sample() -> pd.DataFrame:
    rows = []
    for firm in range(6):
        for offset, year in enumerate(range(2020, 2024)):
            digital_index = round(0.4 + 0.2 * offset + 0.07 * firm, 2)
            investment = round(2.0 + 0.3 * offset + 0.1 * firm, 2)
            productivity = round(5.0 + 0.5 * digital_index + 0.18 * investment + firm * 0.2 + offset * 0.05, 3)
            rows.append(
                {
                    "firm_id": f"F{firm + 1}",
                    "year": year,
                    "productivity": productivity,
                    "digital_index": digital_index,
                    "investment": investment,
                    "sector": "A" if firm % 2 == 0 else "B",
                }
            )
    return pd.DataFrame(rows)


def _did_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "firm_id": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "year": [2020, 2021, 2020, 2021, 2020, 2021, 2020, 2021],
            "outcome": [10, 11, 9, 10, 12, 17, 11, 16],
            "treatment": [0, 0, 0, 0, 1, 1, 1, 1],
            "post": [0, 1, 0, 1, 0, 1, 0, 1],
            "control": [5, 6, 4, 5, 6, 7, 5, 6],
        }
    )


def _iv_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outcome": [9.10, 9.72, 10.34, 10.96, 11.58, 12.20, 16.20, 16.82, 17.44, 18.06, 18.68, 19.30],
            "endogenous": [2.05, 2.28, 2.51, 2.74, 2.97, 3.20, 4.55, 4.78, 5.01, 5.24, 5.47, 5.70],
            "instrument": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "control": [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        }
    )


def _psm_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outcome": [
                5.10,
                5.55,
                5.95,
                6.30,
                6.65,
                7.00,
                7.35,
                7.70,
                5.45,
                6.55,
                7.20,
                7.95,
                7.25,
                7.70,
                8.05,
                8.40,
                8.75,
                9.10,
                9.45,
                9.80,
                7.90,
                8.55,
                9.20,
                9.90,
            ],
            "treatment": [0] * 12 + [1] * 12,
            "covariate_age": [34, 37, 40, 43, 46, 49, 52, 55, 36, 45, 51, 57, 35, 38, 41, 44, 47, 50, 53, 56, 39, 42, 48, 54],
            "covariate_size": [1.0, 1.4, 1.5, 2.0, 2.1, 2.7, 2.6, 3.2, 1.1, 2.3, 2.5, 3.4, 1.2, 1.3, 1.8, 1.9, 2.4, 2.5, 3.0, 3.1, 1.6, 1.7, 2.6, 2.9],
            "covariate_margin": [0.21, 0.25, 0.28, 0.29, 0.35, 0.34, 0.41, 0.40, 0.24, 0.31, 0.39, 0.43, 0.23, 0.26, 0.27, 0.32, 0.33, 0.38, 0.39, 0.45, 0.25, 0.30, 0.36, 0.42],
        }
    )
