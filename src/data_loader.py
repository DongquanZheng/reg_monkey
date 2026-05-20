from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src.file_errors import FriendlyFileError, validate_loaded_dataframe


SUPPORTED_EXCEL_SUFFIXES = {".xlsx", ".xls"}


def _get_file_name(uploaded_file: BinaryIO) -> str:
    return getattr(uploaded_file, "name", "") or ""


def load_dataset(uploaded_file: BinaryIO) -> pd.DataFrame:
    """Load a CSV or Excel upload into a DataFrame.

    Streamlit upload objects behave like binary files and expose a ``name``
    attribute. This function also accepts regular file-like objects in tests.
    """
    if uploaded_file is None:
        raise FriendlyFileError("empty_file")

    file_name = _get_file_name(uploaded_file)
    suffix = Path(file_name).suffix.lower()

    try:
        if suffix == ".csv":
            for encoding in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding, dtype=str, keep_default_na=False, on_bad_lines="error")
                    validate_loaded_dataframe(df)
                    return df
                except UnicodeDecodeError:
                    continue
                except pd.errors.ParserError as exc:
                    raise FriendlyFileError("malformed_csv") from exc
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False, on_bad_lines="error")
            except UnicodeDecodeError as exc:
                raise FriendlyFileError("encoding_error") from exc
            except pd.errors.ParserError as exc:
                raise FriendlyFileError("malformed_csv") from exc
            validate_loaded_dataframe(df)
            return df

        if suffix in SUPPORTED_EXCEL_SUFFIXES:
            uploaded_file.seek(0)
            try:
                df = pd.read_excel(uploaded_file, engine="openpyxl", dtype=str, keep_default_na=False)
            except ValueError as exc:
                raise FriendlyFileError("malformed_excel") from exc
            except Exception as exc:
                raise FriendlyFileError("malformed_excel") from exc
            validate_loaded_dataframe(df)
            return df

        raise FriendlyFileError("unsupported_file_type")
    except pd.errors.EmptyDataError as exc:
        raise FriendlyFileError("empty_file") from exc
    except FriendlyFileError:
        raise
    except Exception as exc:
        raise FriendlyFileError("unknown_file_load_error") from exc
