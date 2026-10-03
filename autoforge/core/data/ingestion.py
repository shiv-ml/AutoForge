"""CSV ingestion with explicit, user-readable failures. Never mutates user data."""
from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from autoforge.config.settings import MAX_UPLOAD_MB, THRESHOLDS
from autoforge.core.errors import DataLoadError

logger = logging.getLogger(__name__)
_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


@dataclass
class LoadResult:
    df: pd.DataFrame
    encoding: str
    delimiter: str
    warnings: list[str] = field(default_factory=list)


def _read_bytes(source: str | Path | bytes | BinaryIO) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataLoadError(f"File not found: {path}")
        return path.read_bytes()
    if hasattr(source, "getvalue"):
        return source.getvalue()
    if hasattr(source, "read"):
        return source.read()
    raise DataLoadError(f"Unsupported data source type: {type(source).__name__}")


def _decode(raw: bytes) -> tuple[str, str]:
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise DataLoadError("The file could not be decoded (tried UTF-8, Windows-1252, Latin-1). "
                        "Is it really a text/CSV file?")


def _sniff_delimiter(text: str) -> str:
    try:
        return csv.Sniffer().sniff(text[:20_000], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def load_csv(source: str | Path | bytes | BinaryIO) -> LoadResult:
    """Load a CSV and return it with diagnostics.

    Raises:
        DataLoadError: for empty, undecodable, malformed or unusable files.
    """
    raw = _read_bytes(source)
    if not raw or not raw.strip():
        raise DataLoadError("The file is empty.")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise DataLoadError(f"The file is larger than the {MAX_UPLOAD_MB} MB limit.")
    if b"\x00" in raw[:4096]:
        raise DataLoadError("The file looks binary (contains NUL bytes), not a CSV text file.")

    text, encoding = _decode(raw)
    delimiter = _sniff_delimiter(text)
    warnings: list[str] = []
    if encoding != "utf-8":
        warnings.append(f"File is not UTF-8; decoded as {encoding}. Check special characters.")

    header = next(csv.reader(io.StringIO(text), delimiter=delimiter), [])
    dupes = sorted({h for h in header if header.count(h) > 1})
    if dupes:
        warnings.append(f"Duplicate column names in header: {dupes}. pandas renamed them (e.g. 'name.1').")

    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError("The file contains no columns/data.") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"Malformed CSV (inconsistent number of fields): {exc}") from exc

    if df.shape[1] < 2:
        raise DataLoadError(
            f"Only {df.shape[1]} column detected using delimiter {delimiter!r}. "
            "A dataset needs at least a feature and a target column.")
    if df.shape[0] == 0:
        raise DataLoadError("The file has a header but no data rows.")
    if df.shape[0] < THRESHOLDS.min_rows:
        warnings.append(f"Only {df.shape[0]} rows; results will be unreliable "
                        f"(recommended >= {THRESHOLDS.min_rows}).")
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        warnings.append(f"Columns without header names detected: {unnamed}.")
    logger.info("Loaded CSV: %s rows x %s cols (encoding=%s, delimiter=%r)",
                df.shape[0], df.shape[1], encoding, delimiter)
    return LoadResult(df=df, encoding=encoding, delimiter=delimiter, warnings=warnings)
