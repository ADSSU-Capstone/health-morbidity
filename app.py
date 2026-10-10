from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1. EMOJI / INVISIBLE CHARACTER PATTERNS
# ------------------------------------------------------------------
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"   # regional indicator (flags)
    "\U0001F300-\U0001F5FF"   # symbols and pictographs
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F680-\U0001F6FF"   # transport and map symbols
    "\U0001F700-\U0001F77F"   # alchemical
    "\U0001F780-\U0001F7FF"   # geometric extended
    "\U0001F800-\U0001F8FF"   # supplemental arrows
    "\U0001F900-\U0001F9FF"   # supplemental symbols
    "\U0001FA00-\U0001FAFF"   # extended-A
    "\U00002600-\U000027BF"   # misc symbols / dingbats
    "\U00002B00-\U00002BFF"   # arrows
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0000200D"              # zero-width joiner
    "\U000020E3"              # combining keycap
    "]+",
    flags=re.UNICODE,
)

_INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200e\u200f\u202a-\u202e\ufeff]")

_WHITESPACE_RE = re.compile(r"\s+")

# ------------------------------------------------------------------
# 2. NON-DISEASE LABELS TO DROP
# ------------------------------------------------------------------
_NON_DISEASE_LABELS = {
    "",
    "OTHER DISEASES",
    "SECTION A.1 MORBIDITY",
    "SECTION A MORBIDITY",
    "PREPARED BY:",
    "PREPARED BY",
    "APPROVED BY:",
    "APPROVED BY",
    "PHN",
    "MHO",
    "RN",
    "MD",
    "ICD-10 CODE",
    "DISEASE",
}

_SECTION_LETTER_RE = re.compile(r"^[A-Z]$")

# ------------------------------------------------------------------
# 3. CANONICAL DISEASE NAME MAP (typos, abbreviations, variants)
# ------------------------------------------------------------------
DISEASE_CANONICAL_MAP: Dict[str, str] = {
    "ABCESS": "ABSCESS",
    "DENTAL ABCESS": "DENTAL ABSCESS",
    "RHINOMEA": "RHINORRHEA",
    "CONJUCTIVITIS": "CONJUNCTIVITIS",
    "TONSILITIS": "TONSILLITIS",
    "INFECTED UMBILLICUS": "INFECTED UMBILICUS",
    "BENIGN PROSTATIC HYPERTHROPHY": "BENIGN PROSTATIC HYPERTROPHY",
    "VARICELLA ZOOSTER": "VARICELLA ZOSTER",
    "BELLS PALSY": "BELL'S PALSY",
    "BELL'S PALSY": "BELL'S PALSY",
    "IMPETIGO CONTIGEOSA": "IMPETIGO",
    "URTI": "UPPER RESPIRATORY TRACT INFECTION",
    "UTI": "URINARY TRACT INFECTION",
    "ACUTE RESPIRATORY INFECTION/ DISEASE": "ACUTE RESPIRATORY INFECTION / DISEASE",
    "PEDIA COMMUNITY ACQUIRED PNEUMONIA": "PEDIATRIC COMMUNITY ACQUIRED PNEUMONIA",
    "HAND, FOOT AND MOUTH DISEASE": "HAND, FOOT AND MOUTH DISEASE",
    "SCROTAL HERNIA": "INGUINAL / SCROTAL HERNIA",
    "URIC ACID": "HYPERURICEMIA",
    "FATTY LIVER": "FATTY LIVER DISEASE",
}

# ------------------------------------------------------------------
# 4. AGE-GROUP GEOMETRY (0-indexed columns C..AL)
# ------------------------------------------------------------------
AGE_GROUPS: List[Tuple[str, int, int]] = [
    ("0-6 days",            2,  3),
    ("7-28 days",           4,  5),
    ("29 days-11 mos",      6,  7),
    ("1-4 y/o",             8,  9),
    ("5-9 y/o",            10, 11),
    ("10-14 y/o",          12, 13),
    ("15-19 y/o",          14, 15),
    ("20-24 y/o",          16, 17),
    ("25-29 y/o",          18, 19),
    ("30-34 y/o",          20, 21),
    ("35-39 y/o",          22, 23),
    ("40-44 y/o",          24, 25),
    ("45-49 y/o",          26, 27),
    ("50-54 y/o",          28, 29),
    ("55-59 y/o",          30, 31),
    ("60-64 y/o",          32, 33),
    ("65-69 y/o",          34, 35),
    ("70 y/o and above",   36, 37),
]

TOTAL_M_COL, TOTAL_F_COL, TOTAL_BOTH_COL = 38, 39, 40
MAX_COL = 41  # columns A .. AO

AGE_ORDER = [a[0] for a in AGE_GROUPS]

# ------------------------------------------------------------------
# 5. SHEET METADATA
# ------------------------------------------------------------------
SHEET_META: Dict[str, Tuple[str, int]] = {
    "ANNUAL TOTAL": ("Annual", 0),
    "JANUARY":      ("Monthly", 1),
    "FEBRUARY":     ("Monthly", 2),
    "MARCH":        ("Monthly", 3),
    "APRIL":        ("Monthly", 4),
    "MAY":          ("Monthly", 5),
    "JUNE":         ("Monthly", 6),
    "JULY":         ("Monthly", 7),
    "AUGUST":       ("Monthly", 8),
    "SEPTEMBER":    ("Monthly", 9),
    "OCTOBER":      ("Monthly", 10),
    "NOVEMBER":     ("Monthly", 11),
    "DECEMBER":     ("Monthly", 12),
    "1ST QUARTER":  ("Quarterly", 1),
    "2ND QUARTER":  ("Quarterly", 2),
    "3RD QUARTER":  ("Quarterly", 3),
    "4TH QUARTER":  ("Quarterly", 4),
}

PERIOD_ORDER = {"Monthly": 0, "Quarterly": 1, "Annual": 2}


# ==================================================================
# TEXT UTILITIES
# ==================================================================
def strip_emojis(value: Any) -> Any:
    """Remove every emoji / pictograph from a string."""
    if not isinstance(value, str):
        return value
    return _EMOJI_RE.sub("", value)


def clean_text(value: Any, *, upper: bool = False, default: str = "") -> str:
    """Full text sanitiser: emoji-free, whitespace-normalised, NFKC."""
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default

    text = str(value)
    text = strip_emojis(text)
    text = _INVISIBLE_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = text.strip(" ,;:")

    if upper:
        text = text.upper()
    return text


def to_number(value: Any) -> Tuple[float, bool]:
    """
    Coerce a cell to float.
    Returns (value, was_coerced) where was_coerced is True when the
    original content was non-numeric and had to be replaced by 0.
    """
    if value is None:
        return 0.0, False
    if isinstance(value, (int, np.integer)):
        return float(value), False
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return 0.0, False
        return float(value), False

    text = clean_text(value)
    if text == "":
        return 0.0, False
    if text.upper() in {"-", "--", "N/A", "NA", "NONE", "NULL", "NIL"}:
        return 0.0, False

    cleaned = text.replace(",", "").replace(" ", "")
    try:
        return float(cleaned), False
    except ValueError:
        return 0.0, True


def canonical_disease(name: str) -> str:
    """Map a cleaned disease label to its canonical form."""
    return DISEASE_CANONICAL_MAP.get(name, name)


# ==================================================================
# RESULT CONTAINER
# ==================================================================
@dataclass
class CleanResult:
    long_df: pd.DataFrame
    wide_df: pd.DataFrame
    quality_log: pd.DataFrame
    sheet_summary: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def quality_score(self) -> float:
        """Percentage of clean rows (no critical issue)."""
        if self.quality_log.empty:
            return 100.0
        critical = self.quality_log[
            self.quality_log["Severity"].isin(["High", "Medium"])
        ]
        if self.long_df.empty:
            return 0.0
        penalty = min(len(critical) / max(len(self.long_df), 1) * 100, 100.0)
        return round(100.0 - penalty, 1)


# ==================================================================
# STEP 1 - READ RAW WORKBOOK
# ==================================================================
def read_workbook(source: Any) -> Dict[str, pd.DataFrame]:
    """Read every sheet with no header so the grid stays intact."""
    sheets = pd.read_excel(source, sheet_name=None, header=None)
    cleaned: Dict[str, pd.DataFrame] = {}
    for name, df in sheets.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        if df.shape[1] < MAX_COL:
            for c in range(df.shape[1], MAX_COL):
                df[c] = np.nan
        df = df.iloc[:, :MAX_COL]
        df.columns = range(MAX_COL)
        df = df.reset_index(drop=True)
        cleaned[clean_text(name, upper=True)] = df
    return cleaned


# ==================================================================
# STEP 2 - DETECT HEADER / DATA BOUNDARIES
# ==================================================================
def _find_header_row(df: pd.DataFrame) -> Optional[int]:
    for i in range(min(30, len(df))):
        if clean_text(df.iat[i, 0], upper=True) == "DISEASE":
            return i
    return None


def _find_total_row(df: pd.DataFrame, start: int) -> int:
    for i in range(start, len(df)):
        if clean_text(df.iat[i, 0], upper=True) == "TOTAL":
            return i
    return len(df)


def _extract_year(df: pd.DataFrame) -> Optional[int]:
    """Pull the reporting year out of the header block."""
    for i in range(min(10, len(df))):
        for c in range(min(6, df.shape[1])):
            text = clean_text(df.iat[i, c], upper=True)
            match = re.search(r"\b(20\d{2})\b", text)
            if match:
                return int(match.group(1))
    return None


def _is_signature_row(name: str) -> bool:
    if name in _NON_DISEASE_LABELS:
        return True
    if _SECTION_LETTER_RE.match(name):
        return True
    if re.search(r",\s*(RN|MD|RMT|RM|RPh)\b", name):
        return True
    if "BUNGCAD" in name or "MACALAGAY" in name:
        return True
    return False


# ==================================================================
# STEP 3 - CLEAN ONE SHEET
# ==================================================================
def clean_sheet(
    df: pd.DataFrame,
    sheet_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return (records, issues) for a single sheet.
    Each record = one disease x age group x sex row.
    """
    records: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []

    period_type, period_order = SHEET_META.get(sheet_name, ("Other", 99))
    reported_year = _extract_year(df)

    header_row = _find_header_row(df)
    if header_row is None:
        issues.append({
            "Sheet": sheet_name, "Row": 1, "Issue Type": "SHEET SKIPPED",
            "Detail": "No 'Disease' header row found.", "Severity": "High",
        })
        return records, issues

    total_row = _find_total_row(df, header_row + 1)
    current_section = "GENERAL"

    # scratch buffer for duplicate disease rows inside this sheet
    buffer: Dict[str, Dict[str, Any]] = {}
    duplicate_names: set[str] = set()

    for i in range(header_row + 1, total_row):
        raw_name = df.iat[i, 0]
        raw_icd = df.iat[i, 1]

        original_name_text = "" if raw_name is None else str(raw_name)
        name = clean_text(raw_name, upper=True)

        # ---- emoji / whitespace audit -----------------------------
        if strip_emojis(original_name_text) != original_name_text:
            issues.append({
                "Sheet": sheet_name, "Row": i + 1,
                "Issue Type": "EMOJI REMOVED",
                "Detail": f"Emoji stripped from '{original_name_text}'.",
                "Severity": "Low",
            })
        if original_name_text.strip() != original_name_text and original_name_text.strip():
            issues.append({
                "Sheet": sheet_name, "Row": i + 1,
                "Issue Type": "WHITESPACE TRIMMED",
                "Detail": f"'{original_name_text}' -> '{original_name_text.strip()}'.",
                "Severity": "Low",
            })

        # ---- numeric row -----------------------------------------
        values: List[float] = []
        coerced_cols: List[str] = []
        for c in range(2, MAX_COL):
            val, coerced = to_number(df.iat[i, c])
            values.append(val)
            if coerced:
                coerced_cols.append(str(c))

        if coerced_cols:
            issues.append({
                "Sheet": sheet_name, "Row": i + 1,
                "Issue Type": "NON-NUMERIC VALUE",
                "Detail": f"Non-numeric entry in column index {', '.join(coerced_cols)} "
                          f"treated as 0 for '{name or 'unlabelled'}'.",
                "Severity": "Medium",
            })

        row_sum = sum(values)
        name_is_section = bool(_SECTION_LETTER_RE.match(name))

        # ---- skip section letters --------------------------------
        if name_is_section and row_sum == 0:
            current_section = name
            continue

        # ---- skip signature / non-disease rows -------------------
        if _is_signature_row(name) and row_sum == 0:
            continue
        if name in {"", "OTHER DISEASES", "SECTION A.1 MORBIDITY"} and row_sum == 0:
            continue

        # ---- unlabelled data row ---------------------------------
        if name == "" and row_sum > 0:
            name = f"UNLABELLED ROW {i + 1}"
            issues.append({
                "Sheet": sheet_name, "Row": i + 1,
                "Issue Type": "UNLABELLED ROW",
                "Detail": f"Cases recorded without a disease label (row total {row_sum:.0f}).",
                "Severity": "High",
            })

        if name == "" and row_sum == 0:
            continue

        # ---- canonical name --------------------------------------
        canonical = canonical_disease(name)
        if canonical != name:
            issues.append({
                "Sheet": sheet_name, "Row": i + 1,
                "Issue Type": "NAME NORMALISED",
                "Detail": f"'{name}' -> '{canonical}'.",
                "Severity": "Low",
            })
        name = canonical

        icd = clean_text(raw_icd, upper=True)

        # ---- duplicate inside sheet ------------------------------
        if name in buffer:
            duplicate_names.add(name)
            prev = buffer[name]
            for k in range(len(prev["values"])):
                prev["values"][k] += values[k]
            continue

        buffer[name] = {
            "row": i + 1,
            "icd": icd,
            "section": current_section,
            "values": values,
        }

    for dup in sorted(duplicate_names):
        issues.append({
            "Sheet": sheet_name, "Row": buffer[dup]["row"],
            "Issue Type": "DUPLICATE DISEASE ROW",
            "Detail": f"'{dup}' appears more than once; rows summed.",
            "Severity": "Medium",
        })

    # ---- explode into long format --------------------------------
    for name, payload in buffer.items():
        values = payload["values"]
        male_total_reported = values[TOTAL_M_COL - 2]
        female_total_reported = values[TOTAL_F_COL - 2]

        male_sum = 0.0
        female_sum = 0.0

        for age_label, m_col, f_col in AGE_GROUPS:
            male = values[m_col - 2]
            female = values[f_col - 2]
            male_sum += male
            female_sum += female

            for sex, cases in (("Male", male), ("Female", female)):
                records.append({
                    "Sheet": sheet_name,
                    "Period Type": period_type,
                    "Period Order": period_order,
                    "Period": sheet_name.title(),
                    "Reported Year": reported_year,
                    "Section": payload["section"],
                    "Disease": name,
                    "ICD Code": payload["icd"],
                    "Age Group": age_label,
                    "Sex": sex,
                    "Cases": cases,
                    "Source Row": payload["row"],
                })

        # ---- reconciliation -------------------------------------
        if male_total_reported and abs(male_total_reported - male_sum) > 0.5:
            issues.append({
                "Sheet": sheet_name, "Row": payload["row"],
                "Issue Type": "TOTAL MISMATCH",
                "Detail": f"'{name}' male column total {male_total_reported:.0f} "
                          f"vs age-group sum {male_sum:.0f}.",
                "Severity": "High",
            })
        if female_total_reported and abs(female_total_reported - female_sum) > 0.5:
            issues.append({
                "Sheet": sheet_name, "Row": payload["row"],
                "Issue Type": "TOTAL MISMATCH",
                "Detail": f"'{name}' female column total {female_total_reported:.0f} "
                          f"vs age-group sum {female_sum:.0f}.",
                "Severity": "High",
            })

    # ---- year reconciliation --------------------------------------
    expected_year = 2026
    if reported_year and reported_year != expected_year:
        issues.append({
            "Sheet": sheet_name, "Row": 1,
            "Issue Type": "YEAR MISMATCH",
            "Detail": f"Sheet reports year {reported_year}; file title states {expected_year}.",
            "Severity": "Medium",
        })

    return records, issues


# ==================================================================
# STEP 4 - BUILD THE FULL DATASET
# ==================================================================
def build_dataset(source: Any) -> CleanResult:
    raw_sheets = read_workbook(source)

    all_records: List[Dict[str, Any]] = []
    all_issues: List[Dict[str, Any]] = []
    sheet_rows: List[Dict[str, Any]] = []

    for sheet_name, df in raw_sheets.items():
        if sheet_name not in SHEET_META:
            all_issues.append({
                "Sheet": sheet_name, "Row": 1,
                "Issue Type": "SHEET NOT RECOGNISED",
                "Detail": "Sheet ignored - not part of the FHSIS morbidity set.",
                "Severity": "Low",
            })
            continue

        records, issues = clean_sheet(df, sheet_name)
        all_records.extend(records)
        all_issues.extend(issues)

        sheet_rows.append({
            "Sheet": sheet_name,
            "Period Type": SHEET_META[sheet_name][0],
            "Diseases": len({r["Disease"] for r in records}),
            "Records": len(records),
            "Cases": sum(r["Cases"] for r in records),
            "Issues": len(issues),
        })

    if not all_records:
        empty = pd.DataFrame(
            columns=["Sheet", "Disease", "Age Group", "Sex", "Cases"]
        )
        return CleanResult(empty, empty, pd.DataFrame(all_issues), pd.DataFrame(sheet_rows))

    long_df = pd.DataFrame(all_records)

    # final safety sweep: emoji-free on every object column
    for col in long_df.select_dtypes(include=["object"]).columns:
        long_df[col] = long_df[col].map(lambda v: strip_emojis(v) if isinstance(v, str) else v)

    # numeric hygiene
    long_df["Cases"] = pd.to_numeric(long_df["Cases"], errors="coerce").fillna(0.0)
    long_df["Cases"] = long_df["Cases"].round(0).astype(int)
    long_df.loc[long_df["Cases"] < 0, "Cases"] = 0

    # ordering helpers
    long_df["Age Group"] = pd.Categorical(
        long_df["Age Group"], categories=AGE_ORDER, ordered=True
    )
    long_df["Period Type"] = pd.Categorical(
        long_df["Period Type"], categories=["Monthly", "Quarterly", "Annual"], ordered=True
    )

    # wide pivot for tabular views
    wide_df = (
        long_df
        .groupby(
            ["Period Type", "Period Order", "Period", "Section", "Disease", "Sex"],
            observed=True,
            as_index=False,
        )["Cases"]
        .sum()
    )

    quality_log = pd.DataFrame(all_issues)
    if quality_log.empty:
        quality_log = pd.DataFrame(
            columns=["Sheet", "Row", "Issue Type", "Detail", "Severity"]
        )
    else:
        quality_log = quality_log.drop_duplicates().reset_index(drop=True)
        for col in quality_log.select_dtypes(include=["object"]).columns:
            quality_log[col] = quality_log[col].map(
                lambda v: strip_emojis(v) if isinstance(v, str) else v
            )

    sheet_summary = pd.DataFrame(sheet_rows).sort_values("Sheet").reset_index(drop=True)

    meta = {
        "sheets_loaded": sorted(raw_sheets.keys()),
        "total_cases": int(long_df["Cases"].sum()),
        "unique_diseases": int(long_df["Disease"].nunique()),
        "period_types": sorted(long_df["Period Type"].dropna().unique().tolist()),
    }

    return CleanResult(long_df, wide_df, quality_log, sheet_summary, meta)
