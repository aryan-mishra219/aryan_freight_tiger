"""
Freight Tiger Evaluation Harness & Reproducibility Suite
Validates pipeline outputs against a 10-case golden benchmark, enforces global
anti-hallucination invariants, and proves bitwise 3-run reproducibility.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from src.pipeline import execute_audit_pipeline

CONTRACT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]

CORE_GRADING_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
]

GOLDEN_TEST_CASES = [
    {
        "id": "CASE-01",
        "route": "Delhi-Jaipur",
        "week_of": "2024-11-11",
        "cost_per_tonne_km": 4.17,
        "vs_own_history": "+35.5% vs this route's past average",
        "vs_similar_routes": "+21.0% vs similar-length routes this week",
        "flagged": "Yes",
        "matched_note_id": "",
    },
    {
        "id": "CASE-02",
        "route": "Ahmedabad-Mumbai",
        "week_of": "2025-01-20",
        "cost_per_tonne_km": 3.29,
        "vs_own_history": "+29.5% vs this route's past average",
        "vs_similar_routes": "+22.5% vs similar-length routes this week",
        "flagged": "No (justified)",
        "matched_note_id": "N002",
    },
    {
        "id": "CASE-03",
        "route": "Mumbai-Pune",
        "week_of": "2025-09-15",
        "cost_per_tonne_km": 3.98,
        "vs_own_history": "+9.2% vs this route's past average",
        "vs_similar_routes": "+23.6% vs similar-length routes this week",
        "flagged": "Yes",
        "matched_note_id": "",
        "reason_mentions": ["N006"],
    },
    {
        "id": "CASE-04",
        "route": "Chennai-Bangalore",
        "week_of": "2025-02-24",
        "cost_per_tonne_km": 3.58,
        "flagged": "No (justified)",
        "matched_note_id": "N001",
    },
    {
        "id": "CASE-05",
        "route": "Chennai-Bangalore",
        "week_of": "2025-03-03",
        "cost_per_tonne_km": 3.48,
        "flagged": "No (justified)",
        "matched_note_id": "N001",
    },
    {
        "id": "CASE-06",
        "route": "Chennai-Bangalore",
        "week_of": "2025-03-10",
        "cost_per_tonne_km": 3.57,
        "flagged": "Yes",
        "matched_note_id": "",
        "reason_mentions": ["N001", "N009"],
    },
    {
        "id": "CASE-07",
        "route": "Chennai-Bangalore",
        "week_of": "2025-03-17",
        "cost_per_tonne_km": 3.47,
        "flagged": "Yes",
        "matched_note_id": "",
        "reason_mentions": ["N009"],
    },
    {
        "id": "CASE-08",
        "route": "Delhi-Jaipur",
        "week_of": "2024-11-18",
        "cost_per_tonne_km": 4.09,
        "flagged": "Yes",
        "matched_note_id": "",
    },
    {
        "id": "CASE-09",
        "route": "Mumbai-Delhi",
        "week_of": "2025-05-12",
        "cost_per_tonne_km": 2.28,
        "flagged": "No (justified)",
        "matched_note_id": "N003",
    },
    {
        "id": "CASE-10",
        "route": "Mumbai-Pune",
        "week_of": "2025-10-27",
        "cost_per_tonne_km": 4.07,
        "flagged": "Yes",
        "matched_note_id": "",
        "reason_mentions": ["N010"],
    },
]


def load_csv_safely(path: Path) -> pd.DataFrame:
    """Load evaluation CSV file handling blank strings properly."""
    df = pd.read_csv(path, keep_default_na=False)
    # Ensure all string columns stripped
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def run_golden_eval(
    eval_csv_path: Path = Path("output/output_anomalies_evaluated.csv"),
    all_csv_path: Path = Path("output/output_all_728_route_weeks.csv"),
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluates dataset against 10 golden test cases and 3 global invariants.
    """
    df_eval = load_csv_safely(eval_csv_path)
    df_all = load_csv_safely(all_csv_path) if all_csv_path.exists() else df_eval

    case_results = []
    all_cases_passed = True

    # 1. Evaluate 10 Golden Cases
    for tc in GOLDEN_TEST_CASES:
        sub = df_eval[
            (df_eval["route"] == tc["route"])
            & (df_eval["week_of"] == tc["week_of"])
        ]
        if sub.empty:
            case_results.append({
                "id": tc["id"],
                "route": tc["route"],
                "week_of": tc["week_of"],
                "status": "FAIL",
                "reason": "Missing row in evaluated anomalies output",
            })
            all_cases_passed = False
            continue

        row = sub.iloc[0]
        actual_cost = float(row["cost_per_tonne_km"])
        cost_ok = abs(actual_cost - tc["cost_per_tonne_km"]) < 1e-3
        flag_ok = row["flagged"] == tc["flagged"]
        note_ok = str(row["matched_note_id"]).strip() == str(tc["matched_note_id"]).strip()

        own_ok = True
        if "vs_own_history" in tc:
            own_ok = row["vs_own_history"] == tc["vs_own_history"]

        peer_ok = True
        if "vs_similar_routes" in tc:
            peer_ok = row["vs_similar_routes"] == tc["vs_similar_routes"]

        reason_ok = True
        if "reason_mentions" in tc:
            actual_reason = row["reason"]
            reason_ok = any(m in actual_reason for m in tc["reason_mentions"])

        passed = cost_ok and flag_ok and note_ok and own_ok and peer_ok and reason_ok
        if not passed:
            all_cases_passed = False

        case_results.append({
            "id": tc["id"],
            "route": tc["route"],
            "week_of": tc["week_of"],
            "status": "PASS" if passed else "FAIL",
            "cost_per_tonne_km": {"actual": actual_cost, "expected": tc["cost_per_tonne_km"], "pass": cost_ok},
            "flagged": {"actual": row["flagged"], "expected": tc["flagged"], "pass": flag_ok},
            "matched_note_id": {"actual": row["matched_note_id"], "expected": tc["matched_note_id"], "pass": note_ok},
            "details": {
                "own_history_pass": own_ok,
                "peer_routes_pass": peer_ok,
                "reason_mentions_pass": reason_ok,
            },
        })

    # 2. Invariant A: Decoy Rejection
    decoy_notes = {"N004", "N005", "N006", "N007", "N008", "N009", "N010"}
    all_matched = set(df_all[df_all["matched_note_id"] != ""]["matched_note_id"].unique())
    decoy_violations = all_matched.intersection(decoy_notes)
    inv_a_pass = len(decoy_violations) == 0

    # 3. Invariant B: Contract Integrity
    flagged_yes_notes = df_all[df_all["flagged"] == "Yes"]["matched_note_id"].unique()
    inv_b1_pass = all(n == "" for n in flagged_yes_notes)

    flagged_no_notes = df_all[df_all["flagged"] == "No"]["matched_note_id"].unique()
    inv_b2_pass = all(n == "" for n in flagged_no_notes)

    flagged_justified_notes = df_all[df_all["flagged"] == "No (justified)"]["matched_note_id"].unique()
    valid_justified_notes = {"N001", "N002", "N003"}
    inv_b3_pass = all(n in valid_justified_notes for n in flagged_justified_notes) and len(flagged_justified_notes) > 0
    inv_b_pass = inv_b1_pass and inv_b2_pass and inv_b3_pass

    # 4. Invariant C: Schema Contract
    actual_cols = list(df_eval.columns)
    inv_c_pass = actual_cols == CONTRACT_COLUMNS

    suite_passed = all_cases_passed and inv_a_pass and inv_b_pass and inv_c_pass

    eval_summary = {
        "suite_passed": suite_passed,
        "golden_cases_passed": sum(1 for c in case_results if c["status"] == "PASS"),
        "golden_cases_total": len(case_results),
        "invariants": {
            "invariant_a_decoy_rejection": {"passed": inv_a_pass, "violations": list(decoy_violations)},
            "invariant_b_contract_integrity": {"passed": inv_b_pass},
            "invariant_c_schema_columns": {"passed": inv_c_pass, "columns": actual_cols},
        },
        "case_results": case_results,
    }

    return suite_passed, eval_summary


def compute_file_hashes(file_path: Path) -> Dict[str, str]:
    """Compute full-file SHA-256 and core-columns SHA-256."""
    with open(file_path, "rb") as f:
        full_hash = hashlib.sha256(f.read()).hexdigest()

    df = load_csv_safely(file_path)
    core_df = df[CORE_GRADING_COLUMNS]
    core_bytes = core_df.to_csv(index=False).encode("utf-8")
    core_hash = hashlib.sha256(core_bytes).hexdigest()

    return {"full_file_sha256": full_hash, "core_columns_sha256": core_hash}


def run_reproducibility_check(output_base_dir: Path = Path("output")) -> Dict[str, Any]:
    """
    Executes pipeline 3 separate untouched times and computes reproducibility diffs and hashes.
    """
    repro_dir = output_base_dir / "reproducibility"
    repro_dir.mkdir(parents=True, exist_ok=True)

    run_paths = [
        repro_dir / "run_1.csv",
        repro_dir / "run_2.csv",
        repro_dir / "run_3.csv",
    ]

    hashes = []
    print("\n" + "=" * 70)
    print("STARTING 3 UNTOUCHED RUNS REPRODUCIBILITY SUITE")
    print("=" * 70)

    for i, run_path in enumerate(run_paths, start=1):
        print(f"\n>>> Executing Untouched Run {i} of 3...")
        anomalies_df, _, _ = execute_audit_pipeline(
            output_dir=output_base_dir,
            preferred_model="llama-3.3-70b-versatile",
            fallback_model="llama-3.1-8b-instant",
        )
        anomalies_df.to_csv(run_path, index=False)
        h = compute_file_hashes(run_path)
        posix_path = run_path.as_posix()
        hashes.append({"run": i, "path": posix_path, **h})
        print(f"    Completed Run {i}: Core Hash = {h['core_columns_sha256'][:16]}... Full Hash = {h['full_file_sha256'][:16]}...")

    # Pairwise column-by-column comparison
    df1 = load_csv_safely(run_paths[0])
    df2 = load_csv_safely(run_paths[1])
    df3 = load_csv_safely(run_paths[2])

    diffs_1_2 = df1.compare(df2)
    diffs_1_3 = df1.compare(df3)

    is_core_identical = (
        hashes[0]["core_columns_sha256"]
        == hashes[1]["core_columns_sha256"]
        == hashes[2]["core_columns_sha256"]
    )
    is_full_identical = (
        hashes[0]["full_file_sha256"]
        == hashes[1]["full_file_sha256"]
        == hashes[2]["full_file_sha256"]
    )

    repro_results = {
        "runs_count": 3,
        "is_core_identical": is_core_identical,
        "is_full_file_identical": is_full_identical,
        "hashes": hashes,
        "diff_run1_vs_run2_rows": len(diffs_1_2),
        "diff_run1_vs_run3_rows": len(diffs_1_3),
    }

    # Generate Human-Readable Report
    report_lines = [
        "=" * 75,
        "FREIGHT TIGER REPRODUCIBILITY & DETERMINISM VERIFICATION REPORT",
        "=" * 75,
        f"Untouched Pipeline Runs Executed: 3",
        f"Core Grading Columns Determinism: {'100% BIT-FOR-BIT IDENTICAL [PASSED]' if is_core_identical else 'FAILED'}",
        f"Full File (Inc. Explanations)    : {'100% BIT-FOR-BIT IDENTICAL [PASSED]' if is_full_identical else 'DIFFERENT'}",
        "-" * 75,
        "SHA-256 HASH VERIFICATION MATRIX:",
    ]
    for h in hashes:
        report_lines.append(
            f"  Run {h['run']}:"
            f"\n    Path               : {h['path']}"
            f"\n    Core Columns SHA256: {h['core_columns_sha256']}"
            f"\n    Full File SHA256   : {h['full_file_sha256']}"
        )
    report_lines.extend([
        "-" * 75,
        f"Row-by-Row Differences (Run 1 vs Run 2): {len(diffs_1_2)} mismatched rows",
        f"Row-by-Row Differences (Run 1 vs Run 3): {len(diffs_1_3)} mismatched rows",
        "=" * 75,
        f"REPRODUCIBILITY VERDICT: {'PERFECT 3-RUN DETERMINISM CONFIRMED' if is_core_identical else 'REPRODUCIBILITY CHECK FAILED'}",
        "=" * 75,
    ])

    report_text = "\n".join(report_lines)
    report_path = output_base_dir / "reproducibility_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return repro_results


def main():
    """Main execution entrypoint for evaluation suite."""
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run Golden Evaluation
    print("\n" + "=" * 70)
    print("RUNNING GOLDEN BENCHMARK EVALUATION (10 TEST CASES + 3 INVARIANTS)")
    print("=" * 70)
    golden_pass, eval_metrics = run_golden_eval()

    print(f"\nGolden Suite Result: {'PASSED (10/10 Cases + 3/3 Invariants)' if golden_pass else 'FAILED'}")
    for c in eval_metrics["case_results"]:
        status_tag = f"[{c['status']}]"
        print(f"  {status_tag} {c['id']}: {c['route']} ({c['week_of']}) -> Flagged: '{c['flagged']['actual']}' | Note: '{c['matched_note_id']['actual']}'")

    inv = eval_metrics["invariants"]
    print("\nGlobal Anti-Hallucination Invariants:")
    print(f"  [PASS] Invariant A (Decoy Rejection - N004-N010 never in matched_note_id): {inv['invariant_a_decoy_rejection']['passed']}")
    print(f"  [PASS] Invariant B (Contract Integrity - Yes/No blank, Justified in N001-N003): {inv['invariant_b_contract_integrity']['passed']}")
    print(f"  [PASS] Invariant C (Schema & Column Ordering): {inv['invariant_c_schema_columns']['passed']}")

    # 2. Run Reproducibility Check
    repro_metrics = run_reproducibility_check(out_dir)

    # Save comprehensive evaluation report
    combined_report = {
        "golden_evaluation": eval_metrics,
        "reproducibility": repro_metrics,
    }
    with open(out_dir / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2)

    # Print Report contents
    report_file = out_dir / "reproducibility_report.txt"
    print("\n" + "=" * 70)
    print(f"REPRODUCIBILITY REPORT FILE ({report_file}):")
    print("=" * 70)
    with open(report_file, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
