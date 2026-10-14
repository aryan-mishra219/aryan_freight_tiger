"""
Freight Tiger Shipping Cost Analytics - Core Math Engine
Deterministic calculations for route baselines, rolling averages, peer divergence, and anomaly candidate identification.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd


def load_and_preprocess_shipments(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load raw shipment records and perform initial standardizations.
    """
    df = pd.read_csv(filepath)
    # Sanitize and title-case route strings
    df["origin"] = df["origin"].astype(str).str.strip().str.title()
    df["destination"] = df["destination"].astype(str).str.strip().str.title()
    df["route"] = df["origin"] + "-" + df["destination"]
    df["route_type"] = df["route_type"].astype(str).str.strip().str.title()
    
    # Coerce numeric values cleanly with fillna(0.0)
    df["quantity_tonnes"] = pd.to_numeric(df["quantity_tonnes"], errors="coerce").fillna(0.0)
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce").fillna(0.0)
    df["freight_cost_inr"] = pd.to_numeric(df["freight_cost_inr"], errors="coerce").fillna(0.0)

    # Standardize dates and Monday-aligned week_of
    df["shipment_date"] = pd.to_datetime(df["shipment_date"], errors="coerce")
    df = df.dropna(subset=["shipment_date"]).copy()
    df["week_of"] = (
        df["shipment_date"] - pd.to_timedelta(df["shipment_date"].dt.weekday, unit="D")
    ).dt.strftime("%Y-%m-%d")
    
    # Calculate tonne-km per shipment
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    return df


def compute_weekly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute deterministic route metrics per (route, route_type, week_of):
    1. cost_per_tonne_km_raw = sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)
    2. vs_own_history: Trailing 8-week rolling average using strictly prior weeks (shift(1), window=8, min_periods=1)
    3. vs_similar_routes: Mean across other routes with identical route_type in the same week_of
    4. is_anomaly_candidate: Flag candidate route-weeks based on historical spike or peer divergence
    """
    # Group by route, route_type, week_of
    grouped = (
        df.groupby(["route", "route_type", "week_of"], as_index=False)
        .agg(
            total_freight_cost=("freight_cost_inr", "sum"),
            total_tonne_km=("tonne_km", "sum"),
            shipment_count=("shipment_id", "count"),
        )
    )

    # 1. Unrounded cost per tonne-km with zero-division guard
    grouped["cost_per_tonne_km_raw"] = np.where(
        grouped["total_tonne_km"] > 0,
        grouped["total_freight_cost"] / grouped["total_tonne_km"],
        0.0,
    )

    # Ensure chronological order per route
    grouped = grouped.sort_values(["route", "week_of"]).reset_index(drop=True)

    # 2. vs_own_history: 8-week trailing rolling average of strictly prior weeks (no lookahead)
    grouped["own_rolling_avg_raw"] = (
        grouped.groupby("route")["cost_per_tonne_km_raw"]
        .transform(lambda s: s.shift(1).rolling(window=8, min_periods=1).mean())
    )

    grouped["pct_own"] = np.where(
        grouped["own_rolling_avg_raw"].notna() & (grouped["own_rolling_avg_raw"] > 0),
        (grouped["cost_per_tonne_km_raw"] - grouped["own_rolling_avg_raw"])
        / grouped["own_rolling_avg_raw"]
        * 100.0,
        np.nan,
    )

    # 3. vs_similar_routes: Mean of cost_per_tonne_km_raw across all OTHER routes sharing the same route_type
    peer_group = grouped.groupby(["week_of", "route_type"])["cost_per_tonne_km_raw"]
    sum_cost = peer_group.transform("sum")
    count_routes = peer_group.transform("count")

    peer_sum = sum_cost - grouped["cost_per_tonne_km_raw"]
    peer_count = count_routes - 1

    grouped["peer_avg_raw"] = np.where(peer_count > 0, peer_sum / peer_count, np.nan)
    grouped["pct_peer"] = np.where(
        grouped["peer_avg_raw"].notna() & (grouped["peer_avg_raw"] > 0),
        (grouped["cost_per_tonne_km_raw"] - grouped["peer_avg_raw"])
        / grouped["peer_avg_raw"]
        * 100.0,
        np.nan,
    )

    # 4. Anomaly Candidate Detection
    # Sharp historical spike (>= 10%) OR subtle creeping divergence (>= 18% peer with >= 2% own) OR pure peer divergence (>= 20%)
    cond_sharp_spike = grouped["pct_own"].ge(10.0)
    cond_creeping_divergence = grouped["pct_peer"].ge(18.0) & grouped["pct_own"].ge(2.0)
    cond_peer_divergence = grouped["pct_peer"].ge(20.0)

    grouped["is_anomaly_candidate"] = (
        cond_sharp_spike | cond_creeping_divergence | cond_peer_divergence
    )

    # 5. Presentation formatting (only at final output time)
    grouped["cost_per_tonne_km"] = grouped["cost_per_tonne_km_raw"].round(2)
    grouped["vs_own_history"] = grouped["pct_own"].apply(
        lambda x: "N/A vs this route's past average"
        if pd.isna(x)
        else f"{x:+.1f}% vs this route's past average"
    )
    grouped["vs_similar_routes"] = grouped["pct_peer"].apply(
        lambda x: "N/A vs similar-length routes this week"
        if pd.isna(x)
        else f"{x:+.1f}% vs similar-length routes this week"
    )

    return grouped


def run_pipeline(
    shipment_records_path: Union[str, Path] = "data/shipment_records.csv"
) -> pd.DataFrame:
    """
    Run complete deterministic preprocessing and metric computation pipeline.
    """
    raw_df = load_and_preprocess_shipments(shipment_records_path)
    metrics_df = compute_weekly_metrics(raw_df)
    return metrics_df


def verify_engine(
    shipment_records_path: Union[str, Path] = "data/shipment_records.csv",
    sample_output_path: Optional[Union[str, Path]] = "data/sample_output_format_v2.csv",
) -> bool:
    """
    Verify engine calculations against sample_output_format_v2.csv ground truth benchmarks.
    """
    df_metrics = run_pipeline(shipment_records_path)

    expected_benchmarks = [
        {
            "route": "Delhi-Jaipur",
            "week_of": "2024-11-11",
            "cost_per_tonne_km": 4.17,
            "vs_own_history": "+35.5% vs this route's past average",
            "vs_similar_routes": "+21.0% vs similar-length routes this week",
            "is_anomaly_candidate": True,
        },
        {
            "route": "Ahmedabad-Mumbai",
            "week_of": "2025-01-20",
            "cost_per_tonne_km": 3.29,
            "vs_own_history": "+29.5% vs this route's past average",
            "vs_similar_routes": "+22.5% vs similar-length routes this week",
            "is_anomaly_candidate": True,
        },
        {
            "route": "Mumbai-Pune",
            "week_of": "2025-09-15",
            "cost_per_tonne_km": 3.98,
            "vs_own_history": "+9.2% vs this route's past average",
            "vs_similar_routes": "+23.6% vs similar-length routes this week",
            "is_anomaly_candidate": True,
        },
    ]

    all_passed = True
    print("=" * 70)
    print("FREIGHT TIGER MATH ENGINE: BENCHMARK VERIFICATION REPORT")
    print("=" * 70)

    for bench in expected_benchmarks:
        sub = df_metrics[
            (df_metrics["route"] == bench["route"])
            & (df_metrics["week_of"] == bench["week_of"])
        ]
        if sub.empty:
            print(f"[FAIL] Missing record: {bench['route']} on {bench['week_of']}")
            all_passed = False
            continue

        row = sub.iloc[0]
        c_match = abs(row["cost_per_tonne_km"] - bench["cost_per_tonne_km"]) < 1e-4
        own_match = row["vs_own_history"] == bench["vs_own_history"]
        peer_match = row["vs_similar_routes"] == bench["vs_similar_routes"]
        cand_match = row["is_anomaly_candidate"] == bench["is_anomaly_candidate"]

        row_passed = c_match and own_match and peer_match and cand_match
        status = "PASSED" if row_passed else "FAILED"
        if not row_passed:
            all_passed = False

        print(f"[{status}] Route: {bench['route']} | Week: {bench['week_of']}")
        print(f"   cost_per_tonne_km : Actual = {row['cost_per_tonne_km']:.2f} | Expected = {bench['cost_per_tonne_km']:.2f} {'[OK]' if c_match else '[FAIL]'}")
        print(f"   vs_own_history    : Actual = '{row['vs_own_history']}' | Expected = '{bench['vs_own_history']}' {'[OK]' if own_match else '[FAIL]'}")
        print(f"   vs_similar_routes : Actual = '{row['vs_similar_routes']}' | Expected = '{bench['vs_similar_routes']}' {'[OK]' if peer_match else '[FAIL]'}")
        print(f"   anomaly_candidate : Actual = {row['is_anomaly_candidate']} | Expected = {bench['is_anomaly_candidate']} {'[OK]' if cand_match else '[FAIL]'}")
        print("-" * 70)

    # Check against sample_output_format_v2.csv directly if file is available
    if sample_output_path and Path(sample_output_path).exists():
        print("Comparing against data/sample_output_format_v2.csv rows:")
        with open(sample_output_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        header = lines[0].split(",")
        for line in lines[1:]:
            parts = line.split(",", 7)
            if len(parts) >= 6:
                s_route = parts[0]
                s_week_of = parts[1]
                s_cost = float(parts[2])
                s_own = parts[3]
                s_peer = parts[4]

                sub = df_metrics[
                    (df_metrics["route"] == s_route)
                    & (df_metrics["week_of"] == s_week_of)
                ]
                if not sub.empty:
                    r = sub.iloc[0]
                    assert abs(r["cost_per_tonne_km"] - s_cost) < 1e-4, f"Mismatch in cost for {s_route} {s_week_of}"
                    assert r["vs_own_history"] == s_own, f"Mismatch in vs_own_history: {r['vs_own_history']} != {s_own}"
                    assert r["vs_similar_routes"] == s_peer, f"Mismatch in vs_similar_routes: {r['vs_similar_routes']} != {s_peer}"
                    print(f"   [OK] Verified sample file row for {s_route} ({s_week_of})")

    print("=" * 70)
    print(f"OVERALL ENGINE STATUS: {'ALL CHECKS PASSED PERFECTLY' if all_passed else 'VERIFICATION FAILED'}")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    verify_engine()
