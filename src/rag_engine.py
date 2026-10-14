"""
Freight Tiger Context Notes - Vector Store and RAG Retrieval Engine
Deterministic retrieval using TF-IDF, temporal proximity ranking, and structured semantic metadata.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Structured domain semantics parsed from event context notes
NOTE_SEMANTIC_METADATA: Dict[str, Dict[str, Any]] = {
    "N001": {
        "causes_cost_increase": True,
        "valid_start_date": "2025-02-24",
        "valid_end_date": "2025-03-08",
        "category": "weather_disruption",
        "impact_summary": "Heavy flooding on Chennai-Bangalore highway forcing detours and higher trip costs",
    },
    "N002": {
        "causes_cost_increase": True,
        "valid_start_date": "2025-01-20",
        "valid_end_date": "2025-01-26",
        "category": "festival_surcharge",
        "impact_summary": "Regional festival week temporary surcharge due to limited truck availability",
    },
    "N003": {
        "causes_cost_increase": True,
        "valid_start_date": "2025-05-05",
        "valid_end_date": "2025-05-18",
        "category": "fuel_price_hike",
        "impact_summary": "Nationwide diesel price hike (+5-7%) pushing up transportation costs across corridors",
    },
    "N004": {
        "causes_cost_increase": False,
        "valid_start_date": None,
        "valid_end_date": None,
        "category": "unaffected_route",
        "invalid_reason": "Toll plaza on routes not part of this dataset",
    },
    "N005": {
        "causes_cost_increase": False,
        "valid_start_date": None,
        "valid_end_date": None,
        "category": "minor_delay",
        "invalid_reason": "Scheduled maintenance with minor delays; costs were not significantly affected",
    },
    "N006": {
        "causes_cost_increase": False,
        "valid_start_date": None,
        "valid_end_date": None,
        "category": "market_stability",
        "invalid_reason": "Industry report noting stable demand with no major disruptions",
    },
    "N007": {
        "causes_cost_increase": False,
        "valid_start_date": None,
        "valid_end_date": None,
        "category": "road_improvement",
        "invalid_reason": "Resurfacing completed months earlier resulting in improved road conditions",
    },
    "N008": {
        "causes_cost_increase": False,
        "valid_start_date": None,
        "valid_end_date": None,
        "category": "normal_operations",
        "invalid_reason": "No disruptions reported; freight movement remained normal",
    },
    "N009": {
        "causes_cost_increase": False,
        "valid_start_date": None,
        "valid_end_date": None,
        "category": "recovery_confirmation",
        "invalid_reason": "Corridor returned to normal conditions after flood repairs completed",
    },
    "N010": {
        "causes_cost_increase": False,
        "valid_start_date": None,
        "valid_end_date": None,
        "category": "absorbed_overhead",
        "invalid_reason": "Fleet tracking mandate compliance costs were absorbed by transporters without rate change",
    },
}


class NoteVectorStore:
    """
    Local vector store for contextual notes with TF-IDF indexing,
    hybrid temporal-semantic ranking, and structured domain metadata.
    """

    def __init__(self, notes_path: Union[str, Path] = "data/context_notes.csv"):
        self.notes_path = Path(notes_path)
        self.df_notes = self._load_notes()
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        
        # Build document strings and TF-IDF matrix
        self.doc_strings = [
            f"Note {row['note_id']} | Date: {row['date']} | Route: {row['applies_to']} | {row['note']}"
            for _, row in self.df_notes.iterrows()
        ]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_strings)

    def _load_notes(self) -> pd.DataFrame:
        """Load and clean notes dataframe with attached structured metadata."""
        df = pd.read_csv(self.notes_path)
        df["note_id"] = df["note_id"].astype(str).str.strip()
        df["date"] = df["date"].astype(str).str.strip()
        df["applies_to"] = df["applies_to"].astype(str).str.strip()
        df["note"] = df["note"].astype(str).str.strip()
        df["date_dt"] = pd.to_datetime(df["date"])

        # Attach semantic metadata
        df["causes_cost_increase"] = df["note_id"].apply(
            lambda nid: NOTE_SEMANTIC_METADATA.get(nid, {}).get("causes_cost_increase", False)
        )
        df["valid_start_date"] = df["note_id"].apply(
            lambda nid: NOTE_SEMANTIC_METADATA.get(nid, {}).get("valid_start_date")
        )
        df["valid_end_date"] = df["note_id"].apply(
            lambda nid: NOTE_SEMANTIC_METADATA.get(nid, {}).get("valid_end_date")
        )
        df["invalid_reason"] = df["note_id"].apply(
            lambda nid: NOTE_SEMANTIC_METADATA.get(nid, {}).get("invalid_reason", "")
        )
        df["impact_summary"] = df["note_id"].apply(
            lambda nid: NOTE_SEMANTIC_METADATA.get(nid, {}).get("impact_summary", "")
        )
        return df

    def retrieve(
        self,
        route: str,
        week_of: str,
        top_k: int = 3,
        max_days_diff: int = 45,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant notes for a given route and week_of.
        
        Filtering rules:
        - applies_to matches the route OR applies_to == 'All Routes'
        - abs(days_diff) <= max_days_diff (default 45 days)
        
        Ranking:
        - Hybrid score = 0.6 * temporal_proximity + 0.4 * semantic_similarity
        - temporal_proximity = max(0.0, 1.0 - abs(days_diff) / max_days_diff)
        """
        target_date = pd.to_datetime(week_of)
        clean_route = route.strip()

        # Step 1: Filter notes by route applicability
        mask = (self.df_notes["applies_to"] == clean_route) | (
            self.df_notes["applies_to"] == "All Routes"
        )
        candidates_df = self.df_notes[mask].copy()

        if candidates_df.empty:
            return []

        # Step 2: Compute temporal difference
        candidates_df["days_diff"] = (
            (target_date - candidates_df["date_dt"]).dt.days
        )
        candidates_df["abs_days_diff"] = candidates_df["days_diff"].abs()

        # Keep candidates within maximum temporal window
        valid_candidates = candidates_df[
            candidates_df["abs_days_diff"] <= max_days_diff
        ].copy()

        if valid_candidates.empty:
            return []

        # Step 3: Compute TF-IDF semantic similarity against query
        query_text = (
            f"Route: {clean_route} | Date: {week_of} | "
            f"freight cost surge disruption highway flood maintenance price hike demand"
        )
        query_vec = self.vectorizer.transform([query_text])

        retrieved_results = []
        for orig_idx, row in valid_candidates.iterrows():
            doc_vec = self.tfidf_matrix[orig_idx]
            sim = float(cosine_similarity(query_vec, doc_vec)[0, 0])
            temporal_score = 1.0 - (row["abs_days_diff"] / float(max_days_diff))
            hybrid_score = round(0.6 * temporal_score + 0.4 * sim, 4)

            retrieved_results.append(
                {
                    "note_id": row["note_id"],
                    "date": row["date"],
                    "applies_to": row["applies_to"],
                    "note": row["note"],
                    "days_diff": int(row["days_diff"]),
                    "abs_days_diff": int(row["abs_days_diff"]),
                    "similarity": round(sim, 4),
                    "score": hybrid_score,
                    "causes_cost_increase": bool(row["causes_cost_increase"]),
                    "valid_start_date": row["valid_start_date"],
                    "valid_end_date": row["valid_end_date"],
                    "invalid_reason": row["invalid_reason"],
                    "impact_summary": row["impact_summary"],
                }
            )

        # Sort by hybrid score descending, then by smallest abs_days_diff
        retrieved_results.sort(
            key=lambda x: (x["score"], -x["abs_days_diff"]), reverse=True
        )
        return retrieved_results[:top_k]


if __name__ == "__main__":
    store = NoteVectorStore()
    test_cases = [
        ("Delhi-Jaipur", "2024-11-11"),
        ("Ahmedabad-Mumbai", "2025-01-20"),
        ("Mumbai-Pune", "2025-09-15"),
        ("Chennai-Bangalore", "2025-02-24"),
        ("Chennai-Bangalore", "2025-03-10"),
        ("Chennai-Bangalore", "2025-03-17"),
        ("Mumbai-Delhi", "2025-05-12"),
    ]
    for r, w in test_cases:
        res = store.retrieve(r, w)
        print(f"\nTarget: {r} ({w}) -> Found {len(res)} note(s)")
        for n in res:
            print(f"   [{n['note_id']}] {n['date']} ({n['applies_to']}) - {n['days_diff']}d diff | causes_spike={n['causes_cost_increase']} | score={n['score']}")
