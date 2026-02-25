#!/usr/bin/env python3
"""
Data Richness Discovery Tool (Template).

This tool connects to an external data source (e.g. a REST API, database,
or SaaS platform) and identifies which records have the most complete data
across multiple domains. The results are saved as curated record lists
so that QA generation targets records with the highest coverage.

Usage:
    python utils/data_discovery.py
    python utils/data_discovery.py --domain employee_directory
    python utils/data_discovery.py --limit 50

NOTE: This is a templatized version. Replace the placeholder API calls
with your actual data-source integration.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

from utils import env_suffix, ENV_TAG

# ---------------------------------------------------------------------------
# Configuration — replace with your data source credentials
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("DATA_SOURCE_URL", "https://api.example.com")
API_KEY = os.getenv("DATA_SOURCE_API_KEY", "")


def get_auth_headers() -> Dict[str, str]:
    """Build authentication headers for API requests."""
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }


# ---------------------------------------------------------------------------
# Record Discovery
# ---------------------------------------------------------------------------


def search_all_records(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Discover records from the data source.

    Replace the body of this function with actual API calls to your
    system (e.g. paginated REST queries, database reads, etc.).
    """
    print(f"Discovering records (limit={limit})...")

    # --- TEMPLATE: replace with real API calls ---
    # Example:
    #   response = requests.get(f"{BASE_URL}/api/v1/records",
    #                           headers=get_auth_headers(),
    #                           params={"limit": limit})
    #   return response.json().get("items", [])

    print("  (No data source configured — returning empty list)")
    return []


# ---------------------------------------------------------------------------
# Domain-specific Richness Scoring
# ---------------------------------------------------------------------------


def score_employee_directory(record: Dict[str, Any]) -> Dict[str, Any]:
    """Score a record's data richness for the employee-directory domain."""
    score = 0
    details: Dict[str, int] = {}

    if record.get("phone"):
        score += 2
        details["phone"] = 1
    if record.get("email"):
        score += 2
        details["email"] = 1
    if record.get("manager"):
        score += 3
        details["manager"] = 1
    if record.get("department"):
        score += 1
        details["department"] = 1
    if record.get("title"):
        score += 1
        details["title"] = 1

    return {"score": score, "details": details}


def score_compensation(record: Dict[str, Any]) -> Dict[str, Any]:
    """Score richness for compensation domain."""
    score = 0
    details: Dict[str, int] = {}

    if record.get("salary"):
        score += 3
        details["salary"] = 1
    if record.get("bonus_history"):
        score += 2
        details["bonus"] = len(record["bonus_history"])
    if record.get("stock_grants"):
        score += 2
        details["stock"] = len(record["stock_grants"])

    return {"score": score, "details": details}


DOMAIN_SCORERS = {
    "employee_directory": score_employee_directory,
    "compensation": score_compensation,
}


# ---------------------------------------------------------------------------
# Analysis & Output
# ---------------------------------------------------------------------------


def analyze_records(records: List[Dict[str, Any]], max_records: int = 50) -> Dict[str, List[Dict]]:
    """Score records and categorize by best domain fit."""
    print(f"\nAnalyzing {min(len(records), max_records)} records...")

    scored: List[Dict] = []

    for rec in records[:max_records]:
        name = rec.get("display_name", rec.get("name", "Unknown"))
        entry = {"name": name, "record": rec, "total_score": 0, "domain_scores": {}}

        for domain, scorer in DOMAIN_SCORERS.items():
            result = scorer(rec)
            entry["domain_scores"][domain] = result
            entry["total_score"] += result["score"]

        scored.append(entry)

    scored.sort(key=lambda x: x["total_score"], reverse=True)

    domain_lists: Dict[str, List[Dict]] = defaultdict(list)
    for entry in scored:
        if entry["total_score"] == 0:
            continue
        best_domain = max(entry["domain_scores"], key=lambda d: entry["domain_scores"][d]["score"])
        if len(domain_lists[best_domain]) < 15:
            domain_lists[best_domain].append(entry)

    return dict(domain_lists)


def save_record_lists(domain_records: Dict[str, List[Dict]]):
    """Persist curated record lists as JSON."""
    out_dir = os.path.join(os.path.dirname(__file__), "..", "record_lists")
    os.makedirs(out_dir, exist_ok=True)

    tag = ENV_TAG or "DEFAULT"
    out_file = os.path.join(out_dir, f"{tag}_record_lists.json")

    payload = {
        "metadata": {
            "env_tag": tag,
            "generated_at": datetime.now().isoformat(),
            "source": "data_discovery.py",
        },
        "domains": {},
    }

    for domain, entries in domain_records.items():
        payload["domains"][domain] = {
            "records": [e["name"] for e in entries],
            "count": len(entries),
        }

    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved record lists -> {out_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Discover records with rich data")
    parser.add_argument("--domain", help="Focus on a single domain")
    parser.add_argument("--limit", type=int, default=100, help="Max records to discover")
    parser.add_argument("--analyze", type=int, default=50, help="Max records to analyze")
    args = parser.parse_args()

    print("Data Richness Discovery Tool")
    print("=" * 40)

    records = search_all_records(limit=args.limit)
    if not records:
        print("No records found. Configure DATA_SOURCE_URL and DATA_SOURCE_API_KEY.")
        return

    domain_records = analyze_records(records, max_records=args.analyze)

    print("\nResults:")
    for domain, entries in domain_records.items():
        print(f"\n  {domain} ({len(entries)} records):")
        for e in entries[:5]:
            print(f"    - {e['name']} (score: {e['total_score']})")

    save_record_lists(domain_records)
    print("\nDone.")


if __name__ == "__main__":
    main()
