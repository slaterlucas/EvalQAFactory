#!/usr/bin/env python3
"""
Concierge QA Remixer.

Generates evaluation questions by remixing existing QA outputs:
- Sampling scenarios: draw questions directly from existing spreadsheets
- Combination scenarios: merge two single-domain questions into one
  multi-topic question using an LLM

This enables creating "concierge" or "generalist" evaluation sets from
domain-specific outputs without re-running the full generation pipeline.

Setup:
    pip install google-generativeai pandas openpyxl xlsxwriter tqdm backoff python-dotenv
    export GOOGLE_API_KEY="..."   # only needed for combination scenarios
"""

import os
import json
import random
import time
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import google.generativeai as genai
import pandas as pd
from tqdm import tqdm
import backoff
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ConciergeConfig:
    eval_name: str
    output_dir: str
    scenario_configs: Dict[str, Dict[str, Any]]


gemini_model = None


def initialize_gemini():
    global gemini_model
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set. Please export it or add to .env")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    print("LLM initialized.")
    return gemini_model


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def generate_with_llm(prompt: str) -> Optional[Dict[str, Any]]:
    global gemini_model
    if gemini_model is None:
        gemini_model = initialize_gemini()
    try:
        response = gemini_model.generate_content(prompt)
        cleaned = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        print(f"Warning: parse error — {e}")
        return None
    except Exception as e:
        print(f"LLM error: {e}")
        return None


def read_excel(path: str) -> pd.DataFrame:
    """Read an existing QA spreadsheet."""
    try:
        if not os.path.exists(path):
            print(f"  Warning: file not found — {path}")
            return pd.DataFrame()
        df = pd.read_excel(path, sheet_name="Questions")
        print(f"  Loaded {len(df)} rows from {os.path.basename(path)}")
        return df
    except Exception as e:
        print(f"  Error reading {path}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Sampling — no LLM needed
# ---------------------------------------------------------------------------


def generate_sampling(config: Dict[str, Any], eval_name: str, scenario_id: str) -> List[Dict]:
    """Sample N questions per source file."""
    sources = config.get("source_spreadsheets", [])
    per_file = config.get("questions_per_file", 3)

    if not sources:
        print(f"  No source spreadsheets for {scenario_id}")
        return []

    results: List[Dict] = []
    counter = 1

    print(f"  Sampling {per_file} questions from {len(sources)} files")
    for path in tqdm(sources, desc="Sampling"):
        df = read_excel(path)
        if df.empty:
            continue
        rows = df.to_dict("records")
        sample = random.sample(rows, min(per_file, len(rows)))
        for row in sample:
            results.append({
                "eval_name": eval_name,
                "eval_scenario": scenario_id,
                "query_num": f"{eval_name}_{scenario_id}_{counter:03d}",
                "query": row["query"],
                "expected_answer": row["expected_answer"],
            })
            counter += 1

    print(f"  Collected {len(results)} questions")
    return results


# ---------------------------------------------------------------------------
# Combinations — uses LLM to merge two QA pairs
# ---------------------------------------------------------------------------


def generate_combinations(config: Dict[str, Any], eval_name: str, scenario_id: str) -> List[Dict]:
    """Combine pairs of single-domain questions into multi-topic questions."""
    combos = config.get("combinations", [])
    target = config.get("target_count", 25)

    if not combos:
        print(f"  No combinations for {scenario_id}")
        return []

    results: List[Dict] = []
    counter = 0

    for _ in tqdm(range(target), desc=f"Generating {scenario_id} combos"):
        cc = random.choice(combos)
        df1 = read_excel(cc["source_1"])
        df2 = read_excel(cc["source_2"])
        if df1.empty or df2.empty:
            continue

        r1 = df1.sample(1).iloc[0].to_dict()
        r2 = df2.sample(1).iloc[0].to_dict()
        desc = cc.get("description", "")

        prompt = f"""You are helping create combined questions for a knowledge assistant.

Source 1 — {desc.split(' + ')[0] if ' + ' in desc else 'Topic A'}:
Question: "{r1['query']}"
Answer: "{r1['expected_answer']}"

Source 2 — {desc.split(' + ')[1] if ' + ' in desc else 'Topic B'}:
Question: "{r2['query']}"
Answer: "{r2['expected_answer']}"

Create ONE new question that naturally asks for BOTH pieces of information.
Merge both answers into a single, natural response preserving all key data.

Return JSON: {{"query": "...", "expected_answer": "..."}}"""

        result = generate_with_llm(prompt)
        if result and "query" in result and "expected_answer" in result:
            counter += 1
            results.append({
                "eval_name": eval_name,
                "eval_scenario": scenario_id,
                "query_num": f"{eval_name}_{scenario_id}_{counter:03d}",
                "query": result["query"],
                "expected_answer": result["expected_answer"],
            })
        else:
            time.sleep(0.5)

    print(f"  Generated {len(results)} combinations")
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_to_excel(data: List[Dict], output_file: str):
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Questions", index=False)
        wb = writer.book
        ws = writer.sheets["Questions"]
        hdr = wb.add_format({"bold": True, "text_wrap": True, "valign": "top"})
        cell = wb.add_format({"text_wrap": True, "valign": "top"})
        for col, w in {"A": 15, "B": 12, "C": 20, "D": 50, "E": 60}.items():
            ws.set_column(f"{col}:{col}", w, cell)
        for ci, cn in enumerate(df.columns):
            ws.write(0, ci, cn, hdr)

    print(f"  Saved {len(df)} questions -> {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_concierge(config: ConciergeConfig):
    print(f"Starting concierge generation: {config.eval_name}")
    all_generated: List[Dict] = []

    for sid, scfg in config.scenario_configs.items():
        if "source_spreadsheets" in scfg:
            all_generated.extend(generate_sampling(scfg, config.eval_name, sid))
        elif "combinations" in scfg:
            all_generated.extend(generate_combinations(scfg, config.eval_name, sid))

    if not all_generated:
        print("No questions generated.")
        return

    by_scenario: Dict[str, List[Dict]] = {}
    for item in all_generated:
        by_scenario.setdefault(item["eval_scenario"], []).append(item)

    for sid, data in by_scenario.items():
        save_to_excel(data, os.path.join(config.output_dir, f"{config.eval_name}_{sid}.xlsx"))

    print(f"\nDone — {len(all_generated)} total questions generated.")


def main():
    parser = argparse.ArgumentParser(description="Concierge QA Remixer")
    parser.add_argument("config_file", help="Path to JSON configuration file")
    args = parser.parse_args()

    with open(args.config_file) as f:
        raw = json.load(f)

    scenario_configs = {}
    for key, val in raw.items():
        if key.endswith("_config") and isinstance(val, dict):
            scenario_configs[key.replace("_config", "").upper()] = val

    config = ConciergeConfig(
        eval_name=raw["eval_name"],
        output_dir=raw["output_dir"],
        scenario_configs=scenario_configs,
    )
    run_concierge(config)


if __name__ == "__main__":
    main()
