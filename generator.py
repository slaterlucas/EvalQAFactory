#!/usr/bin/env python3
"""
Configurable Evaluation QA Generation Pipeline.

This script generates domain-specific question-answer pairs for evaluating
AI assistants. It uses a "ground-truth-first" approach organized by scenarios,
with statistically rigorous sampling via the Coupon Collector Problem.

The pipeline is domain-agnostic: swap in a new config module under configs/
to generate QA pairs for any knowledge domain.

Setup:
1. Place domain data files in data/<your_domain>/.
2. Install dependencies:
   pip install google-generativeai pandas xlsxwriter tqdm backoff python-dotenv
3. Set your LLM API key:
   export GOOGLE_API_KEY="your-api-key-here"
"""

import os
import json
import random
import time
import argparse
from typing import List, Dict, Any, Optional
import concurrent.futures
import importlib

import google.generativeai as genai
import pandas as pd
from tqdm import tqdm
import backoff
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Global configuration — populated at runtime from the selected config module
# ---------------------------------------------------------------------------

SCENARIO_CONFIG: Dict = {}
OUTPUT_BASE_NAME: Optional[str] = None
DATA_SUBDIRECTORY: Optional[str] = None
DOMAIN_NAME: Optional[str] = None
RECORD_TYPE: Optional[str] = None
CONFIG_MODULE = None

GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_CONFIDENCE_LEVEL = 0.90
BATCH_SIZE = 10

# ---------------------------------------------------------------------------
# Data Path Resolution
# ---------------------------------------------------------------------------


def get_data_file_paths(scenario_id: str) -> List[str]:
    """Resolve data file path(s) for a scenario using the config mapping."""
    if not CONFIG_MODULE:
        raise RuntimeError("CONFIG_MODULE not initialized")

    scenario_mapping = getattr(CONFIG_MODULE, "SCENARIO_DATA_MAPPING", {})

    filename_or_list = scenario_mapping.get(scenario_id)
    if not filename_or_list:
        base_scenario = scenario_id.split("_")[0] if "_" in scenario_id else scenario_id
        filename_or_list = scenario_mapping.get(base_scenario)
        if not filename_or_list:
            raise KeyError(
                f"No data mapping for scenario '{scenario_id}'. "
                f"Available: {list(scenario_mapping.keys())}"
            )

    if isinstance(filename_or_list, list):
        return [
            os.path.join(os.path.dirname(__file__), "data", DATA_SUBDIRECTORY, fn)
            for fn in filename_or_list
        ]
    return [
        os.path.join(os.path.dirname(__file__), "data", DATA_SUBDIRECTORY, filename_or_list)
    ]


def get_data_file_path(scenario_id: str) -> str:
    """Return the first data file path (backward-compat helper)."""
    return get_data_file_paths(scenario_id)[0]


def get_data_for_intent(intent_func, scenario_id: str):
    """Load the data file appropriate for a specific intent function."""
    if not CONFIG_MODULE:
        raise RuntimeError("CONFIG_MODULE not initialized")

    intent_name = intent_func.__name__
    intent_data_mapping = getattr(CONFIG_MODULE, "INTENT_DATA_MAPPING", {})

    required = intent_data_mapping.get(intent_name)

    if not required:
        scenario_mapping = getattr(CONFIG_MODULE, "SCENARIO_DATA_MAPPING", {})
        base = scenario_id.split("_")[0] if "_" in scenario_id else scenario_id
        required = scenario_mapping.get(base)
        if not required:
            raise KeyError(
                f"No data mapping for intent '{intent_name}' or scenario '{base}'."
            )

    if isinstance(required, list):
        paths = [
            os.path.join(os.path.dirname(__file__), "data", DATA_SUBDIRECTORY, fn)
            for fn in required
        ]
        return load_combined_data(paths)

    path = os.path.join(os.path.dirname(__file__), "data", DATA_SUBDIRECTORY, required)
    return load_data(path)


# ---------------------------------------------------------------------------
# Record Lookup Helpers
# ---------------------------------------------------------------------------


def find_record_by_name(target_name: str, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find a record by name across varying data-structure formats."""
    for record in records:
        candidates = [
            record.get("display_name"),
            record.get("searched_name"),
            (record.get("response_data", {}).get("items", [{}])[0].get("DisplayName")
             if record.get("response_data", {}).get("items") else None),
        ]
        if target_name in [c for c in candidates if c]:
            return record
    return None


# ---------------------------------------------------------------------------
# Conversation Planning
# ---------------------------------------------------------------------------


def create_person_focused_plan(allowed_intents: List, target_count: int) -> List[Dict]:
    """Build a conversation plan that groups 2-4 questions per entity."""
    plan: List[Dict] = []
    questions_per_person = random.randint(2, 4)
    current_count = 0
    person_id: Optional[str] = None
    used_intents: set = set()

    for i in range(target_count):
        is_new = current_count == 0 or current_count >= questions_per_person
        if is_new:
            current_count = 1
            questions_per_person = random.randint(2, 4)
            person_id = f"person_{i}"
            used_intents = set()
        else:
            current_count += 1

        available = [fn for fn in allowed_intents if fn not in used_intents] or allowed_intents
        intent_func = random.choice(available)
        used_intents.add(intent_func)

        plan.append({
            "intent_func": intent_func,
            "conversation_type": "new_person" if is_new else "same_person",
            "person_question_number": current_count,
            "questions_remaining_for_person": questions_per_person - current_count,
            "person_id": person_id,
            "is_new_person_start": is_new,
        })

    return plan


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSON data file, always returning a list."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        raise ValueError(f"Unexpected data format in {file_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")


def load_combined_data(file_paths: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Load multiple files keyed by scenario name."""
    combined: Dict[str, List[Dict[str, Any]]] = {}
    for fp in file_paths:
        key = os.path.basename(fp).replace(".json", "")
        try:
            combined[key] = load_data(fp)
            print(f"  Loaded {len(combined[key])} records from {key}")
        except Exception as e:
            print(f"  Could not load {key}: {e}")
            combined[key] = []
    return combined


# ---------------------------------------------------------------------------
# LLM Interaction (Google Gemini)
# ---------------------------------------------------------------------------

gemini_model = None


def initialize_gemini():
    """Initialize the Gemini generative model."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set. Please export it or add to .env")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    print(f"LLM model '{GEMINI_MODEL}' initialized.")
    return model


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def generate_with_llm(prompt: str) -> Optional[Dict[str, Any]]:
    """Call the LLM and parse JSON from the response."""
    global gemini_model
    if gemini_model is None:
        gemini_model = initialize_gemini()
    try:
        response = gemini_model.generate_content(prompt)
        cleaned = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        print(f"Warning: could not parse LLM response — {e}")
        return None
    except Exception as e:
        print(f"LLM error: {e}")
        return None


def build_question_prompt(intent_details: Dict[str, Any], is_conversational: bool = False) -> str:
    """Create a prompt for question generation from config-defined template."""
    if not CONFIG_MODULE:
        raise RuntimeError("CONFIG_MODULE not initialized")

    template = getattr(CONFIG_MODULE, "DOMAIN_PROMPT_TEMPLATE", "")
    if not template:
        raise RuntimeError("DOMAIN_PROMPT_TEMPLATE not found in config module")

    guidance = (
        getattr(CONFIG_MODULE, "CONVERSATIONAL_GUIDANCE", "")
        if is_conversational
        else getattr(CONFIG_MODULE, "SINGLE_INTENT_GUIDANCE", "")
    )

    return template.format(
        topic=intent_details["topic"],
        answer=intent_details["answer"],
        context_facts=intent_details.get("context_facts", "N/A"),
        conversational_guidance=guidance,
    )


def build_conversational_prompt(
    intent_details: Dict[str, Any],
    conversation_history: List[Dict],
    step: int,
    total_steps: int,
    conversation_context: Optional[Dict] = None,
) -> str:
    """Build a multi-turn conversational prompt with context."""
    recent_history = ""
    if conversation_history:
        recent_history = "\n".join(
            f"User: {ex.get('query', '')}\nAssistant: {ex.get('expected_answer', '')}"
            for ex in conversation_history[-2:]
        )
        recent_history = f"\nRecent conversation:\n{recent_history}\n"

    transition_styles = [
        "Ask directly without any acknowledgment or thanks.",
        "Jump straight to your question.",
        "Use a natural transition like 'Also', 'By the way', or 'Quick question'.",
        "Reference something from earlier in the conversation.",
        "Sound like you're having a casual workplace conversation.",
    ]

    if conversation_context is None:
        conversation_context = {"conversation_type": "same_person"}

    conv_type = conversation_context.get("conversation_type", "same_person")
    q_num = conversation_context.get("person_question_number", 1)

    if step == 1:
        style = "Start the conversation naturally — just ask your question directly."
    elif conv_type == "new_person":
        style = "You're switching to ask about a different entity. Transition naturally."
    elif q_num == 2:
        style = "You're asking a follow-up about the same entity. Continue naturally."
    elif q_num >= 3:
        style = "Continue asking about the same entity. Keep the conversation flowing."
    else:
        style = random.choice(transition_styles)

    return f"""You are a USER asking questions to a domain-specific assistant. Generate the next USER question.

{recent_history}

You need to ask about: {intent_details['topic']}
The assistant should answer: "{intent_details['answer']}"

CONVERSATION CONTEXT:
- {style}

GUIDELINES:
- Sound natural and human-like
- Vary your conversation style
- Be conversational and professional

Return only JSON: {{"question": "your question here"}}"""


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def generate_natural_answer(question: str, raw_answer: str, context_facts: str) -> Optional[str]:
    """Use the LLM to convert a structured answer into natural language."""
    prompt = f"""You are a helpful assistant providing domain-specific information.

Question: {question}
Raw Answer: {raw_answer}
Context: {context_facts}

Convert the raw answer into a natural, helpful response. The response should:
- Be direct and informative (avoid formulaic openings like "Certainly!" or "Of course!")
- Include the specific data from the raw answer
- Sound like a knowledgeable colleague sharing information
- Be concise but complete (1-2 sentences)
- Vary your response style

Return ONLY valid JSON: {{"natural_answer": "your response here"}}"""

    global gemini_model
    if gemini_model is None:
        gemini_model = initialize_gemini()
    try:
        response = gemini_model.generate_content(prompt)
        text = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        answer = result.get("natural_answer")
        if answer and len(answer.split()) > 3:
            return answer
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Batch Generation
# ---------------------------------------------------------------------------


def generate_question_batch(prompts: List[str]) -> List[Optional[Dict[str, Any]]]:
    """Generate a batch of questions in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
        futures = {pool.submit(generate_with_llm, p): p for p in prompts}
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result(timeout=30))
            except Exception as e:
                print(f"Batch question failed: {e}")
                results.append(None)
        return results


def generate_answer_batch(batch_data: List[Dict[str, str]]) -> List[Optional[str]]:
    """Generate a batch of natural-language answers in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
        futures = {}
        for d in batch_data:
            f = pool.submit(generate_natural_answer, d["question"], d["raw_answer"], d["context_facts"])
            futures[f] = d
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result(timeout=30))
            except Exception as e:
                print(f"Batch answer failed: {e}")
                results.append(None)
        return results


def generate_batch_scenario_questions(
    scenario_id: str,
    scenario_config: Dict,
    record_data: Any,
    allowed_intents: List,
    target_count: int,
    seen_questions: set,
) -> List[Dict]:
    """Generate questions in batches for improved throughput."""
    print(f"  Batch mode (size={BATCH_SIZE})")
    rows: List[Dict] = []
    pbar = tqdm(total=target_count, desc=f"Scenario {scenario_id} (Batch)")

    while len(rows) < target_count:
        remaining = target_count - len(rows)
        batch_size = min(BATCH_SIZE, remaining)

        intent_details_list = []
        prompts = []

        for _ in range(batch_size):
            for _ in range(50):
                try:
                    fn = random.choice(allowed_intents)
                    details = fn(record_data)
                    if details:
                        intent_details_list.append(details)
                        prompts.append(build_question_prompt(details, is_conversational=False))
                        break
                except Exception as e:
                    print(f"  Intent generation error: {e}")

        if not prompts:
            break

        questions = generate_question_batch(prompts)
        answer_batch_data = []
        valid = []

        for details, qr in zip(intent_details_list, questions):
            if qr is None:
                continue
            text = qr.get("question", "").strip()
            if not text or text.lower() in seen_questions:
                continue
            valid.append((details, text))
            answer_batch_data.append({
                "question": text,
                "raw_answer": details["answer"],
                "context_facts": details["context_facts"],
            })

        if answer_batch_data:
            answers = generate_answer_batch(answer_batch_data)
            for (details, text), answer in zip(valid, answers):
                if answer is None:
                    answer = _fallback_answer(details["answer"])
                seen_questions.add(text.lower())
                num = len(rows) + 1
                rows.append({
                    "eval_name": OUTPUT_BASE_NAME,
                    "eval_scenario": scenario_id,
                    "query_num": f"{OUTPUT_BASE_NAME}_{scenario_id}_{num:03d}",
                    "query": text,
                    "expected_answer": answer,
                })
                pbar.write(f"  - {text}")
                pbar.update(1)
                if len(rows) >= target_count:
                    break

        time.sleep(0.5)

    pbar.close()
    return rows


# ---------------------------------------------------------------------------
# Scenario Generation (single + conversational)
# ---------------------------------------------------------------------------


def _fallback_answer(raw: str) -> str:
    """Create a simple fallback when LLM answer generation fails."""
    if raw.startswith("$"):
        return f"The amount is {raw}."
    if raw.isdigit():
        return f"The answer is {raw}."
    return f"Based on the available data, {raw.lower()}."


def generate_conversational_scenario(scenario_id: str, scenario_config: Dict) -> List[Dict]:
    """Generate a flowing multi-turn conversation for a scenario."""
    print(f"\n  Generating conversational flow for {scenario_config['name']}")

    rows: List[Dict] = []
    history: List[Dict] = []
    allowed = scenario_config["intent_functions"]
    target = scenario_config["question_count"]

    single_person_scenarios = getattr(CONFIG_MODULE, "SINGLE_PERSON_CONVERSATION_SCENARIOS", [])

    if scenario_id in single_person_scenarios:
        overview_map = getattr(CONFIG_MODULE, "OVERVIEW_INTENT_BY_SCENARIO", {})
        overview_name = overview_map.get(scenario_id)
        overview_func = next((fn for fn in allowed if fn.__name__ == overview_name), allowed[0])
        remaining = [f for f in allowed if f is not overview_func]
        random.shuffle(remaining)
        ordered = ([overview_func] + remaining)[:target]
        plan = [
            {
                "intent_func": fn,
                "conversation_type": "new_person" if i == 0 else "same_person",
                "person_question_number": i + 1,
                "questions_remaining_for_person": max(0, target - i - 1),
                "person_id": "me",
                "is_new_person_start": i == 0,
            }
            for i, fn in enumerate(ordered)
        ]
    else:
        plan = create_person_focused_plan(allowed, target)

    pbar = tqdm(total=target, desc=f"Conversation {scenario_id}")
    person_map: Dict[str, Dict] = {}
    ok = 0
    attempt = 0

    while ok < target and attempt <= target * 2:
        try:
            step = ok + 1
            if step <= len(plan):
                ps = plan[step - 1]
                intent_func = ps["intent_func"]
                ctx = ps
            else:
                intent_func = random.choice(allowed)
                ctx = {"conversation_type": "same_person"}

            record_data = get_data_for_intent(intent_func, scenario_id)
            pid = ctx.get("person_id", "default")
            is_new = ctx.get("is_new_person_start", False)

            if is_new or pid not in person_map:
                sel = random.choice(record_data)
                person_map[pid] = {
                    "record": sel,
                    "name": sel.get("display_name", sel.get("searched_name", "Unknown")),
                }

            target_name = person_map[pid]["name"]
            match = find_record_by_name(target_name, record_data)
            if match:
                details = intent_func([match])
            else:
                new_rec = random.choice(record_data)
                details = intent_func([new_rec])
                person_map[pid] = {
                    "record": new_rec,
                    "name": new_rec.get("display_name", new_rec.get("searched_name", "Unknown")),
                }

            if not details:
                intent_func = random.choice(allowed)
                record_data = get_data_for_intent(intent_func, scenario_id)
                details = intent_func(record_data)
                if not details:
                    attempt += 1
                    continue

            prompt = build_conversational_prompt(details, history, step, target, ctx)
            variant = generate_with_llm(prompt)
            if variant is None:
                attempt += 1
                time.sleep(1)
                continue

            q_text = variant.get("question", "").strip()
            if not q_text:
                attempt += 1
                continue

            nat_answer = generate_natural_answer(q_text, details["answer"], details["context_facts"])
            if nat_answer is None:
                nat_answer = _fallback_answer(details["answer"])

            row = {
                "eval_name": OUTPUT_BASE_NAME,
                "eval_scenario": scenario_id,
                "query_num": f"{OUTPUT_BASE_NAME}_{scenario_id}_{step:03d}",
                "query": q_text,
                "expected_answer": nat_answer,
            }
            rows.append(row)
            history.append(row)
            pbar.write(f"  - Exchange {step}: {q_text}")
            pbar.update(1)
            ok += 1
            attempt += 1

        except Exception as e:
            print(f"  Error in attempt {attempt}: {e}")
            attempt += 1
            time.sleep(2)

    pbar.close()
    return rows


def generate_scenario_questions(
    scenario_id: str,
    scenario_config: Dict,
    conversational: bool = False,
    batch_mode: bool = False,
) -> List[Dict]:
    """Top-level dispatcher for generating questions for one scenario."""
    if conversational:
        return generate_conversational_scenario(scenario_id, scenario_config)

    print(f"\n  Generating questions for {scenario_config['name']}")
    print(f"   Target: {scenario_config['question_count']} questions")

    paths = get_data_file_paths(scenario_id)
    if len(paths) == 1:
        record_data = load_data(paths[0])
        print(f"   Data source: {paths[0]} ({len(record_data)} records)")
    else:
        record_data = load_combined_data(paths)
        total = sum(len(v) for v in record_data.values())
        print(f"   Data sources: {len(paths)} files ({total} total records)")

    rows: List[Dict] = []
    seen: set = set()
    allowed = scenario_config["intent_functions"]
    target = scenario_config["question_count"]

    if batch_mode:
        return generate_batch_scenario_questions(
            scenario_id, scenario_config, record_data, allowed, target, seen
        )

    pbar = tqdm(total=target, desc=f"Scenario {scenario_id}")

    while len(rows) < target:
        try:
            fn = random.choice(allowed)
            details = fn(record_data)
            if not details:
                continue

            prompt = build_question_prompt(details, is_conversational=conversational)
            variant = generate_with_llm(prompt)
            if variant is None:
                continue

            q_text = variant.get("question", "").strip()
            if not q_text or q_text.lower() in seen:
                continue

            nat_answer = generate_natural_answer(q_text, details["answer"], details["context_facts"])
            if nat_answer is None:
                nat_answer = _fallback_answer(details["answer"])

            seen.add(q_text.lower())
            num = len(rows) + 1
            rows.append({
                "eval_name": OUTPUT_BASE_NAME,
                "eval_scenario": scenario_id,
                "query_num": f"{OUTPUT_BASE_NAME}_{scenario_id}_{num:03d}",
                "query": q_text,
                "expected_answer": nat_answer,
            })
            pbar.write(f"  - Generated: {q_text}")
            pbar.update(1)

        except Exception as e:
            print(f"Error generating scenario {scenario_id}: {e}")
            time.sleep(2)

    pbar.close()
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_organized_output(all_rows: List[Dict], scenarios_config: Dict, conversational: bool = False):
    """Save results as formatted Excel files, one per scenario."""
    if not all_rows:
        print("\nNo questions generated.")
        return

    df = pd.DataFrame(all_rows)
    saved: List[str] = []
    label = "conversation exchanges" if conversational else "questions"

    for sid, cfg in scenarios_config.items():
        sdf = df[df["eval_scenario"] == sid]
        if sdf.empty:
            continue

        base = sid.split("_")[0] if "_" in sid else sid
        tag = cfg.get("tag", "")
        env_tag = cfg.get("env_tag", "")

        subdir = f"{env_tag}_{DATA_SUBDIRECTORY.replace(f'_{env_tag}', '')}" if env_tag else DATA_SUBDIRECTORY
        suffix = f"{base}_{tag}" if tag else base
        clean_name = OUTPUT_BASE_NAME.replace(f"_{env_tag}", "") if env_tag else OUTPUT_BASE_NAME
        out_path = os.path.join(os.path.dirname(__file__), "output", subdir, f"{clean_name}_{suffix}.xlsx")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            sdf.to_excel(writer, sheet_name="Questions", index=False)
            wb = writer.book
            ws = writer.sheets["Questions"]
            hdr_fmt = wb.add_format({"bold": True, "text_wrap": True, "valign": "top"})
            cell_fmt = wb.add_format({"text_wrap": True, "valign": "top"})
            widths = {"A": 15, "B": 12, "C": 20, "D": 50, "E": 60}
            for col, w in widths.items():
                ws.set_column(f"{col}:{col}", w, cell_fmt)
            for ci, cn in enumerate(sdf.columns):
                ws.write(0, ci, cn, hdr_fmt)
            ws.set_default_row(30)
            ws.set_row(0, 25)

        saved.append(out_path)
        print(f"  Saved {len(sdf)} {label} for {sid} -> {out_path}")

    print(f"\nDataset statistics:")
    print(f"  Total {label}: {len(df)}")
    print(f"  Files created: {len(saved)}")
    for sid, cfg in scenarios_config.items():
        cnt = len(df[df["eval_scenario"] == sid])
        print(f"  {cfg.get('name', sid)}: {cnt} {label}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    global OUTPUT_BASE_NAME, DATA_SUBDIRECTORY, DOMAIN_NAME, RECORD_TYPE, CONFIG_MODULE

    parser = argparse.ArgumentParser(description="Generate scenario-based evaluation QA pairs")
    parser.add_argument("--scenarios", nargs="+", help="Specific scenarios to run (e.g. S01 S02)")
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_CONFIDENCE_LEVEL,
        help=f"Statistical confidence level 0-1 (default {DEFAULT_CONFIDENCE_LEVEL})",
    )
    parser.add_argument("--conversational", action="store_true", help="Generate multi-turn conversations")
    parser.add_argument(
        "--config", default="example_config",
        help="Config module name inside configs/ (default: example_config)",
    )
    parser.add_argument("--override-question-count", type=int, help="Override question count for testing")
    parser.add_argument("--batch", action="store_true", help="Enable parallel batch generation")

    args = parser.parse_args()

    if not 0 < args.confidence < 1:
        print(f"Invalid confidence level: {args.confidence}")
        return

    try:
        config_module = importlib.import_module(f"configs.{args.config}")
    except ModuleNotFoundError:
        print(f"Config module 'configs.{args.config}' not found.")
        return

    if not hasattr(config_module, "build_scenario_config"):
        print(f"Config '{args.config}' lacks build_scenario_config().")
        return

    CONFIG_MODULE = config_module
    OUTPUT_BASE_NAME = getattr(config_module, "OUTPUT_BASE_NAME", "EVAL")
    DATA_SUBDIRECTORY = getattr(config_module, "DATA_SUBDIRECTORY", "default")
    DOMAIN_NAME = getattr(config_module, "DOMAIN_NAME", "Default Domain")
    RECORD_TYPE = getattr(config_module, "RECORD_TYPE", "records")

    # Optional environment-tag suffix for multi-env segregation
    try:
        from utils import env_suffix, ENV_TAG
    except ImportError:
        env_suffix = lambda: ""
        ENV_TAG = ""

    OUTPUT_BASE_NAME += env_suffix()
    DATA_SUBDIRECTORY += env_suffix()

    print(f"Starting {DOMAIN_NAME} QA generation pipeline...")
    print(f"Statistical confidence: {args.confidence * 100:.0f}%")
    if args.conversational:
        print("Mode: conversational (multi-turn)")
    else:
        batch_label = " (batch)" if args.batch else ""
        print(f"Mode: single-turn{batch_label}")

    all_scenarios = config_module.build_scenario_config(confidence_level=args.confidence)

    if args.override_question_count:
        print(f"Overriding question count to {args.override_question_count}")
        for cfg in all_scenarios.values():
            cfg["question_count"] = args.override_question_count

    if ENV_TAG:
        for cfg in all_scenarios.values():
            cfg["env_tag"] = ENV_TAG

    if args.scenarios:
        to_run = {k: v for k, v in all_scenarios.items() if k in args.scenarios}
        if not to_run:
            print(f"No matching scenarios. Available: {list(all_scenarios.keys())}")
            return
    else:
        to_run = all_scenarios

    label = "conversation exchanges" if args.conversational else "questions"
    print(f"Running {len(to_run)} scenarios:")
    for sid, cfg in to_run.items():
        print(f"  - {sid}: {cfg['name']} ({cfg['question_count']} {label})")

    all_rows: List[Dict] = []
    for sid, cfg in to_run.items():
        rows = generate_scenario_questions(sid, cfg, conversational=args.conversational, batch_mode=args.batch)
        all_rows.extend(rows)

    save_organized_output(all_rows, to_run, args.conversational)


if __name__ == "__main__":
    main()
