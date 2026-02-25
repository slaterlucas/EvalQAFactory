# Eval QA Factory

A configurable pipeline for generating statistically rigorous evaluation datasets for domain-specific AI assistants. Given structured ground-truth data, the system produces natural-language question-answer pairs organized by scenario, suitable for benchmarking chatbot accuracy and coverage.

## Key Design Decisions

- **Ground-truth-first generation** — every QA pair starts from verified data, guaranteeing answer correctness. An LLM (Google Gemini) is used only to rephrase questions naturally, not to invent answers.
- **Coupon Collector sampling** — question counts are derived from the [Coupon Collector Problem](https://en.wikipedia.org/wiki/Coupon_collector%27s_problem) so that, at a configurable confidence level (default 90%), every intent function is exercised at least once.
- **Pluggable domain configs** — adding a new domain (compensation, benefits, talent, etc.) requires only a new config module; the core pipeline is domain-agnostic.
- **Multi-turn conversations** — an optional conversational mode generates flowing, person-focused dialogue with realistic transitions and follow-ups.
- **Parallel batch generation** — a `--batch` flag enables concurrent LLM calls for higher throughput.
- **Environment-tag segregation** — an `ENV_TAG` environment variable routes data and output into per-environment directories, supporting parallel evaluation across dev/staging/prod.
- **Concierge remixer** — a secondary generator (`concierge_generator.py`) creates cross-domain "generalist" evaluation sets by sampling and combining outputs from multiple domain-specific runs for the purposes of multiagent.

## Architecture

```
eval_qa_factory/
├── generator.py              # Core pipeline — single-turn & conversational
├── concierge_generator.py    # Cross-domain QA remixer
├── configs/
│   └── example_config.py     # Example domain config (employee directory)
├── utils/
│   ├── statistical_rigor.py  # Coupon Collector calculations
│   ├── data_discovery.py     # Data-richness scoring template
│   └── __init__.py           # ENV_TAG helper
├── data/
│   └── example_domain/       # Sample JSON data files
│       ├── S01_search_results.json
│       ├── S02_assignment_details.json
│       ├── S03_contact_details.json
│       └── S04_manager_contacts.json
└── output/                   # Generated Excel files (git-ignored)
```

### How It Works

1. **Config module** defines *intent functions* — small Python functions that pick a random record from structured data and return a `{topic, answer, context_facts}` dict.

2. **Scenarios** group intents. For example, "Employee Search" groups name-lookup, not-found, and team-membership intents. The pipeline calculates how many questions each scenario needs to meet the target confidence.

3. **Generator loop** randomly selects an intent, extracts ground-truth, sends a prompt to the LLM to rephrase the question naturally, then sends another prompt to generate a natural-sounding answer. Results are written to formatted Excel files.

4. **Conversational mode** (`--conversational`) builds a person-focused plan — 2-4 questions per entity with smooth transitions — producing multi-turn dialogues instead of isolated QA pairs.

5. **Batch mode** (`--batch`) parallelizes LLM calls for both question generation and answer rephrasing.

## Quick Start

### 1. Install

```bash
pip install google-generativeai pandas xlsxwriter openpyxl tqdm backoff python-dotenv
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

### 2. Configure

```bash
cp env_example .env
# Edit .env and add your Google API key
```

### 3. Run

```bash
# Generate single-turn questions using the example config
python generator.py --config example_config

# Generate with 95% confidence
python generator.py --config example_config --confidence 0.95

# Multi-turn conversational mode
python generator.py --config example_config --conversational

# Parallel batch mode (faster)
python generator.py --config example_config --batch

# Specific scenarios only
python generator.py --config example_config --scenarios S01 S03

# Override question count for quick testing
python generator.py --config example_config --override-question-count 5
```

### 4. Output

Generated Excel files land in `output/<subdirectory>/`:

```
output/
└── example_domain/
    ├── EVAL_EMPLOYEE_DIR_S01.xlsx
    ├── EVAL_EMPLOYEE_DIR_S02.xlsx
    └── ...
```

Each file contains columns: `eval_name`, `eval_scenario`, `query_num`, `query`, `expected_answer`.

## Adding a New Domain

1. **Prepare data** — create JSON files in `data/<your_domain>/` matching the structures your intent functions expect.

2. **Write a config** — copy `configs/example_config.py` and customize:
   - `OUTPUT_BASE_NAME`, `DATA_SUBDIRECTORY`, `DOMAIN_NAME`
   - `SCENARIO_DATA_MAPPING` and `INTENT_DATA_MAPPING`
   - `DOMAIN_PROMPT_TEMPLATE`
   - Intent functions that extract QA pairs from your data structures

3. **Run** — `python generator.py --config your_new_config`

## Data Discovery

When working with a new data source, use the discovery tool to identify records with the richest data across domains:

```bash
python utils/data_discovery.py --limit 100
```

This scores records by data completeness and saves curated lists for optimal question generation. Customize the scoring functions in `data_discovery.py` for your domain.

## Concierge Remixer

After generating domain-specific outputs, create cross-domain "generalist" evaluation sets:

```bash
python concierge_generator.py config.json
```

The JSON config specifies which source spreadsheets to sample from and which pairs to combine. Sampling scenarios require no LLM; combination scenarios use the LLM to merge two questions into one.

## Multi-Environment Support

Set `ENV_TAG` to segregate data and outputs per environment:

```bash
ENV_TAG=STAGING python generator.py --config example_config
```

Outputs go to `output/STAGING_example_domain/`. This supports running evaluations against multiple environments in parallel.

## Statistical Methodology

The pipeline uses the **Coupon Collector Problem** to determine sample sizes. Given *n* intent functions and a target confidence *p*, the required number of random samples *k* is:

```
k ≈ n × (ln(n) + γ + ln(-ln(1 − p)))
```

where γ ≈ 0.5772 is the Euler–Mascheroni constant.

For example, with 7 intents at 90% confidence → 24 questions; with 16 intents at 90% → 70 questions.

This ensures complete intent coverage without requiring exhaustive enumeration.

## Requirements

- Python 3.10+
- Google Generative AI API key ([get one here](https://makersuite.google.com/app/apikey))
- Structured domain data in JSON format
