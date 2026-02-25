#!/usr/bin/env python3
"""
Example Domain Configuration — Employee Directory Lookup.

Demonstrates how to define:
1. Intent functions that extract ground-truth QA pairs from structured data
2. Scenario configurations grouping intents into evaluation categories
3. Data-file mappings so the generator knows which JSON to load per scenario

To add a new domain, copy this file, rename it, and adjust the intent
functions and data mappings for your use case.
"""

from typing import List, Dict, Any, Optional
import random

from utils.statistical_rigor import calculate_required_questions, validate_scenario_rigor

# ---------------------------------------------------------------------------
# Domain Configuration
# ---------------------------------------------------------------------------

OUTPUT_BASE_NAME = "EVAL_EMPLOYEE_DIR"
DATA_SUBDIRECTORY = "example_domain"
DOMAIN_NAME = "Employee Directory"
RECORD_TYPE = "employee records"

# ---------------------------------------------------------------------------
# Data File Mappings
# ---------------------------------------------------------------------------

SCENARIO_DATA_MAPPING = {
    "S01": "S01_search_results.json",
    "S02": "S02_assignment_details.json",
    "S03": "S03_contact_details.json",
    "S04": "S04_manager_contacts.json",
    "S05": [
        "S01_search_results.json",
        "S02_assignment_details.json",
        "S03_contact_details.json",
    ],
}

INTENT_DATA_MAPPING = {
    "intent_find_employee_by_name": "S01_search_results.json",
    "intent_employee_not_found": "S01_search_results.json",
    "intent_multiple_results": "S01_search_results.json",
    "intent_check_team_membership": "S01_search_results.json",
    "intent_get_job_title": "S02_assignment_details.json",
    "intent_get_department": "S02_assignment_details.json",
    "intent_get_start_date": "S02_assignment_details.json",
    "intent_get_manager_name": "S02_assignment_details.json",
    "intent_get_location": "S02_assignment_details.json",
    "intent_get_employment_type": "S02_assignment_details.json",
    "intent_get_length_of_service": "S02_assignment_details.json",
    "intent_get_work_phone": "S03_contact_details.json",
    "intent_get_work_email": "S03_contact_details.json",
    "intent_contact_not_found": "S03_contact_details.json",
    "intent_get_manager_email": "S04_manager_contacts.json",
    "intent_get_manager_phone": "S04_manager_contacts.json",
    "intent_compare_departments": "S02_assignment_details.json",
    "intent_compare_locations": "S02_assignment_details.json",
}

# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------

DOMAIN_PROMPT_TEMPLATE = """
You are a creative assistant generating training data for an employee directory chatbot.

A user is asking a question about finding employees or getting employee information.
The topic is: {topic}
The exact answer you must target: "{answer}"
Additional context for realism: {context_facts}

Generate 1 natural-sounding question the user could ask to obtain precisely that answer.
- Use varied vocabulary and sentence structures.
- Focus on employee lookup, searching for people, or getting contact details.
- Make it sound conversational.

{conversational_guidance}

IMPORTANT: Return ONLY a valid JSON object (no markdown):
{{"question": "..."}}
"""

SINGLE_INTENT_GUIDANCE = ""
CONVERSATIONAL_GUIDANCE = "Make the question flow naturally as part of a conversation."

# ---------------------------------------------------------------------------
# Helper: extract data for a specific scenario from single/combined formats
# ---------------------------------------------------------------------------


def _extract(data, scenario_key: Optional[str] = None):
    """Handle both single-file list and combined-dict data structures."""
    if isinstance(data, dict):
        if scenario_key and scenario_key in data:
            return data[scenario_key]
        combined = []
        for v in data.values():
            if isinstance(v, list):
                combined.extend(v)
        return combined
    return data


# ---------------------------------------------------------------------------
# S01 — Employee Search Intents
# ---------------------------------------------------------------------------


def intent_find_employee_by_name(records) -> Optional[Dict]:
    """Successful employee lookup by name."""
    search_data = _extract(records, "S01")
    if not search_data:
        return None
    rec = random.choice(search_data)
    info = rec["response_data"]["items"][0]
    first = info["DisplayName"].split()[0]
    return {
        "topic": f"Looking for employee by first name: {first}",
        "answer": (
            f"I found {info['DisplayName']}.\n"
            f"- Person Number: {info['PersonNumber']}\n"
            f"- Email: {info['WorkEmail']}\n"
            f"- Title: {info['BusinessTitle']}\n"
            f"- Manager: {info['ManagerDisplayName']}"
        ),
        "context_facts": f"Employee {info['PersonNumber']} found in directory",
    }


def intent_employee_not_found(records) -> Optional[Dict]:
    """Search for a name that doesn't exist."""
    fake = random.choice(["Alex Rutherford", "Morgan Whitfield", "Sam Nakamura"])
    return {
        "topic": f"Searching for non-existent employee: {fake}",
        "answer": f"I couldn't find any employee named {fake}. Please check the spelling or try a different name.",
        "context_facts": f"No match for '{fake}' in directory",
    }


def intent_multiple_results(records) -> Optional[Dict]:
    """Search returning multiple matches."""
    return {
        "topic": "Searching for a common name that returns multiple results",
        "answer": (
            'I found two employees matching "Alex":\n\n'
            "Alex Chen — Software Engineer (Active)\n"
            "Alex Rivera — Product Designer (Active)\n\n"
            "Which Alex are you looking for?"
        ),
        "context_facts": "Multiple matches for common first name",
    }


def intent_check_team_membership(records) -> Optional[Dict]:
    """Check if someone is on the user's team."""
    search_data = _extract(records, "S01")
    if not search_data:
        return None
    rec = random.choice(search_data)
    info = rec["response_data"]["items"][0]
    return {
        "topic": f"Checking if {info['DisplayName'].split()[0]} is on the team",
        "answer": f"Yes, {info['DisplayName']} is on your team as a {info['BusinessTitle']}.",
        "context_facts": f"Team membership check for {info['DisplayName']}",
    }


# ---------------------------------------------------------------------------
# S02 — Assignment / Employment Details Intents
# ---------------------------------------------------------------------------


def intent_get_job_title(records) -> Optional[Dict]:
    """Get an employee's job title."""
    data = _extract(records, "S02")
    if not data:
        return None
    rec = random.choice(data)
    ad = rec["assignment_details"]
    first = rec["display_name"].split()[0]
    return {
        "topic": f"Getting {first}'s job title",
        "answer": f"{first}'s title is {ad['AssignmentName']}.",
        "context_facts": f"Assignment details for {rec['display_name']}",
    }


def intent_get_department(records) -> Optional[Dict]:
    """Get an employee's department."""
    data = _extract(records, "S02")
    if not data:
        return None
    rec = random.choice(data)
    ad = rec["assignment_details"]
    first = rec["display_name"].split()[0]
    return {
        "topic": f"Getting {first}'s department",
        "answer": f"{first} works in the {ad['DepartmentName']} department.",
        "context_facts": f"Department info for {rec['display_name']}",
    }


def intent_get_start_date(records) -> Optional[Dict]:
    """Get an employee's start date."""
    data = _extract(records, "S02")
    if not data:
        return None
    rec = random.choice(data)
    ad = rec["assignment_details"]
    first = rec["display_name"].split()[0]
    return {
        "topic": f"Getting {first}'s start date",
        "answer": f"{first} started on {ad['StartDate']}.",
        "context_facts": f"Start date for {rec['display_name']}",
    }


def intent_get_manager_name(records) -> Optional[Dict]:
    """Get an employee's manager."""
    data = _extract(records, "S02")
    if not data:
        return None
    rec = random.choice(data)
    ad = rec["assignment_details"]
    first = rec["display_name"].split()[0]
    return {
        "topic": f"Getting {first}'s manager",
        "answer": f"{first}'s manager is {ad['ManagerName']}.",
        "context_facts": f"Manager info for {rec['display_name']}",
    }


def intent_get_location(records) -> Optional[Dict]:
    """Get an employee's work location."""
    data = _extract(records, "S02")
    if not data:
        return None
    rec = random.choice(data)
    ad = rec["assignment_details"]
    first = rec["display_name"].split()[0]
    loc = ad.get("LocationSingleLineAddress", ad.get("LocationName", "Unknown"))
    return {
        "topic": f"Getting {first}'s work location",
        "answer": f"{first} works at {loc}.",
        "context_facts": f"Location info for {rec['display_name']}",
    }


def intent_get_employment_type(records) -> Optional[Dict]:
    """Get full-time / part-time status."""
    data = _extract(records, "S02")
    if not data:
        return None
    rec = random.choice(data)
    ad = rec["assignment_details"]
    first = rec["display_name"].split()[0]
    status = ad.get("FullPartTime", "FULL_TIME")
    text = "full-time" if status == "FULL_TIME" else "part-time"
    return {
        "topic": f"Getting {first}'s employment type",
        "answer": f"{first} works {text}.",
        "context_facts": f"Employment type for {rec['display_name']}",
    }


def intent_get_length_of_service(records) -> Optional[Dict]:
    """Get how long someone has been with the company."""
    data = _extract(records, "S02")
    if not data:
        return None
    rec = random.choice(data)
    ad = rec["assignment_details"]
    first = rec["display_name"].split()[0]
    years = ad["LengthOfServiceYears"]
    months = ad["LengthOfServiceMonths"]
    return {
        "topic": f"Getting {first}'s tenure",
        "answer": f"{first} has been with the company for {years} years and {months} months.",
        "context_facts": f"Service duration for {rec['display_name']}",
    }


# ---------------------------------------------------------------------------
# S03 — Contact Details Intents
# ---------------------------------------------------------------------------


def intent_get_work_phone(records) -> Optional[Dict]:
    """Get an employee's work phone."""
    data = _extract(records, "S03")
    if not data:
        return None
    rec = random.choice(data)
    first = rec["display_name"].split()[0]
    phones = rec.get("phones_data", [])
    if phones:
        p = next((ph for ph in phones if ph.get("PrimaryFlag")), phones[0])
        number = f"({p['AreaCode']}) {p['PhoneNumber']}"
        answer = f"{first}'s work phone is {number}."
    else:
        answer = f"{first} doesn't have a work phone on file."
    return {
        "topic": f"Getting {first}'s work phone",
        "answer": answer,
        "context_facts": f"Phone lookup for {rec['display_name']}",
    }


def intent_get_work_email(records) -> Optional[Dict]:
    """Get an employee's work email."""
    data = _extract(records, "S03")
    if not data:
        return None
    rec = random.choice(data)
    first = rec["display_name"].split()[0]
    email = rec.get("email_data", {}).get("WorkEmail")
    if email:
        answer = f"{first}'s email is {email}."
    else:
        answer = f"{first} doesn't have a work email on file."
    return {
        "topic": f"Getting {first}'s work email",
        "answer": answer,
        "context_facts": f"Email lookup for {rec['display_name']}",
    }


def intent_contact_not_found(records) -> Optional[Dict]:
    """Search for contact details of a non-existent person."""
    fake = random.choice(["Jordan West", "Taylor Okafor", "Casey Lin"])
    kind = random.choice(["phone", "email"])
    return {
        "topic": f"Searching for {kind} of non-existent employee: {fake}",
        "answer": f"I couldn't find any {kind} information for {fake}. Please check the spelling.",
        "context_facts": f"No contact data for '{fake}'",
    }


# ---------------------------------------------------------------------------
# S04 — Manager Contact Intents
# ---------------------------------------------------------------------------


def intent_get_manager_email(records) -> Optional[Dict]:
    """Get the email address of an employee's manager."""
    data = _extract(records, "S04")
    if not data:
        return None
    rec = random.choice(data)
    first = rec["searched_name"].split()[0]
    mgr = rec.get("manager_contact_details", [])
    if mgr:
        ed = mgr[0].get("email_data", {})
        name = ed.get("DisplayName", "Unknown")
        email = ed.get("WorkEmail", "")
        if email:
            answer = f"{first}'s manager is {name}, reachable at {email}."
        else:
            answer = f"{first}'s manager is {name}, but no email is on file."
    else:
        answer = f"Manager contact info for {first} is not available."
    return {
        "topic": f"Getting manager email for {first}",
        "answer": answer,
        "context_facts": f"Manager email lookup for {rec['searched_name']}",
    }


def intent_get_manager_phone(records) -> Optional[Dict]:
    """Get the phone number of an employee's manager."""
    data = _extract(records, "S04")
    if not data:
        return None
    rec = random.choice(data)
    first = rec["searched_name"].split()[0]
    mgr = rec.get("manager_contact_details", [])
    if mgr:
        ed = mgr[0].get("email_data", {})
        phones = mgr[0].get("phones_data", [])
        name = ed.get("DisplayName", "Unknown")
        if phones:
            p = phones[0]
            num = f"({p.get('AreaCode', '')}) {p.get('PhoneNumber', '')}"
            answer = f"{first}'s manager is {name}, phone: {num}."
        else:
            answer = f"{first}'s manager is {name}, but no phone is on file."
    else:
        answer = f"Manager contact info for {first} is not available."
    return {
        "topic": f"Getting manager phone for {first}",
        "answer": answer,
        "context_facts": f"Manager phone lookup for {rec['searched_name']}",
    }


# ---------------------------------------------------------------------------
# S05 — Cross-domain Comparison Intents
# ---------------------------------------------------------------------------


def intent_compare_departments(records) -> Optional[Dict]:
    """Compare departments of two employees."""
    data = _extract(records, "S02")
    if not data or len(data) < 2:
        return None
    pair = random.sample(data, 2)
    names = [r["display_name"].split()[0] for r in pair]
    depts = [r["assignment_details"].get("DepartmentName", "Unknown") for r in pair]
    answer = f"{names[0]} works in {depts[0]}. {names[1]} works in {depts[1]}."
    return {
        "topic": f"Comparing departments for {names[0]} and {names[1]}",
        "answer": answer,
        "context_facts": f"Department comparison: {pair[0]['display_name']} vs {pair[1]['display_name']}",
    }


def intent_compare_locations(records) -> Optional[Dict]:
    """Compare work locations of two employees."""
    data = _extract(records, "S02")
    if not data or len(data) < 2:
        return None
    pair = random.sample(data, 2)
    names = [r["display_name"].split()[0] for r in pair]
    locs = [r["assignment_details"].get("LocationName", "Unknown") for r in pair]
    answer = f"{names[0]} works at {locs[0]}. {names[1]} works at {locs[1]}."
    return {
        "topic": f"Comparing locations for {names[0]} and {names[1]}",
        "answer": answer,
        "context_facts": f"Location comparison: {pair[0]['display_name']} vs {pair[1]['display_name']}",
    }


# ---------------------------------------------------------------------------
# Scenario Configuration Builder
# ---------------------------------------------------------------------------


def build_scenario_config(confidence_level: float = 0.90):
    """
    Build scenario configs with question counts derived from the
    Coupon Collector Problem to guarantee the target confidence that
    every intent is sampled at least once.
    """
    scenarios = {
        "S01": {
            "name": "Employee Search",
            "intent_functions": [
                intent_find_employee_by_name,
                intent_employee_not_found,
                intent_multiple_results,
                intent_check_team_membership,
            ],
        },
        "S02": {
            "name": "Assignment & Employment Details",
            "intent_functions": [
                intent_get_job_title,
                intent_get_department,
                intent_get_start_date,
                intent_get_manager_name,
                intent_get_location,
                intent_get_employment_type,
                intent_get_length_of_service,
            ],
        },
        "S03": {
            "name": "Employee Contact Info",
            "intent_functions": [
                intent_get_work_phone,
                intent_get_work_email,
                intent_contact_not_found,
            ],
        },
        "S04": {
            "name": "Manager Contact Info",
            "intent_functions": [
                intent_get_manager_email,
                intent_get_manager_phone,
            ],
        },
        "S05": {
            "name": "Cross-domain Comparisons",
            "intent_functions": [
                intent_find_employee_by_name,
                intent_get_job_title,
                intent_get_work_email,
                intent_get_manager_name,
                intent_get_location,
                intent_compare_departments,
                intent_compare_locations,
            ],
        },
    }

    print("Calculating statistically rigorous question counts...")
    print(f"  Target confidence: {confidence_level * 100:.0f}%\n")

    for sid, cfg in scenarios.items():
        n = len(cfg["intent_functions"])
        required = calculate_required_questions(n, confidence_level)
        cfg["question_count"] = required

        val = validate_scenario_rigor(cfg["intent_functions"], required, confidence_level)
        print(f"  {sid}: {cfg['name']}")
        print(f"    Intents: {n}  |  Questions: {required}  |  Confidence: {val['confidence_percentage']}")
        print()

    return scenarios

