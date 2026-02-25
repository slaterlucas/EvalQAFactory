#!/usr/bin/env python3
"""
Statistical rigor utilities for question generation.

Uses the Coupon Collector Problem to determine how many questions are
needed so that every intent function is sampled at least once with a
given confidence level.

References:
    https://en.wikipedia.org/wiki/Coupon_collector%27s_problem
"""

import math
from typing import List, Callable


def calculate_required_questions(num_intents: int, confidence_level: float = 0.90) -> int:
    """
    Minimum questions to achieve *confidence_level* probability that all
    *num_intents* are sampled at least once.

    Formula: k ≈ n * (ln(n) + γ + ln(-ln(1-p)))
    where n = intents, p = confidence, γ ≈ 0.5772 (Euler–Mascheroni).
    """
    if num_intents <= 0:
        raise ValueError("num_intents must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")

    n = num_intents
    gamma = 0.5772156649015329
    k = n * (math.log(n) + gamma + math.log(-math.log(1 - confidence_level)))
    return math.ceil(k)


def calculate_expected_questions(num_intents: int) -> int:
    """Expected value E[T] = n * H_n (n-th harmonic number)."""
    if num_intents <= 0:
        raise ValueError("num_intents must be positive")
    return math.ceil(num_intents * sum(1 / i for i in range(1, num_intents + 1)))


def get_confidence_for_questions(num_intents: int, num_questions: int) -> float:
    """Confidence achieved given a fixed number of questions."""
    if num_intents <= 0 or num_questions <= 0:
        raise ValueError("Both values must be positive")
    gamma = 0.5772156649015329
    exp_arg = (num_questions / num_intents) - math.log(num_intents) - gamma
    return max(0.0, min(1.0, 1 - math.exp(-math.exp(exp_arg))))


def validate_scenario_rigor(
    intent_functions: List[Callable],
    question_count: int,
    target_confidence: float = 0.90,
) -> dict:
    """Check whether a scenario has enough questions for the target confidence."""
    n = len(intent_functions)
    required = calculate_required_questions(n, target_confidence)
    expected = calculate_expected_questions(n)
    actual = get_confidence_for_questions(n, question_count)

    return {
        "num_intents": n,
        "current_questions": question_count,
        "required_questions": required,
        "expected_questions": expected,
        "target_confidence": target_confidence,
        "actual_confidence": actual,
        "is_sufficient": question_count >= required,
        "confidence_percentage": f"{actual * 100:.1f}%",
        "recommendation": (
            f"Sufficient: {question_count} questions -> {actual * 100:.1f}% confidence"
            if question_count >= required
            else f"Need {required} questions for {target_confidence * 100:.0f}% (currently {actual * 100:.1f}%)"
        ),
    }


if __name__ == "__main__":
    print("Statistical Rigor Calculator")
    print("=" * 45)
    for intents, questions in [(4, 12), (7, 24), (16, 70)]:
        print(f"\n{intents} intents, {questions} questions:")
        r = validate_scenario_rigor(list(range(intents)), questions)
        print(f"  Required: {r['required_questions']}")
        print(f"  Confidence: {r['confidence_percentage']}")
        print(f"  {r['recommendation']}")
