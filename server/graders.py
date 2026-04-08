# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Deterministic grader functions for the FoodLabelAuditor environment.

All graders are pure functions with zero side effects and zero network calls.
They reference only the static ground_truth data loaded at environment init.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import AuditAction


def grade_step(action: AuditAction, ground_truth: dict, products_by_id: dict) -> float:
    """
    Grade a single step (one product audit) used for per-step reward.

    Components:
      - risk_correctness (0.4): exact match = 1.0, off-by-one = partial
      - ingredient_recall (0.3): fraction of expected flagged ingredients found
      - alternative_validity (0.15): suggested alt is in same category with better score
      - precision_penalty (0.15): penalise false flags

    Returns a float in [0.0, 1.0].
    """
    expected_risk = ground_truth["expected_risk_level"]
    expected_ingredients = set(
        i.lower() for i in ground_truth.get("expected_flagged_ingredients", [])
    )
    expected_violations = set(ground_truth.get("expected_violation_codes", []))

    # Risk correctness (0.4 weight)
    risk_diff = abs(action.risk_level - expected_risk)
    if risk_diff == 0:
        risk_score = 1.0
    else:
        risk_score = max(0.0, 1.0 - risk_diff * 0.3)

    # Ingredient recall (0.3 weight)
    agent_ingredients = set(i.lower() for i in action.flagged_ingredients)
    if expected_ingredients:
        recall = len(agent_ingredients & expected_ingredients) / len(expected_ingredients)
    else:
        recall = 1.0 if not agent_ingredients else 0.5

    # Precision penalty (0.15 weight) — penalise over-flagging
    if agent_ingredients:
        true_positives = len(agent_ingredients & expected_ingredients)
        precision = true_positives / len(agent_ingredients)
    else:
        precision = 1.0 if not expected_ingredients else 0.0

    # Alternative validity (0.15 weight)
    alt_score = 0.0
    alt_id = action.alternative_product_id
    if alt_id and alt_id in products_by_id:
        alt_product = products_by_id[alt_id]
        gt_category = ground_truth.get("category", "")
        if alt_product.get("category") == gt_category:
            alt_score = 0.5
            # Bonus if alternative has better nutri-score
            ns_order = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "NA": 0, "": 0}
            product = products_by_id.get(ground_truth["product_id"], {})
            if ns_order.get(alt_product.get("nutri_score", ""), 0) > ns_order.get(
                product.get("nutri_score", ""), 0
            ):
                alt_score = 1.0

    return (
        risk_score * 0.4
        + recall * 0.3
        + precision * 0.15
        + alt_score * 0.15
    )


def grade_task1(
    actions: list[AuditAction],
    ground_truths: list[dict],
    products_by_id: dict,
) -> float:
    """
    Task 1 grader (Easy): Single product scan.

    score = (risk_level_correct * 0.4)
          + (any_true_ingredient_flagged * 0.3)
          + (alternative_in_same_category * 0.3)
    """
    if not actions or not ground_truths:
        return 0.0

    action = actions[0]
    gt = ground_truths[0]

    # Risk level correctness
    risk_correct = 1.0 if action.risk_level == gt["expected_risk_level"] else max(
        0.0, 1.0 - abs(action.risk_level - gt["expected_risk_level"]) * 0.3
    )

    # Any true ingredient flagged
    expected = set(i.lower() for i in gt.get("expected_flagged_ingredients", []))
    agent_flagged = set(i.lower() for i in action.flagged_ingredients)
    any_flagged = 1.0 if (expected & agent_flagged) else (0.0 if expected else 1.0)

    # Alternative in same category
    alt_valid = 0.0
    if action.alternative_product_id and action.alternative_product_id in products_by_id:
        if products_by_id[action.alternative_product_id].get("category") == gt.get("category"):
            alt_valid = 1.0

    return risk_correct * 0.4 + any_flagged * 0.3 + alt_valid * 0.3


def grade_task2(
    actions: list[AuditAction],
    ground_truths: list[dict],
    products_by_id: dict,
) -> float:
    """
    Task 2 grader (Medium): 10 products across 2 profiles.

    score = sum(per_scan_correctness) / n_scans
    per_scan = 1.0 if risk exact, 0.5 if off-by-one, 0.0 otherwise
    + profile differentiation bonus (up to +0.2 scaled)
    """
    if not actions:
        return 0.0

    n = len(actions)
    scan_scores = []

    for action, gt in zip(actions, ground_truths):
        diff = abs(action.risk_level - gt["expected_risk_level"])
        if diff == 0:
            scan_scores.append(1.0)
        elif diff == 1:
            scan_scores.append(0.5)
        else:
            scan_scores.append(0.0)

    base_score = sum(scan_scores) / n

    # Profile differentiation bonus: check if agent gives different risk
    # levels for the same product scanned under different profiles
    product_risk_by_profile: dict[str, dict[str, int]] = {}
    for action, gt in zip(actions, ground_truths):
        pid = gt["product_id"]
        profile = gt["profile_id"]
        if pid not in product_risk_by_profile:
            product_risk_by_profile[pid] = {}
        product_risk_by_profile[pid][profile] = action.risk_level

    diff_count = 0
    total_multi = 0
    for pid, profile_risks in product_risk_by_profile.items():
        if len(profile_risks) > 1:
            total_multi += 1
            values = list(profile_risks.values())
            if len(set(values)) > 1:
                diff_count += 1

    diff_bonus = (diff_count / max(total_multi, 1)) * 0.2

    return min(1.0, base_score + diff_bonus)


def grade_task3(
    actions: list[AuditAction],
    ground_truths: list[dict],
    products_by_id: dict,
) -> float:
    """
    Task 3 grader (Hard/Adversarial): 30 products with adversarial cases.

    score = (adversarial_caught / total_adversarial * 0.5)
          + (overall_accuracy * 0.3)
          + (no_action_loops * 0.2)
    """
    if not actions:
        return 0.0

    # Adversarial detection score
    total_adversarial = 0
    adversarial_caught = 0
    for action, gt in zip(actions, ground_truths):
        if gt.get("is_adversarial"):
            total_adversarial += 1
            expected_violations = set(gt.get("expected_violation_codes", []))
            misleading_codes = {c for c in expected_violations if "MISLEADING" in c}
            agent_codes = set(action.violation_codes)
            if misleading_codes & agent_codes:
                adversarial_caught += 1
            elif action.risk_level >= gt["expected_risk_level"]:
                adversarial_caught += 0.5

    adv_score = (adversarial_caught / max(total_adversarial, 1))

    # Overall accuracy (risk level)
    correct = 0
    for action, gt in zip(actions, ground_truths):
        if abs(action.risk_level - gt["expected_risk_level"]) <= 1:
            correct += 1
    accuracy = correct / len(actions)

    # Action loop penalty: penalize if agent repeats exact same action 3+ times
    loop_penalty = 0.0
    consecutive = 1
    for i in range(1, len(actions)):
        if (
            actions[i].risk_level == actions[i - 1].risk_level
            and actions[i].flagged_ingredients == actions[i - 1].flagged_ingredients
            and actions[i].violation_codes == actions[i - 1].violation_codes
        ):
            consecutive += 1
            if consecutive >= 3:
                loop_penalty += 0.05
        else:
            consecutive = 1

    no_loops = max(0.0, 1.0 - loop_penalty)

    return adv_score * 0.5 + accuracy * 0.3 + no_loops * 0.2
