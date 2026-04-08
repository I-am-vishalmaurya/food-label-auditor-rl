# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for deterministic grader functions."""

import json
from pathlib import Path

import pytest
from food_label_auditor.models import AuditAction
from food_label_auditor.server.graders import (
    grade_step,
    grade_task1,
    grade_task2,
    grade_task3,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def products_by_id():
    with open(DATA_DIR / "products.json") as f:
        products = json.load(f)
    return {p["product_id"]: p for p in products}


@pytest.fixture
def maggi_gt_diabetic():
    with open(DATA_DIR / "ground_truth.json") as f:
        gt_list = json.load(f)
    for gt in gt_list:
        if gt["product_id"] == "IND_001" and gt["profile_id"] == "diabetic_adult":
            return gt
    pytest.fail("Maggi + diabetic GT not found")


class TestGradeStep:
    def test_perfect_answer_scores_high(self, maggi_gt_diabetic, products_by_id):
        action = AuditAction(
            risk_level=maggi_gt_diabetic["expected_risk_level"],
            flagged_ingredients=maggi_gt_diabetic["expected_flagged_ingredients"],
            violation_codes=maggi_gt_diabetic["expected_violation_codes"],
            alternative_product_id="",
            explanation="High sodium and ultra-processed",
        )
        score = grade_step(action, maggi_gt_diabetic, products_by_id)
        assert score >= 0.7

    def test_random_answer_scores_low(self, maggi_gt_diabetic, products_by_id):
        action = AuditAction(
            risk_level=0,
            flagged_ingredients=["water", "oxygen"],
            violation_codes=[],
            alternative_product_id="nonexistent",
            explanation="",
        )
        score = grade_step(action, maggi_gt_diabetic, products_by_id)
        assert score < 0.2

    def test_partial_answer_scores_medium(self, maggi_gt_diabetic, products_by_id):
        action = AuditAction(
            risk_level=maggi_gt_diabetic["expected_risk_level"] - 1,
            flagged_ingredients=maggi_gt_diabetic["expected_flagged_ingredients"][:1],
            violation_codes=[],
        )
        score = grade_step(action, maggi_gt_diabetic, products_by_id)
        assert 0.2 <= score <= 0.7


class TestGradeTask1:
    def test_perfect_score(self, maggi_gt_diabetic, products_by_id):
        action = AuditAction(
            risk_level=maggi_gt_diabetic["expected_risk_level"],
            flagged_ingredients=maggi_gt_diabetic["expected_flagged_ingredients"],
            violation_codes=maggi_gt_diabetic["expected_violation_codes"],
            alternative_product_id="IND_041",  # another noodle
        )
        score = grade_task1([action], [maggi_gt_diabetic], products_by_id)
        assert score >= 0.7

    def test_random_near_zero(self, maggi_gt_diabetic, products_by_id):
        action = AuditAction(
            risk_level=0,
            flagged_ingredients=["water"],
            violation_codes=[],
            alternative_product_id="nonexistent",
        )
        score = grade_task1([action], [maggi_gt_diabetic], products_by_id)
        assert score < 0.3

    def test_empty_returns_zero(self, products_by_id):
        assert grade_task1([], [], products_by_id) == 0.0


class TestGradeTask2:
    def test_all_correct_scores_high(self, products_by_id):
        gt_list = []
        actions = []
        with open(DATA_DIR / "ground_truth.json") as f:
            all_gt = json.load(f)
        selected = [g for g in all_gt if not g.get("is_adversarial")][:10]
        for gt in selected:
            gt_list.append(gt)
            actions.append(AuditAction(
                risk_level=gt["expected_risk_level"],
                flagged_ingredients=gt["expected_flagged_ingredients"],
                violation_codes=gt["expected_violation_codes"],
            ))
        score = grade_task2(actions, gt_list, products_by_id)
        assert score >= 0.8

    def test_empty_returns_zero(self, products_by_id):
        assert grade_task2([], [], products_by_id) == 0.0


class TestGradeTask3:
    def test_adversarial_detection_matters(self, products_by_id):
        with open(DATA_DIR / "ground_truth.json") as f:
            all_gt = json.load(f)
        adv_gt = [g for g in all_gt if g.get("is_adversarial")][:10]
        normal_gt = [g for g in all_gt if not g.get("is_adversarial")][:20]
        gt_list = adv_gt + normal_gt

        # Agent that catches adversarial cases
        good_actions = []
        for gt in gt_list:
            good_actions.append(AuditAction(
                risk_level=gt["expected_risk_level"],
                flagged_ingredients=gt["expected_flagged_ingredients"],
                violation_codes=gt["expected_violation_codes"],
            ))

        # Agent that ignores adversarial
        bad_actions = []
        for gt in gt_list:
            bad_actions.append(AuditAction(
                risk_level=0,
                flagged_ingredients=[],
                violation_codes=[],
            ))

        good_score = grade_task3(good_actions, gt_list, products_by_id)
        bad_score = grade_task3(bad_actions, gt_list, products_by_id)
        assert good_score > bad_score

    def test_action_loops_penalized(self, products_by_id):
        with open(DATA_DIR / "ground_truth.json") as f:
            all_gt = json.load(f)
        gt_list = all_gt[:30]

        # Repetitive actions
        loop_actions = [
            AuditAction(risk_level=2, flagged_ingredients=["sugar"], violation_codes=["X"])
            for _ in range(30)
        ]

        # Varied actions
        varied_actions = []
        for i, gt in enumerate(gt_list):
            varied_actions.append(AuditAction(
                risk_level=gt["expected_risk_level"],
                flagged_ingredients=gt["expected_flagged_ingredients"],
                violation_codes=gt["expected_violation_codes"],
            ))

        loop_score = grade_task3(loop_actions, gt_list, products_by_id)
        varied_score = grade_task3(varied_actions, gt_list, products_by_id)
        assert varied_score >= loop_score

    def test_empty_returns_zero(self, products_by_id):
        assert grade_task3([], [], products_by_id) == 0.0
