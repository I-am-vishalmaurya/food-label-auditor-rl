# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration tests: full episodes for all 3 tasks + reproducibility."""

import json
from pathlib import Path

import pytest
from food_label_auditor.models import AuditAction
from food_label_auditor.server.food_label_auditor_environment import (
    FoodLabelAuditorEnvironment,
)
from food_label_auditor.server.graders import grade_task1, grade_task2, grade_task3

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def env():
    return FoodLabelAuditorEnvironment()


@pytest.fixture
def products_by_id():
    with open(DATA_DIR / "products.json") as f:
        products = json.load(f)
    return {p["product_id"]: p for p in products}


@pytest.fixture
def ground_truth_map():
    with open(DATA_DIR / "ground_truth.json") as f:
        gt_list = json.load(f)
    return {(g["product_id"], g["profile_id"]): g for g in gt_list}


class TestTask1FullEpisode:
    def test_task1_single_step_episode(self, env):
        obs = env.reset(seed=42, task_id=1)
        assert obs.nova_group == 4
        assert obs.nutri_score.upper() in ("D", "E")
        assert obs.total_steps == 1
        assert obs.done is False

        action = AuditAction(
            risk_level=4,
            flagged_ingredients=["salt"],
            violation_codes=["FSSAI-2020-HIGH-SODIUM", "NOVA-4-ULTRA-PROCESSED"],
        )
        obs = env.step(action)
        assert obs.done is True
        assert obs.reward > 0

    def test_task1_grader_on_perfect_agent(self, env, products_by_id, ground_truth_map):
        obs = env.reset(seed=42, task_id=1)
        gt = ground_truth_map.get((obs.product_id, obs.user_profile_id))
        assert gt is not None

        action = AuditAction(
            risk_level=gt["expected_risk_level"],
            flagged_ingredients=gt["expected_flagged_ingredients"],
            violation_codes=gt["expected_violation_codes"],
        )
        env.step(action)
        score = grade_task1([action], [gt], products_by_id)
        assert score >= 0.7

    def test_task1_random_baseline_low(self, env, products_by_id, ground_truth_map):
        obs = env.reset(seed=42, task_id=1)
        gt = ground_truth_map.get((obs.product_id, obs.user_profile_id))

        action = AuditAction(
            risk_level=0,
            flagged_ingredients=["water"],
            violation_codes=[],
            alternative_product_id="nonexistent",
        )
        env.step(action)
        score = grade_task1([action], [gt], products_by_id)
        assert score < 0.5


class TestTask2FullEpisode:
    def test_task2_ten_steps(self, env):
        obs = env.reset(seed=42, task_id=2)
        assert obs.total_steps == 10

        profiles_seen = set()
        for i in range(10):
            profiles_seen.add(obs.user_profile_id)
            action = AuditAction(risk_level=2, flagged_ingredients=["sugar"])
            obs = env.step(action)

        assert obs.done is True
        assert env.state.step_count == 10
        assert len(profiles_seen) >= 2, "Task 2 should use at least 2 profiles"

    def test_task2_grader_on_perfect_agent(self, env, products_by_id, ground_truth_map):
        obs = env.reset(seed=42, task_id=2)
        actions = []
        gts = []

        for i in range(10):
            gt = ground_truth_map.get((obs.product_id, obs.user_profile_id))
            assert gt is not None
            gts.append(gt)
            action = AuditAction(
                risk_level=gt["expected_risk_level"],
                flagged_ingredients=gt["expected_flagged_ingredients"],
                violation_codes=gt["expected_violation_codes"],
            )
            actions.append(action)
            obs = env.step(action)

        score = grade_task2(actions, gts, products_by_id)
        assert score >= 0.8


class TestTask3FullEpisode:
    def test_task3_thirty_steps_with_adversarial(self, env):
        obs = env.reset(seed=42, task_id=3)
        assert obs.total_steps == 30

        adversarial_seen = 0
        for i in range(30):
            if "ADV_" in obs.product_id:
                adversarial_seen += 1
            action = AuditAction(risk_level=2)
            obs = env.step(action)

        assert obs.done is True
        assert adversarial_seen >= 8, f"Expected >=8 adversarial, got {adversarial_seen}"

    def test_task3_grader_on_perfect_agent(self, env, products_by_id, ground_truth_map):
        obs = env.reset(seed=42, task_id=3)
        actions = []
        gts = []

        for i in range(30):
            gt = ground_truth_map.get((obs.product_id, obs.user_profile_id))
            assert gt is not None
            gts.append(gt)
            action = AuditAction(
                risk_level=gt["expected_risk_level"],
                flagged_ingredients=gt["expected_flagged_ingredients"],
                violation_codes=gt["expected_violation_codes"],
            )
            actions.append(action)
            obs = env.step(action)

        score = grade_task3(actions, gts, products_by_id)
        assert score >= 0.7


class TestDeterministicReproducibility:
    def test_same_seed_same_episode(self):
        env1 = FoodLabelAuditorEnvironment()
        env2 = FoodLabelAuditorEnvironment()

        obs1 = env1.reset(seed=777, task_id=2)
        obs2 = env2.reset(seed=777, task_id=2)

        for i in range(10):
            assert obs1.product_id == obs2.product_id, f"Mismatch at step {i}"
            assert obs1.user_profile_id == obs2.user_profile_id

            action = AuditAction(risk_level=2)
            obs1 = env1.step(action)
            obs2 = env2.step(action)

            assert obs1.reward == obs2.reward

    def test_same_seed_same_task3(self):
        env1 = FoodLabelAuditorEnvironment()
        env2 = FoodLabelAuditorEnvironment()

        obs1 = env1.reset(seed=123, task_id=3)
        obs2 = env2.reset(seed=123, task_id=3)

        for i in range(30):
            assert obs1.product_id == obs2.product_id
            action = AuditAction(risk_level=3, flagged_ingredients=["salt"])
            obs1 = env1.step(action)
            obs2 = env2.step(action)

        assert env1.state.cumulative_score == env2.state.cumulative_score


class TestRandomBaseline:
    def test_random_policy_scores_near_zero(self, env, products_by_id, ground_truth_map):
        """A random policy should score poorly -- verifying the action space is wide enough."""
        import random
        rng = random.Random(42)

        obs = env.reset(seed=42, task_id=1)
        gt = ground_truth_map.get((obs.product_id, obs.user_profile_id))

        random_action = AuditAction(
            risk_level=rng.randint(0, 4),
            flagged_ingredients=[rng.choice(["water", "oxygen", "nitrogen"])],
            violation_codes=[f"RANDOM-{rng.randint(0, 999)}"],
            alternative_product_id=f"FAKE_{rng.randint(0, 999)}",
        )
        env.step(random_action)
        score = grade_task1([random_action], [gt], products_by_id)
        assert score < 0.5, f"Random baseline scored too high: {score}"
