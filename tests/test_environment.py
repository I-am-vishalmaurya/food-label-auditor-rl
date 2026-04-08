# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for FoodLabelAuditorEnvironment."""

import pytest
from food_label_auditor.server.food_label_auditor_environment import (
    FoodLabelAuditorEnvironment,
)
from food_label_auditor.models import AuditAction


@pytest.fixture
def env():
    return FoodLabelAuditorEnvironment()


class TestReset:
    def test_reset_produces_valid_observation(self, env):
        obs = env.reset(seed=42, task_id=1)
        assert obs.product_id != ""
        assert obs.ingredients_text != ""
        assert obs.user_profile_id in [
            "diabetic_adult", "healthy_child",
            "hypertensive_senior", "healthy_adult",
        ]
        assert obs.done is False

    def test_reset_task1_picks_nova4(self, env):
        obs = env.reset(seed=42, task_id=1)
        assert obs.nova_group == 4
        assert obs.nutri_score.upper() in ("D", "E")
        assert obs.total_steps == 1

    def test_reset_task2_has_10_steps(self, env):
        obs = env.reset(seed=42, task_id=2)
        assert obs.total_steps == 10

    def test_reset_task3_has_30_steps(self, env):
        obs = env.reset(seed=42, task_id=3)
        assert obs.total_steps == 30

    def test_reset_deterministic_with_seed(self, env):
        env2 = FoodLabelAuditorEnvironment()
        obs1 = env.reset(seed=42, task_id=2)
        obs2 = env2.reset(seed=42, task_id=2)
        assert obs1.product_id == obs2.product_id
        assert obs1.user_profile_id == obs2.user_profile_id
        assert obs1.ingredients_text == obs2.ingredients_text

    def test_different_seeds_give_different_episodes(self, env):
        obs1 = env.reset(seed=42, task_id=2)
        obs2 = env.reset(seed=99, task_id=2)
        # Very unlikely to pick same first product
        assert obs1.product_id != obs2.product_id or obs1.user_profile_id != obs2.user_profile_id


class TestStep:
    def test_step_returns_observation(self, env):
        env.reset(seed=42, task_id=1)
        action = AuditAction(
            risk_level=4,
            flagged_ingredients=["salt"],
            violation_codes=["FSSAI-2020-HIGH-SODIUM"],
            alternative_product_id="",
            explanation="High sodium",
        )
        obs = env.step(action)
        assert obs.done is True  # Task 1 has only 1 product

    def test_step_computes_reward(self, env):
        env.reset(seed=42, task_id=1)
        action = AuditAction(
            risk_level=4,
            flagged_ingredients=["salt"],
            violation_codes=["FSSAI-2020-HIGH-SODIUM"],
        )
        obs = env.step(action)
        assert obs.reward >= 0.0

    def test_task2_full_episode(self, env):
        obs = env.reset(seed=42, task_id=2)
        for i in range(10):
            action = AuditAction(
                risk_level=2,
                flagged_ingredients=["sugar"],
            )
            obs = env.step(action)

        assert obs.done is True
        assert env.state.step_count == 10

    def test_task3_includes_adversarial(self, env):
        obs = env.reset(seed=42, task_id=3)
        adversarial_seen = False
        for i in range(30):
            if "ADV_" in obs.product_id:
                adversarial_seen = True
            action = AuditAction(risk_level=2)
            obs = env.step(action)

        assert adversarial_seen, "Task 3 should include adversarial products"

    def test_step_after_done_returns_empty(self, env):
        env.reset(seed=42, task_id=1)
        action = AuditAction(risk_level=2)
        env.step(action)  # done
        obs = env.step(action)  # after done
        assert obs.done is True
        assert obs.product_name == "Episode ended"


class TestState:
    def test_state_tracks_episode(self, env):
        env.reset(seed=42, task_id=2)
        state = env.state
        assert state.task_id == 2
        assert state.seed == 42
        assert state.products_remaining == 10
        assert state.step_count == 0

    def test_state_updates_after_step(self, env):
        env.reset(seed=42, task_id=2)
        action = AuditAction(risk_level=2)
        env.step(action)
        state = env.state
        assert state.step_count == 1
        assert state.products_remaining == 9
        assert len(state.scan_history) == 1
