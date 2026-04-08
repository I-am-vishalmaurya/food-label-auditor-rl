# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for FoodLabelAuditor Pydantic models."""

import pytest


def test_audit_action_valid():
    from food_label_auditor.models import AuditAction

    action = AuditAction(
        risk_level=3,
        flagged_ingredients=["maltodextrin", "sodium benzoate"],
        violation_codes=["FSSAI-2020-3.1.2"],
        alternative_product_id="prod_042",
        explanation="Contains hidden sugar (maltodextrin) flagged for diabetic profile",
    )
    assert action.risk_level == 3
    assert len(action.flagged_ingredients) == 2
    assert action.violation_codes == ["FSSAI-2020-3.1.2"]


def test_audit_action_risk_level_bounds():
    from food_label_auditor.models import AuditAction

    with pytest.raises(ValueError):
        AuditAction(
            risk_level=5,
            flagged_ingredients=[],
            violation_codes=[],
            alternative_product_id="",
            explanation="",
        )

    with pytest.raises(ValueError):
        AuditAction(
            risk_level=-1,
            flagged_ingredients=[],
            violation_codes=[],
            alternative_product_id="",
            explanation="",
        )


def test_audit_action_defaults():
    from food_label_auditor.models import AuditAction

    action = AuditAction(risk_level=0)
    assert action.flagged_ingredients == []
    assert action.violation_codes == []
    assert action.alternative_product_id == ""
    assert action.explanation == ""


def test_audit_observation_required_fields():
    from food_label_auditor.models import AuditObservation

    obs = AuditObservation(
        product_id="OFF_001",
        product_name="Test Product",
        category="snacks",
        ingredients_text="wheat flour, sugar, salt",
        user_profile_id="diabetic_adult",
    )
    assert obs.product_id == "OFF_001"
    assert obs.nova_group == 0
    assert obs.marketing_claims == []
    assert obs.done is False


def test_audit_observation_full():
    from food_label_auditor.models import AuditObservation

    obs = AuditObservation(
        product_id="OFF_002",
        product_name="Maggi Noodles",
        brand="Nestle",
        category="instant_noodles",
        ingredients_text="Refined wheat flour (maida), palm oil, salt",
        nutrition_per_100g={"sugars_g": 1.2, "sodium_mg": 1040},
        marketing_claims=["No added MSG"],
        nutri_score="D",
        nova_group=4,
        user_profile_id="healthy_child",
        user_conditions=["none"],
        step_number=3,
        total_steps=10,
        done=False,
        reward=0.5,
    )
    assert obs.nova_group == 4
    assert obs.nutri_score == "D"
    assert obs.step_number == 3


def test_audit_state_defaults():
    from food_label_auditor.models import AuditState

    state = AuditState()
    assert state.task_id == 1
    assert state.seed == 0
    assert state.products_remaining == 0
    assert state.cumulative_score == 0.0
    assert state.scan_history == []


def test_audit_state_custom():
    from food_label_auditor.models import AuditState

    state = AuditState(
        task_id=3,
        seed=42,
        products_remaining=28,
        cumulative_score=12.5,
        scan_history=["OFF_001", "OFF_002"],
        episode_id="ep-123",
        step_count=2,
    )
    assert state.task_id == 3
    assert state.seed == 42
    assert len(state.scan_history) == 2


def test_user_profile_model():
    from food_label_auditor.models import UserProfile

    profile = UserProfile(
        profile_id="diabetic_adult",
        age=45,
        conditions=["diabetes"],
        dietary_restrictions=["low_sugar"],
        thresholds={"sugars_g_max": 5.0, "sodium_mg_max": 600},
    )
    assert profile.profile_id == "diabetic_adult"
    assert profile.thresholds["sugars_g_max"] == 5.0
