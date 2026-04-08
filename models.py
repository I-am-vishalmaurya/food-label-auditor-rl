# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the FoodLabelAuditor Environment.

Defines typed Action, Observation, and State models for auditing
Indian food product labels against FSSAI 2020 regulations and
ICMR-NIN dietary guidelines.
"""

from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation, State


class UserProfile(BaseModel):
    """Health profile for the simulated consumer, with ICMR-NIN thresholds."""

    profile_id: str = Field(..., description="Unique profile identifier")
    age: int = Field(..., ge=1, le=120, description="Age in years")
    conditions: list[str] = Field(
        default_factory=list, description="Health conditions e.g. diabetes, hypertension"
    )
    dietary_restrictions: list[str] = Field(
        default_factory=list, description="Dietary restrictions e.g. low_sugar, gluten_free"
    )
    thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="ICMR-NIN per-condition thresholds e.g. sugars_g_max, sodium_mg_max",
    )


class AuditAction(Action):
    """Agent's audit decision for a food product label."""

    risk_level: int = Field(
        ..., ge=0, le=4, description="0=none, 1=low, 2=moderate, 3=high, 4=critical"
    )
    flagged_ingredients: list[str] = Field(
        default_factory=list, description="Problematic ingredient names from the label"
    )
    violation_codes: list[str] = Field(
        default_factory=list,
        description="FSSAI violation codes e.g. FSSAI-2020-HIGH-SODIUM",
    )
    alternative_product_id: str = Field(
        default="", description="Product ID of a safer alternative in the same category"
    )
    explanation: str = Field(
        default="", description="Reasoning for the audit decision"
    )


class AuditObservation(Observation):
    """What the agent sees: a food label + user profile context."""

    product_id: str = Field(..., description="Unique product identifier")
    product_name: str = Field(..., description="Product name as printed on label")
    brand: str = Field(default="", description="Brand name")
    category: str = Field(..., description="Food category e.g. instant_noodles, biscuits")
    ingredients_text: str = Field(
        ..., description="Raw ingredients list as printed on the product label"
    )
    nutrition_per_100g: dict = Field(
        default_factory=dict,
        description="Nutritional values per 100g: energy_kcal, sugars_g, sodium_mg, fat_g",
    )
    marketing_claims: list[str] = Field(
        default_factory=list, description="Front-of-pack marketing claims"
    )
    nutri_score: str = Field(default="", description="Nutri-Score grade A/B/C/D/E")
    nova_group: int = Field(
        default=0, ge=0, le=4, description="NOVA processing classification 1-4"
    )
    user_profile_id: str = Field(..., description="Active user health profile ID")
    user_conditions: list[str] = Field(
        default_factory=list, description="User's health conditions for this step"
    )
    step_number: int = Field(default=0, description="Current step in the episode")
    total_steps: int = Field(default=1, description="Total steps in this task")


class AuditState(State):
    """Full episode state for reproducibility across reset/step cycles."""

    task_id: int = Field(default=1, description="Task difficulty: 1=easy, 2=medium, 3=hard")
    seed: int = Field(default=0, description="RNG seed for deterministic reproduction")
    products_remaining: int = Field(default=0, description="Products left to scan")
    cumulative_score: float = Field(default=0.0, description="Running grader score")
    scan_history: list[str] = Field(
        default_factory=list, description="Product IDs already scanned in this episode"
    )
