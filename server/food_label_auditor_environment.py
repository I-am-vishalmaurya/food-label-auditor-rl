# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FoodLabelAuditor Environment Implementation.

An RL environment where the agent audits Indian food product labels against
FSSAI 2020 regulations and ICMR-NIN dietary guidelines.

The agent receives a raw ingredient string + user health profile and must:
  1. Classify risk level (0-4)
  2. Flag problematic ingredients
  3. Cite FSSAI violation codes
  4. Suggest safer alternatives
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AuditAction, AuditObservation, AuditState, UserProfile
    from .graders import grade_step, grade_task1, grade_task2, grade_task3
except ImportError:
    from models import AuditAction, AuditObservation, AuditState, UserProfile
    from server.graders import grade_step, grade_task1, grade_task2, grade_task3

EPISODE_GRADERS = {
    1: grade_task1,
    2: grade_task2,
    3: grade_task3,
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TASK_CONFIGS = {
    1: {"n_products": 1, "n_profiles": 1, "include_adversarial": False},
    2: {"n_products": 10, "n_profiles": 2, "include_adversarial": False},
    3: {"n_products": 30, "n_profiles": 2, "include_adversarial": True},
}


class FoodLabelAuditorEnvironment(Environment):
    """
    Environment for auditing food product labels against FSSAI/ICMR standards.

    Supports 3 task difficulty levels:
      - Task 1 (Easy): single NOVA-4 product, 1 profile
      - Task 2 (Medium): 10 products across 2 profiles
      - Task 3 (Hard): 30 products including adversarial cases, 2+ profiles
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Load static dataset from JSON files."""
        super().__init__()

        with open(DATA_DIR / "products.json") as f:
            products_list = json.load(f)
        self._products_list = products_list
        self._products_by_id = {p["product_id"]: p for p in products_list}

        with open(DATA_DIR / "user_profiles.json") as f:
            profiles_list = json.load(f)
        self._profiles = {p["profile_id"]: p for p in profiles_list}

        with open(DATA_DIR / "ground_truth.json") as f:
            gt_list = json.load(f)
        self._ground_truth = {}
        for gt in gt_list:
            key = (gt["product_id"], gt["profile_id"])
            self._ground_truth[key] = gt

        # Episode state
        self._state = AuditState(episode_id=str(uuid4()), step_count=0)
        self._rng: random.Random = random.Random(0)
        self._episode_queue: list[tuple[str, str]] = []  # (product_id, profile_id) pairs
        self._current_idx: int = 0
        self._actions_history: list[AuditAction] = []
        self._gt_history: list[dict] = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: int = 1,
        **kwargs: Any,
    ) -> AuditObservation:
        """
        Reset the environment for a new episode.

        Args:
            seed: RNG seed for deterministic reproduction.
            episode_id: Optional episode identifier.
            task_id: Difficulty level (1=easy, 2=medium, 3=hard).

        Returns:
            First AuditObservation for the episode.
        """
        effective_seed = seed if seed is not None else random.randint(0, 2**31)
        self._rng = random.Random(effective_seed)

        config = TASK_CONFIGS.get(task_id, TASK_CONFIGS[1])
        n_products = config["n_products"]
        n_profiles = config["n_profiles"]
        include_adversarial = config["include_adversarial"]

        # Select profiles deterministically
        profile_ids = list(self._profiles.keys())
        self._rng.shuffle(profile_ids)
        selected_profiles = profile_ids[:n_profiles]

        # Select products deterministically
        if task_id == 1:
            # Task 1: always pick a clearly bad product (NOVA 4, Nutri-Score D/E)
            bad_products = [
                p for p in self._products_list
                if p["nova_group"] == 4
                and p.get("nutri_score", "").upper() in ("D", "E")
                and not p.get("is_adversarial", False)
            ]
            self._rng.shuffle(bad_products)
            selected_products = bad_products[:1]
        elif task_id == 3 and include_adversarial:
            # Task 3: include all adversarial + fill rest from normal
            adv_products = [p for p in self._products_list if p.get("is_adversarial", False)]
            normal_products = [p for p in self._products_list if not p.get("is_adversarial", False)]
            self._rng.shuffle(adv_products)
            self._rng.shuffle(normal_products)
            n_adv = min(len(adv_products), 10)
            n_normal = n_products - n_adv
            selected_products = adv_products[:n_adv] + normal_products[:n_normal]
            self._rng.shuffle(selected_products)
        else:
            # Task 2: mixed products
            normal_products = [p for p in self._products_list if not p.get("is_adversarial", False)]
            self._rng.shuffle(normal_products)
            selected_products = normal_products[:n_products]

        # Build episode queue: pair each product with a profile
        self._episode_queue = []
        for i, product in enumerate(selected_products):
            profile_id = selected_profiles[i % len(selected_profiles)]
            self._episode_queue.append((product["product_id"], profile_id))

        self._current_idx = 0
        self._actions_history = []
        self._gt_history = []

        self._state = AuditState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            seed=effective_seed,
            products_remaining=len(self._episode_queue),
            cumulative_score=0.0,
            scan_history=[],
        )

        return self._build_observation()

    def step(self, action: AuditAction, timeout_s: Optional[float] = None, **kwargs: Any) -> AuditObservation:
        """
        Process one audit action and advance to the next product.

        Args:
            action: The agent's audit decision.

        Returns:
            Next AuditObservation (or final observation with done=True).
        """
        if self._current_idx >= len(self._episode_queue):
            return AuditObservation(
                product_id="",
                product_name="Episode ended",
                category="",
                ingredients_text="",
                user_profile_id="",
                done=True,
                reward=0.0,
            )

        product_id, profile_id = self._episode_queue[self._current_idx]
        gt = self._ground_truth.get((product_id, profile_id), {})

        # Grade this step
        reward = grade_step(action, gt, self._products_by_id)

        # Record history
        self._actions_history.append(action)
        self._gt_history.append(gt)

        # Update state
        self._state.step_count += 1
        self._state.cumulative_score += reward
        self._state.scan_history.append(product_id)
        self._state.products_remaining = len(self._episode_queue) - self._current_idx - 1

        self._current_idx += 1
        is_done = self._current_idx >= len(self._episode_queue)

        if is_done:
            episode_grader = EPISODE_GRADERS.get(self._state.task_id)
            episode_score = 0.0
            if episode_grader is not None:
                episode_score = episode_grader(
                    self._actions_history,
                    self._gt_history,
                    self._products_by_id,
                )

            return AuditObservation(
                product_id=product_id,
                product_name="Episode complete",
                category="",
                ingredients_text="",
                user_profile_id=profile_id,
                step_number=self._state.step_count,
                total_steps=len(self._episode_queue),
                done=True,
                reward=reward,
                metadata={
                    "cumulative_score": self._state.cumulative_score,
                    "episode_score": episode_score,
                    "task_id": self._state.task_id,
                    "grader": f"grade_task{self._state.task_id}",
                    "steps_completed": self._state.step_count,
                },
            )

        obs = self._build_observation()
        obs.reward = reward
        return obs

    @property
    def state(self) -> AuditState:
        """Get the current environment state for reproducibility."""
        return self._state

    def _build_observation(self) -> AuditObservation:
        """Build an AuditObservation for the current queue position."""
        product_id, profile_id = self._episode_queue[self._current_idx]
        product = self._products_by_id[product_id]
        profile = self._profiles[profile_id]

        return AuditObservation(
            product_id=product["product_id"],
            product_name=product["product_name"],
            brand=product.get("brand", ""),
            category=product["category"],
            ingredients_text=product["ingredients_text"],
            nutrition_per_100g=product.get("nutrition_per_100g", {}),
            marketing_claims=product.get("marketing_claims", []),
            nutri_score=product.get("nutri_score", ""),
            nova_group=product.get("nova_group", 0),
            user_profile_id=profile_id,
            user_conditions=profile.get("conditions", []),
            step_number=self._current_idx,
            total_steps=len(self._episode_queue),
            done=False,
            reward=0.0,
        )
