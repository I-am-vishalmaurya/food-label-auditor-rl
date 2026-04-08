# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FoodLabelAuditor Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AuditAction, AuditObservation, AuditState


class FoodLabelAuditorEnv(
    EnvClient[AuditAction, AuditObservation, AuditState]
):
    """
    Client for the FoodLabelAuditor Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step food label auditing sessions.

    Example:
        >>> with FoodLabelAuditorEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.product_name)
        ...
        ...     result = client.step(AuditAction(
        ...         risk_level=3,
        ...         flagged_ingredients=["palm oil", "salt"],
        ...         violation_codes=["FSSAI-2020-HIGH-SODIUM"],
        ...     ))
        ...     print(result.observation.reward)

    Example with Docker:
        >>> client = FoodLabelAuditorEnv.from_docker_image("food-label-auditor:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(AuditAction(risk_level=2))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: AuditAction) -> Dict:
        """
        Convert AuditAction to JSON payload for step message.

        Args:
            action: AuditAction instance with risk_level, flagged_ingredients, etc.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "risk_level": action.risk_level,
            "flagged_ingredients": action.flagged_ingredients,
            "violation_codes": action.violation_codes,
            "alternative_product_id": action.alternative_product_id,
            "explanation": action.explanation,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AuditObservation]:
        """
        Parse server response into StepResult[AuditObservation].

        Args:
            payload: JSON response data from server.

        Returns:
            StepResult with AuditObservation.
        """
        obs_data = payload.get("observation", {})
        observation = AuditObservation(
            product_id=obs_data.get("product_id", ""),
            product_name=obs_data.get("product_name", ""),
            brand=obs_data.get("brand", ""),
            category=obs_data.get("category", ""),
            ingredients_text=obs_data.get("ingredients_text", ""),
            nutrition_per_100g=obs_data.get("nutrition_per_100g", {}),
            marketing_claims=obs_data.get("marketing_claims", []),
            nutri_score=obs_data.get("nutri_score", ""),
            nova_group=obs_data.get("nova_group", 0),
            user_profile_id=obs_data.get("user_profile_id", ""),
            user_conditions=obs_data.get("user_conditions", []),
            step_number=obs_data.get("step_number", 0),
            total_steps=obs_data.get("total_steps", 1),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> AuditState:
        """
        Parse server response into AuditState object.

        Args:
            payload: JSON response from state request.

        Returns:
            AuditState with full episode context.
        """
        return AuditState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", 1),
            seed=payload.get("seed", 0),
            products_remaining=payload.get("products_remaining", 0),
            cumulative_score=payload.get("cumulative_score", 0.0),
            scan_history=payload.get("scan_history", []),
        )
