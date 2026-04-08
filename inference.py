# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference Script for FoodLabelAuditor Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The name of the local image to use for the environment

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

STDOUT FORMAT
- The script emits exactly three line types to stdout:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import re
import textwrap
from typing import List, Optional

from openai import OpenAI

from food_label_auditor import AuditAction, FoodLabelAuditorEnv

IMAGE_NAME = os.getenv("IMAGE_NAME", "food-label-auditor:latest")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SEED = int(os.getenv("SEED", "42"))
BENCHMARK = os.getenv("FOOD_LABEL_AUDITOR_BENCHMARK", "food_label_auditor")
MAX_STEPS = int(os.getenv("MAX_STEPS", "35"))
TEMPERATURE = 0.3
MAX_TOKENS = 300
LLM_TIMEOUT = 30

ALL_TASKS = [
    {"task_id": 1, "task_name": "food_audit_task1"},
    {"task_id": 2, "task_name": "food_audit_task2"},
    {"task_id": 3, "task_name": "food_audit_task3"},
]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a food safety auditor for Indian food products. You audit food labels against 
    FSSAI 2020 regulations and ICMR-NIN dietary guidelines.

    For each product, you receive:
    - Product name, brand, category
    - Raw ingredients list
    - Nutrition per 100g (energy_kcal, sugars_g, sodium_mg, fat_g)
    - Marketing claims (front-of-pack)
    - Nutri-Score (A-E) and NOVA group (1-4, where 4 is ultra-processed)
    - User's health profile and conditions

    Your task is to:
    1. Assess the risk level (0-4: none/low/moderate/high/critical)
    2. Flag problematic ingredients for the user's health profile
    3. Identify FSSAI violation codes if applicable
    4. Suggest a safer alternative product ID if available

    FSSAI Violation Codes:
    - FSSAI-2020-HIGH-SODIUM: Sodium > 600mg per 100g
    - FSSAI-2020-HIGH-SUGAR: Added sugars misleading claims
    - FSSAI-2020-TRANS-FAT: Trans fat content violations
    - ICMR-NIN-SUGAR-LIMIT: Exceeds ICMR daily sugar guidelines
    - ICMR-NIN-SODIUM-LIMIT: Exceeds ICMR daily sodium guidelines

    Risk Level Guidelines:
    - 0 (none): Product safe for user's profile
    - 1 (low): Minor concerns, generally acceptable
    - 2 (moderate): Some nutrients exceed user's thresholds
    - 3 (high): Multiple violations or significant health risk
    - 4 (critical): Severe risk for user's conditions, avoid completely

    Watch for adversarial cases:
    - "Multigrain" products with refined flour as first ingredient
    - "No added sugar" with maltodextrin, dextrose, or corn syrup
    - "Low salt" claims with sodium > 600mg
    - "Natural/organic" labels on NOVA 4 ultra-processed foods

    Respond ONLY with a valid JSON object (no markdown, no explanation outside JSON):
    {
        "risk_level": <int 0-4>,
        "flagged_ingredients": ["ingredient1", "ingredient2"],
        "violation_codes": ["FSSAI-2020-HIGH-SODIUM"],
        "alternative_product_id": "<product_id or empty string>",
        "explanation": "Brief reasoning"
    }
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_summary = action[:80] + "..." if len(action) > 80 else action
    print(
        f"[STEP] step={step} action={action_summary} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(obs) -> str:
    """Build the prompt from the AuditObservation."""
    nutrition = obs.nutrition_per_100g
    nutrition_str = ", ".join(f"{k}: {v}" for k, v in nutrition.items()) if nutrition else "Not available"
    claims_str = ", ".join(obs.marketing_claims) if obs.marketing_claims else "None"
    conditions_str = ", ".join(obs.user_conditions) if obs.user_conditions else "None"

    return textwrap.dedent(
        f"""
        **Step {obs.step_number} of {obs.total_steps}**

        **Product Information:**
        - Product ID: {obs.product_id}
        - Name: {obs.product_name}
        - Brand: {obs.brand}
        - Category: {obs.category}

        **Ingredients (as on label):**
        {obs.ingredients_text}

        **Nutrition per 100g:**
        {nutrition_str}

        **Marketing Claims:**
        {claims_str}

        **Nutri-Score:** {obs.nutri_score or "Not available"}
        **NOVA Group:** {obs.nova_group} (1=minimally processed, 4=ultra-processed)

        **User Profile:**
        - Profile ID: {obs.user_profile_id}
        - Health Conditions: {conditions_str}

        Analyze this product for the given user profile and provide your audit decision as JSON.
        """
    ).strip()


def parse_model_response(response_text: str) -> AuditAction:
    """Parse the model's JSON response into an AuditAction."""
    try:
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(response_text)

        return AuditAction(
            risk_level=max(0, min(4, int(data.get("risk_level", 2)))),
            flagged_ingredients=data.get("flagged_ingredients", []),
            violation_codes=data.get("violation_codes", []),
            alternative_product_id=data.get("alternative_product_id", ""),
            explanation=data.get("explanation", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[DEBUG] JSON parse error: {e}, using default action", flush=True)
        return AuditAction(
            risk_level=2,
            flagged_ingredients=[],
            violation_codes=[],
            alternative_product_id="",
            explanation="Parse error, defaulting to moderate risk",
        )


def get_model_action(client: OpenAI, obs) -> AuditAction:
    """Query the model and get an AuditAction."""
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            timeout=LLM_TIMEOUT,
        )
        response_text = (completion.choices[0].message.content or "").strip()
        return parse_model_response(response_text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return AuditAction(
            risk_level=2,
            flagged_ingredients=[],
            violation_codes=[],
            alternative_product_id="",
            explanation="Model request failed",
        )


async def run_task(
    env: FoodLabelAuditorEnv,
    client: OpenAI,
    task_id: int,
    task_name: str,
) -> None:
    """Run a single task end-to-end, emitting [START]/[STEP]/[END] logs."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    max_possible_score = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(seed=SEED, task_id=task_id)
        obs = result.observation
        total_steps = obs.total_steps
        max_possible_score = total_steps * 1.0
        print(f"[DEBUG] Task {task_id}: {total_steps} products to audit", flush=True)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_action(client, obs)
            action_str = f"risk={action.risk_level},flags={len(action.flagged_ingredients)}"

            result = await env.step(action)
            obs = result.observation

            reward = result.reward if result.reward is not None else 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        total_reward = sum(rewards)
        score = total_reward / max_possible_score if max_possible_score > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.3

    except Exception as e:
        print(f"[DEBUG] Task {task_id} episode error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await FoodLabelAuditorEnv.from_docker_image(
        IMAGE_NAME,
        connect_timeout_s=30.0,
        message_timeout_s=120.0,
    )

    try:
        for task_cfg in ALL_TASKS:
            await run_task(
                env=env,
                client=client,
                task_id=task_cfg["task_id"],
                task_name=task_cfg["task_name"],
            )
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
