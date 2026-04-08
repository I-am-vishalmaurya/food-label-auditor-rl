---
title: Food Label Auditor Environment
emoji: 🏷️
colorFrom: green
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - health
  - food-safety
  - india
  - fssai
  - rl-environment
---

# Food Label Auditor Environment

An OpenEnv environment where an RL agent audits Indian food product labels against **FSSAI 2020 regulations** and **ICMR-NIN dietary guidelines**. The agent reads raw ingredient labels and must classify risk, flag violations, and suggest safer alternatives — personalized to different health profiles.

**Domain gap filled:** The first OpenEnv environment for health decision-making, food safety, and regulatory compliance. No existing environment in the OpenEnv Hub covers this domain.

## Quick Start

```python
from food_label_auditor import FoodLabelAuditorEnv, AuditAction

# Connect to a running server
with FoodLabelAuditorEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Product: {result.observation.product_name}")
    print(f"Ingredients: {result.observation.ingredients_text}")

    result = env.step(AuditAction(
        risk_level=3,
        flagged_ingredients=["palm oil", "salt"],
        violation_codes=["FSSAI-2020-HIGH-SODIUM"],
        alternative_product_id="IND_011",
        explanation="High sodium for diabetic profile, NOVA 4 ultra-processed",
    ))
    print(f"Reward: {result.reward}")
```

## Building the Docker Image

```bash
docker build -t food-label-auditor:latest -f server/Dockerfile .
```

## Who Is the Agent?

The agent IS the human standing in a supermarket reading a food label. Every step, it receives:
- A raw ingredient list (exactly as printed on the packet)
- Nutritional data per 100g
- Front-of-pack marketing claims
- The user's health profile (age, conditions, dietary restrictions)

The agent must decide: **How risky is this product for this person?**

This is a task humans do thousands of times across their lives. FSSAI compliance officers, nutritionists, and health-conscious consumers all perform this exact decision loop daily.

## Three Tasks with Wide Difficulty Spread

### Task 1: Easy — Single Product Scan
- 1 product (always NOVA-4, Nutri-Score D/E), 1 health profile
- Any reasonable risk flag passes
- Tests basic food safety understanding

### Task 2: Medium — Multi-Product, Multi-Profile
- 10 products across 2 health profiles (e.g., diabetic adult + healthy child)
- Agent must give **different risk levels for the same product** depending on the user's health profile
- Tests personalized health reasoning

### Task 3: Hard — Adversarial Label Auditing
- 30 products including 8-10 **adversarial cases** where marketing claims contradict actual ingredients
- Examples of adversarial cases:
  - "Multigrain" bread where refined wheat flour (maida) is the first ingredient
  - "No added sugar" products containing maltodextrin, dextrose, or corn syrup
  - "Low salt" products with >700mg sodium per 100g
  - "100% natural" products that are NOVA group 4 (ultra-processed)
- Requires FSSAI-specific regulatory reasoning that frontier models cannot have memorized
- Tests contradiction detection between marketing language and ingredient reality

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `risk_level` | int (0-4) | 0=none, 1=low, 2=moderate, 3=high, 4=critical |
| `flagged_ingredients` | list[str] | Problematic ingredient names from the label |
| `violation_codes` | list[str] | FSSAI violation codes (e.g., `FSSAI-2020-HIGH-SODIUM`) |
| `alternative_product_id` | str | Product ID of a safer alternative in the same category |
| `explanation` | str | Reasoning for the audit decision |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `product_id` | str | Unique product identifier |
| `product_name` | str | Product name as printed on label |
| `brand` | str | Brand name |
| `category` | str | Food category (e.g., instant_noodles, biscuits) |
| `ingredients_text` | str | Raw ingredients list as printed on the label |
| `nutrition_per_100g` | dict | Nutritional values: energy_kcal, sugars_g, sodium_mg, fat_g |
| `marketing_claims` | list[str] | Front-of-pack marketing claims |
| `nutri_score` | str | Nutri-Score grade (A/B/C/D/E) |
| `nova_group` | int | NOVA processing level (1-4) |
| `user_profile_id` | str | Active health profile ID |
| `user_conditions` | list[str] | User's health conditions |

## Reward Function

The reward fires at **every step** (not sparse). Three components:

| Component | Weight | Signal |
|-----------|--------|--------|
| Risk correctness | 0.40 | Exact match = 1.0, off-by-one = partial, off-by-many = 0.0 |
| Ingredient recall | 0.30 | Fraction of expected flagged ingredients correctly identified |
| Precision + Alternative | 0.30 | Penalizes over-flagging; rewards valid same-category alternatives |

## Grader Formulas (Deterministic)

All graders reference a pre-computed `ground_truth.json` — zero runtime API calls, zero randomness.

**Task 1:** `score = (risk_correct * 0.4) + (any_true_ingredient_flagged * 0.3) + (alternative_in_same_category * 0.3)`

**Task 2:** `score = mean(per_scan_accuracy) + profile_differentiation_bonus` where per_scan = 1.0 exact, 0.5 off-by-one, 0.0 otherwise.

**Task 3:** `score = (adversarial_caught / total_adversarial * 0.5) + (overall_accuracy * 0.3) + (no_action_loops * 0.2)`

## Dataset

- **492 products** — 50 curated Indian products (Maggi, Parle-G, Amul, etc.) + 410 synthetic + 32 adversarial
- **4 health profiles** — diabetic adult, healthy child, hypertensive senior, healthy adult
- **1968 ground truth entries** — all (product, profile) pairs with pre-computed expected risk, violations, and flagged ingredients
- Thresholds derived from **ICMR-NIN RDA 2020** guidelines
- Violation codes mapped to **FSSAI Food Products Standards 2020**
- Product data seeded from **OpenFoodFacts India** (10,000+ Indian products)

## Deployment Pathway

An agent trained in FoodLabelAuditorEnv can be exported as a policy and deployed inside any food scanning application via a simple API call. The environment is not a proxy for the real task — it **IS** the real task, simulated. The inference interface accepts a product label + user profile and returns a structured audit decision.

## Research Applications

- **LLM safety alignment**: Teach models to reason about regulatory compliance, not just general nutrition knowledge
- **Personalized health AI evaluation**: Benchmark adaptive personalization over long horizons
- **Global South AI benchmarking**: First environment using FSSAI/ICMR-NIN (Indian regulatory standards) instead of Western-only data

## Development

```bash
# Run server locally
uvicorn server.app:app --reload

# Test environment logic directly
python3 -c "
from server.food_label_auditor_environment import FoodLabelAuditorEnvironment
from models import AuditAction
env = FoodLabelAuditorEnvironment()
obs = env.reset(seed=42, task_id=1)
print(obs.product_name, obs.nutri_score, obs.nova_group)
"

# Run tests
pytest tests/ -v
```

## Project Structure

```
food_label_auditor/
├── __init__.py
├── client.py                        # FoodLabelAuditorEnv client
├── models.py                        # AuditAction, AuditObservation, AuditState
├── openenv.yaml                     # OpenEnv manifest
├── pyproject.toml                   # Package config
├── README.md                        # This file
├── data/
│   ├── products.json                # 492 Indian food products
│   ├── user_profiles.json           # 4 health profiles
│   ├── ground_truth.json            # Pre-computed correct answers
│   └── adversarial_cases.json       # 32 adversarial label cases
├── scripts/
│   └── build_dataset.py             # Dataset generation script
├── server/
│   ├── __init__.py
│   ├── food_label_auditor_environment.py  # Core environment logic
│   ├── graders.py                   # 3 deterministic grader functions
│   ├── app.py                       # FastAPI application
│   ├── Dockerfile                   # Container image
│   └── requirements.txt             # Server dependencies
└── tests/
    ├── test_models.py
    ├── test_environment.py
    ├── test_graders.py
    └── test_integration.py
```
