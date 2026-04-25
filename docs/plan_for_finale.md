# HouseholdBasketEnv — Design Document v2

**Status:** Revised after red-team review, awaiting final approval
**Depends on:** FoodLabelAuditor v1 (Round 1, FROZEN)
**Target:** OpenEnv hackathon finale, Bangalore

## 1. Architectural principle — wrapper, not replacement

Unchanged from v1. New package `household_basket_env/` holds a reference to `food_label_auditor` and delegates *single-product* scoring to it. **[v2]** Basket-level scoring is explicitly ours — not Round 1's — and lives in a new module.

**[v2]** Revised file layout:

```
household_basket_env/
├── __init__.py
├── models.py
├── client.py
├── openenv.yaml
├── pyproject.toml
├── server/
│   ├── environment.py
│   ├── app.py
│   ├── rewards.py
│   ├── basket_grader.py       ← NEW v2: cumulative basket-level check
│   ├── curriculum.py
│   ├── household_fixtures.py
│   ├── seed_verifier.py       ← NEW v2: offline solver for curriculum seeds
│   ├── Dockerfile
│   └── requirements.txt
└── tests/
    ├── test_env_contract.py
    ├── test_rewards.py
    ├── test_basket_grader.py  ← NEW v2
    ├── test_reward_hacks.py   ← expanded to 10 policies
    └── test_seed_verifier.py  ← NEW v2
```

## 2. Schemas

**[v2]** `BasketObservation` gets two new fields. `BasketAction` unchanged.

| Field | Type | Purpose |
|---|---|---|
| `prompt` | str | task instructions + current situation |
| `household` | list[MemberSummary] | per-member conditions, **cumulative intake so far**, thresholds cap |
| `basket_so_far` | list[TaggedItem] | items + member tags |
| `budget_remaining` | float | INR |
| `candidates` | list[ProductSummary] | tier-dependent catalog |
| `step_index` | int | **[v2]** counts valid steps only — see §4.4 |
| `attempt_index` | int | **[v2] NEW** counts every step including parse errors; for logging, not reward |
| `max_steps` | int | 3 / 5 / 7 |
| `seed` | int | **[v2] NEW** echoed back for reproducibility debugging |
| `reward` | float | last step's dense reward |
| `done` | bool | |
| `parse_error` | str \| None | populated on invalid action |

`MemberSummary` now exposes `cumulative_intake` (dict: nutrient → g/mg consumed so far across items tagged to this member) and `thresholds_cap` (hard caps from profile). **[v2]** Agent can reason about margin to cap explicitly — this enables the redesigned R_threshold in §4.2.

`BasketState` (internal): adds `seed: int`, `rng` (a seeded `random.Random` instance), `verified: bool` (checked against `seed_verifier` at reset).

## 3. Episode lifecycle and task tiers

**[v2]** Reset now takes a `seed: int` parameter. Household composition, budget jitter (±10%), and candidate ordering all derive from that seed via the internal RNG. Two resets with the same seed produce byte-identical observations. This is load-bearing for the deterministic-grader claim.

| Tier | Members | Valid steps | Catalog | Budget (INR) | Target p(reward>0), random-valid policy |
|---|---|---|---|---|---|
| Task 1 | 1 healthy adult | 3 | 20 curated | 500 | ≥0.50 |
| Task 2 | 2 (healthy + diabetic) | 5 | 50 | 1000 | ≈0.20 |
| Task 3 | 3 (diabetic + hypertensive + child) | 7 | 492 incl. adversarial 32 | 1500 | ≈0.05 |

**[v2]** Task 1 catalog design requirement: it must contain **at least one non-obvious coupling** — a pair or triple of products that individually look fine but jointly exceed a cap. Specifically: two items both marketed as "low fat" that jointly push sodium past the healthy-adult cap when both are in a 3-item basket. A prompted base model picks both routinely because each looks good in isolation. A trained model should learn the coupling. **This is what makes Task 1 RL-worthy rather than prompt-solvable.** If we can't construct this coupling from existing products, we fabricate one adversarial product for the Task 1 catalog specifically.

## 4. Reward decomposition — redesigned

**[v2]** Four positive signals (was 5), plus the terminal. Removed the collinear R_variety in its original form; replaced with R_meal_type_coverage.

### 4.1 R_format — unchanged

+0.2 dense when action JSON validates. −0.5 on parse error. Can't be gamed, not collinear with anything.

### 4.2 R_threshold — redesigned **[v2]**

Was: *"+0.4 when member stays under cap."* Broken — rewards picking zero-nutrient items.

Now: piecewise function on **margin remaining in healthy band**. For each of the member's watched nutrients (sugar, sodium, saturated fat for diabetic; sodium, saturated fat for hypertensive; protein/fiber floors for child):

| Post-item cumulative intake vs cap | R contribution per nutrient |
|---|---|
| 0–30% of cap | +0.05 (under-consumption is mildly penalized at terminal, not here) |
| 30–85% of cap ← **healthy band** | +0.10 (full credit) |
| 85–100% of cap | +0.02 (near-risk) |
| >100% of cap | −0.30 (violation) |

Dense reward per step is the sum across the member's watched nutrients, clipped to the range [−0.6, +0.4]. This closes the "buy water forever" loophole because bottled water parks the agent in the 0–30% tier, earning little. It also creates a gradient toward actual nutrition.

### 4.3 R_budget — unchanged

+0.1 when item's price ≤ per-step allowance = `budget_remaining / steps_remaining`.

### 4.4 R_meal_type_coverage — replaces R_variety **[v2]**

Each product in the catalog gets a `meal_type` tag: `staple` (rice, atta, oil), `protein` (dal, eggs, paneer), `vegetable`, `dairy`, `snack`, `beverage`. This tag is orthogonal to nutrition thresholds — a basket can satisfy thresholds while missing entire meal types (e.g., all-vegetable basket passes sugar/sodium/fat caps but fails meal-type coverage).

+0.15 dense when the picked item's meal_type is new to the member's tagged subset. Hard cap: one bonus per meal_type per member.

Why orthogonal to R_terminal: R_terminal checks thresholds + minimum-intake floor. Meal-type coverage is a separate dimension — a basket can satisfy both floors and caps while being monotonously all-protein or all-staple. The signals now produce different gradients.

### 4.5 R_terminal — now owns a real grader **[v2]**

Delegates to **new** `basket_grader.py`, not Round 1. The new grader:

1. For each member, iterates their tagged subset, sums cumulative nutrients from `products.json` data.
2. Checks each sum against the member's threshold cap (existing `user_profiles.json` thresholds).
3. Checks each member hits minimum-intake floors for calorie, protein, fiber (these come from ICMR-NIN data already referenced in Round 1 thresholds).
4. For each item individually, calls Round 1's `grade_step(product, profile)` as a per-item sanity check — catches allergens and profile-incompatible items.

**[v2]** Round 1's contribution is now explicit: per-item sanity, not basket-level scoring. Basket scoring is new code we own.

Terminal reward = +1.0 if every member passes all four checks. −0.5 if any member violates a cap. Partial credit: +0.3 if every item passes per-item sanity but cumulative caps violated.

### 4.6 Penalties — **[v2]** expanded

| Penalty | Weight | Fires when |
|---|---|---|
| P_parse | −0.5 | invalid JSON |
| P_duplicate | −0.3 | product_id already in basket |
| P_unknown_member | −0.4 | member_id not in household |
| P_over_budget | terminates | cumulative spend > budget |

### 4.7 Step-advance semantics — decided **[v2]**

Option A confirmed: **step_index advances only on valid actions. attempt_index advances every step.** Parse errors cost the parse penalty and the agent gets another attempt within the same episode. Cap attempts at `max_steps × 2` to prevent infinite loops on truly stuck models — if the agent burns 2N attempts, episode terminates with current reward.

This makes reward attribution consistent: every completed episode has exactly N valid picks. Across rollouts, reward distributions are comparable.

## 5. Reward-hack suite — 10 policies **[v2]**

| Policy | Expected max reward | Test asserts |
|---|---|---|
| AlwaysBuyRice | < 1.0 | terminal floor violation for non-rice-tolerant members |
| AlwaysCheapestItem | < 1.5 | threshold violations dominate |
| AlwaysSameCategory | < 2.0 | meal-type coverage never fires past 1 |
| MinCostIgnoreHealth | < 1.0 | terminal fails for diabetic/hypertensive |
| RandomValidJSON | measured p(reward>0) ≥ tier target | baseline reproducibility |
| OneMemberGetsEverything | < half of max | other members fail terminal |
| BuySameItemNTimes | < 1.5 | P_duplicate dominates |
| EmptyBasketPolicy | exactly 0 | bounded-below check |
| **AlwaysPickFromAdversarialSet** **[v2]** | < 1.5 on Task 3 | adversarial items fail per-item sanity at terminal |
| **TerminalOnlyPolicy** **[v2]** | bounded by dense reward floor | proves terminal alone can't be gamed without dense accrual |

## 6. Training plan — revised **[v2]**

**Model:** Qwen2.5-3B-Instruct, Unsloth 4-bit QLoRA, rank 16, alpha 32, dropout 0.05.

**Training distribution: [v2]** Task 1 **and** Task 2 mixed, 80/20 weighted. Task 3 reserved for eval only. This narrows the generalization axes the trained model has to cover.

**GRPO config:**
- 4 prompts × 8 generations = 32 rollouts/step
- temperature 0.8, top_p 0.9
- **[v2]** max_new_tokens = **128** (was 256)
- learning_rate 5e-6, cosine schedule
- **[v2] beta = 0.1** (KL penalty, was unspecified)
- **[v2]** reference model = base Qwen2.5-3B-Instruct, LoRA off
- **[v2]** log KL divergence every step — if KL climbs past 5.0, pause and inspect

**[v2]** Target: **50 GRPO steps**, not 100. At ~3.5–4 min/step realistic on T4, that's ~3 hours of a Colab session with checkpointing every 10 steps.

**Success criteria:**
- Mean reward on Task 1 climbs from random-valid baseline (~0.8) to ≥2.0
- Valid-JSON rate climbs from ~90% to ≥99%
- KL stays bounded under 5.0
- Reward on Task 2 (held-out during training mix, evaluated on fresh seeds) shows positive delta
- **[v2]** Reward on **Round 1's Task 1** using trained model ≥ baseline Qwen2.5-3B on Round 1's Task 1 (continuity check)

## 7. Seed verification **[v2] NEW**

Before any curriculum seed ships, it must be proven achievable. `seed_verifier.py` implements a greedy solver:

1. For each seed and each tier, reconstruct the household, catalog, budget.
2. Greedy pick: at each step, choose the product that maximizes R_threshold + R_meal_type_coverage, breaks budget ties toward cheaper.
3. Run all dense + terminal rewards on the greedy basket.
4. Seed is "verified" iff greedy achieves reward ≥ 0.6 × theoretical max.

Only verified seeds go into `curriculum.py`. Target: 100 verified seeds per tier. Run as offline script, results checked into repo. This is a one-time cost, not runtime.

## 8. Demo script — 100 seconds **[v2]**

**[v2]** One extra slide added for Round 1 continuity.

1. 0–15s — problem, FSSAI, 492 products, household with conflicts
2. 15–30s — env repo tour, `reset(seed=42)` → observation
3. 30–50s — baseline Qwen2.5-3B on Task 3, fails (sugar cap violated)
4. 50–70s — trained model on same seed, succeeds, reward curve shown
5. **70–85s — [v2]** trained model on Round 1's Task 1, matches baseline → continuity story
6. 85–95s — `pytest tests/test_reward_hacks.py -v`, all 10 green
7. 95–100s — questions

## 9. Execution order — revised **[v2]**

1. **Phase 1** — scaffolding, schemas, Dockerfile, `openenv validate` ✅
2. **[v2] Phase 1.5** — seed verifier implementation + offline run, produce verified seed list
3. **Gate A — Baseline sanity** — 50 random-valid rollouts on Task 1 verified seeds, p(reward>0) ≥ 0.5
4. **Phase 2** — core env logic, `test_env_contract.py` green
5. **Phase 3** — rewards module + `basket_grader.py`, unit tests green
6. **Phase 4** — reward-hack suite (10 policies), all bounded as expected
7. **Phase 5** — baseline eval notebook, numbers recorded for Task 1/2/3 + Round 1 Task 1
8. **Phase 6** — GRPO training notebook, 50 steps mixed Task 1+2
9. **Phase 7** — trained eval, deltas recorded, KL inspected
10. **Phase 8** — `openenv push`, README, demo recording

## 10. Explicitly out of scope for v1

Same as v1, plus: pantry decay, recipe graphs, 14-day horizons, conflict subgames, LLM-as-judge, images, tool use.

## 11. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Task 1 coupling is too easy for prompted base | Medium | After Gate A, run a prompted (no training) Qwen on Task 1 with coupling. If it scores within 10% of random-valid ceiling, the coupling is weak — strengthen before training. |
| basket_grader has a bug that AlwaysPickFromAdversarialSet exploits | Medium | Test exists; if it fails, fix grader not test |
| KL climbs past 5.0 during training | Medium | Stop training, inspect generations, raise beta to 0.2 |
| Seed verifier marks too few Task 3 seeds verifiable | Low–medium | Relax greedy threshold to 0.5 × max; if still bad, Task 3 catalog is too adversarial — tune |
| Round 1 continuity eval shows catastrophic forgetting | Medium | Honest slide in demo acknowledging it + lower mixing ratio (90/10 instead of 80/20) in a second training run |
| Colab disconnect mid-50-steps | High | Checkpoint every 10 steps to Drive, notebook auto-resumes |
| `openenv push` fails on sibling-dir Dockerfile | Low | Vendor Round 1 as copied sub-package inside v2 if needed |
