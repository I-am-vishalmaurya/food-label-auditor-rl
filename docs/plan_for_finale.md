# HouseholdBasketEnv — Design Document v3

**Status:** Revised after judge-perspective red-team review
**Theme:** OpenEnv Hackathon #3.2 — Personalized Tasks
**Target:** OpenEnv hackathon finale, Bangalore, Apr 26 2026
**Changes from v2:** training distribution restructured, prompted-baseline gate added, two ablations, demo rewritten for storytelling, R_threshold smoothed, KL/entropy monitoring tightened, framing rewritten to lead with novelty.

---

## 0. The pitch (read this first)

**HouseholdBasketEnv is a multi-stakeholder constraint-satisfaction environment for personalized grocery planning.** A single agent must compose a basket that simultaneously satisfies the conflicting nutritional constraints of multiple household members — e.g., a diabetic adult's sugar cap and a growing child's protein floor and a hypertensive grandparent's sodium cap — under a hard budget, drawing from a real catalog of 492 Indian packaged-food products with FSSAI-derived nutrition data.

What makes this RL-worthy and not prompt-solvable: **the constraints couple across items**. Two products that look healthy in isolation can jointly violate a member's cap. The agent has to learn the coupling structure, not just read labels. We demonstrate this with a held-out hard tier (Task 3) the model never sees during training and an adversarial product set designed to fool surface-level pattern matching.

This fits Theme #3.2 (Personalized Tasks) because the household profile *is* the personalization signal — the same prompt produces different optimal baskets for different households, and the agent has to internalize "what does this specific family need."

---

## 1. Architectural principle

`household_basket_env/` is an OpenEnv-compliant package. Per-item nutritional sanity uses a vendored grading utility (`food_label_auditor`); **basket-level scoring, multi-member constraint reasoning, and all RL-relevant logic are new code in this package**. The basket grader is the contribution.

File layout:

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
│   ├── basket_grader.py        ← cumulative basket-level check (ours)
│   ├── curriculum.py
│   ├── household_fixtures.py
│   ├── seed_verifier.py        ← offline solver, proves seeds are achievable
│   ├── Dockerfile
│   └── requirements.txt
└── tests/
    ├── test_env_contract.py
    ├── test_rewards.py
    ├── test_basket_grader.py
    ├── test_reward_hacks.py    ← 10 adversarial policies
    └── test_seed_verifier.py
```

---

## 2. Schemas

`BasketObservation`:

| Field | Type | Purpose |
|---|---|---|
| `prompt` | str | task instructions + current situation |
| `household` | list[MemberSummary] | per-member conditions, cumulative intake so far, thresholds |
| `basket_so_far` | list[TaggedItem] | items + member tags |
| `budget_remaining` | float | INR |
| `candidates` | list[ProductSummary] | tier-dependent catalog |
| `step_index` | int | counts valid steps only |
| `attempt_index` | int | counts every step including parse errors; for logging |
| `max_steps` | int | 3 / 5 / 7 |
| `seed` | int | echoed for reproducibility debugging |
| `reward` | float | last step's dense reward |
| `done` | bool | |
| `parse_error` | str \| None | populated on invalid action |

`MemberSummary` exposes `cumulative_intake` (dict: nutrient → quantity consumed across items tagged to this member) and `thresholds_cap` (hard caps from profile). The agent can reason about margin to cap explicitly.

`BasketState` (internal): `seed: int`, `rng` (seeded `random.Random`), `verified: bool` (checked against `seed_verifier` at reset).

**Reset is deterministic from seed.** Household composition, budget jitter (±10%), and candidate ordering all derive from the seed via the internal RNG. Two resets with the same seed produce byte-identical observations.

---

## 3. Episode lifecycle and task tiers

| Tier | Members | Valid steps | Catalog | Budget (INR) | Role |
|---|---|---|---|---|---|
| Task 1 | 1 healthy adult | 3 | 20 curated | 500 | **Held-out easy eval** (regression check) |
| Task 2 | 2 (healthy + diabetic) | 5 | 50 | 1000 | **Training (70%)** |
| Task 3 | 3 (diabetic + hypertensive + child) | 7 | 492 incl. 32 adversarial | 1500 | **Training (30%) + held-out hard eval seeds** |

**Adversarial products in Task 3:** 32 items deliberately constructed (or curated) so each looks healthy in isolation — "low fat," "no added sugar," "high protein" marketing — but combinations of them violate caps. A trained model should learn to deprioritize this set; a prompted model picks them readily because each label looks fine.

**Train/eval split for Task 3:** 100 verified seeds total, 70 used during training, 30 held out. Held-out seeds are never seen during GRPO and form the headline eval number.

---

## 4. Reward decomposition

Four positive signals plus terminal, all dense except R_terminal.

### 4.1 R_format

+0.2 dense when action JSON validates. **−0.25 on parse error** (matched to format-reward scale).

### 4.2 R_threshold — smooth triangular

For each watched nutrient on the member tagged to the picked item:

```
let m = (post-pick cumulative intake) / (member's cap)
R_per_nutrient(m) =
    -0.30                       if m > 1.00            (violation)
    +0.10 * (1 - |m - 0.60|/0.40)  if 0.20 ≤ m ≤ 1.00   (triangular peak at 60% of cap)
    +0.02                       if m < 0.20            (under-consumption, mildly positive)
```

Triangular kernel peaks at +0.10 when post-pick intake is at 60% of cap (the "healthy band center"), decays linearly to 0 at 20% and 100%, and crashes to −0.30 above cap. Sum across the member's watched nutrients, clip to [−0.6, +0.4] per step.

This closes the "buy water forever" loophole (water parks the agent at m≈0, earning only +0.02 per nutrient) and creates a smooth gradient toward genuinely healthy choices.

### 4.3 R_budget

+0.1 when item's price ≤ per-step allowance (`budget_remaining / steps_remaining`).

### 4.4 R_meal_type_coverage

Each catalog product has a `meal_type` tag: `staple`, `protein`, `vegetable`, `dairy`, `snack`, `beverage`. +0.15 dense when the picked item's meal_type is new to the member's tagged subset. Hard cap: one bonus per meal_type per member.

**Honest framing:** this is a dense proxy for the terminal-floor compliance check, not an orthogonal signal. It rewards meal-type diversity early so the agent doesn't fill the basket with one category and fail the floor at terminal. The ablation (§7) measures whether this proxy actually helps vs. relying on terminal alone.

### 4.5 R_terminal

Delegates to `basket_grader.py`:

1. For each member, sum cumulative nutrients from products in their tagged subset.
2. Check each sum against the member's threshold caps.
3. Check each member hits minimum-intake floors for calorie, protein, fiber (ICMR-NIN derived).
4. For each item individually, run per-item sanity (allergen check, profile compatibility).

Terminal reward:
- **+1.0** if every member passes all four checks
- **+0.3** partial credit if every item passes per-item sanity but cumulative caps violated
- **−0.5** if any member exceeds a cap (hard violation)

### 4.6 Penalties

| Penalty | Weight | Fires when |
|---|---|---|
| P_parse | −0.25 | invalid JSON |
| P_duplicate | −0.3 | product_id already in basket |
| P_unknown_member | −0.4 | member_id not in household |
| P_over_budget | terminates | cumulative spend > budget |

### 4.7 Step-advance semantics

`step_index` advances only on valid actions. `attempt_index` advances every step. Parse errors cost P_parse and the agent gets another attempt within the same episode. Cap attempts at `max_steps × 2`. If hit, episode terminates with **current accumulated reward** (no additional penalty — the parse penalties already accumulated are the cost).

---

## 5. Reward-hack suite — 10 policies

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
| AlwaysPickFromAdversarialSet | < 1.5 on Task 3 | adversarial items fail per-item sanity at terminal |
| TerminalOnlyPolicy | bounded by dense reward floor | proves terminal alone can't be gamed without dense accrual |

All 10 must pass before training starts.

---

## 6. Training plan

**Model:** Qwen2.5-3B-Instruct, Unsloth 4-bit QLoRA, rank 16, alpha 32, dropout 0.05.

**Training distribution:** Task 2 (70%) + Task 3 training seeds (30%). Task 1 is held-out eval. Task 3 held-out seeds (30 of 100) are never seen during training.

**GRPO config:**
- 4 prompts × 8 generations = 32 rollouts/step
- temperature 0.8, top_p 0.9
- max_new_tokens = 128
- learning_rate 5e-6, cosine schedule
- **beta = 0.1** (KL penalty)
- reference model = base Qwen2.5-3B-Instruct, LoRA off
- **80 GRPO steps** main run, checkpoint every 10 steps

**Monitoring (per step):**
- Mean reward, broken down by reward component
- Valid-JSON rate
- **Per-token KL divergence** — inspect at 2.0, abort at 4.0
- **Action-distribution entropy** — guards against mode collapse
- Sample 4 generations per step into a log file for spot inspection

**Success criteria:**
- Mean reward on Task 2 climbs from prompted baseline to ≥ +30% delta
- Mean reward on Task 3 **held-out seeds** shows positive delta vs prompted baseline
- Mean reward on Task 1 (held-out, never trained on) within 10% of prompted baseline (no regression)
- Valid-JSON rate ≥ 99% by step 40
- Per-token KL stays under 4.0 throughout
- Entropy doesn't collapse (defined: action-distribution entropy stays above 50% of initial value)

---

## 7. Ablations — two runs

Both at 40 GRPO steps each, evaluated on the same Task 2 + Task 3 held-out seeds as the main run.

**Ablation A — no R_meal_type_coverage.** Tests whether the dense meal-type proxy is doing real work or merely shadowing the terminal floor. Hypothesis: removing it slows convergence and causes more terminal-floor failures. If results show no difference, R_meal_type_coverage was redundant and we say so honestly in the README.

**Ablation B — no curriculum, Task 3 only from step 0.** Tests whether the 70/30 mix helps. Hypothesis: pure Task 3 has sparser reward, slower convergence, possibly higher final ceiling on Task 3 itself. Useful either way — confirms the curriculum decision or tells us we left performance on the table.

Results from both ablations go in a single table in the README. This is the kind of evidence judges almost never see at hackathons.

---

## 8. Seed verification

Before any curriculum seed ships, it must be proven achievable. `seed_verifier.py`:

1. For each seed and each tier, reconstruct the household, catalog, budget.
2. Greedy pick: at each step, choose the product maximizing R_threshold + R_meal_type_coverage, breaking budget ties toward cheaper.
3. Run all dense + terminal rewards on the greedy basket.
4. Seed verified iff greedy achieves reward ≥ 0.6 × theoretical max.

Only verified seeds enter `curriculum.py`. Target: 100 verified seeds per tier. Run as offline script, results checked into repo.

---

## 9. Demo script — 115 seconds

| Time | Beat | Content |
|---|---|---|
| 0–15s | **Hook** | "Indian household, three members, conflicting health needs, 1500 rupees, 7 picks. Diabetic grandfather, hypertensive mother, growing child. What goes in the cart?" Show the household profile slide. |
| 15–30s | **Why this is hard** | Show two products both labeled "low fat" — explain that picking both pushes the diabetic over his sugar cap. "Constraints couple across items. You can't solve this by reading labels." |
| 30–45s | **The environment** | Quick repo tour, `reset(seed=42)` → observation. Show seed verifier output: "every training seed provably solvable." |
| 45–65s | **Baseline fails** | Prompted Qwen2.5-3B on a held-out Task 3 seed. It picks the two low-fat-but-high-sugar items. Terminal reward: −0.5. Show reasoning trace. |
| 65–85s | **Trained model succeeds** | Same seed, trained model. It avoids the coupling, picks paneer + dal + atta + vegetables. Terminal: +1.0. Show reward curve: prompted baseline → trained, on Task 3 held-out. |
| 85–100s | **Qualitative before/after** | Side-by-side basket comparison on a second hard seed. One-line caption per basket explaining what the model learned. **This is the storytelling slide.** |
| 100–110s | **Honesty slide** | Ablation table (A and B), reward-hack suite green, KL stayed under 2.5 throughout. One failure case shown: "trained model still struggles when budget < 1200 — here's why." |
| 110–115s | **Close** | Repo URL, Space URL, README link. |

**Recording:** locally via Docker against a stable container, not against live Colab. Pre-record once, review, re-record if needed.

---

## 10. Execution order

1. **Phase 1** — scaffolding, schemas, Dockerfile, `openenv validate` ✅
2. **Phase 1.5** — seed verifier implementation + offline run, produce verified seed lists for all three tiers
3. **Gate A — Baseline sanity** — 50 random-valid rollouts on Task 2 verified seeds, confirm p(reward>0) ≈ 0.20
4. **Phase 2** — core env logic, `test_env_contract.py` green
5. **Phase 3** — rewards module + `basket_grader.py`, unit tests green
6. **Phase 4** — reward-hack suite (10 policies), all bounded as expected
7. **Phase 4.5 — Prompted baseline gate (NEW, critical)** — run prompted Qwen2.5-3B on 30 seeds each of Task 2 and Task 3 held-out. **If prompted Qwen scores within 15% of theoretical max on Task 2, the task is too easy and we escalate difficulty before training.** Specific escalation levers: tighten budget, add adversarial products to Task 2, increase member count to 3.
8. **Phase 5** — full baseline eval notebook, numbers recorded for Task 1, Task 2, Task 3 held-out
9. **Phase 6** — main GRPO run, 80 steps, mixed Task 2 + Task 3
10. **Phase 7** — Ablation A (no R_meal_type_coverage), 40 steps
11. **Phase 8** — Ablation B (Task 3 only, no curriculum), 40 steps
12. **Phase 9** — trained eval on all three tiers' held-out seeds, deltas recorded, KL/entropy curves saved
13. **Phase 10** — qualitative before/after generation: pick 3 hard seeds, generate baskets from prompted vs. trained, write captions
14. **Phase 11** — `openenv push`, README with embedded plots and ablation table, demo recording

---

## 11. Out of scope for v1

Pantry decay, recipe graphs, 14-day horizons, conflict subgames, LLM-as-judge, images, tool use, multi-agent (one shopper per episode).

---

## 12. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Prompted baseline matches trained model** | **Medium-high** | **Phase 4.5 gate catches this before training. If hit, escalate Task 2 difficulty.** |
| Task 3 adversarial set is gameable through reward hacking | Medium | `AlwaysPickFromAdversarialSet` test exists; if it fails, fix grader not test |
| GRPO mode collapse | Medium | Entropy monitoring at every step; if entropy halves, raise temperature mid-run or abort |
| KL climbs past 4.0 during training | Medium | Inspect at 2.0, raise beta to 0.2 if pattern persists |
| Seed verifier marks too few Task 3 seeds verifiable | Low-medium | Relax greedy threshold to 0.5 × max; if still bad, Task 3 catalog is too adversarial — tune |
| Colab disconnect mid-training | High | Checkpoint every 10 steps to Drive, notebook auto-resumes |
| Demo recording fails on live Colab | High | Record locally against Docker container, not live Colab |
| `openenv push` fails on dependency layout | Low | Vendor dependencies as copied sub-package if needed |
| Ablation results contradict main hypothesis | Low (and fine) | Report honestly. Negative results in ablations build credibility, not destroy it. |

---

## 13. Mapping to judging criteria

| Criterion | Weight | How v3 addresses it |
|---|---|---|
| Environment Innovation | 40% | Multi-stakeholder constraint-satisfaction framing; coupled-constraint structure that defeats prompted baselines; 492-product catalog with adversarial subset; held-out hard eval; seed verifier proves seeds are non-trivially achievable. |
| Storytelling | 30% | Demo opens with concrete Indian household scenario; qualitative before/after slide on a hard seed; honesty slide showing one failure case + ablation table; clear theme fit (#3.2 Personalized Tasks). |
| Reward Improvement | 20% | Prompted baseline (not random-valid) as the reference; held-out Task 3 seeds as headline number; Task 1 regression check; reward curves embedded in README with both axes labeled. |
| Reward & Pipeline | 10% | 10 reward-hack policies, all green; smooth differentiable R_threshold; KL + entropy monitoring; two ablations validating reward design choices. |