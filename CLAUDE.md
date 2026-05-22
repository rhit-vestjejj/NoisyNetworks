# NoisyNetworks — workshop paper project notes

## Project goal

A small empirical workshop paper studying the **interaction** between two
established components of value-based deep RL:

- Exploration: NoisyNet (Fortunato et al. 2017) vs ε-greedy
- Replay: Prioritized Experience Replay (Schaul et al. 2016) vs uniform

Target venues: NeurIPS Deep RL Workshop, RLC workshops, ICLR Blog Posts /
Tiny Papers. Not a main-conference contribution.

## The novelty angle (lock this in)

Nobody has published the controlled **2×2 factorial** of
{NoisyNet, ε-greedy} × {Uniform, PER} with a proper interaction-effect
estimate. Rainbow (Hessel 2018) does leave-one-out ablations.
Revisiting Rainbow (Obando-Ceron & Castro 2021) does careful component-wise
sweeps. Neither isolates the pairwise interaction. Deep-literature scan
confirmed the gap.

**Framing rule:** the paper must be sold as a *targeted interaction study*,
not as "another Rainbow ablation." Title, abstract, and section headers must
make the interaction term the central object.

**Headline statistic** (commit to this before running analysis):

    Δ_interaction = (AUC_noisy_per − AUC_noisy_uniform)
                  − (AUC_eps_per   − AUC_eps_uniform)

Reported per environment with 95% bootstrap CI across seeds.

## Experimental design

### Main 2×2 factorial (`run_factorial.sh`)

- 4 conditions: {noisy_uniform, noisy_per, eps_uniform, eps_per}
- 5 seeds per condition
- 3 MiniGrid envs spanning reward sparsity:
  - `MiniGrid-Empty-8x8-v0` (dense, control)
  - `MiniGrid-DoorKey-6x6-v0` (medium)
  - `MiniGrid-MultiRoom-N2-S4-v0` (sparse)
- = 60 runs

### α-sweep (`run_alpha_sweep.sh`)

The cleanest novelty slice per the deep-research scan: does PER's optimal
prioritization exponent α depend on the exploration regime?

- α ∈ {0.4, 0.8} (α=0.6 reused from the factorial)
- 2 PER conditions × 2 new α values × 5 seeds = 20 runs
- Env: doorkey6 only (extend if interaction confirmed)

### Hypothesis behind the sweep

ε-greedy → uniformly random exploratory actions → TD-error distribution is
mostly low with occasional genuine high spikes → aggressive prioritization
(high α) helps. NoisyNet → *structured* correlated weight perturbations →
TD-error distribution may be noisier in a misleading way (cf. UPER,
Carrasco-Davis 2025) → optimal α may be lower. If the per-regime optimal α
differs, that's the clean novel finding.

## Scope decisions (do not re-litigate without strong reason)

- **Drop Atari.** Too expensive for server compute limits. Gridworlds only.
- **No DDQN as a third factor.** Vanilla DQN backbone only. A 2×2×2 design
  doubles runs without strengthening the interaction story.
- **MiniGrid runs on CPU, not GPU.** Network and obs are tiny (7×7×3,
  2 Conv + 2 FC, batch 64). GPU launch overhead exceeds compute. Force CPU
  on a GPU server with `CUDA_VISIBLE_DEVICES=""`. Atari would have been
  GPU-bound; MiniGrid is not.

## What's implemented

### `train_minigrid.py` diagnostic logging (already wired up)

Saved to each run's `log.npz`:
- `step, mean_reward, sigma, epsilon` — training curves
- `sigma_per_layer` — (n_log_steps × n_noisy_layers) trajectory. Lets us ask
  whether `fc1` and `fc2` decay differently under PER vs uniform.
- `td_mean, td_max, td_std` — TD-error stats per log step
- `noisy_layer_names` — names of the NoisyLinear layers, in order

Saved to `per_diag.npz` (PER runs only):
- `sample_counts` — per-slot sampling frequency over the full run
- `insert_step` — env step at which each slot was filled
- `sigma_at_insert` — σ level at insertion (UPER-style diagnostic: does PER
  over-sample high-σ transitions?)

Plus `args.json` per run for unambiguous regrouping in analysis.

### Driver scripts

- `run_factorial.sh` — 2×2 × seeds × envs, skip-existing/resumable, supports
  `PARALLEL=N`, `SMOKE=1`, `SEEDS=...`, `ENVS=...`.
- `run_alpha_sweep.sh` — α sweep with the same conventions.

## Mechanistic figures (what `analyze.py` will produce)

Headline:
1. **Per-env rliable IQM bar chart** across the 4 cells, 5 seeds, with 95%
   bootstrap CIs.
2. **Interaction term plot** — Δ_interaction per env, with CI bars.

Mechanistic:
3. **σ trajectory split by replay regime** — fc1 and fc2 separately,
   NoisyNet only, with vs without PER. Tests "does PER distort σ learning?"
4. **TD-error distribution per condition** — histogram of `td_mean` over
   training, four cells overlaid.
5. **PER sample-count vs σ-at-insert scatter** — does PER systematically
   over-represent high-σ transitions? (UPER mechanism check.)
6. **α-sweep curve** — best-α per exploration regime, with CIs. If the
   curves peak at different α, that's the paper's secondary finding.

## Must-cite list (from the deep-research literature scan)

- Hessel et al. 2018 (Rainbow) — leave-one-out ablation; the obvious comparator.
- Obando-Ceron & Castro 2021 (Revisiting Rainbow) — methodology bar.
- Clark et al. 2025 (Beyond The Rainbow) — appendix observation that
  NoisyNet+ε-greedy combined helps some Atari envs; motivates the question.
- Perkins et al. 2025 (DQN×ε×PER) — adjacent without NoisyNet.
- Plappert et al. 2018 (parameter-space noise) — explicitly leaves PER
  combination as future work.
- Carrasco-Davis et al. 2025 (UPER) — TD-error PER over-samples noisy
  transitions; the mechanism we're testing.
- Panahi et al. 2024 (PER × generalization) — PER often fails under NNs.
- Fortunato et al. 2017 (NoisyNet original) — exploration mechanism.
- Schaul et al. 2016 (PER original).
- Fedus et al. 2020 (Revisiting Fundamentals of Experience Replay).
- Fujimoto et al. 2020 (PER ↔ loss-shaping equivalence).

## Glossary

- **2×2 factorial design.** Cross every level of every factor with every
  level of every other. Lets you estimate not just main effects but the
  *interaction* — the difference-of-differences. A one-at-a-time ablation
  cannot see interactions.
- **α (PER alpha).** Prioritization exponent. Sample probability ∝
  |TD-error|^α. α=0 is uniform, α=1 is full priority, paper default 0.6.
- **α-sweep.** Try multiple α values systematically and plot performance vs
  α. Reveals the shape of the curve and where it peaks.
- **Interaction term.** (D−C) − (B−A) for the 2×2. Non-zero means PER's
  effect depends on the exploration method, which is what the paper claims.

## Workflow

1. Kick off runs on the server (your action):
   ```bash
   PARALLEL=4 CUDA_VISIBLE_DEVICES="" ./run_factorial.sh
   PARALLEL=4 CUDA_VISIBLE_DEVICES="" ./run_alpha_sweep.sh
   ```
   Both are resumable; lose-power-and-restart is safe.

2. While runs go: write `analyze.py` (rliable + interaction term +
   mechanistic plots) and the 4-page paper outline.

3. After runs land: regenerate all figures, lock the headline number,
   draft the paper.

## Out of scope for this paper

- DDQN as a factor (would make it 2×2×2).
- Atari (compute-prohibitive; deferred).
- Dueling architecture, distributional RL, n-step beyond default.
- Anything past 4 pages of empirical content.
