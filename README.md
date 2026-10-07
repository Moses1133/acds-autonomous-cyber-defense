# ACDS - Autonomous Cyber Defense Simulator

![CI](https://github.com/Moses1133/acds-autonomous-cyber-defense/actions/workflows/python-app.yml/badge.svg)

👉 **[Try the live demo](https://acds-autonomous-cyber-defense-g8n2yia4ngxdlu3mmkrbdn.streamlit.app)**

A reinforcement-learning driven cyber defense simulator where a PPO agent
learns to defend a 10-host network against attacks calibrated to the
**CICIDS2017** intrusion detection dataset.

## What It Does

- Simulates a 10-host network (router + servers + workstations)
- Generates structured attacks (DoS, brute force, CVE exploits, botnet, port scan)
  whose intensity and frequency distributions are **calibrated to real
  statistics extracted from 2.83M labeled flows** in the CICIDS2017 dataset
- Uses **multi-discrete action space**: the agent picks both `action_type`
  AND `target_host` each step
- Attacks follow a **3-stage kill chain** (recon -> exploit -> escalate) so the
  agent must anticipate escalation

### Actions

| ID | Action | Effect |
|----|--------|--------|
| 0 | noop | do nothing |
| 1 | patch | remove a CVE from the target host |
| 2 | block_port | close one port on the target host |
| 3 | isolate | remove all ports from target (small health cost) |
| 4 | scan | reveal hidden target state |

## Architecture

- **Simulation layer** - `Host`, `Network`, `CyberDefenseEnvV7` (Gymnasium)
- **Cybersecurity layer** - `CICIDSAttackGenerator` calibrated to real data
- **ML/AI layer** - PPO (Stable-Baselines3) with `MultiDiscrete([5, 10])`
- **Dashboard** - Streamlit live visualization

## Results (v7, 400k PPO timesteps)

| Metric | Random | Trained v7 | Improvement |
|--------|--------|-----------|-------------|
| Mean episode reward | +26.80 | **+75.32** | **+48.52** |
| Hosts compromised at end | 2.10 | **0.00** | **100% reduction** |
| Episode length | ~90 | **100** | full survival |
| Reward std (20 seeds) | 5-10 | **0.91** | very stable |
| Value fn `explained_variance` | - | **0.85** | strong signal |

### What the agent learned

The trained policy uses **`isolate` 74% of the time**, `noop` 25%, with
negligible `patch`/`block_port`/`scan`. Interpretation: in this reward
landscape, isolating the most-at-risk host dominates because it removes
*all* attack vectors at once. This mirrors real-world "kill switch" defense
tactics - powerful but expensive, which is why production systems gate it
behind human approval.

### Training curve

![Training curve](docs/training_curve.png)

## Live Dashboard

![Dashboard](docs/v7_demo.png)

Run the interactive demo:

    streamlit run src\dashboard\app.py

Then open http://localhost:8501

Click **Run Episode(s)** to watch the trained agent defend the network in real time.

## Dataset: CICIDS2017

Attack distributions are calibrated from the **Canadian Institute for
Cybersecurity CICIDS2017** dataset:

- **2,830,743 labeled network flows** across 8 CSV files
- Attack groups extracted: `dos` (380K flows), `recon` (158K), `brute_force`
  (13K), `bot` (2K), `web` (2K)
- Statistics (mean/std/median/p95) computed per group for features like
  `Flow Bytes/s`, `Flow Duration`, `Flow Packets/s`
- Regenerate with:

      python scripts/extract_cicids_stats.py

Source: [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)

## What Was Hard

The project went through **seven environment versions**:

1. **v1** - placeholder random attacks. No causal chain from action to reward.
2. **v2** - real attacks with defense-modulated damage. Agent learned, but
   skill gap over random was tiny (+7).
3. **v3** - over-hardened. Signal died, policy collapsed to `noop`.
4. **v4** - rebalanced. Signal returned, gap still small.
5. **v5** - reward rebalance + entropy regularization. Stable, but limited by
   the discrete action space.
6. **v6** - switched to `MultiDiscrete([5, 10])` and added kill-chain attacks.
   Big jump in expressiveness but attacks were still synthetic.
7. **v7** - added CICIDS2017-calibrated attacks + real flow statistics.
   **Final result: +48.5 gap, zero hosts compromised.**

The lesson: **RL is a signal problem, not a code problem.** The environment
must give the agent a causal chain from action to reward, the reward must be
tuned to allow learning, and the action space must be expressive enough to
solve the task. Tuning these is 80% of applied RL.

## Setup

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

**Optional:** download CICIDS2017 (2 GB) and extract statistics:

    python scripts\download_cicids.py
    python scripts\extract_cicids_stats.py

The extracted `data/cicids_stats.json` (5.7 KB) is committed, so the
generator works without the full dataset.

## Usage

**Train:**

    python scripts\train_ppo.py --env v7 --steps 400000

**Evaluate:**

    python scripts\evaluate_ppo.py --env v7 --episodes 20

**Dashboard:**

    streamlit run src\dashboard\app.py

**TensorBoard:**

    tensorboard --logdir logs\v7

## Tests

    python tests\test_host.py
    python tests\test_network.py
    python tests\test_env_v3.py
    python tests\test_env_v6.py
    python tests\test_env_v7.py
    python tests\test_cicids_generator.py

## Project Layout

    acds-autonomous-cyber-defense/
    |-- src/
    |   |-- sim/            # Host, Network, CyberDefenseEnvV7
    |   |-- attacks/        # CICIDSAttackGenerator
    |   |-- dashboard/      # Streamlit app
    |-- scripts/
    |   |-- train_ppo.py
    |   |-- evaluate_ppo.py
    |   |-- extract_cicids_stats.py
    |-- tests/
    |-- data/
    |   `-- cicids_stats.json  (committed; full dataset gitignored)
    |-- models/             # trained checkpoints
    `-- README.md

## Author

Moses ([@Moses1133](https://github.com/Moses1133))
