# Deployment Analysis: From Simulator to Production SOC

This document describes what it would take to move the ACDS simulator
from a research prototype toward a real Security Operations Center (SOC)
deployment. It is deliberately honest about the gap and does not claim
the current system is production-ready.

**Audience:** engineers and researchers evaluating whether this approach
could be applied to real networks.

**Status of the project:** research prototype. The simulator, the trained
PPO policy, and the evaluation harness are complete and reproducible.
Nothing in this document has been implemented.

---

## 1. What the current project actually is

Strip away the RL vocabulary and the project is a discrete-time model of
a security triage loop:

- **State:** a 104-dimensional vector describing 10 simulated hosts
  (health, open ports, vulnerabilities, compromise status, kill-chain stage)
- **Action:** `MultiDiscrete([5, 10])` -- pick one of 5 action types
  (noop / patch / block_port / isolate / scan) and one of 10 target hosts
- **Reward:** attack damage avoided, minus action cost, minus compromise penalty
- **Policy:** PPO, trained for 800k timesteps

The attack generator is calibrated to **CICIDS2017** flow statistics:
damage, duration, and inter-arrival distributions per attack group
(dos, recon, brute_force, bot, web) come from real data.

The result is a **toy model of the SOC triage loop** with realistic
attack statistics. It is not a network simulator in the NS-3 / Mininet
sense, and it does not interact with real infrastructure.

---

## 2. What is captured, and what is not

| Aspect | Simulator | Real SOC |
|--------|-----------|----------|
| Attack distributions | CICIDS-calibrated | actual |
| Attack sequencing | 3-stage kill chain | varies, often nonlinear |
| State observability | full | partial, delayed, noisy |
| Action consequences | scalar reward | outage, downtime, user impact |
| Adversary | fixed distribution | adaptive, may read defense |
| Observability of attacker | perfect | incomplete, encrypted |
| Time scale | discrete steps | seconds to hours |
| Human in loop | none | always in modern SOCs |
| Compliance / audit | none | mandatory (PCI, HIPAA, SOC2) |
| Blast radius | bounded by sim | unbounded |

The most important mismatch: **in the simulator, a wrong action costs
reward points. In production, a wrong action costs an outage.** That
single difference invalidates any direct deployment of the current
policy.

---

## 3. The five layers of real-world deployment

Each of these is a substantial engineering project. None are solved by
more RL training.

### Layer 1: Real telemetry

The simulator's state vector is not available on a real network. Real
sources of observation include:

- **NetFlow / IPFIX** from routers and switches
- **EDR telemetry** from endpoints (CrowdStrike, Defender, SentinelOne)
- **Firewall logs** (Palo Alto, Fortinet, iptables)
- **DNS logs** (query patterns, beaconing)
- **Authentication logs** (AD, Okta, SSH)
- **IDS alerts** (Suricata, Zeek)

Each is a stream of events, not a state vector. A production system needs
a **feature pipeline** that turns raw events into a fixed-shape
observation at a fixed cadence. Typically 100-500 dimensions, updated
every 1-60 seconds.

Open challenges in this layer:

- **Noise:** IDS false positives dominate many real feeds
- **Incompleteness:** encrypted traffic, unmanaged endpoints
- **Latency:** batch collection pipelines introduce minutes of delay
- **Adversarial shaping:** attackers can generate cover traffic

### Layer 2: Real interventions

The simulator's five actions map to real interventions, but the blast
radius is entirely different:

| Sim action | Real equivalent | Blast radius | Reversible? |
|-----------|-----------------|--------------|-------------|
| `noop` | do nothing | none | -- |
| `patch` | push software update | app restart required | partial |
| `block_port` | firewall / ACL change | may break legit traffic | yes, with rollback |
| `isolate` | EDR network containment | user cannot work | yes, but disruptive |
| `scan` | vuln scan / active probe | generates noise, detectable | n/a |

Every action above except `noop` and `scan` has real consequences for
users. Modern SOCs gate all of these behind human approval or strict
policy.

**Implication:** production deployment requires either human-in-the-loop
for anything with blast radius, or formal action masks that prevent the
policy from choosing high-impact actions outside a whitelist.

### Layer 3: Real objectives

The simulator optimizes a scalar reward. Real SOCs optimize a vector:

- **MTTD** -- mean time to detect
- **MTTR** -- mean time to respond
- **False positive rate** -- SOC analyst time wasted
- **Business impact** -- downtime minutes, users affected
- **Compliance posture** -- audit trail, evidence preservation

Multi-objective optimization replaces scalar RL. Applicable approaches:

- **Constrained RL** (CPO, PPO-Lagrangian)
- **Safe RL** with action shielding
- **Multi-objective PPO** with Pareto-front tracking

None of these are drop-in replacements for PPO. Each requires retraining
and re-evaluation.

### Layer 4: Safety and auditability

Any system that can act on production infrastructure needs:

- **Replayability:** every decision reproducible from a full state snapshot
- **Rollback:** undo any action within a bounded time window
- **Hard constraints:** never isolate a domain controller, never block
  port 443 outside change windows, never take action on hosts with active
  incident tickets
- **Explainability:** a human reviewer must understand *why* an action
  was chosen, not just what

PPO is a black box. Current approaches to explainability for policy
networks:

- **Decision tree distillation** (train a surrogate on policy rollouts)
- **SHAP / integrated gradients** on the policy network
- **Attention over observation dimensions**
- **Action masking as policy constraint** (removes need to explain
  forbidden actions)

Hard constraints should be enforced by the **execution layer**, not the
policy. The policy should not be able to violate a safety rule, even by
mistake.

### Layer 5: Adversarial robustness

The simulator trains against a fixed attack distribution. Real attackers:

- **Read the defense.** If isolating on port scan becomes known,
  attackers stop scanning and go quiet.
- **Adapt.** Multi-armed bandit attackers that probe the defense.
- **Poison observations.** Generate traffic to make benign hosts look
  compromised, triggering expensive actions.

The current PPO policy would likely fail against an adaptive attacker.
Approaches to robustness:

- **Self-play training** with a learned attacker
- **Robust RL** with adversarial perturbations during training
- **OOD detection** at runtime to catch state distributions the policy
  was not trained on
- **Ensemble policies** with disagreement-based abstention

---

## 4. What current production systems actually do

Nothing today autonomously defends a network with RL. What exists:

**Research testbeds:**
- DARPA CAGE Challenge (CybORG environment)
- Autonomous Cyber Operation environments (MIT, CMU)
- Academic ACD research, mostly at defense contractors and labs

**Production reality:**
- **Alert triage:** ML prioritizes alerts. Human decides.
- **Recommendation systems:** ML suggests actions. Human approves.
- **Contextual bandits** for narrow, low-blast-radius actions
  (e.g., which endpoint to quarantine). Heavily policy-gated.
- **Auto-remediation** for well-understood cases (patch deployment,
  certificate renewal). Not RL-driven.

The closest thing to autonomous cyber defense in production is
**automated containment in EDR products** (e.g., CrowdStrike RTR,
Defender ATP). These are rule-based, not learned, and gated by strict
policy.

---

## 5. A realistic path from this prototype

Nothing below has been done. Each step is a project on its own.

### Step 1: Red-team simulation
Have a human attacker attempt to penetrate the simulated network while
the PPO policy defends. If a smart human can reliably defeat the policy,
the current approach has a ceiling. Publish either result.

### Step 2: Digital twin
Stand up 3-5 real VMs with a network tap. Feed real NetFlow into the
observation pipeline. Run the agent in shadow mode (proposes, does not
act). Measure agreement with a human SOC analyst. Disagreement is often
more interesting than agreement.

### Step 3: Recommendation mode
Human approves every action. Collect 6 months of data. Measure approval
rate, precision, MTTD impact.

### Step 4: Hard constraints
Formalize safety rules as action masks in the execution layer. The
policy cannot choose forbidden actions because they do not exist in its
action space.

### Step 5: Semi-autonomous on narrow actions
Pick one low-blast-radius action (e.g., temporary port block for
5 minutes) and let the agent act without approval on that action only.
Measure. Expand slowly.

### Step 6: Bounded autonomy
Autonomy scales with: low blast radius, high reversibility, well-defined
action space, and long track record. Never full autonomy on all actions.

---

## 6. Engineering reality: timelines

Production RL deployments in comparable domains (industrial control,
autonomous vehicles, high-frequency trading) took:

| Phase | Typical duration |
|-------|-----------------|
| Research prototype -> validated simulation | 2-3 years |
| Simulation -> digital twin | 1 year |
| Twin -> shadow mode | 6 months |
| Shadow -> recommendation | 6 months |
| Recommendation -> narrow autonomy | 1-2 years |
| Narrow -> broad autonomy | not recommended |

That is 5-7 years from prototype to live system, with a team of
10-20 engineers, in a well-resourced organization. **The model is
typically the smallest part of the work.** The surrounding
infrastructure -- telemetry, policy, approval workflow, audit, rollback --
is 90% of the effort.

---

## 7. What this project demonstrates

Given the gap above, what is the value of the current project?

1. **The architecture is correct.** State -> policy -> action is the right
   loop for a defense system. The observation, action, and reward
   design generalize.

2. **The calibration instinct is correct.** Training on CICIDS-derived
   distributions is more realistic than synthetic attacks. This is the
   right direction, even if CICIDS itself is dated (2017).

3. **The evaluation is honest.** Three-seed cross-validation, baseline
   comparison, and explicit acknowledgment of the round_robin heuristic
   are the right posture. Papers and production systems often fail this.

4. **The failure mode is instructive.** The fact that a simple heuristic
   (`round_robin`) matches PPO on this environment is a *finding*, not a
   bug. It says the environment admits a simple dominant strategy. That
   is exactly the kind of thing a real evaluation would catch -- if it ran
   baselines.

What the project does *not* demonstrate:

- That PPO can defend a real network
- That RL is the right tool (it may not be)
- That the environment captures adversarial adaptation
- That the policy is safe, explainable, or auditable

---

## 8. The most valuable next artifact

If this project is going to advance, the next artifact is not another
environment version, another training run, or another algorithm. It is:

**A red-team evaluation.** Train a human attacker against the current
policy in the simulator. Measure whether the policy can be defeated.
Publish the result.

Two outcomes:

- **Policy survives.** Surprising and worth investigating further.
- **Policy is defeated.** Expected; the interesting question is *how*
  and what defense would have prevented it.

Either way, the result is more useful than another PPO run. This is
what a serious next step would look like.

---

## References

- CICIDS2017 dataset: https://www.unb.ca/cic/datasets/ids-2017.html
- DARPA CAGE Challenge: https://github.com/cage-challenge
- CybORG simulator: https://github.com/cage-challenge/CybORG
- Constrained Policy Optimization (Achiam et al., 2017)
- Safe RL survey (Garcia and Fernandez, 2015)

---

*Written as a deployment analysis for the ACDS project. Not a
deployment plan.*