# Approach

## Team: Arm pullers

## Problem Overview

The task is a stationary Bernoulli Multi-Armed Bandit problem with 10 arms and a horizon of 5000 impressions. The objective is to maximize the total number of clicks by balancing exploration (learning arm quality) and exploitation (selecting the currently best arm).

Since the true click-through rates (CTRs) are unknown, the policy must learn them online through interaction with the environment.

---

# Baseline: Thompson Sampling

We began with the provided Thompson Sampling implementation.

For each arm, a Beta distribution is maintained:

* `alpha = 1 + clicks`
* `beta = 1 + failures`

At each round:

1. Sample a CTR estimate from each arm's Beta posterior.
2. Select the arm with the highest sampled value.
3. Update the posterior using the observed reward.

This provided a strong baseline and naturally balanced exploration and exploitation.

### Baseline Performance

| Method            | Avg Clicks | Regret |
| ----------------- | ---------: | -----: |
| Thompson Sampling |      477.3 |   72.3 |

---

# Experiments

## 1. Forced Initial Exploration

Pure Thompson Sampling can occasionally ignore some arms due to early randomness.

To ensure that every arm receives at least one observation, we forced each arm to be selected once during the first 10 rounds:

```python
if self.t < self.n_arms:
    return self.t
```

### Result

| Method                        | Avg Clicks | Regret |
| ----------------------------- | ---------: | -----: |
| Thompson + Forced Exploration |      491.8 |   63.8 |

This improved performance by providing an initial estimate for every arm before posterior sampling took over.

---

## 2. More Conservative Beta Priors

We experimented with stronger priors:

```python
alpha = 2
beta = 2
```

The goal was to reduce sensitivity to lucky early rewards.

### Result

| Method               | Avg Clicks | Regret |
| -------------------- | ---------: | -----: |
| Thompson + Beta(2,2) |      471.0 |   77.6 |

Performance decreased because the policy became slower to adapt to observed rewards.

### Decision

Rejected.

---

## 3. Additional Forced Exploration

We increased the initial exploration phase so that each arm was sampled multiple times before Thompson Sampling was allowed to dominate.

### Result

| Method                    | Avg Clicks | Regret |
| ------------------------- | ---------: | -----: |
| Thompson + 5 Forced Pulls |      488.2 |   66.1 |

This performed worse than forcing only one pull per arm.

### Observation

The extra exploration consumed impressions that could have been spent exploiting promising arms.

### Decision

Rejected.

---

## 4. UCB1

We implemented the Upper Confidence Bound (UCB1) algorithm as an alternative to Thompson Sampling.

The algorithm selects the arm maximizing:

```text
mean reward + exploration bonus
```

where the exploration bonus decreases as an arm is sampled more frequently.

### Result

| Method | Avg Clicks | Regret |
| ------ | ---------: | -----: |
| UCB1   |      382.9 |  164.0 |

UCB significantly underperformed Thompson Sampling on the evaluation environment.

### Decision

Rejected.

---

## 5. Horizon-Aware Thompson Sampling

The problem provides the total horizon (5000 impressions).

This suggests that exploration should be more valuable early in the run and less valuable near the end.

To incorporate this information, we introduced a progress-dependent weighting:

```python
progress = self.t / self.horizon
```

and combined Thompson samples with estimated arm quality.

### Result

| Method                 | Avg Clicks | Regret |
| ---------------------- | ---------: | -----: |
| Horizon-Aware Thompson |      512.2 |   43.0 |

This produced the largest improvement observed during testing.

---

## 6. Empirical CTR Exploitation

Instead of using the posterior mean:

```python
alpha / (alpha + beta)
```

we used the empirical click-through rate:

```python
(clicks) / (pulls)
```

implemented as:

```python
means = (self.alpha - 1) / (self.alpha + self.beta - 2 + 1e-9)
```

This provided a more direct estimate of observed arm quality.

### Result

| Method                        | Avg Clicks | Regret |
| ----------------------------- | ---------: | -----: |
| Horizon-Aware + Empirical CTR |      523.9 |   31.0 |

This was the strongest-performing variant during local evaluation.

---

## Final Policy

The final submission combines:

1. Thompson Sampling
2. Forced initial exploration (one pull per arm)
3. Horizon-aware exploration/exploitation scheduling
4. Empirical CTR-based exploitation

The progress variable is defined as:

```python
progress = (self.t / self.horizon) ** 1.5
```

The exponent of 1.5 was selected after experimentation because it delays exploitation compared to a linear schedule while still allowing the policy to commit to strong arms later in the run.

For each arm:

```python
samples = np.random.beta(alpha, beta)
means = clicks / pulls

score = (1 - progress) * samples + progress * means
```

The arm with the highest score is selected.

---

# Final Results
### Local Evaluation

| Method                     | Avg Clicks | Regret |
| -------------------------- | ---------: | -----: |
| Baseline Thompson Sampling |      477.3 |   72.3 |
| Final Policy               |      523.9 |   31.0 |

The final policy improved average clicks by approximately 9.8% and reduced regret by approximately 57.1% compared to the original Thompson Sampling baseline.

### Hidden Leaderboard
The final policy achieved:
* Average Clicks: 389.5
* CTR: 7.79%
* Regret: 37.7
---

# Conclusion

The most effective improvements were:

* Ensuring every arm was sampled at least once.
* Exploiting knowledge of the fixed horizon.
* Gradually transitioning from uncertainty-driven exploration to empirical performance-driven exploitation.

The final approach retained Thompson Sampling's strong exploration properties while using horizon-aware scheduling to improve decision quality in the later stages of the run.
