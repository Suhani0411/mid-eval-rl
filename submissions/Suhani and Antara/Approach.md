Approach- Team "Suhani and Antara"

1. Understanding the problem:
10 ads, 5,000 impressions, hidden fixed click-rates, Bernoulli rewards. Score is total clicks averaged over 10 seeds, so we need a policy that finds the best ad quickly and stops wasting impressions once it's confident. The problem statement warns the top ads have very close click-rates on purpose, so the hard part isn't finding a decent ad, it's telling the best ad apart from the runner-up without overspending on exploration.

2. What we chose and why:
We started from the given baseline, plain Thompson Sampling with a uniform 'Beta(1, 1)' prior, redrawing one random sample per ad each round and playing the highest. This explores well early but has no sense of "the clock", it treats round 1 and round 4,999 the same way.

Our first improvement made the policy clock-aware: blend a random Thompson
sample with the plain running average, weighted by how far through the
5,000 rounds we are ('progress = (t / horizon) ** exponent'). Early on, trust
the random sample (explore); late on, trust the average (exploit). This
already beat the baseline clearly.

We then refined this further with two changes:

- Lowered the exploitation exponent. We first tried steepening it
  (exponent = 2.0), expecting faster commitment to help instead it made things worse, because it started trusting the average too early, before there was enough evidence. This told us to push the exponent down instead, which we kept doing until it stopped helping (final value: '0.2').
- Switched the prior from uniform to Jeffreys ('Beta(0.5, 0.5)')** and added a forced warm-up, every ad is guaranteed 5 shows before the policy trusts any comparison between them. Since the top ads are deliberately close, leaving the first few looks entirely to random sampling risks unluckily under-trying the actual best ad early on. A guaranteed minimum exposure protects against that.

3. What else we tried:
The baseline Thompson Sampling algorithm, using a uniform prior, achieved a local average of 477.3 clicks, corresponding to a regret of 72.3 and 66% of the performance gap closed. While it explores effectively, it does not account for the remaining time horizon, treating the very first round and the final round (round 4,999) identically. As a result, it continues exploring even when it would be more beneficial to exploit the best-performing arm near the end.

The initial clock-aware Thompson Sampling approach, which uses a Jeffreys prior with an exponent of 1.5, improved performance to a local average of 512.6 clicks, with a regret of 41.3 and 82% of the gap closed. This version blends a random Thompson Sampling draw with the running average of each arm, where the balance between exploration and exploitation changes based on progress through the time horizon. This simple modification already outperformed the baseline by a clear margin.

A more aggressive version of the clock-aware Thompson Sampling algorithm, using a steeper exploitation schedule (exponent = 2.0), resulted in a local average of 497.9 clicks, a regret of 51.7, and 76% of the gap closed. Despite intending to exploit earlier, this approach actually performed worse than the previous version. By committing to exploitation too soon, it sacrificed valuable exploration before collecting enough evidence to confidently identify the best advertisements.

The final approach, which combines a Jeffreys prior, a forced warm-up phase of five pulls per arm, and a gentle clock-aware exponent of 0.2, produced the best overall performance. It achieved a local average of 526.7 clicks, a regret of 27.6, and approximately 89% of the performance gap closed. The forced warm-up ensures that every advertisement is explored fairly during the early stages, which is particularly important because the click-through rates of the best ads are very close to one another. After this initial exploration, the algorithm gradually shifts toward exploitation while still maintaining sufficient exploration, leading to the strongest overall results among all the tested variants.

4. Results:
 - Local practice score: 526.7 clicks, regret 27.6 (closed ~89% of the
  random→optimal gap).
 - Best website leaderboard score: 409.8, rank #5

5. What we'd do next:
Tune the warm-up length ('min_rounds') more precisely we picked 5 somewhat heuristically and didn't have time to sweep it properly. We'd also like to try scaling the warm-up with 'n_arms' and 'horizon' directly, so it generalizes better to bandit instances with a different number of arms or horizon, rather than using a fixed constant.

6. How to reproduce:
cd starter
python local_test.py "../submissions/Antara and Suhani/policy.py"


