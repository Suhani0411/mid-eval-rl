## Mid-Eval RL Submission

## Team
* Team Name:Antara and Suhani
* Member 1: Suhani (Roll No. 241049)
* **Member 2: Antara Biswas (Roll No. 250168)

## Repository Structure

```
Antara and Suhani/
├── policy.py
├── APPROACH.md
└── README.md
```

## Files

### `policy.py`
Contains our implementation of the `Policy` class for the multi-armed bandit ad recommendation problem.

The class implements:
* `__init__()`
* `select_arm()`
* `update()`

and follows the interface specified in the assignment.

### `APPROACH.md`
Explains the algorithm, design choices, implementation details, and observations.

## Algorithm Used

Our solution uses a reinforcement learning approach to balance **exploration** and **exploitation** while maximizing the total click-through rate. The policy updates its estimates online after every user interaction without relying on any hard-coded click probabilities.

*(See `APPROACH.md` for detailed methodology.)*

## Requirements

* Python 3.x
* NumPy

No external libraries are required.

## Running Locally

From the repository root:

```bash
python starter/local_test.py "submissions/Antara and Suhani/policy.py"
```

## Notes

* Uses only NumPy and the Python standard library.
* No hard-coded ad indices or click probabilities.
* Designed to learn solely from observed rewards.
