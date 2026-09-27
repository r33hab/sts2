"""Every random choice a run makes comes from streams derived from its seed (5.10).

A run is then reproducible, which the worker retry rule relies on, and no two runs share
a stream. The whole canonical seed is hashed, so `E1ABCDEFGH` and `E2ABCDEFGH` draw
different numbers (the probe's first helper hashed only the last 8 bytes, and they did
not), and search never draws from the stream exploration uses.
"""

from __future__ import annotations

import hashlib

import numpy as np

from sts2_gym.seeds import canonicalize

# `run_search` seeds simulation j of combat decision `move` with `move * sims + j`, and the
# native clone takes that seed as a C int. `move` is `move_base + k` with k below the run's
# step cap, so a base below this bound keeps the product inside int32 for any budget up to
# `MAX_SIMS` (DESIGN.md 5.10).
MOVE_BASE_LIMIT = 1_000_000
# The run's engine-step cap (DESIGN.md 1.4 item 6) and the largest per-decision budget;
# `CombatBudget` refuses anything above `MAX_SIMS`.
RUN_STEP_CAP = 20_000  # a full run (Acts 1-3); Act 1 runs took about 250
MAX_SIMS = 2_000
assert (MOVE_BASE_LIMIT + RUN_STEP_CAP) * MAX_SIMS < 2**31


def run_streams(seed: str) -> tuple[np.random.Generator, np.random.Generator, int]:
    """Return the run's MCTS stream, its exploration stream and its determinization base.

    The seed is canonicalized the way the game does (`SeedHelper.CanonicalizeSeed`: upper
    case, O -> 0, I -> 1, stripped), hashed with blake2b to 64 bits and fed to a
    `SeedSequence`, which spawns one child per consumer. The MCTS stream feeds UCT ties,
    the tempo rollout and Dirichlet noise; the exploration stream feeds the softmax and
    epsilon sampling of options (DESIGN.md 5.5). The base offsets `run_search`'s `move`.
    """
    canonical = canonicalize(seed)
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).digest()
    mcts_seq, explore_seq = np.random.SeedSequence(
        int.from_bytes(digest, "little"),
    ).spawn(2)
    mcts = np.random.default_rng(mcts_seq)
    move_base = int(mcts.integers(0, MOVE_BASE_LIMIT))
    return mcts, np.random.default_rng(explore_seq), move_base
