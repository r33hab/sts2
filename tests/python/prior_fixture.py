"""A small combat prior exported the way the prior loop exports its champion, for tests."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / "rl", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import combat_prior  # noqa: E402
import torch  # noqa: E402
from combat_net import CombatNet  # noqa: E402
from combat_tokens import VOCAB  # noqa: E402

N_BUCKETS = 19
SHAPE = {"d_model": 32, "layers": 1, "heads": 2}


def export_prior(
    root: Path,
    *,
    probe: bool = False,
    prior_min_visits: int = 3,
    c_puct: float = 0.2,
    seed: int = 0,
    **meta: Any,
) -> Path:
    """Write `combat_policy.pt` and `model.json` into `root` and return it.

    A one-layer network with seeded random weights: what matters to a test is that it is a
    network the real loader plays, not that it plays well. `meta` overrides fields of
    `model.json`.
    """
    torch.manual_seed(seed)
    model = CombatNet(VOCAB, N_BUCKETS, **SHAPE).eval()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    policy = root / combat_prior.POLICY
    torch.save(
        {
            "model": model.state_dict(),
            "n_buckets": N_BUCKETS,
            "width": 5,
            "max_hp": 80,
            **SHAPE,
        },
        policy,
    )
    data = {
        "kind": combat_prior.KIND,
        "iteration": 1,
        "vocab_sha256": combat_prior.vocab_sha256(),
        "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
        "search": {"c": 0.1, "c_puct": c_puct, "prior_min_visits": prior_min_visits},
        "probe": probe,
        **meta,
    }
    (root / combat_prior.MODEL_JSON).write_text(json.dumps(data), encoding="utf-8")
    return root


def network(root: Path) -> CombatNet:
    """Load the exported network itself, to hold a prior's answers against."""
    blob = torch.load(Path(root) / combat_prior.POLICY, weights_only=True)
    model = CombatNet(VOCAB, blob["n_buckets"], **SHAPE)
    model.load_state_dict(blob["model"])
    return model.eval()
