"""The exported combat prior (stage 3's champion) for the searches outside the prior loop.

The prior loop (`rl/prior_loop/`) exports its champion to `DEFAULT_DIR`: `combat_policy.pt`,
`combat_net.pt` and a `model.json` recording how the gate played it (DESIGN.md 6.4). Two
searches can use it:

- the coach's fight search, `live_search.search_live`, whose `prior_fn(handle, legal)` is a
  `LivePrior`. It pickles to a path and a flag, because it travels to the search workers,
  and each process loads the network once, on first use or at `warm()`.
- stage-2 self-play, `combat_mcts.run_search`, whose `prior_fn(world)` returns
  `{action: p}`: a `WorldPrior`.

Both play the network the way the gate did: the policy head, a softmax over the legal
actions only, torch on one thread, and the end-turn probe only if the network read it
(`model.json`'s `probe`). `Champion.search` holds the rest of the gate's search settings,
`c_puct` and `prior_min_visits`.

`find(directory)` reads an export: None when there is none, `PriorError` when there is one
this build cannot play (another kind, another token vocabulary, a policy file that is not
the one `model.json` describes).
"""

from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from combat_tokens import VOCAB, Tokens, encode

from sts2_gym import native

DEFAULT_DIR = (
    Path.home()
    / "Documents"
    / "git"
    / "spire-coach"
    / "data"
    / "models"
    / "combat_prior"
)
MODEL_JSON = "model.json"
POLICY = "combat_policy.pt"
KIND = "combat_prior"


class PriorError(Exception):
    """An exported prior that is there but cannot be played by this build."""


def vocab_sha256(vocab: Any = VOCAB) -> str:
    """Hash every id space the tokens use, as the prior loop's export records it."""
    payload = {
        field.name: sorted(getattr(vocab, field.name).items())
        for field in dataclasses.fields(vocab)
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Champion:
    """An export this build can play, and how the gate searched with it."""

    directory: str
    iteration: int | None
    policy_sha256: str
    probe: bool
    c_puct: float
    prior_min_visits: int

    @property
    def policy(self) -> Path:
        return Path(self.directory) / POLICY

    def describe(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def live_prior(self) -> LivePrior:
        return LivePrior(str(self.policy), probe=self.probe)

    def world_prior(self) -> WorldPrior:
        return WorldPrior(str(self.policy), probe=self.probe)

    def live_settings(self) -> dict[str, Any]:
        """Return what `live_search.search_live` takes to search with this prior."""
        return {
            "prior_fn": self.live_prior(),
            "c_puct": self.c_puct,
            "prior_min_visits": self.prior_min_visits,
        }


def find(directory: Path | str | None = None) -> Champion | None:
    """Read the export in `directory` (`DEFAULT_DIR` by default); None if there is none.

    Raises:
        PriorError: If `model.json` and the policy are there but cannot be played here.

    """
    root = Path(directory) if directory is not None else DEFAULT_DIR
    meta_path, policy = root / MODEL_JSON, root / POLICY
    if not meta_path.is_file() or not policy.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PriorError(f"{meta_path} cannot be read: {error}") from error
    if meta.get("kind") != KIND:
        raise PriorError(f"{meta_path} is a {meta.get('kind')!r}, not a {KIND}")
    if meta.get("vocab_sha256") != vocab_sha256():
        raise PriorError(
            f"{root} was trained on another token vocabulary than this build's "
            "combat_tokens.VOCAB, so it would read other tokens than it learned",
        )
    digest = _sha256_file(policy)
    if meta.get("policy_sha256") not in (None, digest):
        raise PriorError(f"{policy} is not the policy {meta_path} describes")
    search = meta.get("search") or {}
    return Champion(
        directory=str(root),
        iteration=meta.get("iteration"),
        policy_sha256=digest,
        probe=bool(meta.get("probe", True)),
        c_puct=float(search.get("c_puct", 0.2)),
        prior_min_visits=int(search.get("prior_min_visits", 0)),
    )


class PolicyNet:
    """The policy head on the CPU, one torch thread: token sequences in, probabilities out."""

    def __init__(self, checkpoint: Path | str):
        import torch  # noqa: PLC0415 -- only a process that plays the prior pays for torch
        from combat_net import CombatNet, collate  # noqa: PLC0415

        torch.set_num_threads(1)
        blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = CombatNet(
            VOCAB,
            blob["n_buckets"],
            d_model=blob.get("d_model", 128),
            layers=blob.get("layers", 4),
            heads=blob.get("heads", 4),
        )
        self.model.load_state_dict(blob["model"])
        self.model.eval()
        self._torch = torch
        self._collate = collate
        self._cpu = torch.device("cpu")

    def weights(
        self,
        tokens: Sequence[Tokens],
        legals: Sequence[Sequence[int]],
    ) -> list[np.ndarray]:
        """Return each position's probabilities over its legal actions, in their order.

        A position whose head puts no finite score on any legal action gets a uniform prior.
        """
        with self._torch.inference_mode():
            _, logits = self.model(**self._collate(list(tokens), self._cpu))
        out = []
        for row, legal in zip(logits.numpy(), legals, strict=True):
            index = np.asarray(legal, dtype=np.int64)
            scores = np.full(len(index), -np.inf)
            inside = index < row.shape[0]
            scores[inside] = row[index[inside]]
            finite = np.isfinite(scores)
            if not finite.any():
                out.append(np.full(len(index), 1.0 / max(1, len(index))))
                continue
            scores = np.where(finite, scores, -np.inf)
            weights = np.exp(scores - scores[finite].max())
            out.append(weights / weights.sum())
        return out


_NETS: dict[str, PolicyNet] = {}


def network(checkpoint: Path | str) -> PolicyNet:
    """Return the network at `checkpoint`, loaded once per process."""
    key = str(checkpoint)
    net = _NETS.get(key)
    if net is None:
        net = _NETS[key] = PolicyNet(key)
    return net


@dataclass(frozen=True)
class LivePrior:
    """`live_search`'s `prior_fn(handle, legal)`: the policy's probabilities, in `legal`'s order.

    The observation `encode` reads comes from a faithful clone of the handle, made and
    destroyed here: the handle is only read. A position with no known encounter (the coach's
    live snapshots carry none) encodes it as the unknown token.
    """

    policy: str
    probe: bool = False

    def warm(self) -> None:
        """Load the network now, so the first search does not spend its budget on it."""
        network(self.policy)

    def __call__(self, handle: int, legal: Sequence[int]) -> np.ndarray:
        obs = (ctypes.c_int * native.OBS_SIZE)()
        view = native.clone(handle, False, 0, obs)
        native.destroy(view)
        tokens = encode(handle, obs, native.encounter_id(handle), probe=self.probe)
        return network(self.policy).weights([tokens], [list(legal)])[0]


@dataclass(frozen=True)
class WorldPrior:
    """`combat_mcts`'s `prior_fn(world)`: `{action: p}` over a combat handle's legal actions."""

    policy: str
    probe: bool = False

    def __call__(self, world: Any) -> dict[int, float]:
        legal = world.legal()
        tokens = encode(world.handle, world.obs, world.encounter_id, probe=self.probe)
        weights = network(self.policy).weights([tokens], [legal])[0]
        return dict(zip(legal, weights.tolist(), strict=True))
