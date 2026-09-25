"""The value models: FightNet and RunValueNet (DESIGN.md 5.1, 5.2).

Torch trains them (`train.py`); numpy runs them everywhere else. Self-play workers and the
engine service import this module for inference and never import torch: the torch modules
`FightNet` and `RunValueNet` are built on first use (the module's `__getattr__`), and a
checkpoint's `.npz` export holds every weight the numpy nets need (`load_numpy`).

Both networks cut the sparse blocks of the feature vector down with bias-free linear
embeddings -- deck counts E 1154 -> 64, relics F 296 -> 32, potions G 63 -> 8, so a deck's
embedding is the count-weighted sum of its cards' vectors -- before one MLP:

- **FightNet** reads blocks A and D of the view at fight start, a one-hot over the 88
  encounter ids and a weak flag (136 wide), with the embeddings (240) -> 256 -> 128 -> 22
  logits: class 0 is death, classes 1-21 the HP lost in buckets (-inf, 0], 1-4, 5-8, ...,
  73-76 and 77+ (`fight_class`).
- **RunValueNet (V)** reads blocks A, B, C, D and the fight block H (338 wide), with the
  embeddings (442) -> 256 -> dropout 0.1 -> 256 -> 18 floor logits (class k-1: the run ended
  on floor k; class 17: the Act 1 boss died) and a sigmoid end-HP fraction.

The inputs are assembled here, in numpy, for training and inference alike, so the two
cannot drift (`fight_inputs`, `value_inputs`). The one transformation is a clamp on the HP
scalars (`HP_BOUND`); nothing assumes an input is non-negative.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import FEATURE_VERSION, features
from .features import (
    A_OFFSET,
    A_SIZE,
    B_OFFSET,
    BASE_SIZE,
    D_OFFSET,
    D_SIZE,
    E_OFFSET,
    E_SIZE,
    F_SIZE,
    FIGHT_BLOCK_SIZE,
    G_SIZE,
)
from .shards import write_atomic

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch
    from numpy.typing import ArrayLike

    from .features import FightQuery

__all__ = [
    "BOSS_CLASS",
    "COUNTS",
    "FIGHT_CLASSES",
    "FIGHT_DENSE",
    "FLOOR_CLASSES",
    "HP_BOUND",
    "LOSS_CENTRES",
    "N_ENCOUNTERS",
    "VALUE_DENSE",
    "VALUE_INPUT",
    "NumpyFightNet",
    "NumpyNets",
    "NumpyValueNet",
    "boss_chance",
    "expected_floors",
    "expected_loss",
    "fight_class",
    "fight_inputs",
    "fight_state_inputs",
    "floor_class",
    "load_numpy",
    "load_torch",
    "npz_for",
    "numpy_params",
    "save",
    "softmax",
    "value_inputs",
]

# --------------------------------------------------------------------------------------
# Classes and the values read off them (DESIGN.md 5.1, 5.2)
# --------------------------------------------------------------------------------------

# Class 0 is death, 1 lost nothing, 2-20 lost 1-4 ... 73-76, and 21 lost 77 or more.
FIGHT_CLASSES = 22
# Class k-1: the run ended on floor k (1-17); class 17: the Act 1 boss died.
FLOOR_CLASSES = 18
BOSS_CLASS = FLOOR_CLASSES - 1
_LOSS_BUCKET = 4  # HP per loss bucket

# The HP a loss class stands for, for E[HP lost]: bucket centres (2.5 for 1-4), 0 for
# "lost nothing", and 78.5 for 77+ (the next bucket's centre). A death stands for the
# whole HP at stake (`expected_loss`), so its entry is unused.
LOSS_CENTRES = np.array(
    [0.0, 0.0, *(_LOSS_BUCKET * j + 2.5 for j in range(FIGHT_CLASSES - 2))],
)
_FLOORS = np.arange(1, FLOOR_CLASSES, dtype=np.float64)  # the floors of classes 0-16


def fight_class(hp_before: int, hp_after: int) -> int:
    """Return a fight's class (5.1): death, or the HP it cost, after end-of-fight heals.

    The label reads HP alone, never a step reward, so a fight a potion ends or heals
    through is labelled like any other.
    """
    if hp_after <= 0:
        return 0
    lost = hp_before - hp_after
    if lost <= 0:
        return 1
    return 1 + min(FIGHT_CLASSES - 2, (lost + _LOSS_BUCKET - 1) // _LOSS_BUCKET)


def floor_class(final_floor: int) -> int:
    """Return the floor class of a run's final floor: 1-17, or 18 for a boss kill (E10).

    Raises:
        ValueError: For a floor outside 1-18.

    """
    if not 1 <= final_floor <= FLOOR_CLASSES:
        raise ValueError(
            f"a final floor is 1-18 (18 = the boss died), not {final_floor}",
        )
    return final_floor - 1


def expected_floors(p: np.ndarray, w: float) -> np.ndarray:
    """Return `V_W = sum_k k * p_k + W * p_18` over the last axis (5.2).

    Expected final floor, with a boss kill worth `w` floors (the design's W = 26: the floor
    Act 2 opens on plus half of it). W is in no training target, so it changes the ranking
    of two options only through their boss-kill chances.
    """
    p = np.asarray(p)
    return p[..., :BOSS_CLASS] @ _FLOORS + w * p[..., BOSS_CLASS]


def boss_chance(p: np.ndarray) -> np.ndarray:
    """Return `p_18`, the chance the Act 1 boss dies, over the last axis (5.2)."""
    return np.asarray(p)[..., BOSS_CLASS]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Return the softmax over the last axis."""
    shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def expected_loss(probs: np.ndarray, hp: ArrayLike) -> np.ndarray:
    """Return E[HP lost] from fight-class probabilities, a death costing all of `hp`.

    The loss classes stand for their bucket centres (`LOSS_CENTRES`); a death loses the
    whole HP the player brought, so a deadlier fight never reads as a cheaper one.
    """
    probs = np.asarray(probs)
    return probs[..., 1:] @ LOSS_CENTRES[1:] + probs[..., 0] * hp


# --------------------------------------------------------------------------------------
# Inputs: assembled once, in numpy, for torch and numpy alike
# --------------------------------------------------------------------------------------

# `sts2_gym.env.ENCOUNTER_NAMES` holds ids 0-87 (a test holds the two equal).
N_ENCOUNTERS = 88
DECK_EMBED, RELIC_EMBED, POTION_EMBED = 64, 32, 8
EMBED = DECK_EMBED + RELIC_EMBED + POTION_EMBED
COUNTS = E_SIZE + F_SIZE + G_SIZE  # blocks E, F and G, which close the base vector
_STATE = A_SIZE + D_SIZE  # FightNet's per-view dense part
FIGHT_DENSE = _STATE + N_ENCOUNTERS + 1
_BCD = slice(B_OFFSET, D_OFFSET + D_SIZE)  # blocks B, C and D sit side by side
VALUE_DENSE = A_SIZE + (_BCD.stop - _BCD.start) + FIGHT_BLOCK_SIZE
VALUE_INPUT = BASE_SIZE + FIGHT_BLOCK_SIZE
assert E_OFFSET + COUNTS == BASE_SIZE

# HP past which the number means nothing: more than any creature in the game holds. The
# Waterfall Giant's 999999999 is the sentinel the emulator has; a sentinel must never
# reach a network as a raw magnitude. Block A opens with the four HP scalars:
# hp/max_hp, hp/100, max_hp/100 and (max_hp - hp)/100 (DESIGN.md 3.2).
HP_BOUND = 600
_HP_CAPS = np.array([1.0, HP_BOUND / 100, HP_BOUND / 100, HP_BOUND / 100], np.float32)


def _scalars(rows: np.ndarray) -> np.ndarray:
    """Block A with its HP scalars clamped from above; no lower bound is assumed."""
    scalars = np.array(rows[:, A_OFFSET : A_OFFSET + A_SIZE], dtype=np.float32)
    scalars[:, : len(_HP_CAPS)] = np.minimum(scalars[:, : len(_HP_CAPS)], _HP_CAPS)
    return scalars


def _counts(rows: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rows[:, E_OFFSET:BASE_SIZE], dtype=np.float32)


def _encounter_one_hot(encounters: np.ndarray) -> np.ndarray:
    """Return the one-hot over the encounter ids FightNet is built over.

    Raises:
        ValueError: For an id outside them.

    """
    encounters = np.asarray(encounters, dtype=np.int64).reshape(-1)
    outside = encounters[(encounters < 0) | (encounters >= N_ENCOUNTERS)]
    if outside.size:
        raise ValueError(
            f"encounter ids {sorted(set(outside.tolist()))} are outside the "
            f"0-{N_ENCOUNTERS - 1} FightNet is built over",
        )
    one_hot = np.zeros((len(encounters), N_ENCOUNTERS), dtype=np.float32)
    one_hot[np.arange(len(encounters)), encounters] = 1.0
    return one_hot


def fight_state_inputs(bases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return FightNet's per-view inputs: blocks A (clamped) and D, and the counts E-G."""
    bases = np.asarray(bases, dtype=np.float32)
    dense = np.concatenate(
        [_scalars(bases), bases[:, D_OFFSET : D_OFFSET + D_SIZE]],
        axis=1,
    )
    return dense, _counts(bases)


def fight_inputs(
    bases: np.ndarray,
    encounters: np.ndarray,
    weak: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return FightNet's (dense, counts) for base vectors, encounter ids and weak flags.

    An encounter id outside the one-hot raises `ValueError`.
    """
    state, counts = fight_state_inputs(bases)
    dense = np.concatenate(
        [
            state,
            _encounter_one_hot(encounters),
            np.asarray(weak, dtype=np.float32).reshape(-1, 1),
        ],
        axis=1,
    )
    return dense, counts


def value_inputs(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return V's (dense, counts) for rows of the 1851-wide vector (base plus H).

    Raises:
        ValueError: For rows of another width.

    """
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != VALUE_INPUT:
        raise ValueError(f"V reads {VALUE_INPUT}-wide rows, not shape {rows.shape}")
    dense = np.concatenate([_scalars(rows), rows[:, _BCD], rows[:, BASE_SIZE:]], axis=1)
    return dense, _counts(rows)


# --------------------------------------------------------------------------------------
# Numpy inference (the checkpoint's .npz export)
# --------------------------------------------------------------------------------------


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # The tanh form never overflows, so large logits raise no warning.
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class _NumpyEmbeddings:
    """The three bias-free embeddings, applied to the counts E, F and G."""

    def __init__(self, params: dict[str, np.ndarray]):
        self.deck = params["embed.deck.weight"].T
        self.relics = params["embed.relics.weight"].T
        self.potions = params["embed.potions.weight"].T

    def __call__(self, counts: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                counts[:, :E_SIZE] @ self.deck,
                counts[:, E_SIZE : E_SIZE + F_SIZE] @ self.relics,
                counts[:, E_SIZE + F_SIZE :] @ self.potions,
            ],
            axis=1,
        )


class NumpyFightNet:
    """FightNet in numpy: the torch module's forward pass, weight for weight."""

    # Rows per pass of `summary`: a pass holds (rows x queries x 256) floats.
    _CHUNK = 2048

    def __init__(self, params: dict[str, np.ndarray]):
        self.embed = _NumpyEmbeddings(params)
        self.w1, self.b1 = params["hidden1.weight"], params["hidden1.bias"]
        self.w2, self.b2 = params["hidden2.weight"], params["hidden2.bias"]
        self.w3, self.b3 = params["out.weight"], params["out.bias"]

    def logits(self, dense: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Return the 22 logits for assembled inputs (`fight_inputs`)."""
        joined = np.concatenate([dense, self.embed(counts)], axis=1)
        hidden = _relu(joined @ self.w1.T + self.b1)
        return self._tail(hidden)

    def query_logits(
        self,
        bases: np.ndarray,
        encounters: np.ndarray,
        weak: np.ndarray,
    ) -> np.ndarray:
        """Return (rows, queries, 22) logits: every base vector against every query.

        The first layer is linear, so each row's part of it (its blocks and embeddings) is
        computed once and each query adds its encounter's column and its weak flag.
        """
        state, counts = fight_state_inputs(bases)
        one_hot = _encounter_one_hot(encounters)
        w1 = self.w1
        shared = (
            state @ w1[:, :_STATE].T
            + self.embed(counts) @ w1[:, FIGHT_DENSE:].T
            + self.b1
        )
        per_query = (
            one_hot @ w1[:, _STATE : _STATE + N_ENCOUNTERS].T
            + np.asarray(weak, dtype=np.float32).reshape(-1, 1) * w1[:, FIGHT_DENSE - 1]
        )
        return self._tail(_relu(shared[:, None, :] + per_query[None, :, :]))

    def summary(self, bases: np.ndarray, queries: Sequence[FightQuery]) -> np.ndarray:
        """Return the fight block H (3.3) of each base vector over `queries`."""
        bases = np.asarray(bases, dtype=np.float32)
        encounters = np.array([query.encounter_id for query in queries])
        weak = np.array([query.weak for query in queries], dtype=np.float32)
        blocks = []
        for start in range(0, len(bases), self._CHUNK):
            chunk = bases[start : start + self._CHUNK]
            probs = softmax(self.query_logits(chunk, encounters, weak))
            hp = np.minimum(chunk[:, A_OFFSET + 1] * 100, HP_BOUND)
            losses = expected_loss(probs, hp[:, None])
            blocks.append(features.fight_blocks(queries, losses, probs[..., 0]))
        if not blocks:
            return np.zeros((0, FIGHT_BLOCK_SIZE), dtype=np.float32)
        return np.concatenate(blocks)

    def _tail(self, hidden: np.ndarray) -> np.ndarray:
        hidden = _relu(hidden @ self.w2.T + self.b2)
        return hidden @ self.w3.T + self.b3


class NumpyValueNet:
    """RunValueNet in numpy, in evaluation mode (no dropout)."""

    def __init__(self, params: dict[str, np.ndarray]):
        self.embed = _NumpyEmbeddings(params)
        self.w1, self.b1 = params["hidden1.weight"], params["hidden1.bias"]
        self.w2, self.b2 = params["hidden2.weight"], params["hidden2.bias"]
        self.floor_w, self.floor_b = (
            params["floor_head.weight"],
            params["floor_head.bias"],
        )
        self.hp_w, self.hp_b = params["end_hp_head.weight"], params["end_hp_head.bias"]

    def forward(
        self,
        dense: np.ndarray,
        counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the 18 floor logits and the end-HP fraction for assembled inputs."""
        joined = np.concatenate([dense, self.embed(counts)], axis=1)
        hidden = _relu(joined @ self.w1.T + self.b1)
        hidden = _relu(hidden @ self.w2.T + self.b2)
        end_hp = _sigmoid(hidden @ self.hp_w.T + self.hp_b)[:, 0]
        return hidden @ self.floor_w.T + self.floor_b, end_hp

    def floor_probs(self, rows: np.ndarray) -> np.ndarray:
        """Return the floor distribution `p_1 .. p_18` of each 1851-wide row."""
        logits, _ = self.forward(*value_inputs(rows))
        return softmax(logits)


@dataclass(frozen=True)
class NumpyNets:
    """A checkpoint loaded for inference: both nets and the metadata it was saved with."""

    fight: NumpyFightNet
    value: NumpyValueNet
    meta: dict[str, Any]


CHECKPOINT_FORMAT = "run_train/value-models"


def npz_for(pt: Path | str) -> Path:
    """Return the numpy export that sits beside a `.pt` checkpoint."""
    return Path(pt).with_suffix(".npz")


def _checked(meta: dict[str, Any], path: Path | str) -> dict[str, Any]:
    """Return a checkpoint's metadata once it is known to be loadable here.

    Raises:
        ValueError: For another format, or another `FEATURE_VERSION` (3.2).

    """
    if meta.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    if meta.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"{path} was trained on feature version {meta.get('feature_version')}, and "
            f"this build encodes feature version {FEATURE_VERSION} (DESIGN.md 3.2)",
        )
    return meta


def load_numpy(npz: Path | str) -> NumpyNets:
    """Load a checkpoint's numpy export for inference.

    A file that is not a value-model checkpoint, or was trained on another
    `FEATURE_VERSION`, raises `ValueError`.

    Raises:
        ValueError: If the file holds no checkpoint metadata (a `.pt`, say).

    """
    with np.load(npz, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if "meta" not in arrays:  # a `.pt` is a zip archive too, and np.load opens it
        raise ValueError(
            f"{npz} is not the .npz export of a {CHECKPOINT_FORMAT} checkpoint",
        )
    meta = _checked(json.loads(str(arrays.pop("meta"))), npz)

    def part(prefix: str) -> dict[str, np.ndarray]:
        return {
            key.removeprefix(prefix): value
            for key, value in arrays.items()
            if key.startswith(prefix)
        }

    return NumpyNets(
        NumpyFightNet(part("fight/")),
        NumpyValueNet(part("value/")),
        meta,
    )


# --------------------------------------------------------------------------------------
# Torch: the modules, built on first use, and the checkpoint files
# --------------------------------------------------------------------------------------


@cache
def _torch_nets() -> dict[str, type]:
    import torch  # noqa: PLC0415 -- numpy-only processes never import torch (5.1)
    from torch import nn  # noqa: PLC0415

    class Embeddings(nn.Module):
        """The bias-free embeddings of the counts E, F and G."""

        def __init__(self) -> None:
            super().__init__()
            self.deck = nn.Linear(E_SIZE, DECK_EMBED, bias=False)
            self.relics = nn.Linear(F_SIZE, RELIC_EMBED, bias=False)
            self.potions = nn.Linear(G_SIZE, POTION_EMBED, bias=False)

        def forward(self, counts: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [
                    self.deck(counts[:, :E_SIZE]),
                    self.relics(counts[:, E_SIZE : E_SIZE + F_SIZE]),
                    self.potions(counts[:, E_SIZE + F_SIZE :]),
                ],
                dim=1,
            )

    class FightNet(nn.Module):
        """A fight's outcome for (view at fight start, encounter, weak): 22 logits."""

        def __init__(self) -> None:
            super().__init__()
            self.embed = Embeddings()
            self.hidden1 = nn.Linear(FIGHT_DENSE + EMBED, 256)
            self.hidden2 = nn.Linear(256, 128)
            self.out = nn.Linear(128, FIGHT_CLASSES)

        def forward(self, dense: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
            joined = torch.cat([dense, self.embed(counts)], dim=1)
            hidden = torch.relu(self.hidden1(joined))
            return self.out(torch.relu(self.hidden2(hidden)))

    class RunValueNet(nn.Module):
        """V: an afterstate's final-floor logits and its end-HP fraction."""

        def __init__(self) -> None:
            super().__init__()
            self.embed = Embeddings()
            self.hidden1 = nn.Linear(VALUE_DENSE + EMBED, 256)
            self.dropout = nn.Dropout(0.1)
            self.hidden2 = nn.Linear(256, 256)
            self.floor_head = nn.Linear(256, FLOOR_CLASSES)
            self.end_hp_head = nn.Linear(256, 1)

        def forward(
            self,
            dense: torch.Tensor,
            counts: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            joined = torch.cat([dense, self.embed(counts)], dim=1)
            hidden = self.dropout(torch.relu(self.hidden1(joined)))
            hidden = torch.relu(self.hidden2(hidden))
            end_hp = torch.sigmoid(self.end_hp_head(hidden)).squeeze(1)
            return self.floor_head(hidden), end_hp

    nets: dict[str, type] = {"FightNet": FightNet, "RunValueNet": RunValueNet}
    for cls in (Embeddings, FightNet, RunValueNet):
        cls.__qualname__ = cls.__name__
        cls.__module__ = __name__
    return nets


def __getattr__(name: str) -> Any:
    """Build `FightNet` and `RunValueNet` on first use, importing torch only then.

    Raises:
        AttributeError: For any other name, as a module does.

    """
    if name in {"FightNet", "RunValueNet"}:
        return _torch_nets()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def numpy_params(net: torch.nn.Module) -> dict[str, np.ndarray]:
    """Return a module's weights as float32 arrays, by state-dict name."""
    return {
        name: tensor.detach().cpu().numpy().astype(np.float32, copy=True)
        for name, tensor in net.state_dict().items()
    }


def save(
    pt: Path,
    fight: torch.nn.Module,
    value: torch.nn.Module,
    meta: dict[str, Any],
) -> Path:
    """Write a checkpoint: `pt` for torch, then its `.npz` export beside it (5.1).

    Each file is written whole or not at all, and the export goes last, so an `.npz`
    means its `.pt` is there too. `meta` must be plain JSON; it gains the format and the
    `FEATURE_VERSION`. Returns the `.npz` path.
    """
    import torch  # noqa: PLC0415 -- numpy-only processes never import torch (5.1)

    meta = {**meta, "format": CHECKPOINT_FORMAT, "feature_version": FEATURE_VERSION}
    buffer = io.BytesIO()
    torch.save(
        {"meta": meta, "fight": fight.state_dict(), "value": value.state_dict()},
        buffer,
    )
    write_atomic(Path(pt), buffer.getvalue())
    arrays: dict[str, np.ndarray] = {"meta": np.array(json.dumps(meta, sort_keys=True))}
    for prefix, net in (("fight/", fight), ("value/", value)):
        arrays.update(
            {prefix + name: array for name, array in numpy_params(net).items()},
        )
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=False, **arrays)
    npz = npz_for(pt)
    write_atomic(npz, buffer.getvalue())
    return npz


def load_torch(
    pt: Path | str,
    fight: torch.nn.Module,
    value: torch.nn.Module,
) -> dict[str, Any]:
    """Load a `.pt` checkpoint's weights into the two modules; return its metadata.

    A file that is not a value-model checkpoint, or was trained on another
    `FEATURE_VERSION`, raises `ValueError`.
    """
    import torch  # noqa: PLC0415 -- numpy-only processes never import torch (5.1)

    state = torch.load(pt, weights_only=True)
    meta = _checked(state["meta"], pt)
    fight.load_state_dict(state["fight"])
    value.load_state_dict(state["value"])
    return meta
