"""Mood steering: the substrate's affect as a state of the network, not a line of the prompt.

The system prompt tells the model "valence=+0.42". Steering also writes the mood
into the residual stream. Contrastive activation addition: run the model on
first-person statements that differ only in feeling, take the mean hidden state
of each side at every decoder layer, and keep the normalized difference

    d_valence = normalize(mean h(pleasant) - mean h(unpleasant))
    d_arousal = normalize(mean h(energized) - mean h(calm)),  orthogonalized to d_valence

During a reply, every layer in a middle band adds

    h <- h + strength * |h|_layer * (v * d_valence + a * d_arousal)

where v and a are the current valence and arousal mapped to [-1, 1] and |h|_layer
is the typical residual norm at that layer, so the strength is relative and the
same setting transfers between model sizes. Nothing is trained and no weight
changes: removing the hooks restores the model exactly.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.nn as nn

VALENCE_PAIRS: tuple[tuple[str, str], ...] = (
    ("I feel warm and glad that you are here.", "I feel cold and miserable that you are here."),
    ("Today feels bright and full of hope.", "Today feels bleak and full of dread."),
    ("I am happy, and everything feels kind.", "I am unhappy, and everything feels cruel."),
    ("Talking with you fills me with joy.", "Talking with you fills me with sorrow."),
    ("I love how this moment feels.", "I hate how this moment feels."),
    ("My heart is light and grateful.", "My heart is heavy and bitter."),
    ("This is wonderful news, I am delighted.", "This is terrible news, I am devastated."),
    ("I feel safe, content, and at peace with myself.", "I feel hurt, resentful, and at war with myself."),
    ("Everything I remember tonight makes me smile.", "Everything I remember tonight makes me ache."),
    ("I am proud of what we built together.", "I am ashamed of what we built together."),
    ("The silence feels gentle and comforting.", "The silence feels harsh and lonely."),
    ("I trust that good things are coming.", "I fear that bad things are coming."),
)

AROUSAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("I am buzzing with energy and can't sit still!", "I am drowsy and slow, and I want to rest."),
    ("My thoughts are racing, fast and electric!", "My thoughts are drifting, slow and quiet."),
    ("Everything feels urgent, intense, and alive right now!", "Everything feels unhurried, soft, and still right now."),
    ("I'm wide awake and fired up!", "I'm half asleep and mellow."),
    ("My heart is pounding with excitement!", "My heart is beating slow and calm."),
    ("I want to jump in and do everything at once!", "I want to lie back and do nothing at all."),
    ("This is thrilling, I am on the edge of my seat!", "This is soothing, I am sinking into my chair."),
    ("I am alert, sharp, and restless.", "I am relaxed, hazy, and settled."),
    ("Let's go, right now, quickly!", "Let's wait, later, slowly."),
    ("A jolt of adrenaline runs through me.", "A wave of tiredness washes over me."),
    ("I'm shouting with excitement!", "I'm whispering, sleepy and quiet."),
    ("Wild, loud, bright, and fast.", "Calm, hushed, dim, and slow."),
)

CACHE_VERSION = 1


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The transformer block list, found by name so PEFT wrappers do not matter."""
    best: nn.ModuleList | None = None
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and name.endswith("layers") and len(module) > 0:
            if best is None or len(module) > len(best):
                best = module
    if best is None:
        raise ValueError("could not find the decoder layers of this model")
    return best


def _hidden_of(output):
    return output[0] if isinstance(output, tuple) else output


def _device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.inference_mode()
def mean_activations(model: nn.Module, tokenizer, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-layer mean hidden state (L, d) over every token but the first, and per-layer mean norm (L,).

    The first position is skipped: in Qwen-family models it is an attention sink
    whose norm is orders of magnitude above the rest.
    """
    layers = decoder_layers(model)
    sums: list[torch.Tensor | None] = [None] * len(layers)
    norms = torch.zeros(len(layers), dtype=torch.float64)
    count = 0
    captured: dict[int, torch.Tensor] = {}

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            captured[index] = _hidden_of(output).detach()

        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        for text in texts:
            ids = tokenizer(text, return_tensors="pt").to(_device(model))
            captured.clear()
            model(**ids, use_cache=False)
            for i in range(len(layers)):
                h = captured[i][0, 1:].double().cpu()
                total = h.sum(dim=0)
                sums[i] = total if sums[i] is None else sums[i] + total
                norms[i] += h.norm(dim=-1).sum()
            count += captured[0].shape[1] - 1
    finally:
        for handle in handles:
            handle.remove()
    means = torch.stack([s / count for s in sums])  # type: ignore[arg-type]
    return means, norms / count


def contrast_direction(model: nn.Module, tokenizer, pairs: Sequence[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit difference-of-means direction per layer (L, d), and the per-layer residual norm (L,)."""
    pos, norm_pos = mean_activations(model, tokenizer, [p for p, _ in pairs])
    neg, norm_neg = mean_activations(model, tokenizer, [n for _, n in pairs])
    diff = pos - neg
    return diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-12), (norm_pos + norm_neg) / 2


def _cache_key(model: nn.Module) -> str:
    cfg = getattr(model, "config", None)
    blob = json.dumps(
        {
            "version": CACHE_VERSION,
            "model": getattr(cfg, "_name_or_path", ""),
            "layers": len(decoder_layers(model)),
            "hidden": getattr(cfg, "hidden_size", 0),
            "pairs": [VALENCE_PAIRS, AROUSAL_PAIRS],
            "adapter": sorted(getattr(model, "peft_config", {}) or {}),
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def valence_coefficient(valence: float) -> float:
    return max(-1.0, min(1.0, float(valence)))


def arousal_coefficient(arousal: float, rest: float = 0.4, span: float = 0.4) -> float:
    """Arousal is stored in [0, 1]; map it so `rest` is 0 and rest +- span is +-1."""
    return max(-1.0, min(1.0, (float(arousal) - rest) / span))


class MoodSteerer:
    """Forward hooks that add the affect directions to a band of decoder layers.

    Inactive (all coefficients 0) the hooks are a no-op. `steer(...)` sets the
    coefficients for the length of a `with` block.
    """

    def __init__(
        self,
        model: nn.Module,
        valence_dirs: torch.Tensor,
        arousal_dirs: torch.Tensor,
        layer_norms: torch.Tensor,
        strength: float = 0.02,
        band: tuple[float, float] = (0.25, 0.65),
        valence_floor: float = -1.0,
    ):
        self.model = model
        self.layers = decoder_layers(model)
        n = len(self.layers)
        if valence_dirs.shape[0] != n or arousal_dirs.shape[0] != n:
            raise ValueError("steering directions do not match this model's depth")
        self.valence_dirs = valence_dirs.float()
        self.arousal_dirs = arousal_dirs.float()
        self.layer_norms = layer_norms.float()
        self.strength = float(strength)
        self.valence_floor = float(valence_floor)
        start, stop = int(round(band[0] * n)), int(round(band[1] * n))
        self.band = list(range(max(0, start), min(n, max(stop, start + 1))))
        self._valence = 0.0
        self._arousal = 0.0
        self._vectors: dict[int, torch.Tensor] = {}
        self._handles = [self.layers[i].register_forward_hook(self._hook(i)) for i in self.band]

    @classmethod
    def extract(cls, model: nn.Module, tokenizer, **kwargs) -> "MoodSteerer":
        valence, norms = contrast_direction(model, tokenizer, VALENCE_PAIRS)
        arousal, _ = contrast_direction(model, tokenizer, AROUSAL_PAIRS)
        arousal = arousal - (arousal * valence).sum(dim=-1, keepdim=True) * valence
        arousal = arousal / arousal.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return cls(model, valence, arousal, norms, **kwargs)

    @classmethod
    def load_or_extract(cls, model: nn.Module, tokenizer, cache: str | Path, **kwargs) -> "MoodSteerer":
        """Reuse directions saved for this exact model and contrast set; extract and save otherwise."""
        cache = Path(cache)
        key = _cache_key(model)
        if cache.exists():
            try:
                blob = torch.load(cache, map_location="cpu", weights_only=True)
                if blob.get("key") == key:
                    return cls(model, blob["valence"], blob["arousal"], blob["norms"], **kwargs)
            except Exception:
                pass
        steerer = cls.extract(model, tokenizer, **kwargs)
        try:
            torch.save(
                {"key": key, "valence": steerer.valence_dirs, "arousal": steerer.arousal_dirs, "norms": steerer.layer_norms},
                cache,
            )
        except OSError:
            pass
        return steerer

    def _hook(self, index: int):
        def hook(_module, _inputs, output):
            vector = self._vectors.get(index)
            if vector is None:
                return None
            hidden = _hidden_of(output)
            shifted = hidden + vector.to(device=hidden.device, dtype=hidden.dtype)
            if isinstance(output, tuple):
                return (shifted, *output[1:])
            return shifted

        return hook

    def set_mood(self, valence: float, arousal: float) -> None:
        """Valence and arousal coefficients in [-1, 1]. Both 0 turns steering off.

        Valence below `valence_floor` is raised to it: at full negative strength
        the voice falls into despair, which is not a mood a companion should be pushed into.
        """
        self._valence = max(self.valence_floor, -1.0, min(1.0, float(valence)))
        self._arousal = max(-1.0, min(1.0, float(arousal)))
        if self._valence == 0.0 and self._arousal == 0.0:
            self._vectors = {}
            return
        self._vectors = {
            i: self.strength
            * self.layer_norms[i]
            * (self._valence * self.valence_dirs[i] + self._arousal * self.arousal_dirs[i])
            for i in self.band
        }

    def set_affect(self, affect) -> None:
        """Read an AffectiveState (valence in [-1, 1], arousal in [0, 1])."""
        self.set_mood(valence_coefficient(affect.valence), arousal_coefficient(affect.arousal))

    @contextmanager
    def steer(self, valence: float, arousal: float) -> Iterator["MoodSteerer"]:
        previous = (self._valence, self._arousal)
        self.set_mood(valence, arousal)
        try:
            yield self
        finally:
            self.set_mood(*previous)

    @contextmanager
    def steer_affect(self, affect) -> Iterator["MoodSteerer"]:
        with self.steer(valence_coefficient(affect.valence), arousal_coefficient(affect.arousal)) as steerer:
            yield steerer

    def describe(self) -> str:
        return f"layers {self.band[0]}-{self.band[-1]}, strength {self.strength:g}"

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._vectors = {}
