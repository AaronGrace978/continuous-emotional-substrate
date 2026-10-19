"""
Quick look-back over older conversation turns.

A literal grep (rare shared words, IDF-weighted) combined with sentence-level embedding
similarity, so a turn surfaces both when it repeats a distinctive word and when one of its
sentences is about the same thing. Long, multi-topic turns are judged by their best sentence.
"""

import math
import re
import threading
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from memory_store import Embedder, keywords, stem


@dataclass
class Hit:
    role: str
    content: str   # the best-matching excerpt, not the whole turn
    t: float       # Unix time of the turn, 0 when unknown
    score: float


def _terms(text: str) -> Set[str]:
    return {stem(w) for w in keywords(text)}


# Words common to any chat; sharing one says little about topic, so they count for less.
GENERIC = {stem(w) for w in """
today feel feeling think thing things talk talks something anything everything going doing tell
know want need make good great nice well back time way kind sure thank thanks hello hey
remember recall mention mentioned said told conversation chat
""".split()}


def _split(text: str) -> List[str]:
    text = re.sub(r"[*_`#>]+", "", text)
    parts = [p.strip()[:300] for p in re.split(r"(?<=[.!?])\s+", text) if len(p.split()) >= 3]
    return parts[:16] or [text.strip()[:300]]


class ConversationGrep:
    LEX_SCALE = 2.5        # IDF mass at which the lexical score reaches ~63%
    SEM_FLOOR, SEM_SPAN = 0.55, 0.25   # cosine mapped linearly from [floor, floor+span] to [0, 1]
    MIN_SCORE = 0.45
    OVERLAP_NUDGE = 0.15   # per shared term when choosing a turn's excerpt sentence

    def __init__(self, embedder: Embedder, limit: int = 3, snippet_chars: int = 220):
        self.embedder = embedder
        self.limit = limit
        self.snippet_chars = snippet_chars
        self._vecs: Dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    def _vectors(self, sentences: Sequence[str]) -> np.ndarray:
        keys = [hash(s) for s in sentences]
        with self._lock:
            missing = [i for i, k in enumerate(keys) if k not in self._vecs]
            if missing:
                for i, vec in zip(missing, self.embedder.encode([sentences[i] for i in missing])):
                    self._vecs[keys[i]] = vec
            self._vecs = {k: self._vecs[k] for k in keys}  # forget sentences that left the transcript
            return np.vstack([self._vecs[k] for k in keys]) if keys else np.zeros((0, 1), dtype=np.float32)

    def prime(self, turns: Sequence[dict]):
        """Embed ahead of time so the first real search is as quick as the rest."""
        self._vectors([s for t in turns for s in _split(t["content"])])

    def search(self, query: str, turns: Sequence[dict]) -> List[Hit]:
        """Up to `limit` turns that pertain to `query`, oldest first; empty when nothing does."""
        if not turns or not query.strip():
            return []
        q_terms = _terms(query)
        sents = [_split(t["content"]) for t in turns]
        flat = [(i, s) for i, ss in enumerate(sents) for s in ss]
        cosine = self._vectors([s for _, s in flat]) @ self.embedder.encode_one(query[:1000])

        semantic = np.zeros(len(turns))
        best: Dict[int, Tuple[float, int]] = {}   # turn -> (pick score, sentence index)
        for k, (i, s) in enumerate(flat):
            semantic[i] = max(semantic[i], cosine[k])
            pick = cosine[k] + self.OVERLAP_NUDGE * len(q_terms & _terms(s))
            if i not in best or pick > best[i][0]:
                best[i] = (pick, sents[i].index(s))
        semantic = np.clip((semantic - self.SEM_FLOOR) / self.SEM_SPAN, 0.0, 1.0)

        docs = [_terms(t["content"]) for t in turns]
        counts = {w: sum(w in d for d in docs) for w in q_terms}
        weight = {w: math.log(1.0 + len(turns) / c) * (0.3 if w in GENERIC else 1.0)
                  for w, c in counts.items() if c}
        lexical = np.array([
            1.0 - math.exp(-sum(weight.get(w, 0.0) for w in q_terms & d) / self.LEX_SCALE) for d in docs
        ])
        score = 1.0 - (1.0 - lexical) * (1.0 - semantic)

        hits: List[Hit] = []
        seen: Set[str] = set()
        for i in np.argsort(-score):
            if score[i] < self.MIN_SCORE or len(hits) >= self.limit:
                break
            ss, j = sents[i], best[i][1]
            snippet = ss[j]
            if len(snippet) < self.snippet_chars // 2 and j + 1 < len(ss):
                snippet += " " + ss[j + 1]
            if len(snippet) > self.snippet_chars:
                snippet = snippet[:self.snippet_chars].rsplit(" ", 1)[0] + "…"
            if snippet[:60] in seen:
                continue
            seen.add(snippet[:60])
            hits.append(Hit(turns[i]["role"], snippet, float(turns[i].get("t", 0.0)), float(score[i])))
        return sorted(hits, key=lambda h: h.t)
