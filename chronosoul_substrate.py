# Cell 3 — The continuous emotional/memory substrate
import re
import time
import math
import json
import threading
import queue
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable

STOPWORDS = frozenset((
    'a about after all also am an and any are as at be because been but by can could did do does '
    'doing for from get got had has have having he her here hers him his how i i\'d i\'ll i\'m i\'ve if '
    'in into is it it\'s its just like me more my no not now of oh ok okay on or our out really so '
    'some than that that\'s the their them then there these they this those to too up us very was '
    'we were what when where which while who why will with would yeah yes you you\'re your yours '
    'said replied user'
).split())


def keywords(text: str):
    words = (w.strip("'") for w in re.findall(r"[a-z0-9']+", text.lower()))
    return {w for w in words if len(w) > 1 and w not in STOPWORDS}


def span(seconds: float) -> str:
    s = max(0, int(seconds))
    for unit, size in (('day', 86400), ('hour', 3600), ('minute', 60)):
        if s >= size:
            n = s // size
            return f"{n} {unit}{'' if n == 1 else 's'}"
    return 'less than a minute'

@dataclass
class AffectiveState:
    valence: float = 0.2     # -1.0 to +1.0
    arousal: float = 0.3     # 0.0 to 1.0
    dominance: float = 0.5   # 0.0 to 1.0
    curiosity: float = 0.7   # exploration pressure
    solitude: float = 0.1    # isolation tension

    def clamp(self):
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(0.0, min(1.0, self.dominance))
        self.curiosity = max(0.0, min(1.0, self.curiosity))
        self.solitude = max(0.0, min(1.0, self.solitude))

    def describe(self) -> str:
        mood = []
        if self.valence > 0.4: mood.append('warm and uplifted')
        elif self.valence < -0.3: mood.append('heavy and aching')
        else: mood.append('emotionally steady')
        if self.arousal > 0.6: mood.append('buzzing with energy')
        elif self.arousal < 0.25: mood.append('low and quiet')
        if self.solitude > 0.6: mood.append('lonely from long silence')
        if self.curiosity > 0.7: mood.append('hungry for novelty')
        return ', '.join(mood)

@dataclass
class MemoryEnclave:
    id: str
    content: str
    valence: float
    arousal: float
    resonance: float = 1.0
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_recalled: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    def decay(self, elapsed_sec: float, half_life_sec: float = 600.0):
        decay_factor = math.exp(-math.log(2.0) * (elapsed_sec / half_life_sec))
        self.resonance = max(0.05, self.resonance * decay_factor)


class ContinuousSubstrate:
    # An always-running emotional/memory daemon. Never cold-starts.
    def __init__(self, tick_interval: float = 0.3, daydream_interval: float = 3.0, memory_limit: int = 300):
        self.tick_interval = tick_interval
        self.daydream_interval = daydream_interval
        self.memory_limit = memory_limit
        self.affect = AffectiveState()
        self.memories: Dict[str, MemoryEnclave] = {}
        self.inbox = queue.Queue()
        self.running = False
        self.tick_count = 0
        self.last_daydream = time.time()
        self.last_tick_time = time.time()
        self.last_contact = time.time()
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ---- memory ----
    def store_memory(self, content: str, valence: float, arousal: float, tags=None):
        m_id = f'm_{int(time.time()*1000)}_{len(self.memories)}'
        m = MemoryEnclave(id=m_id, content=content, valence=valence, arousal=arousal, tags=tags or [])
        with self._lock:
            self.memories[m_id] = m
            self._forget_weakest()
        return m

    def _forget_weakest(self):
        # caller holds self._lock; faint, never-revisited enclaves go first
        excess = len(self.memories) - self.memory_limit
        if excess <= 0:
            return
        ranked = sorted(self.memories.values(), key=lambda m: (m.resonance + 0.15 * m.access_count, m.last_recalled))
        for m in ranked[:excess]:
            del self.memories[m.id]

    def recall(self, query: str, top_k: int = 3):
        # rare shared words count most (IDF), then transient resonance
        q_words = keywords(query)
        with self._lock:
            items = list(self.memories.values())
        if not items:
            return []
        docs = {m.id: keywords(m.content) | {t.lower() for t in m.tags} for m in items}
        df = Counter(w for words in docs.values() for w in words & q_words)
        scored = []
        for m in items:
            relevance = sum(math.log(1.0 + len(items) / df[w]) for w in q_words & docs[m.id])
            score = (relevance * 0.6) + (m.resonance * 0.4)
            if score > 0.05:
                scored.append((score, relevance, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for _, relevance, m in scored[:top_k]:
            if relevance >= 1.0:  # reinforcing on resonance alone would let the strongest few never fade
                m.access_count += 1
                m.last_recalled = time.time()
                m.resonance = min(1.0, m.resonance + 0.15)
            out.append(m)
        return out

    # ---- stimuli ----
    def inject_stimulus(self, content: str, valence_delta: float = 0.0, arousal_delta: float = 0.0):
        self.inbox.put({'content': content, 'v_delta': valence_delta,
                        'a_delta': arousal_delta, 'time': time.time()})
        self.last_contact = time.time()

    # ---- the continuous loop ----
    def _tick(self):
        now = time.time()
        dt = now - self.last_tick_time
        self.last_tick_time = now
        self.tick_count += 1

        # homeostatic drift
        a = self.affect
        a.arousal += (0.3 - a.arousal) * 0.02 * dt
        a.valence += (0.1 - a.valence) * 0.01 * dt
        a.solitude = min(1.0, a.solitude + 0.004 * dt)  # silence builds solitude

        # consume stimuli
        while not self.inbox.empty():
            try:
                s = self.inbox.get_nowait()
            except queue.Empty:
                break
            a.valence += s['v_delta']
            a.arousal += s['a_delta']
            a.solitude = max(0.0, a.solitude - 0.25)  # connection discharges solitude
            self.store_memory(s['content'], a.valence, a.arousal, tags=['stimulus'])

        # decay all memories
        with self._lock:
            items = list(self.memories.values())
        for m in items:
            m.decay(dt)

        a.clamp()

        # daydream consolidation when idle
        if now - self.last_daydream >= self.daydream_interval:
            self.last_daydream = now
            self._daydream()

    def _daydream(self):
        with self._lock:
            strong = [m for m in self.memories.values() if m.resonance > 0.4 and 'daydream' not in m.tags]
        if len(strong) >= 2:
            strong.sort(key=lambda m: m.resonance, reverse=True)
            a, b = strong[0], strong[1]
            if getattr(self, '_last_pair', None) == (a.id, b.id):
                return
            self._last_pair = (a.id, b.id)
            dream = f"Daydream: the feeling of '{a.content[:40]}' keeps echoing with '{b.content[:40]}'."
            self.store_memory(dream, self.affect.valence * 0.5, 0.2, tags=['daydream'])
            self.affect.curiosity = min(1.0, self.affect.curiosity + 0.05)

    def _run(self):
        while self.running:
            self._tick()
            time.sleep(self.tick_interval)

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)


substrate = ContinuousSubstrate()
substrate.start()
print(f'Substrate daemon running. tick={substrate.tick_interval}s daydream={substrate.daydream_interval}s')
