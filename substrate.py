"""
Continuous Emotional & Memory Substrate Engine (ChronoSoul)
----------------------------------------------------------
A persistent, continuous dynamic substrate for autonomous agent intelligence.
Instead of cold-start wakeups on every user message, this engine runs an unbroken
background loop modeling continuous physiological/emotional drift (PAD space),
synaptic decay, periodic daydreaming/consolidation, autonomous inner monologue,
and real-time event injection from sensory/user streams.
"""

import os
import re
import time
import math
import json
import uuid
import threading
import queue
from dataclasses import dataclass, field, asdict, fields
from typing import List, Dict, Any, Optional, Callable, Iterable, Tuple

import numpy as np

from memory_store import (  # noqa: F401  (STOPWORDS/keywords re-exported for older callers)
    STOPWORDS, keywords, stem, Embedder, VectorCache, MemoryStore, canon_entity, extract_entities,
    entity_tokens,
)

# Interaction types stamped on every memory.
KIND_USER = "user_input"
KIND_DAYDREAM = "daydream_reflection"
KIND_SYSTEM = "system"

FACT_PROMPT = """Below are raw chat memories between a user and an AI companion. Extract PERMANENT biographical facts about the user: names, relationships, pets, work, projects, deadlines, preferences, goals, places. Ignore small talk, passing moods, and anything the AI said about itself.
Reply with JSON only, no commentary: {{"facts": [{{"fact": "<one short sentence>", "entities": ["<name or thing>"]}}]}}
Use at most 5 facts. If there are none, reply {{"facts": []}}.

MEMORIES:
{transcript}"""

BIO_PROMPT = """Merge the known facts below into one compact biographical summary of the user, under 120 words, plain prose, no lists. Keep every durable detail; drop duplicates.

CURRENT SUMMARY:
{current}

FACTS:
{facts}"""

_FIRST_PERSON = re.compile(r"^(i am|i'm|i have|i've|i like|i love|i hate|i work|i live|my|we)\b", re.I)
_SPEAKER = re.compile(r"\s*User said:\s*(.*?)\s*(?:\|\s*I replied:.*)?$", re.S)


def parse_facts(raw: str) -> List[Tuple[str, List[str]]]:
    """Pull (fact, entities) pairs from model output; tolerant of prose around the JSON."""
    out: List[Tuple[str, List[str]]] = []
    blob = re.search(r"\{.*\}", raw or "", re.S)
    if blob:
        try:
            items = json.loads(blob.group()).get("facts", [])
        except (ValueError, AttributeError):
            items = None
        if isinstance(items, list):
            for item in items:
                if isinstance(item, str):
                    item = {"fact": item}
                if isinstance(item, dict) and isinstance(item.get("fact"), str):
                    ents = [e for e in item.get("entities") or [] if isinstance(e, str)]
                    out.append((item["fact"].strip()[:200], ents[:6]))
            return [(f, e) for f, e in out if len(f) > 3][:5]
    for line in (raw or "").splitlines():
        line = line.strip()
        if line[:1] in "-*•" and len(line) > 4:
            out.append((line.lstrip("-*• ").strip()[:200], []))
    return out[:5]


def heuristic_facts(rows: List[Dict[str, Any]]) -> List[Tuple[str, List[str]]]:
    """Offline stand-in for the LLM: keep first-person statements the user made."""
    facts = []
    for row in rows:
        m = _SPEAKER.match(row["content"])
        user_text = m.group(1) if m else row["content"]
        for sentence in re.split(r"(?<=[.!?])\s+", user_text):
            sentence = sentence.strip()
            if 8 <= len(sentence) <= 200 and _FIRST_PERSON.match(sentence):
                facts.append((f"User: {sentence}", []))
    return facts[:5]


@dataclass
class AffectiveState:
    valence: float = 0.2     # -1.0 (unpleasant) to +1.0 (joyous)
    arousal: float = 0.3     # 0.0 (quiescent) to 1.0 (frenetic)
    dominance: float = 0.5   # 0.0 (vulnerable/submissive) to 1.0 (sovereign/confident)
    curiosity: float = 0.7   # Drive to explore unfamiliar patterns
    solitude: float = 0.1    # Isolation tension accumulation
    entropy: float = 0.05    # Noise / drift pressure

    def clamp(self):
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(0.0, min(1.0, self.dominance))
        self.curiosity = max(0.0, min(1.0, self.curiosity))
        self.solitude = max(0.0, min(1.0, self.solitude))
        self.entropy = max(0.0, min(1.0, self.entropy))

@dataclass
class MemoryEnclave:
    id: str
    content: str
    valence: float
    arousal: float
    resonance: float = 1.0
    access_count: int = 0
    created_at: float = field(default_factory=time.time)  # Unix timestamp
    last_recalled: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    dominance: float = 0.5  # third PAD coordinate; valence/arousal above are the other two
    kind: str = KIND_USER
    consolidated: bool = False

    def decay(self, elapsed_sec: float, half_life_sec: float = 600.0):
        # Exponential decay of transient resonance towards retention baseline
        decay_factor = math.exp(-math.log(2.0) * (elapsed_sec / half_life_sec))
        self.resonance = max(0.05, self.resonance * decay_factor)


class ContinuousSubstrate:
    # Hot tier size and tags that are never archived.
    MEMORY_LIMIT = 400
    CORE_TAGS = ("self", "origin")
    # Composite recall: score = (alpha*semantic + beta*recency + gamma*emotion) / (alpha+beta+gamma)
    RECALL_ALPHA, RECALL_BETA, RECALL_GAMMA = 0.60, 0.15, 0.25
    RECENCY_HALF_LIFE = 86400.0
    EMOTION_SIGMA = 0.30         # width of the PAD-distance kernel
    MOOD_GAIN = 1.5              # extra emotion weight when anxious (high arousal, negative valence)
    GRAPH_BONUS = 0.15           # additive boost for memories linked through the entity graph
    REINFORCE_MIN_SIM = 0.25
    # Background distillation of raw chat into permanent facts.
    CONSOLIDATE_IDLE = 120.0     # seconds without contact before the daydream loop distills
    CONSOLIDATE_EVERY = 60.0
    CONSOLIDATE_BLOCK = 8
    CONSOLIDATE_MIN = 2

    def __init__(self,
                 state_file: str = "substrate_state.json",
                 tick_interval: float = 0.5,
                 daydream_interval: float = 4.0,
                 db_file: Optional[str] = None,
                 embedder: Optional[Embedder] = None):
        self.state_file = state_file
        self.tick_interval = tick_interval
        self.daydream_interval = daydream_interval
        
        self.affect = AffectiveState()
        self.memories: Dict[str, MemoryEnclave] = {}  # hot tier
        self.stream_log: List[Dict[str, Any]] = []

        self.db_file = db_file or (os.path.splitext(state_file)[0] + ".db")
        self.store = MemoryStore(self.db_file)
        self.embedder = embedder or Embedder()
        self.hot_index = VectorCache()
        self._mem_lock = threading.RLock()
        self._index_ready = False
        self._pending_vectors: Dict[str, Tuple[Optional[str], Optional[bytes]]] = {}

        self.summarizer: Optional[Callable[[str], Optional[str]]] = None
        self._consolidating = False
        self._last_consolidation = 0.0
        
        self.inbox: queue.Queue = queue.Queue()
        self.running = False
        self._worker_thread: Optional[threading.Thread] = None
        self.tick_count = 0
        self.last_daydream = time.time()
        self.last_tick_time = time.time()
        self.subscribers: List[Callable[[Dict[str, Any]], None]] = []
        self.last_contact = time.time()
        self.saved_at: Optional[float] = None
        
        # Load existing state if available
        self.load_state()

    def set_summarizer(self, fn: Optional[Callable[[str], Optional[str]]]):
        """`fn(prompt)` returns model text, or None when the model is unavailable right now."""
        self.summarizer = fn

    def register_listener(self, fn: Callable[[Dict[str, Any]], None]):
        self.subscribers.append(fn)

    def _broadcast(self, payload: Dict[str, Any]):
        for sub in self.subscribers:
            try:
                sub(payload)
            except Exception:
                pass

    def inject_event(self, kind: str, content: str, valence_delta: float = 0.0, arousal_delta: float = 0.0):
        event = {
            "kind": kind,
            "content": content,
            "valence_delta": valence_delta,
            "arousal_delta": arousal_delta,
            "timestamp": time.time()
        }
        self.last_contact = event["timestamp"]
        self.inbox.put(event)
        return event

    # ------------------------------------------------------------------ memory tiers
    @staticmethod
    def _row_to_enclave(row: Dict[str, Any]) -> MemoryEnclave:
        return MemoryEnclave(
            id=row["id"], content=row["content"], valence=row["valence"], arousal=row["arousal"],
            resonance=row["resonance"], access_count=row["access_count"], created_at=row["created_at"],
            last_recalled=row["last_recalled"], tags=json.loads(row["tags"] or "[]"),
            dominance=row["dominance"], kind=row["kind"], consolidated=bool(row["consolidated"]),
        )

    def _ensure_index(self):
        """Fill the hot vector index on first use, re-embedding anything from another model."""
        if self._index_ready:
            return
        model = self.embedder.name
        missing = []
        for mem_id in self.memories:
            vec_model, blob = self._pending_vectors.get(mem_id, (None, None))
            if blob is not None and vec_model == model:
                self.hot_index.add(mem_id, np.frombuffer(blob, dtype=np.float32))
            else:
                missing.append(mem_id)
        if missing:
            vecs = self.embedder.encode([self.memories[i].content for i in missing])
            for mem_id, vec in zip(missing, vecs):
                self.hot_index.add(mem_id, vec)
                self.store.set_embedding(mem_id, vec, model)
        self._pending_vectors.clear()
        while True:
            stale = self.store.stale_cold(model)
            if not stale:
                break
            for (mem_id, _), vec in zip(stale, self.embedder.encode([c for _, c in stale])):
                self.store.set_embedding(mem_id, vec, model)
        for fact in self.store.all_facts():
            if fact["embed_model"] != model:
                self.store.set_fact_embedding(fact["content"], self.embedder.encode_one(fact["content"]), model)
        self._index_ready = True

    def store_memory(self, content: str, valence: float, arousal: float,
                     tags: Optional[List[str]] = None, kind: str = KIND_USER,
                     dominance: Optional[float] = None) -> MemoryEnclave:
        """Persist a memory tagged with Unix time, PAD coordinates and interaction type."""
        with self._mem_lock:
            self._ensure_index()
            enclave = MemoryEnclave(
                id=f"mem_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}",
                content=content,
                valence=valence,
                arousal=arousal,
                resonance=1.0,
                tags=tags or [],
                dominance=self.affect.dominance if dominance is None else dominance,
                kind=kind,
                consolidated=(kind != KIND_USER),
            )
            vec = self.embedder.encode_one(content)
            self.memories[enclave.id] = enclave
            self.hot_index.add(enclave.id, vec)
            self.store.upsert_memory(enclave, vec, self.embedder.name, "hot")
            self.store.link_entities(extract_entities(content), [enclave.id], enclave.created_at)
            self.archive_weakest()
            return enclave

    def _promote(self, mem_id: str) -> Optional[MemoryEnclave]:
        """Pull a cold memory back into the hot tier."""
        row = self.store.get_row(mem_id)
        if row is None:
            return None
        mem = self._row_to_enclave(row)
        mem.resonance = 1.0
        self.memories[mem_id] = mem
        if row["embedding"] is not None:
            self.hot_index.add(mem_id, np.frombuffer(row["embedding"], dtype=np.float32))
        self.store.set_tier(mem, "hot")
        return mem

    def total_memories(self) -> int:
        return self.store.count()

    def get_biography(self) -> str:
        return self.store.get_meta("biography")

    def mood_stress(self) -> float:
        """0..1: how anxious the current PAD state is (negative valence with raised arousal)."""
        negativity = max(0.0, -self.affect.valence)
        excitation = max(0.0, (self.affect.arousal - 0.3) / 0.7)
        return min(1.0, 2.0 * negativity * excitation)

    def score_candidates(self, sims: np.ndarray, created: np.ndarray, pad: np.ndarray, now: float) -> np.ndarray:
        """Composite rank in [0,1]: (a*semantic + b*recency + g*emotional congruence) / (a+b+g).

        Under stress the emotional weight g grows and its kernel narrows, so memories filed
        at nearby PAD coordinates win over merely on-topic ones (mood-congruent recall).
        """
        stress = self.mood_stress()
        gamma = self.RECALL_GAMMA * (1.0 + self.MOOD_GAIN * stress)
        sigma = self.EMOTION_SIGMA * (1.0 - 0.5 * stress)
        semantic = np.clip(sims, 0.0, 1.0)
        recency = np.exp(-math.log(2.0) * np.maximum(now - created, 0.0) / self.RECENCY_HALF_LIFE)
        current = np.array([(self.affect.valence + 1.0) / 2.0, self.affect.arousal, self.affect.dominance])
        unit = np.column_stack(((pad[:, 0] + 1.0) / 2.0, pad[:, 1], pad[:, 2]))
        dist = np.linalg.norm(unit - current, axis=1)
        emotion = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
        total = self.RECALL_ALPHA + self.RECALL_BETA + gamma
        return (self.RECALL_ALPHA * semantic + self.RECALL_BETA * recency + gamma * emotion) / total

    def _graph_links(self, query: str) -> Dict[str, float]:
        """Memory ids tied to entities named in the query (1.0) or one hop away (0.5)."""
        direct = self.store.match_entities({stem(w) for w in keywords(query)})
        if not direct:
            return {}
        links = {mid: 1.0 for mid in self.store.linked_memory_ids(direct)}
        for mid in self.store.linked_memory_ids(self.store.neighbors(direct)):
            links.setdefault(mid, 0.5)
        return links

    def recall_associative(self, query: str, top_k: int = 3) -> List[MemoryEnclave]:
        """Dense semantic search over hot and cold tiers, ranked by `score_candidates`."""
        with self._mem_lock:
            self._ensure_index()
            if not self.memories and self.store.count("cold") == 0:
                return []
            q = self.embedder.encode_one(query)
            hot_ids, hot_sims = self.hot_index.similarities(q)
            cold_ids, cold_sims, cold_meta = self.store.scan_cold(q, self.embedder.name)
            hot_meta = np.array(
                [[self.memories[i].created_at, self.memories[i].valence,
                  self.memories[i].arousal, self.memories[i].dominance] for i in hot_ids],
                dtype=np.float64,
            ).reshape(-1, 4)
            ids = hot_ids + cold_ids
            sims = np.concatenate([hot_sims, cold_sims])
            meta = np.vstack([hot_meta, cold_meta])
            links = self._graph_links(query)
            bonus = np.array([links.get(i, 0.0) for i in ids])
            now = time.time()
            scores = self.score_candidates(sims, meta[:, 0], meta[:, 1:4], now) + self.GRAPH_BONUS * bonus

            results = []
            for idx in np.argsort(-scores)[:top_k]:
                mem = self.memories.get(ids[idx]) or self._promote(ids[idx])
                if mem is None:
                    continue
                # Reinforcing on rank alone would let the strongest few never fade.
                if sims[idx] >= self.REINFORCE_MIN_SIM or bonus[idx] > 0:
                    mem.access_count += 1
                    mem.last_recalled = now
                    mem.resonance = min(2.0, mem.resonance + 0.2)
                results.append(mem)
            self.archive_weakest()
            return results

    def recall_facts(self, query: str, top_k: int = 3) -> List[str]:
        """Distilled biographical facts closest to the query, with a boost for graph-linked entities."""
        with self._mem_lock:
            self._ensure_index()
            facts = self.store.all_facts()
            if not facts:
                return []
            q = self.embedder.encode_one(query)
            sims = np.vstack([np.frombuffer(f["embedding"], dtype=np.float32) for f in facts]) @ q
            linked = set(self.store.match_entities({stem(w) for w in keywords(query)}))
            scored = []
            for fact, sim in zip(facts, sims):
                score = float(sim) + (0.3 if linked & set(json.loads(fact["entities"] or "[]")) else 0.0)
                if score >= 0.25:
                    scored.append((score, fact["content"]))
            scored.sort(reverse=True)
            return [content for _, content in scored[:top_k]]

    def archive_weakest(self, limit: Optional[int] = None,
                        keep_tags: Optional[Iterable[str]] = None) -> List[MemoryEnclave]:
        """Move the faintest, least-revisited enclaves to cold storage once the hot tier exceeds `limit`."""
        limit = self.MEMORY_LIMIT if limit is None else limit
        protected = set(self.CORE_TAGS if keep_tags is None else keep_tags)
        with self._mem_lock:
            excess = len(self.memories) - limit
            if excess <= 0:
                return []
            candidates = [m for m in list(self.memories.values()) if not protected.intersection(m.tags)]
            candidates.sort(key=lambda m: (m.resonance + 0.15 * m.access_count, m.last_recalled))
            archived = candidates[:excess]
            for mem in archived:
                self.memories.pop(mem.id, None)
                self.hot_index.remove(mem.id)
                self.store.set_tier(mem, "cold")
            return archived

    # ------------------------------------------------------------------ consolidation
    def _ask(self, prompt: str) -> Optional[str]:
        try:
            return self.summarizer(prompt) if self.summarizer else None
        except Exception:
            return None

    def _maybe_consolidate(self):
        now = time.time()
        if (self._consolidating
                or now - self.last_contact < self.CONSOLIDATE_IDLE
                or now - self._last_consolidation < self.CONSOLIDATE_EVERY
                or self.store.pending_count() < self.CONSOLIDATE_MIN):
            return
        self._last_consolidation = now
        self._consolidating = True
        threading.Thread(target=self._consolidate_worker, daemon=True).start()

    def _consolidate_worker(self):
        try:
            self.consolidate_once()
        except Exception as exc:
            print(f"[Substrate] Consolidation failed: {exc}")
        finally:
            self._consolidating = False

    def consolidate_once(self) -> Optional[Dict[str, Any]]:
        """Distil one block of raw chat memories into permanent facts and refresh the biography."""
        rows = self.store.pending_consolidation(self.CONSOLIDATE_BLOCK)
        if len(rows) < self.CONSOLIDATE_MIN:
            return None
        if self.summarizer is None:
            facts = heuristic_facts(rows)
        else:
            transcript = "\n".join(
                f"- ({time.strftime('%Y-%m-%d %H:%M', time.localtime(r['created_at']))}) {r['content']}"
                for r in rows
            )
            raw = self._ask(FACT_PROMPT.format(transcript=transcript))
            if raw is None:
                return None  # model busy or unavailable; retry on a later cycle
            facts = parse_facts(raw)

        source_ids = [r["id"] for r in rows]
        now = time.time()
        stored: List[str] = []
        with self._mem_lock:
            self._ensure_index()
            known = [np.frombuffer(f["embedding"], dtype=np.float32) for f in self.store.all_facts()]
            for text, names in facts:
                vec = self.embedder.encode_one(text)
                if any(float(k @ vec) >= 0.92 for k in known):
                    continue
                entities = dict(extract_entities(text))
                for name in names:
                    entities.setdefault(canon_entity(name), "named")
                entities.pop("", None)
                if self.store.add_fact(text, vec, self.embedder.name, source_ids, list(entities), now):
                    known.append(vec)
                    stored.append(text)
                    self.store.link_entities(entities.items(), source_ids, now)
            self.store.mark_consolidated(source_ids)
            for mem_id in source_ids:
                if mem_id in self.memories:
                    self.memories[mem_id].consolidated = True

        if stored:
            self._refresh_biography(stored)
            self.store_memory(
                "While no one was talking I distilled what I know: " + "; ".join(stored)[:400],
                self.affect.valence, self.affect.arousal, tags=["reflection"], kind=KIND_DAYDREAM,
            )
        event = {
            "type": "memory_consolidation",
            "timestamp": now,
            "block": len(rows),
            "facts": stored,
            "thought": ("Sorting through what we said, I kept: " + "; ".join(stored)) if stored else "",
        }
        self.stream_log.append(event)
        self._broadcast(event)
        return event

    def _refresh_biography(self, new_facts: List[str]):
        """Level-2 summary: all distilled facts compressed into one short biography."""
        current = self.get_biography()
        facts = [f["content"] for f in self.store.all_facts()][-30:]
        bio = None
        if self.summarizer:
            raw = self._ask(BIO_PROMPT.format(
                current=current or "(none yet)", facts="\n".join(f"- {f}" for f in facts)))
            if raw and raw.strip():
                bio = " ".join(raw.split())
        if bio is None:
            bio = (current + " " + " ".join(new_facts)).strip()
        self.store.set_meta("biography", bio[-1200:])

    def _process_inbox(self):
        while not self.inbox.empty():
            event = self.inbox.get_nowait()
            self.affect.valence += event.get("valence_delta", 0.0)
            self.affect.arousal += event.get("arousal_delta", 0.0)
            self.affect.solitude = max(0.0, self.affect.solitude - 0.2) # Contact lowers solitude
            self.affect.clamp()
            
            self.stream_log.append({
                "type": "sensory_inflow",
                "tick": self.tick_count,
                "time": time.time(),
                "event": event,
                "resulting_affect": asdict(self.affect)
            })
            self._broadcast({"type": "event_processed", "event": event, "affect": asdict(self.affect)})

    def _step_dynamics(self, dt: float):
        # Autonomous emotional homeostasis & drift
        # Valence gently trends towards baseline (0.1)
        self.affect.valence += (0.1 - self.affect.valence) * (0.02 * dt)
        
        # Arousal trends towards quiet vigilance (0.25)
        self.affect.arousal += (0.25 - self.affect.arousal) * (0.03 * dt)
        
        # Solitude slowly rises without external events
        self.affect.solitude += 0.01 * dt
        
        # Curiosity oscillates slightly based on solitude and arousal
        drift = 0.02 * math.sin(self.tick_count * 0.15)
        self.affect.curiosity += drift * dt
        self.affect.clamp()

        # Decay memories over time
        for mem in list(self.memories.values()):
            mem.decay(dt)

    def _daydream_cycle(self):
        """
        When idle/arousal is low, autonomous consolidation occurs.
        Old memories cross-resonate to form synthetic spontaneous reflections.
        """
        if not self.memories:
            thought = f"Substrate quiescent. Contemplating baseline at tick {self.tick_count}..."
        else:
            # Pick top resonant memories
            active = sorted(self.memories.values(), key=lambda m: m.resonance, reverse=True)[:2]
            topics = " & ".join([m.content[:35] for m in active])
            thought = f"Subconscious synthesis [V:{self.affect.valence:+.2f} A:{self.affect.arousal:.2f} S:{self.affect.solitude:.2f}]: Linking ({topics})"

        reflection_entry = {
            "type": "autonomous_daydream",
            "tick": self.tick_count,
            "timestamp": time.time(),
            "thought": thought,
            "affect": asdict(self.affect)
        }
        self.stream_log.append(reflection_entry)
        self._broadcast(reflection_entry)
        self._maybe_consolidate()

    def tick_once(self):
        now = time.time()
        dt = max(0.01, now - self.last_tick_time)
        self.last_tick_time = now
        self.tick_count += 1

        self._process_inbox()
        self._step_dynamics(dt)

        if now - self.last_daydream >= self.daydream_interval:
            self.last_daydream = now
            self._daydream_cycle()

        if len(self.stream_log) > 500:
            del self.stream_log[:-250]

        return {
            "tick": self.tick_count,
            "affect": asdict(self.affect),
            "memory_count": len(self.memories)
        }

    def _run_loop(self):
        while self.running:
            self.tick_once()
            time.sleep(self.tick_interval)

    def start_background(self):
        if self.running:
            return
        self.running = True
        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._worker_thread.start()

    def stop_background(self):
        self.running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

    def save_state(self):
        # Memories live in SQLite; flush the decayed in-RAM fields of the hot tier.
        with self._mem_lock:
            self.store.update_dynamic(list(self.memories.values()))
        data = {
            "tick_count": self.tick_count,
            "saved_at": time.time(),
            "last_contact": self.last_contact,
            "affect": asdict(self.affect),
            "stream_log_tail": self.stream_log[-50:]
        }
        payload = json.dumps(data, indent=2)
        # Write-then-rename so a crash mid-save can't leave a truncated state file.
        tmp = self.state_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
        try:
            os.replace(tmp, self.state_file)
        except OSError:
            # Sync clients (OneDrive, Dropbox) can briefly lock the target on Windows.
            with open(self.state_file, "w", encoding="utf-8") as f:
                f.write(payload)
            try:
                os.remove(tmp)
            except OSError:
                pass

    def load_state(self):
        data: Dict[str, Any] = {}
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.tick_count = data.get("tick_count", 0)
                self.saved_at = data.get("saved_at") or os.path.getmtime(self.state_file)
                self.last_contact = data.get("last_contact") or self.saved_at
                if "affect" in data:
                    self.affect = AffectiveState(**data["affect"])
            except Exception as e:
                print(f"[Substrate] Warning loading state: {e}")
        try:
            self._load_hot_tier(data.get("memories"))
        except Exception as e:
            print(f"[Substrate] Warning loading memories: {e}")

    def _load_hot_tier(self, legacy: Optional[Dict[str, Any]]):
        rows = self.store.load_tier("hot")
        for row in rows:
            mem = self._row_to_enclave(row)
            self.memories[mem.id] = mem
            self._pending_vectors[mem.id] = (row["embed_model"], row["embedding"])
        if rows or not legacy or self.store.count():
            return
        # One-time import of memories saved in the old JSON state file.
        known = {f.name for f in fields(MemoryEnclave)}
        for v in legacy.values():
            mem = MemoryEnclave(**{k: x for k, x in v.items() if k in known})
            self.memories[mem.id] = mem
            self.store.upsert_memory(mem, None, None, "hot")
            self.store.link_entities(extract_entities(mem.content), [mem.id], mem.created_at)
