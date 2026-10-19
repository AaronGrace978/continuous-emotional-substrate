"""
Memory storage layer for the ChronoSoul substrate.

- Embedder:     all-MiniLM-L6-v2 (sentence-transformers) with an offline hashing fallback.
- VectorCache:  hot-tier vector index (FAISS flat inner-product, NumPy when FAISS is absent).
- MemoryStore:  SQLite durable store holding hot + cold tiers, distilled facts and the
                entity knowledge graph.
"""

import itertools
import json
import os
import re
import sqlite3
import sys
import threading
import zlib
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

STOPWORDS = frozenset("""
a about after all also am an and any are as at be because been but by can could did do does
doing for from get got had has have having he her here hers him his how i i'd i'll i'm i've if
in into is it it's its just like me more my no not now of oh ok okay on or our out really so
some than that that's the their them then there these they this those to too up us very was
we were what when where which while who why will with would yeah yes you you're your yours
said replied user
""".split())


def keywords(text: str) -> Set[str]:
    words = (w.strip("'") for w in re.findall(r"[a-z0-9']+", text.lower()))
    return {w for w in words if len(w) > 1 and w not in STOPWORDS}


def stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


# --------------------------------------------------------------------------- embeddings

def _hash_embed(text: str) -> np.ndarray:
    """Signed feature hashing of word stems and char trigrams; deterministic and offline."""
    vec = np.zeros(EMBED_DIM, dtype=np.float32)

    def add(token: str, weight: float):
        h = zlib.crc32(token.encode("utf-8"))
        vec[h % EMBED_DIM] += weight if (h >> 16) & 1 else -weight

    for word in keywords(text):
        s = stem(word)
        add("w:" + s, 1.0)
        padded = f"<{s}>"
        for i in range(len(padded) - 2):
            add("c:" + padded[i:i + 3], 0.3)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class Embedder:
    """Lazy embedding backend; `CHRONOSOUL_EMBEDDER=hash` forces the offline fallback."""

    def __init__(self, backend: Optional[str] = None):
        self.backend = (backend or os.environ.get("CHRONOSOUL_EMBEDDER", "auto")).lower()
        self._model = None
        self._name: Optional[str] = None
        self._lock = threading.Lock()

    def _resolve(self):
        with self._lock:
            if self._name is not None:
                return
            if self.backend in ("auto", "minilm"):
                try:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(EMBED_MODEL, device="cpu")
                    self._name = "minilm"
                except Exception as exc:
                    sys.stderr.write(f"[substrate] MiniLM unavailable, using hashing embedder: {exc}\n")
            if self._name is None:
                self._name = "hash"

    def warm(self):
        self._resolve()

    @property
    def name(self) -> str:
        self._resolve()
        return self._name or "hash"

    def encode(self, texts: List[str]) -> np.ndarray:
        self._resolve()
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        if self._model is not None:
            out = self._model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )
            return np.asarray(out, dtype=np.float32)
        return np.vstack([_hash_embed(t) for t in texts])

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class VectorCache:
    """Hot-tier index. Callers serialise access; the index is rebuilt lazily after changes."""

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self._vecs: Dict[str, np.ndarray] = {}
        self._ids: List[str] = []
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._index = None
        self._dirty = False

    def __len__(self) -> int:
        return len(self._vecs)

    def __contains__(self, mem_id: str) -> bool:
        return mem_id in self._vecs

    def add(self, mem_id: str, vec: np.ndarray):
        self._vecs[mem_id] = np.asarray(vec, dtype=np.float32)
        self._dirty = True

    def remove(self, mem_id: str):
        if self._vecs.pop(mem_id, None) is not None:
            self._dirty = True

    def _build(self):
        self._ids = list(self._vecs)
        if self._ids:
            self._matrix = np.vstack([self._vecs[i] for i in self._ids])
        else:
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        self._index = None
        if faiss is not None and self._ids:
            self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(self._matrix)
        self._dirty = False

    def similarities(self, query: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """Cosine similarity (vectors are unit length) of the query against every cached id."""
        if self._dirty:
            self._build()
        if not self._ids:
            return [], np.zeros(0, dtype=np.float32)
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if self._index is not None:
            dist, idx = self._index.search(q, len(self._ids))
            sims = np.zeros(len(self._ids), dtype=np.float32)
            sims[idx[0]] = dist[0]
            return self._ids, sims
        return self._ids, (self._matrix @ q[0]).astype(np.float32)


# --------------------------------------------------------------------------- entities

_ROLE_SKIP = frozenset("""
thought thoughts mind life way time day opinion question point idea turn answer mistake fault
best own favorite favourite new old little big first last good great other same
""".split())
_NOT_ENTITIES = frozenset("""
user i chronosoul said replied monday tuesday wednesday thursday friday saturday sunday
january february march april may june july august september october november december
""".split())
_ROLE_MODIFIERS = frozenset("best own favorite favourite new old little big first last good great other same".split())
_NAMED_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_SPEAKER_RE = re.compile(r"\s*User said:\s*(.*?)\s*\|\s*I replied:\s*(.*)", re.S)


def canon_entity(name: str) -> str:
    n = re.sub(r"\s+", " ", name.strip().strip(".,;:!?\"()[]").lower())
    n = re.sub(r"^(?:the user's|users'|user's|my|our|your|their)\s+", "user's ", n)
    if not entity_tokens(n):
        return ""
    return n[:60]


def entity_tokens(name: str) -> Set[str]:
    return {stem(w) for w in keywords(re.sub(r"\buser's\b", " ", name.lower()))}


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Cheap heuristic entity spotting: 'my dog' -> user's dog, plus mid-sentence proper nouns."""
    user_part, bot_part = text, ""
    m = _SPEAKER_RE.match(text)
    if m:
        user_part, bot_part = m.groups()
    found: Dict[str, str] = {}

    for part, owners in ((user_part, "my|our"), (bot_part, "your")):
        pattern = re.compile(rf"\b(?:{owners})\s+([a-z][a-z-]+)(?:\s+([a-z][a-z-]+))?", re.I)
        for first, second in pattern.findall(part):
            word = (second if first.lower() in _ROLE_MODIFIERS and second else first).lower()
            if len(word) < 3 or word in STOPWORDS or word in _ROLE_SKIP:
                continue
            found.setdefault(canon_entity(f"user's {word}"), "role")

    for match in _NAMED_RE.finditer(user_part):
        words = match.group().split()
        while words and (words[0].lower() in STOPWORDS or words[0].lower() in _NOT_ENTITIES):
            words.pop(0)
        if not words or any(w.lower() in _NOT_ENTITIES for w in words):
            continue
        lead = user_part[:match.start()].rstrip()
        if len(words) == 1 and (not lead or lead[-1] in ".!?:"):
            continue
        found.setdefault(canon_entity(" ".join(words)), "named")

    return [(n, k) for n, k in found.items() if n][:8]


# --------------------------------------------------------------------------- SQLite store

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories(
    id TEXT PRIMARY KEY, content TEXT NOT NULL,
    valence REAL, arousal REAL, dominance REAL,
    resonance REAL, access_count INTEGER, created_at REAL, last_recalled REAL,
    kind TEXT, tags TEXT, tier TEXT, consolidated INTEGER,
    embed_model TEXT, embedding BLOB);
CREATE INDEX IF NOT EXISTS idx_mem_tier ON memories(tier, created_at);
CREATE TABLE IF NOT EXISTS facts(
    id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT UNIQUE, created_at REAL,
    source_ids TEXT, entities TEXT, embed_model TEXT, embedding BLOB);
CREATE TABLE IF NOT EXISTS entities(
    name TEXT PRIMARY KEY, kind TEXT, tokens TEXT, mentions INTEGER,
    first_seen REAL, last_seen REAL);
CREATE TABLE IF NOT EXISTS entity_memories(
    entity TEXT, memory_id TEXT, PRIMARY KEY(entity, memory_id));
CREATE INDEX IF NOT EXISTS idx_em_entity ON entity_memories(entity);
CREATE TABLE IF NOT EXISTS entity_edges(
    a TEXT, b TEXT, weight REAL, last_seen REAL, PRIMARY KEY(a, b));
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""

_MEM_COLS = ("id, content, valence, arousal, dominance, resonance, access_count, created_at, "
             "last_recalled, kind, tags, tier, consolidated, embed_model, embedding")


def _blob(vec: Optional[np.ndarray]) -> Optional[bytes]:
    return None if vec is None else np.asarray(vec, dtype=np.float32).tobytes()


class MemoryStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            try:
                self._db.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self):
        with self._lock:
            self._db.commit()
            self._db.close()

    # ---- memories
    def upsert_memory(self, mem, vec: Optional[np.ndarray], embed_model: Optional[str], tier: str):
        with self._lock:
            self._db.execute(
                f"INSERT OR REPLACE INTO memories({_MEM_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (mem.id, mem.content, mem.valence, mem.arousal, mem.dominance, mem.resonance,
                 mem.access_count, mem.created_at, mem.last_recalled, mem.kind,
                 json.dumps(mem.tags), tier, int(mem.consolidated),
                 embed_model if vec is not None else None, _blob(vec)),
            )
            self._db.commit()

    def update_dynamic(self, mems: Iterable):
        with self._lock:
            self._db.executemany(
                "UPDATE memories SET resonance=?, access_count=?, last_recalled=?, consolidated=? WHERE id=?",
                [(m.resonance, m.access_count, m.last_recalled, int(m.consolidated), m.id) for m in mems],
            )
            self._db.commit()

    def set_tier(self, mem, tier: str):
        with self._lock:
            self._db.execute(
                "UPDATE memories SET tier=?, resonance=?, access_count=?, last_recalled=? WHERE id=?",
                (tier, mem.resonance, mem.access_count, mem.last_recalled, mem.id),
            )
            self._db.commit()

    def get_row(self, mem_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._db.execute(f"SELECT {_MEM_COLS} FROM memories WHERE id=?", (mem_id,)).fetchone()
        return dict(row) if row else None

    def load_tier(self, tier: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                f"SELECT {_MEM_COLS} FROM memories WHERE tier=? ORDER BY created_at", (tier,)
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self, tier: Optional[str] = None) -> int:
        with self._lock:
            if tier is None:
                return self._db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            return self._db.execute("SELECT COUNT(*) FROM memories WHERE tier=?", (tier,)).fetchone()[0]

    def stale_cold(self, embed_model: str, limit: int = 64) -> List[Tuple[str, str]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, content FROM memories WHERE tier='cold' "
                "AND (embed_model IS NULL OR embed_model != ?) LIMIT ?", (embed_model, limit),
            ).fetchall()
        return [(r["id"], r["content"]) for r in rows]

    def set_embedding(self, mem_id: str, vec: np.ndarray, embed_model: str):
        with self._lock:
            self._db.execute(
                "UPDATE memories SET embedding=?, embed_model=? WHERE id=?", (_blob(vec), embed_model, mem_id)
            )
            self._db.commit()

    def scan_cold(self, query: np.ndarray, embed_model: str, chunk: int = 2000):
        """Stream the cold tier from disk; returns ids, cosine sims, (created, V, A, D) metadata and interaction kinds."""
        ids: List[str] = []
        sims: List[np.ndarray] = []
        meta: List[Tuple[float, float, float, float]] = []
        kinds: List[str] = []
        with self._lock:
            cur = self._db.execute(
                "SELECT id, embedding, created_at, valence, arousal, dominance, kind FROM memories "
                "WHERE tier='cold' AND embed_model=? AND embedding IS NOT NULL", (embed_model,),
            )
            while True:
                rows = cur.fetchmany(chunk)
                if not rows:
                    break
                mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                sims.append(mat @ np.asarray(query, dtype=np.float32))
                ids.extend(r["id"] for r in rows)
                meta.extend((r["created_at"], r["valence"], r["arousal"], r["dominance"]) for r in rows)
                kinds.extend(r["kind"] for r in rows)
        if not ids:
            return [], np.zeros(0, dtype=np.float32), np.zeros((0, 4)), []
        return ids, np.concatenate(sims), np.array(meta, dtype=np.float64), kinds

    # ---- consolidation bookkeeping
    def pending_consolidation(self, limit: int) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, content, created_at FROM memories WHERE consolidated=0 AND kind='user_input' "
                "ORDER BY created_at LIMIT ?", (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def pending_count(self) -> int:
        with self._lock:
            return self._db.execute(
                "SELECT COUNT(*) FROM memories WHERE consolidated=0 AND kind='user_input'"
            ).fetchone()[0]

    def mark_consolidated(self, ids: Iterable[str]):
        with self._lock:
            self._db.executemany("UPDATE memories SET consolidated=1 WHERE id=?", [(i,) for i in ids])
            self._db.commit()

    # ---- facts
    def add_fact(self, content: str, vec: np.ndarray, embed_model: str,
                 source_ids: List[str], entities: List[str], now: float) -> bool:
        with self._lock:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO facts(content, created_at, source_ids, entities, embed_model, embedding) "
                "VALUES (?,?,?,?,?,?)",
                (content, now, json.dumps(source_ids), json.dumps(entities), embed_model, _blob(vec)),
            )
            self._db.commit()
            return cur.rowcount > 0

    def all_facts(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT content, created_at, entities, embed_model, embedding FROM facts ORDER BY created_at"
            ).fetchall()
        return [dict(r) for r in rows]

    def set_fact_embedding(self, content: str, vec: np.ndarray, embed_model: str):
        with self._lock:
            self._db.execute(
                "UPDATE facts SET embedding=?, embed_model=? WHERE content=?", (_blob(vec), embed_model, content)
            )
            self._db.commit()

    # ---- knowledge graph
    def link_entities(self, entities: Iterable[Tuple[str, str]], memory_ids: Iterable[str], now: float):
        by_name: Dict[str, str] = {}
        for name, kind in entities:
            canon = canon_entity(name)
            if canon:
                by_name.setdefault(canon, kind)
        if not by_name:
            return
        mids = list(memory_ids)
        with self._lock:
            for name, kind in by_name.items():
                self._db.execute(
                    "INSERT INTO entities(name, kind, tokens, mentions, first_seen, last_seen) VALUES (?,?,?,?,?,?) "
                    "ON CONFLICT(name) DO UPDATE SET mentions=mentions+1, last_seen=excluded.last_seen",
                    (name, kind, " ".join(sorted(entity_tokens(name))), 1, now, now),
                )
                self._db.executemany(
                    "INSERT OR IGNORE INTO entity_memories(entity, memory_id) VALUES (?,?)",
                    [(name, mid) for mid in mids],
                )
            for a, b in itertools.combinations(sorted(by_name), 2):
                self._db.execute(
                    "INSERT INTO entity_edges(a, b, weight, last_seen) VALUES (?,?,1,?) "
                    "ON CONFLICT(a, b) DO UPDATE SET weight=weight+1, last_seen=excluded.last_seen",
                    (a, b, now),
                )
            self._db.commit()

    def match_entities(self, query_tokens: Set[str]) -> List[str]:
        if not query_tokens:
            return []
        with self._lock:
            rows = self._db.execute("SELECT name, tokens FROM entities").fetchall()
        return [r["name"] for r in rows if query_tokens & set(r["tokens"].split())]

    def neighbors(self, names: Iterable[str]) -> List[str]:
        names = list(names)
        if not names:
            return []
        marks = ",".join("?" * len(names))
        with self._lock:
            rows = self._db.execute(
                f"SELECT a, b FROM entity_edges WHERE a IN ({marks}) OR b IN ({marks})", names + names
            ).fetchall()
        near = {r["a"] for r in rows} | {r["b"] for r in rows}
        return sorted(near - set(names))

    def linked_memory_ids(self, names: Iterable[str]) -> Set[str]:
        names = list(names)
        if not names:
            return set()
        marks = ",".join("?" * len(names))
        with self._lock:
            rows = self._db.execute(
                f"SELECT DISTINCT memory_id FROM entity_memories WHERE entity IN ({marks})", names
            ).fetchall()
        return {r["memory_id"] for r in rows}

    # ---- meta
    def get_meta(self, key: str, default: str = "") -> str:
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (key, value))
            self._db.commit()
