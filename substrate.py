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
import time
import math
import json
import threading
import queue
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable

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
    created_at: float = field(default_factory=time.time)
    last_recalled: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    def decay(self, elapsed_sec: float, half_life_sec: float = 600.0):
        # Exponential decay of transient resonance towards retention baseline
        decay_factor = math.exp(-math.log(2.0) * (elapsed_sec / half_life_sec))
        self.resonance = max(0.05, self.resonance * decay_factor)


class ContinuousSubstrate:
    def __init__(self,
                 state_file: str = "substrate_state.json",
                 tick_interval: float = 0.5,
                 daydream_interval: float = 4.0):
        self.state_file = state_file
        self.tick_interval = tick_interval
        self.daydream_interval = daydream_interval
        
        self.affect = AffectiveState()
        self.memories: Dict[str, MemoryEnclave] = {}
        self.stream_log: List[Dict[str, Any]] = []
        
        self.inbox: queue.Queue = queue.Queue()
        self.running = False
        self._worker_thread: Optional[threading.Thread] = None
        self.tick_count = 0
        self.last_daydream = time.time()
        self.last_tick_time = time.time()
        self.subscribers: List[Callable[[Dict[str, Any]], None]] = []
        
        # Load existing state if available
        self.load_state()

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
        self.inbox.put(event)
        return event

    def store_memory(self, content: str, valence: float, arousal: float, tags: Optional[List[str]] = None) -> MemoryEnclave:
        mem_id = f"mem_{int(time.time() * 1000)}_{len(self.memories)}"
        enclave = MemoryEnclave(
            id=mem_id,
            content=content,
            valence=valence,
            arousal=arousal,
            resonance=1.0,
            tags=tags or []
        )
        self.memories[mem_id] = enclave
        return enclave

    def recall_associative(self, query: str, top_k: int = 3) -> List[MemoryEnclave]:
        # Associative resonance ranking based on emotional alignment and tags
        scored = []
        q_tokens = set(query.lower().split())
        for mem in self.memories.values():
            mem_tokens = set(mem.content.lower().split() + [t.lower() for t in mem.tags])
            overlap = len(q_tokens.intersection(mem_tokens))
            # Emotional proximity to current affect
            aff_dist = math.sqrt(
                (mem.valence - self.affect.valence) ** 2 +
                (mem.arousal - self.affect.arousal) ** 2
            )
            proximity = 1.0 / (1.0 + aff_dist)
            score = (overlap * 2.0) + (mem.resonance * 1.5) + proximity
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for s, mem in scored[:top_k]:
            mem.access_count += 1
            mem.last_recalled = time.time()
            mem.resonance = min(2.0, mem.resonance + 0.2) # Synaptic reinforcement
            results.append(mem)
        return results

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
        for mem in self.memories.values():
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
        data = {
            "tick_count": self.tick_count,
            "affect": asdict(self.affect),
            "memories": {k: asdict(v) for k, v in self.memories.items()},
            "stream_log_tail": self.stream_log[-50:]
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_state(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.tick_count = data.get("tick_count", 0)
            if "affect" in data:
                self.affect = AffectiveState(**data["affect"])
            if "memories" in data:
                for k, v in data["memories"].items():
                    self.memories[k] = MemoryEnclave(**v)
        except Exception as e:
            print(f"[Substrate] Warning loading state: {e}")
