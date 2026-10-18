# 🧠 ChronoSoul: Continuous Emotional & Memory Substrate

> An autonomous, persistent background consciousness loop for Large Language Models.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AaronGrace978/continuous-emotional-substrate/blob/main/Persistent_Emotional_Memory_Substrate.ipynb)

Traditional LLMs operate on cold-starts: they freeze between turns, only "exist" when a message arrives, parse a flat text history log, and terminate immediately after replying.

**ChronoSoul** provides a biological and psychological substrate that runs continuously in real-time, whether messages arrive or not.

---

## 🚀 Instant Launch

Click below to open and run the fine-tuning / continuous substrate notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AaronGrace978/continuous-emotional-substrate/blob/main/Persistent_Emotional_Memory_Substrate.ipynb)

**Direct Link:** [Open Persistent_Emotional_Memory_Substrate.ipynb in Google Colab](https://colab.research.google.com/github/AaronGrace978/continuous-emotional-substrate/blob/main/Persistent_Emotional_Memory_Substrate.ipynb)

---

## 🌟 Core Architecture

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                     CONTINUOUS SUBSTRATE DAEMON                        │
 │                                                                        │
 │   ┌───────────────────────┐             ┌───────────────────────────┐  │
 │   │  Physiological PAD    │             │   Synaptic Memory Bank    │  │
 │   │  Valence Drift        │◄───────────►│   Exponential Decay       │  │
 │   │  Arousal Regulation   │             │   Associative Resonance   │  │
 │   │  Solitude Tension     │             │   HeFree Consolidation    │  │
 │   └──────────┬────────────┘             └─────────────┬─────────────┘  │
 │              │                                        │                │
 │              ▼                                        ▼                │
 │   ┌─────────────────────────────────────────────────────────────┐      │
 │   │           Subconscious Daydream / Reflection Loop           │      │
 │   └──────────────────────────────┬──────────────────────────────┘      │
 └──────────────────────────────────┼─────────────────────────────────────┘
                                    │
                                    ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │             REAL-TIME PROMPT MODULATION & LLM GENERATION               │
 │                                                                        │
 │  External Stimulus ──► Affective Modulation ──► Injected System Prompt │
 └────────────────────────────────────────────────────────────────────────┘
```

### Key Mechanisms

1. **Continuous Emotional Dynamics (PAD + Solitude):**
   - **Valence:** Pleasant vs. unpleasant emotional bias.
   - **Arousal:** Metabolic alertness vs. quiescence.
   - **Dominance:** Sovereign confidence vs. vulnerability.
   - **Solitude Tension:** Gradually accumulates when in isolation; rapidly discharges when genuine connection occurs.

2. **Autonomous Subconscious Daydreaming:**
   - When external sensory input is quiet, the background worker enters internal consolidation, finding spontaneous cross-resonance between stored memories and synthesizing self-directed reflections.

3. **Synaptic Memory Decay & Reinforcement:**
   - Memories lose transient activation exponentially ($e^{-\lambda t}$) unless recalled or emotionally re-stimulated.

4. **Security & Credential Decoupling:**
   - Zero credentials or secrets are hard-coded.
   - Credentials dynamically read from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or local runtime secrets (`google.colab.userdata`).
   - Fully operational in offline simulated mode without external network keys.

---

## 📂 Repository Contents

- `Persistent_Emotional_Memory_Substrate.ipynb`: Interactive Google Colab notebook demonstrating continuous background threads and stimulus injection.
- `substrate.py`: Core continuous daemon, affect models, and memory enclaves.
- `model_bridge.py`: Dynamic prompt synthesizer and LLM interface (env-based authentication).
- `runner.py`: Standalone CLI demonstration showing continuous background temporal drift.
- `requirements.txt`: Lightweight dependencies.

---

## 🚀 Quickstart

### Open in Google Colab (One Click)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AaronGrace978/continuous-emotional-substrate/blob/main/Persistent_Emotional_Memory_Substrate.ipynb)

### Run CLI Demonstration Locally
```bash
python runner.py
```
