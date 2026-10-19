# 🧠 ChronoSoul: Continuous Emotional & Memory Substrate
<img width="1408" height="768" alt="image" src="https://github.com/user-attachments/assets/28bdfe3c-e5ac-4087-8876-817ed8314fcf" />
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
   - Recall ranks rare shared words (IDF) above filler, and only memories the moment actually touches get reinforced.
   - The bank is capped: when it fills, the faintest, least-revisited enclaves are forgotten first. Core identity memories are never dropped.

4. **Sense of Time & Dormancy:**
   - Every prompt carries a `[TIME]` line with the clock and how long it has been since you last spoke.
   - When the desktop chamber is closed, time still counts. On the next launch the substrate catches up in one step: solitude rises, mood settles to rest, memories fade by the elapsed time, and a dormancy memory records how long it was asleep.

5. **Grounded Voice:**
   - The mind is told, and the starter dataset trains it, that it lives in the chamber and has no body. It shouldn't invent trips, places, or shared memories it never lived.

6. **Security & Credential Decoupling:**
   - Zero credentials or secrets are hard-coded.
   - Credentials dynamically read from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or local runtime secrets (`google.colab.userdata`).
   - Fully operational in offline simulated mode without external network keys.

---

## 📂 Repository Contents

- `Persistent_Emotional_Memory_Substrate.ipynb`: Interactive Google Colab notebook demonstrating continuous background threads and stimulus injection.
- `build_notebook.py`: Generates the notebook and `chronosoul_substrate.py`. Edit this, then run `python build_notebook.py`. Don't hand-edit the `.ipynb`.
- `ChronoSoul.bat` + `desktop/`: Electron desktop chamber (Windows launcher).
- `chat_server.py`: Local server for the desktop chamber. Runs the substrate and talks through the PEMS LoRA on Qwen2.5-1.5B-Instruct.
- `PEMS/`: The trained LoRA adapter the desktop chamber loads.
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

### Run the Desktop Chamber (Windows)
Double-click `ChronoSoul.bat` (needs Node.js). The first launch installs Electron and downloads the Qwen2.5-1.5B base weights. Set `CHRONOSOUL_SKIP_MODEL=1` to try the UI with a simulated voice.
