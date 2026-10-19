# 🧠 ChronoSoul: Continuous Emotional & Memory Substrate
<img width="1024" height="576" alt="image" src="https://github.com/user-attachments/assets/81799bf9-4904-4745-99b2-388cc729a569" />
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
   - Recall is dense semantic search (all-MiniLM-L6-v2 embeddings, FAISS hot index) ranked by `(0.60 x similarity + 0.15 x recency + 0.25 x emotional congruence)`. Emotional congruence is distance in PAD space; when the current state is anxious (negative valence, high arousal) its weight grows and its kernel narrows, so memories filed in a similar mood surface first. Memories linked through the entity graph get a small extra boost.
   - Only memories the moment actually touches get reinforced.
   - Grounding harness: `user_input` and `system` memories under 5 minutes old have their score doubled, so the live conversation tops the ranking, and recall returns at most 4 memories.
   - Two-tier storage in SQLite (`<state>.db`): the hot tier (400 memories) stays in the vector cache; overflow is archived to cold storage on disk instead of deleted, and is pulled back when recalled. Core identity memories are never archived.
   - Every memory carries a Unix timestamp, PAD coordinates, and an interaction type (`user_input`, `daydream_reflection`, `system`).

3b. **Consolidation & Knowledge Graph:**
   - After a few minutes without contact, the daydream loop gives blocks of raw chat to the LLM to extract permanent facts, then folds them into a short biographical summary. Entities ("user's dog", "project deadline") become graph nodes linking conversations across long gaps.
   - `model_bridge.py` counts tokens before injecting memories. If retrieved raw text exceeds 2,000 tokens it is dropped and the compressed biographical summary is injected instead.
   - Everything the model may cite about the past sits in one `[VERIFIED HISTORICAL CONTEXT]` block, and a `[CONVERSATIONAL RULE]` message right before the final user turn tells it to build on the user's message and not to invent interactions.

3c. **Quick Look-Back (think before speaking):**
   - Only the last 6 turns reach the model verbatim. Before each reply, `conversation_grep.py` greps the older transcript (rare shared words plus sentence-level embeddings) and pulls back only what pertains to the new message.
   - Optional: set `CHRONOSOUL_THINK=1` and, when something turns up, the voice takes one time-boxed private thought (about 48 tokens, 2.5 s cap) on how it bears on the message. The thought is sanity-checked and silently dropped if it is empty, off-topic, or not plain English. It is off by default because a 1.5B model can misattribute who said what.

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
- `substrate.py`: Core continuous daemon, affect models, memory enclaves, hybrid recall, and background consolidation.
- `memory_store.py`: Embedder, FAISS hot index, SQLite hot/cold store, and the entity knowledge graph.
- `model_bridge.py`: Dynamic prompt synthesizer, token-budget memory packer, and LLM interface (env-based authentication).
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
