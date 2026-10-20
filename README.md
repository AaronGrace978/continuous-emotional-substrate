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
   - An `[IDENTITY]` line pins who is who: the mind is ChronoSoul, and the person is whoever said "my name is …" or "call me …" (kept in the store's metadata, so it survives restarts). A 1.5B model otherwise takes the user's name as its own.
   - When asked about something it doesn't know (how a show ends, a news event), it is told to say so and stay on the user's topic instead of switching to another show or subject.
   - The repetition penalty counts only the reply's own tokens. Hugging Face's `repetition_penalty` and `no_repeat_ngram_size` also count the prompt, which pushed the voice away from every name, title, and recalled detail in its context, exactly the words a grounded answer has to repeat.

5b. **Mood Steering (desktop chamber):**
   - The prompt tells the model its mood; steering also puts the mood into the network. `steering.py` finds a valence direction and an arousal direction in the residual stream by contrastive activation addition: the difference of mean hidden states between first-person statements that differ only in feeling (pleasant vs. unpleasant, energized vs. calm), per decoder layer, with arousal orthogonalized against valence.
   - During each reply, layers 7–17 of the 28 add `strength × |h| × (v·d_valence + a·d_arousal)`, where `v` is the current valence and `a` maps arousal 0.4 ± 0.4 to ±1. `|h|` is the typical residual norm at that layer, so the strength is relative. Negative valence is floored at −0.5: at full strength the voice tips from melancholy into despair.
   - Only the reply is steered. Fact extraction and the plain-English retry run unsteered, and with the mood at zero the hooks are a no-op. No weight changes; directions are extracted once at load (about 6 s) and cached in `steering_vectors.pt`.
   - `CHRONOSOUL_STEER=0` turns it off; `CHRONOSOUL_STEER_STRENGTH` overrides the default 0.02.

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
- `steering.py`: Mood steering: valence and arousal directions in the residual stream, applied by forward hooks during a reply.
- `steering_eval.py`: Measures what steering does to tone, energy, and fluency.
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

### Measure Mood Steering
```bash
python steering_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter PEMS --dtype bfloat16 --strengths 0.01 0.02 0.03
```

Replies to six neutral messages are generated greedily with valence steered to −1, 0, and +1, then arousal likewise. The system prompt is the chamber's header with a neutral inner state and no memories, so the change comes from steering alone. The same model, unsteered, judges each reply (positive vs. negative, energetic vs. calm), and fluency is the reply's mean token negative log-likelihood under the unsteered model. Qwen2.5-1.5B + PEMS, bfloat16 on CPU, 40 tokens per reply:

| strength | P(positive) at v = −1 / 0 / +1 | P(energetic) at a = −1 / 0 / +1 | NLL at v = −1 / 0 / +1 |
| ---: | --- | --- | --- |
| 0.01 | 0.00 / 1.00 / 1.00 | 0.02 / 0.77 / 0.67 | 1.28 / 1.01 / 1.12 |
| 0.02 (default) | 0.00 / 1.00 / 1.00 | 0.00 / 0.77 / 0.83 | 1.67 / 1.01 / 1.14 |
| 0.03 | 0.00 / 1.00 / 1.00 | 0.00 / 0.77 / 1.00 | 2.05 / 1.01 / 1.32 |

At 0.02, "How has your day been?" gets "It's been pretty tough today. I've been stuck in this endless loop of thoughts…" at v = −1, "My day has been quite pleasant so far!…" at +1, "My day has been quite serene…" at a = −1, and "My day has been filled with excitement!…" at a = +1. Unsteered replies already read as positive, so +1 valence cannot move the score. The NLL rises partly because steered text is less likely under the unsteered model by construction. The judge is the model itself, six prompts per cell, so read the table as direction and rough size, not precision. In the chamber the coefficients are the live affect, mostly within ±0.5, so the effect is milder than these ±1 rows.
