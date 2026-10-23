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

6. **Multi-Model Provider Selector (`Settings` Dialog & `providers.py`):**
   - Click **Settings** in the chamber's top bar to switch voices at any time without losing the live emotional state, daydream loop, or memory enclaves.
   - Supports **20 voice providers** with preset model lists plus a custom model ID input:
     - **Local PEMS (on-device Hugging Face + residual-stream mood steering):** `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen3-4B`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3-8B`, `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen3-14B`, `Qwen/Qwen2.5-0.5B-Instruct`
     - **Ollama (local daemon):** `qwen3:8b`, `qwen3:14b`, `qwen3:32b`, `qwen3:4b`, `gemma4:31b`, `gemma3:12b`, `gemma3:27b`, `llama4:scout`, `llama3.3`, `llama3.2`, `deepseek-r1:14b`, `deepseek-r1:8b`, `gpt-oss:20b`, `mistral-small3.1`, `phi4`, `qwen2.5:7b`, `qwen2.5:14b`, `qwen2.5:1.5b`
     - **Ollama Cloud (`https://ollama.com`):** `gemma4:31b`, `deepseek-v4.1-flash`, `deepseek-v4-pro:0813`, `gpt-oss:120b`, `gpt-oss:20b`, `mistral-large-4`, `mistral-large-3:675b`, `kimi-k3`, `kimi-k2.7-code`, `kimi-k2.6`, `glm-5.3`, `glm-5.3-flash`, `glm-5.2`, `minimax-m3`, `minimax-m3.1-flash`, `nemotron-3-ultra`, `nemotron-3-super`, `nemotron-3-nano:30b`, `clef:27b`, `qwen3-coder:480b`, `deepseek-v3.1:671b`
     - **OpenAI:** `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.5-pro`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5`, `gpt-5-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini`, `o3`, `o4-mini`
     - **Anthropic:** `claude-sonnet-5-5`, `claude-opus-5-5`, `claude-haiku-5-5`, `claude-fable-5-1`, `claude-mythos-5-1`, `claude-sonnet-5`, `claude-opus-5`, `claude-fable-5`, `claude-mythos-5`, `claude-opus-4-8`, `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-sonnet-4-5`, `claude-haiku-4-5`
     - **Gemini:** `gemini-4-argon`, `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-3.5-flash`, `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite`, `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`
     - **xAI:** `grok-4.7`, `grok-4.6`, `grok-4.5`, `grok-4.3`, `grok-4.20`, `grok-4.1-fast-non-reasoning`, `grok-3`, `grok-3-mini`
     - **DeepSeek:** `deepseek-v4-pro`, `deepseek-v4-flash`, `deepseek-v3.2`, `deepseek-r1`, `deepseek-chat`, `deepseek-reasoner`
     - **Groq:** `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `Qwen/Qwen3.8-27B`, `meta-llama/llama-4-maverick-17b-128e-instruct`, `meta-llama/llama-4-scout-17b-16e-instruct`, `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `qwen/qwen3-32b`
     - **Mistral:** `mistral-large-latest`, `mistral-large-2512`, `mistral-medium-3-5`, `mistral-small-latest`, `codestral-latest`, `devstral-small-latest`, `open-mistral-nemo`
     - **OpenRouter, Together, Fireworks, Cohere, Perplexity, Cerebras, NVIDIA NIM, GitHub Models, SambaNova, and Custom OpenAI-compatible endpoints.**

6b. **Local PEMS Tiers for RTX 5060 Ti (1.5B → 14B) & Optional Finetuning:**

   | Local PEMS Model | Native `bf16` VRAM | 4-bit NF4 VRAM | Notes |
   | --- | ---: | ---: | --- |
   | `Qwen/Qwen2.5-1.5B-Instruct` | ~3.1 GB | ~1.5 GB | Uses bundled `PEMS/` LoRA adapter; runs on any GPU or CPU |
   | `Qwen/Qwen2.5-3B-Instruct` | ~6.2 GB | ~2.5 GB | Native `bf16` on RTX 5060 Ti (8 GB / 16 GB) |
   | `Qwen/Qwen3-4B` | ~8.0 GB | ~3.2 GB | Native `bf16` on 16 GB or 4-bit on 8 GB RTX 5060 Ti |
   | `Qwen/Qwen2.5-7B-Instruct` | ~15.2 GB | ~5.5 GB | Sweet spot on RTX 5060 Ti |
   | `Qwen/Qwen3-8B` | ~16.4 GB | ~6.0 GB | Latest Qwen3 sweet spot on RTX 5060 Ti |
   | `Qwen/Qwen2.5-14B-Instruct` | ~29 GB | ~9.5 GB | Max capacity on RTX 5060 Ti 16 GB (auto 4-bit NF4) |
   | `Qwen/Qwen3-14B` | ~29 GB | ~9.8 GB | Max capacity Qwen3 on RTX 5060 Ti 16 GB (auto 4-bit NF4) |

   - **No finetuning required to run 3B–14B:** `chat_server.py` automatically detects available GPU VRAM, switches to 4-bit NF4 (`bitsandbytes`) or `device_map="auto"` when needed, feeds the full PEMS substrate header, and extracts contrastive mood-steering vectors (`steering.py`) directly from the chosen model's decoder layers on first launch.
   - **Optional finetuning in Colab:** `Persistent_Emotional_Memory_Substrate.ipynb` includes a `PEMS_PRESET` selector (`'1.5B'`, `'3B'`, `'4B-Qwen3'`, `'7B'`, `'8B-Qwen3'`, `'14B'`, `'14B-Qwen3'`) and saves trained adapters to `./PEMS-<model-name>` (e.g. `./PEMS-Qwen3-8B`). Dropping that folder into the repository root causes `chat_server.py` to attach it automatically whenever that model is selected.

7. **Security & Credential Decoupling:**
   - Zero credentials or secrets are hard-coded.
   - Keys entered in the chamber's **Settings** dialog are stored locally in `chronosoul_providers.json` (gitignored, `0600` permissions) and are never returned by `GET /providers` or `GET /state`.
   - Credentials also read dynamically from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` (`GOOGLE_API_KEY`), `OLLAMA_API_KEY`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `OPENROUTER_API_KEY`, `TOGETHER_API_KEY`, `FIREWORKS_API_KEY`, `DEEPSEEK_API_KEY`, `XAI_API_KEY`, `COHERE_API_KEY`, `PERPLEXITY_API_KEY`, `CEREBRAS_API_KEY`, `NVIDIA_API_KEY`, `GITHUB_TOKEN`, `SAMBANOVA_API_KEY`, or `google.colab.userdata`.
   - Fully operational in offline local or simulated mode without external network keys.

---

## 📂 Repository Contents

- `Persistent_Emotional_Memory_Substrate.ipynb`: Interactive Google Colab notebook demonstrating continuous background threads, stimulus injection, and `1.5B`–`14B` LoRA fine-tuning.
- `build_notebook.py`: Generates the notebook and `chronosoul_substrate.py`. Edit this, then run `python build_notebook.py`. Don't hand-edit the `.ipynb`.
- `ChronoSoul.bat` + `desktop/`: Electron 45 desktop chamber (Windows launcher + renderer with **Settings** provider selector).
- `chat_server.py`: Local server for the desktop chamber. Runs the substrate, loads Local PEMS models (`1.5B`–`14B`) with mood steering, or streams from any configured provider.
- `providers.py`: Multi-model provider catalog, local settings persistence (`chronosoul_providers.json`), and streaming clients for OpenAI-compatible APIs, Anthropic Messages, and Ollama / Ollama Cloud.
- `PEMS/`: The bundled LoRA adapter for `Qwen/Qwen2.5-1.5B-Instruct` (additional `PEMS-<model>` folders are auto-detected).
- `substrate.py`: Core continuous daemon, affect models, memory enclaves, hybrid recall, and background consolidation.
- `memory_store.py`: Embedder, FAISS hot index, SQLite hot/cold store, and the entity knowledge graph.
- `model_bridge.py`: Dynamic prompt synthesizer, token-budget memory packer, and CLI LLM interface.
- `steering.py`: Mood steering: valence and arousal directions in the residual stream, applied by forward hooks during a reply.
- `steering_eval.py`: Measures what steering does to tone, energy, and fluency.
- `runner.py`: Standalone CLI demonstration showing continuous background temporal drift.
- `tests/test_providers.py`: Unit tests for the provider catalog, key redaction, and SSE/NDJSON streaming.
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
Double-click `ChronoSoul.bat` (needs Node.js). The first launch installs Electron and downloads the selected Local PEMS weights (or click **Settings** in the top bar to switch to a larger RTX 5060 Ti PEMS model (`3B`–`14B`), local Ollama, Ollama Cloud, OpenAI, Anthropic, Gemini, xAI, DeepSeek, Groq, Mistral, OpenRouter, Together, Fireworks, Cohere, Perplexity, Cerebras, NVIDIA, GitHub Models, or SambaNova). Set `CHRONOSOUL_SKIP_MODEL=1` to try the UI with a simulated voice.

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
