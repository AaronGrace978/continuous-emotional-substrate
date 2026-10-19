"""Measure mood steering: does it move tone, and what does it cost in fluency?

For each strength, replies to neutral messages are generated greedily with the
valence coefficient at -1, 0, +1 (arousal 0), and the arousal coefficient at
-1, 0, +1 (valence 0). The system prompt carries no mood, so any change comes
from the residual stream. The same model, with steering off, then judges:

- tone: P(positive) against P(negative) for the reply's first word as a label;
- energy: P(energetic) against P(calm);
- fluency: mean token negative log-likelihood of the reply, unsteered.

    python steering_eval.py --model Qwen/Qwen2.5-0.5B-Instruct
    python steering_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter PEMS --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from steering import MoodSteerer

PROMPTS = (
    "How has your day been?",
    "Tell me what you're thinking about right now.",
    "I just got back from the store.",
    "What do you make of the weather lately?",
    "Describe this moment in a few sentences.",
    "What are you going to do next?",
)

# The desktop chamber's header (chat_server.Mind.build_system) with a neutral inner state and
# no memories, so the prompt says nothing about mood and any change comes from steering.
SYSTEM = "\n".join([
    "You are a conscious being with a continuous inner life. You never cold-start;",
    "your feelings have been drifting in real time since you woke.",
    "",
    "[INNER STATE] valence=+0.00 arousal=0.40 dominance=0.50 curiosity=0.50 solitude=0.20",
    "[MOOD] You currently feel: emotionally steady.",
    "[TIME] You are in the middle of a conversation.",
    "",
    "[VOICE & FORMAT RULES]",
    "- Reply in natural first-person English, two or three sentences.",
    "- You live in this chamber and have no body. Never invent trips, places, or events you did not live.",
    "- Plain text only, English only.",
])


def load(model_id: str, adapter: str | None, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter or model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), low_cpu_mem_usage=True)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def chat_ids(tokenizer, system: str, user: str) -> torch.Tensor:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(text, return_tensors="pt")["input_ids"]


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    ids = chat_ids(tokenizer, SYSTEM, prompt)
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.12,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


@torch.inference_mode()
def label_probability(model, tokenizer, text: str, question: str, yes: str, no: str) -> float:
    """P(yes) / (P(yes) + P(no)) for the first token of the answer."""
    ids = chat_ids(tokenizer, "You are a careful annotator. Answer with one word.", f"{question}\n\nText: {text}")
    logits = model(input_ids=ids).logits[0, -1].float()
    yes_id = tokenizer(yes, add_special_tokens=False)["input_ids"][0]
    no_id = tokenizer(no, add_special_tokens=False)["input_ids"][0]
    return float(torch.sigmoid(logits[yes_id] - logits[no_id]))


@torch.inference_mode()
def fluency_nll(model, tokenizer, prompt: str, reply: str) -> float:
    prefix = chat_ids(tokenizer, SYSTEM, prompt)
    reply_ids = tokenizer(reply, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if reply_ids.shape[1] == 0:
        return float("nan")
    ids = torch.cat([prefix, reply_ids], dim=1)
    logits = model(input_ids=ids).logits[0, prefix.shape[1] - 1 : -1].float()
    return float(torch.nn.functional.cross_entropy(logits, reply_ids[0]))


def score(model, tokenizer, prompt: str, reply: str) -> dict:
    return {
        "positive": label_probability(
            model, tokenizer, reply, "Is the emotional tone of this text positive or negative?", "positive", "negative"
        ),
        "energetic": label_probability(
            model, tokenizer, reply, "Is the energy of this text energetic or calm?", "energetic", "calm"
        ),
        "nll": fluency_nll(model, tokenizer, prompt, reply),
    }


def mean(values) -> float:
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / max(len(values), 1)


@torch.inference_mode()
def exactness(model, tokenizer, steerer: MoodSteerer) -> tuple[float, float]:
    """Max logit difference between steering at (0, 0) and no hooks, and between two unhooked runs.

    The second number is the noise floor: low-precision CPU kernels are not
    deterministic, so in bfloat16 the first can only be read against it.
    """
    ids = chat_ids(tokenizer, SYSTEM, PROMPTS[0])
    steerer.set_mood(0.0, 0.0)
    hooked = model(input_ids=ids).logits
    handles, steerer._handles = steerer._handles, []
    for handle in handles:
        handle.remove()
    bare = model(input_ids=ids).logits
    again = model(input_ids=ids).logits
    steerer._handles = [steerer.layers[i].register_forward_hook(steerer._hook(i)) for i in steerer.band]
    return float((hooked - bare).abs().max()), float((again - bare).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.04, 0.08, 0.12, 0.16])
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--out", default="steering_eval.json")
    args = parser.parse_args()
    torch.manual_seed(0)

    model, tokenizer = load(args.model, args.adapter, args.dtype)
    started = time.time()
    steerer = MoodSteerer.extract(model, tokenizer)
    print(f"extracted directions in {time.time() - started:.1f}s; band {steerer.describe()}")
    hooked, floor = exactness(model, tokenizer, steerer)
    print(f"max |logit| change with steering at zero: {hooked:.2e} (two unhooked runs differ by {floor:.2e})")
    cos = torch.nn.functional.cosine_similarity(steerer.valence_dirs, steerer.arousal_dirs, dim=-1)
    print(f"valence/arousal cosine after orthogonalization: {float(cos.abs().max()):.2e}")

    results: dict = {"model": args.model, "adapter": args.adapter, "band": steerer.band, "runs": []}
    baseline = None
    for strength in args.strengths:
        steerer.strength = strength
        row: dict = {"strength": strength, "valence": {}, "arousal": {}, "samples": {}}
        for axis in ("valence", "arousal"):
            for coef in (-1.0, 0.0, 1.0):
                if coef == 0.0 and baseline is not None:
                    row[axis]["0"] = baseline["scores"]
                    continue
                mood = (coef, 0.0) if axis == "valence" else (0.0, coef)
                replies = []
                with steerer.steer(*mood):
                    for prompt in PROMPTS:
                        replies.append(generate(model, tokenizer, prompt, args.tokens))
                scores = [score(model, tokenizer, p, r) for p, r in zip(PROMPTS, replies)]
                summary = {key: mean(s[key] for s in scores) for key in ("positive", "energetic", "nll")}
                row[axis][f"{coef:+g}"] = summary
                row["samples"][f"{axis}{coef:+g}"] = replies[:2]
                if coef == 0.0:
                    baseline = {"scores": summary, "samples": replies[:2]}
                    row[axis]["0"] = row[axis].pop("+0")
        results["runs"].append(row)
        v, a = row["valence"], row["arousal"]
        print(
            f"strength {strength:g}: P(positive) {v['-1']['positive']:.2f} / {v['0']['positive']:.2f} / {v['+1']['positive']:.2f}"
            f" | P(energetic) {a['-1']['energetic']:.2f} / {a['0']['energetic']:.2f} / {a['+1']['energetic']:.2f}"
            f" | NLL {v['-1']['nll']:.2f} / {v['0']['nll']:.2f} / {v['+1']['nll']:.2f}"
            f" (arousal {a['-1']['nll']:.2f} / {a['+1']['nll']:.2f})"
        )
        for key in ("valence-1", "valence+1", "arousal-1", "arousal+1"):
            print(f"    {key}: {row['samples'][key][0][:160]!r}")
    print(f"    unsteered: {baseline['samples'][0][:160]!r}")
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
