#!/usr/bin/env python3
"""Generate a 128k-context SFT dataset (chat `messages`) for GLM-5.2.

Needle-in-a-haystack style: each sample is a long (~SEQ tokens) user turn made of
filler passages with a few embedded facts + a question about one fact; the
assistant turn is the answer. This exercises the true long-context path (real
~128k-token samples, not short samples padded).

Usage:
  python make_128k_sft_data.py --tokenizer <hf_model_dir> --seq 131072 \
         --n 64 --out data/sft_128k.parquet
"""
import argparse, random, pandas as pd
from transformers import AutoTokenizer

FILLER = (
    "The maintenance log records routine telemetry from the orbital relay. "
    "Signal integrity remained nominal across all bands during this interval. "
    "Thermal margins stayed within the expected envelope for the array. "
)
FACTS = [
    ("the access code for vault {i}", "The access code for vault {i} is {code}."),
    ("the calibration constant for sensor {i}", "The calibration constant for sensor {i} is {code}."),
    ("the docking bay assigned to shuttle {i}", "Shuttle {i} is assigned to docking bay {code}."),
]

def build_sample(idx, target_tokens, tok, rng):
    code = rng.randint(10000, 99999)
    q_tmpl, a_tmpl = rng.choice(FACTS)
    needle = f" IMPORTANT RECORD: {a_tmpl.format(i=idx, code=code)} "
    # grow filler until we hit ~target tokens, insert the needle at a random spot
    body, n = [], 0
    while n < target_tokens:
        body.append(FILLER)
        n += len(tok(FILLER, add_special_tokens=False)["input_ids"])
    pos = rng.randint(0, len(body))
    body.insert(pos, needle)
    question = f"\n\nBased on the records above, what is {q_tmpl.format(i=idx)}?"
    user = "".join(body) + question
    assistant = a_tmpl.format(i=idx, code=code)
    return [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seq", type=int, default=131072)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    rng = random.Random(a.seed)
    # leave headroom for chat template + assistant turn
    target = int(a.seq * 0.92)
    rows = []
    for i in range(a.n):
        msgs = build_sample(i, target, tok, rng)
        ntok = len(tok.apply_chat_template(msgs, tokenize=True, return_dict=True)["input_ids"])
        rows.append({"messages": msgs})
        if i < 3 or i == a.n - 1:
            print(f"sample {i}: ~{ntok} tokens")
    pd.DataFrame(rows).to_parquet(a.out)
    print(f"wrote {len(rows)} samples -> {a.out}")

if __name__ == "__main__":
    main()
