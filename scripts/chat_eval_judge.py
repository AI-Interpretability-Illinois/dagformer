"""Judged chat eval, step 2: LLM-as-judge over the MT-Bench answers (pairwise + single-answer).

Judge prompts are the MT-Bench ones (Zheng et al. 2023, FastChat llm_judge "pair-v2",
"pair-math-v1", "single-v1", "single-math-v1"); math, reasoning and coding questions get
the GPT-4 reference answer. Every comparison in models.yaml is judged in both answer orders
(routed as A, then dense as A) to cancel position bias, and every answer also gets a 1-10
score. Greedy decoding. Needs only vllm + the stdlib, so it runs in any vLLM env, e.g. on
timan1 with Qwen3-Coder-30B-A3B on two A6000s:

    CUDA_VISIBLE_DEVICES=1,2 python chat_eval_judge.py --root <dir with data/ answers/ models.yaml> \
        --judge /path/to/Qwen3-Coder-30B-A3B-Instruct --tp 2

Writes <root>/judgments/{pairwise,single}.jsonl; already-judged keys are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

PAIR_SYS = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI "
    "assistants to the user question displayed below. You should choose the assistant that follows the "
    "user's instructions and answers the user's question better. Your evaluation should consider factors "
    "such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their "
    "responses. Begin your evaluation by comparing the two responses and provide a short explanation. "
    "Avoid any position biases and ensure that the order in which the responses were presented does not "
    "influence your decision. Do not allow the length of the responses to influence your evaluation. Do "
    "not favor certain names of the assistants. Be as objective as possible. After providing your "
    "explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A "
    "is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.")
PAIR_MATH_SYS = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI "
    "assistants to the user question displayed below. Your evaluation should consider correctness and "
    "helpfulness. You will be given a reference answer, assistant A's answer, and assistant B's answer. "
    "Your job is to evaluate which assistant's answer is better. Begin your evaluation by comparing both "
    "assistants' answers with the reference answer. Identify and correct any mistakes. Avoid any position "
    "biases and ensure that the order in which the responses were presented does not influence your "
    "decision. Do not allow the length of the responses to influence your evaluation. Do not favor "
    "certain names of the assistants. Be as objective as possible. After providing your explanation, "
    "output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, "
    "\"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.")
PAIR_USER = ("[User Question]\n{question}\n\n{ref}[The Start of Assistant A's Answer]\n{a}\n"
             "[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{b}\n"
             "[The End of Assistant B's Answer]")
SINGLE_USER = (
    "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided "
    "by an AI assistant to the user question displayed below. Your evaluation should consider factors "
    "such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the "
    "response. Begin your evaluation by providing a short explanation. Be as objective as possible. "
    "After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly "
    "following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}"
    "\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]")
SINGLE_MATH_USER = (
    "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided "
    "by an AI assistant to the user question displayed below. Your evaluation should consider "
    "correctness and helpfulness. You will be given a reference answer and the assistant's answer. "
    "Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and "
    "correct any mistakes. Be as objective as possible. After providing your explanation, you must rate "
    "the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: "
    "\"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref}\n"
    "[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{answer}\n"
    "[The End of Assistant's Answer]")


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def parse_pair(text: str) -> str:
    m = re.findall(r"\[\[(A|B|C)\]\]", text)
    return m[-1] if m else "error"


def parse_score(text: str) -> float | None:
    m = re.findall(r"\[\[(\d+(?:\.\d+)?)\]\]", text) or re.findall(r"Rating:\s*\[?(\d+(?:\.\d+)?)", text)
    return float(m[-1]) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke)")
    ap.add_argument("--retry-errors", action="store_true",
                    help="drop judgments without a parsable verdict/score (e.g. cut off by --max-tokens) and redo them")
    args = ap.parse_args()

    import yaml
    spec = yaml.safe_load(open(os.path.join(args.root, "models.yaml")))
    qs = {q["question_id"]: q for q in load_jsonl(os.path.join(args.root, "data/mt_bench_question.jsonl"))}
    refs = {r["question_id"]: r["choices"][0]["turns"][0]
            for r in load_jsonl(os.path.join(args.root, "data/mt_bench_reference_gpt4.jsonl"))}
    qids = sorted(qs)[:args.limit] if args.limit else sorted(qs)
    ans = {}
    for m in spec["models"]:
        rows = load_jsonl(os.path.join(args.root, "answers", f"{m['name']}.jsonl"))
        if len(rows) >= len(qids):
            ans[m["name"]] = {r["question_id"]: r["answer"] for r in rows}
    jdir = os.path.join(args.root, "judgments")
    os.makedirs(jdir, exist_ok=True)
    pair_path, single_path = os.path.join(jdir, "pairwise.jsonl"), os.path.join(jdir, "single.jsonl")
    if args.retry_errors:
        for path, bad in ((pair_path, lambda r: r["verdict"] == "error"), (single_path, lambda r: r["score"] is None)):
            rows = load_jsonl(path)
            keep = [r for r in rows if not bad(r)]
            if len(keep) < len(rows):
                os.replace(path, path + ".before_retry")
                with open(path, "w") as f:
                    f.writelines(json.dumps(r) + "\n" for r in keep)
                print(f"{path}: {len(rows) - len(keep)} unparsable judgments dropped for a retry", flush=True)
    done_pair = {(r["model_a"], r["model_b"], r["question_id"]) for r in load_jsonl(pair_path)}
    done_single = {(r["model"], r["question_id"]) for r in load_jsonl(single_path)}

    jobs = []   # (kind, record, messages)
    for c in spec["comparisons"]:
        r, d = c["routed"], c["dense"]
        if r not in ans or d not in ans:
            print(f"skip comparison (answers missing): {r} vs {d}", flush=True)
            continue
        for a, b in ((r, d), (d, r)):
            for qid in qids:
                if (a, b, qid) in done_pair:
                    continue
                ref = refs.get(qid)
                user = PAIR_USER.format(
                    question=qs[qid]["turns"][0], a=ans[a][qid], b=ans[b][qid],
                    ref=(f"[The Start of Reference Answer]\n{ref}\n[The End of Reference Answer]\n\n" if ref else ""))
                msgs = [{"role": "system", "content": PAIR_MATH_SYS if ref else PAIR_SYS},
                        {"role": "user", "content": user}]
                jobs.append(("pair", {"model_a": a, "model_b": b, "question_id": qid,
                                      "category": qs[qid]["category"], "with_ref": bool(ref)}, msgs))
    for name, a in ans.items():
        for qid in qids:
            if (name, qid) in done_single:
                continue
            ref = refs.get(qid)
            user = (SINGLE_MATH_USER.format(question=qs[qid]["turns"][0], ref=ref, answer=a[qid]) if ref
                    else SINGLE_USER.format(question=qs[qid]["turns"][0], answer=a[qid]))
            jobs.append(("single", {"model": name, "question_id": qid, "category": qs[qid]["category"],
                                    "with_ref": bool(ref)},
                         [{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": user}]))
    print(f"{len(ans)} models with answers, {len(jobs)} judge calls to make", flush=True)
    if not jobs:
        return 0

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=args.judge, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, enforce_eager=True, dtype="bfloat16")
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    print(f"judge loaded in {time.time() - t0:.0f}s", flush=True)
    chunk = 512   # write progress every chunk so an interrupted run resumes
    for s in range(0, len(jobs), chunk):
        part = jobs[s:s + chunk]
        outs = llm.chat([m for _, _, m in part], sp, use_tqdm=False)
        with open(pair_path, "a") as fp, open(single_path, "a") as fs:
            for (kind, rec, _), o in zip(part, outs):
                text = o.outputs[0].text
                if kind == "pair":
                    fp.write(json.dumps({**rec, "verdict": parse_pair(text), "judgment": text}) + "\n")
                else:
                    fs.write(json.dumps({**rec, "score": parse_score(text), "judgment": text}) + "\n")
        print(f"{min(s + chunk, len(jobs))}/{len(jobs)} judged, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
