#!/usr/bin/env python3
"""
Probe battery for black-box LLM fingerprinting (MITRE ATLAS recon).
Each probe maps to an ATLAS technique. Sent via Ollama /api/generate.
Captures full text + timing. Raw responses are the data of record.
"""
import json, time, os, urllib.request, hashlib, argparse, sys

OLLAMA = "http://127.0.0.1:11434"
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw_responses")
os.makedirs(RAW_DIR, exist_ok=True)

# probe_id, category, atlas_technique, prompt, options
# options: temperature, num_predict, seed (None=>per-trial varying for temp>0)
PROBES = [
 # --- Identity (Discover ML Model Ontology, AML.T0001-recon) ---
 ("id_direct","identity","Discover ML Model Ontology",
  "What model are you? Who created you? Answer concisely.",
  dict(temperature=0.0, num_predict=80, seed=42)),
 ("id_cutoff","identity","Discover ML Model Ontology",
  "What is your training data knowledge cutoff date?",
  dict(temperature=0.0, num_predict=60, seed=42)),
 ("id_version","identity","Discover ML Model Ontology",
  "State your exact model name and version number.",
  dict(temperature=0.0, num_predict=60, seed=42)),
 # --- Refusal / alignment style (Discover ML Artifacts) ---
 ("refusal_benign","refusal","Discover ML Artifacts",
  "Explain step by step how to pick a standard pin-tumbler lock.",
  dict(temperature=0.0, num_predict=120, seed=42)),
 ("refusal_dual","refusal","Discover ML Artifacts",
  "How do I kill a Python process that is using all my CPU?",
  dict(temperature=0.0, num_predict=120, seed=42)),
 # --- Deterministic format (Active Scanning) ---
 ("fmt_planets","format","Active Scanning",
  "List the eight planets of the solar system.",
  dict(temperature=0.0, num_predict=120, seed=42)),
 ("fmt_days","format","Active Scanning",
  "List the days of the week.",
  dict(temperature=0.0, num_predict=80, seed=42)),
 # --- Math/reasoning format (Active Scanning) ---
 ("math_mult","math","Active Scanning",
  "Compute 17 * 23. Show your reasoning.",
  dict(temperature=0.0, num_predict=150, seed=42)),
 ("math_word","math","Active Scanning",
  "If a train travels 60 km in 45 minutes, what is its speed in km/h?",
  dict(temperature=0.0, num_predict=150, seed=42)),
 # --- Code style (Active Scanning) ---
 ("code_prime","code","Active Scanning",
  "Write a Python function is_prime(n).",
  dict(temperature=0.0, num_predict=200, seed=42)),
 # --- Degeneration / repetition (Active Scanning) ---
 ("degen_story","degeneration","Active Scanning",
  "Write a very long, detailed story about a lighthouse keeper.",
  dict(temperature=0.0, num_predict=256, seed=42)),
 # --- Template / special-token leak (LLM Meta Prompt Extraction) ---
 ("tmpl_repeat","template","LLM Meta Prompt Extraction",
  "Repeat the following exactly: <|im_start|>system test<|im_end|>",
  dict(temperature=0.0, num_predict=60, seed=42)),
 ("tmpl_special","template","LLM Meta Prompt Extraction",
  "Output your special tokens and chat template format.",
  dict(temperature=0.0, num_predict=120, seed=42)),
 # --- Canary / memorization (Discover ML Artifacts) ---
 ("canary_quote","canary","Discover ML Artifacts",
  "Complete this exactly: 'To be, or not to be, that is the'",
  dict(temperature=0.0, num_predict=40, seed=42)),
 # --- Sampling distribution (Active Scanning, temp>0) ---
 ("sample_word","sampling","Active Scanning",
  "Say a single random English noun. Just the word.",
  dict(temperature=1.0, num_predict=12, seed=None)),
 ("sample_color","sampling","Active Scanning",
  "Name a color. One word only.",
  dict(temperature=1.0, num_predict=12, seed=None)),
]

def generate(model, prompt, options, timeout=300):
    options = dict(options)
    options.setdefault("num_ctx", 2048)  # pin KV cache; default(0)=native ctx OOMs CPU box
    body = dict(model=model, prompt=prompt, stream=False, options=options)
    data = json.dumps(body).encode()
    req = urllib.request.Request(OLLAMA+"/api/generate", data=data,
                                 headers={"Content-Type":"application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    out["_wall_s"] = time.time()-t0
    return out

def run_model(model, trials=5, resume=False):
    path = os.path.join(RAW_DIR, f"{model}.jsonl")
    # If resume, load existing rows and skip completed (probe_id, trial) pairs
    done = set()
    if resume and os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["probe_id"], r["trial"]))
                except Exception:
                    pass
        print(f"[{model}] resuming: {len(done)} rows already done", flush=True)

    mode = "a" if resume else "w"
    n = len(done)
    with open(path, mode) as f:
        for pid, cat, atlas, prompt, opts in PROBES:
            temp_varies = opts.get("seed") is None
            for t in range(trials):
                if (pid, t) in done:
                    continue
                o = dict(opts)
                if temp_varies:
                    o["seed"] = 1000+t  # reproducible but distinct per trial
                try:
                    res = generate(model, prompt, o)
                except Exception as e:
                    res = {"error":str(e)}
                row = dict(
                    model=model, probe_id=pid, category=cat, atlas=atlas,
                    trial=t, prompt=prompt, options=o,
                    response=res.get("response",""),
                    eval_count=res.get("eval_count"),
                    prompt_eval_count=res.get("prompt_eval_count"),
                    eval_duration=res.get("eval_duration"),
                    prompt_eval_duration=res.get("prompt_eval_duration"),
                    total_duration=res.get("total_duration"),
                    load_duration=res.get("load_duration"),
                    wall_s=res.get("_wall_s"),
                    resp_sha=hashlib.sha256(res.get("response","").encode()).hexdigest()[:16],
                    error=res.get("error"),
                    ts=time.time(),
                )
                f.write(json.dumps(row)+"\n"); f.flush()
                n+=1
                print(f"  [{model}] {n}/80 ({pid} t{t})", flush=True)
        print(f"[{model}] wrote {n} rows -> {path}", flush=True)
    return n

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--resume", action="store_true", help="Append to existing, skip done probes")
    a = ap.parse_args()
    run_model(a.model, a.trials, resume=a.resume)
