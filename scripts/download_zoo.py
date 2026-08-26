#!/usr/bin/env python3
"""
Download the model zoo as GGUF from HuggingFace CDN (Ollama R2 is blocked here).
Verifies exact byte size (silent truncation is the failure mode), resumes,
retries, then imports into Ollama via a Modelfile.

Each entry: ollama_name, family, params_b, developer, arch, repo, file
"""
import json, os, subprocess, sys, time, urllib.request, urllib.error

HF = "https://huggingface.co"
GGUF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/gguf")
MF_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/modelfiles")
LOG      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs/zoo_download.log")
os.makedirs(GGUF_DIR, exist_ok=True)
os.makedirs(MF_DIR, exist_ok=True)

# repo, file are HF resolve path components. q4_K_M unless noted.
ZOO = [
  # name, family, params_b, developer, arch, repo, file
  ("qwen2.5-0.5b","qwen2.5",0.5,"Alibaba","qwen2","Qwen/Qwen2.5-0.5B-Instruct-GGUF","qwen2.5-0.5b-instruct-q4_k_m.gguf"),
  ("qwen2.5-1.5b","qwen2.5",1.5,"Alibaba","qwen2","Qwen/Qwen2.5-1.5B-Instruct-GGUF","qwen2.5-1.5b-instruct-q4_k_m.gguf"),
  ("qwen2.5-3b","qwen2.5",3.0,"Alibaba","qwen2","Qwen/Qwen2.5-3B-Instruct-GGUF","qwen2.5-3b-instruct-q4_k_m.gguf"),
  ("llama3.2-1b","llama3.2",1.0,"Meta","llama","bartowski/Llama-3.2-1B-Instruct-GGUF","Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
  ("llama3.2-3b","llama3.2",3.0,"Meta","llama","bartowski/Llama-3.2-3B-Instruct-GGUF","Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
  ("smollm2-360m","smollm2",0.36,"HuggingFace","llama","HuggingFaceTB/SmolLM2-360M-Instruct-GGUF","smollm2-360m-instruct-q4_k_m.gguf"),
  ("smollm2-1.7b","smollm2",1.7,"HuggingFace","llama","HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF","smollm2-1.7b-instruct-q4_k_m.gguf"),
  ("gemma2-2b","gemma2",2.0,"Google","gemma2","bartowski/gemma-2-2b-it-GGUF","gemma-2-2b-it-Q4_K_M.gguf"),
  ("phi3.5-mini","phi3.5",3.8,"Microsoft","phi3","bartowski/Phi-3.5-mini-instruct-GGUF","Phi-3.5-mini-instruct-Q4_K_M.gguf"),
  ("tinyllama-1.1b","tinyllama",1.1,"community","llama","TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF","tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
  ("stablelm2-1.6b","stablelm2",1.6,"Stability","stablelm","second-state/stablelm-2-zephyr-1.6b-GGUF","stablelm-2-zephyr-1_6b-Q4_K_M.gguf"),
  ("granite3.1-2b","granite3.1",2.0,"IBM","granite","bartowski/granite-3.1-2b-instruct-GGUF","granite-3.1-2b-instruct-Q4_K_M.gguf"),
  ("falcon3-1b","falcon3",1.0,"TII","llama","bartowski/Falcon3-1B-Instruct-GGUF","Falcon3-1B-Instruct-Q4_K_M.gguf"),
  ("mistral-7b","mistral",7.0,"Mistral","llama","bartowski/Mistral-7B-Instruct-v0.3-GGUF","Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
]

def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG,"a") as f: f.write(line+"\n")

def remote_size(url):
    """Follow redirect, read content-length of the actual blob."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            cl = r.headers.get("X-Linked-Size") or r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception as e:
        log(f"  HEAD failed: {e}")
        return None

def download(url, dest, expected, tries=6):
    for attempt in range(1, tries+1):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if expected and have == expected:
            return True
        if expected and have > expected:  # corrupt, restart
            os.remove(dest); have = 0
        mode = "ab" if have else "wb"
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, mode) as f:
                while True:
                    chunk = r.read(1<<20)
                    if not chunk: break
                    f.write(chunk)
        except Exception as e:
            log(f"  attempt {attempt}: stream error @ {os.path.getsize(dest) if os.path.exists(dest) else 0}B: {e}")
            time.sleep(3); continue
        have = os.path.getsize(dest)
        if expected and have == expected:
            return True
        log(f"  attempt {attempt}: have {have}/{expected}, retrying")
        time.sleep(2)
    return expected and os.path.getsize(dest) == expected

def main():
    manifest = []
    for name, fam, pb, dev, arch, repo, fn in ZOO:
        url = f"{HF}/{repo}/resolve/main/{fn}"
        dest = os.path.join(GGUF_DIR, f"{name}.gguf")
        log(f"=== {name} ({fam}, {pb}B, {dev}) ===")
        exp = remote_size(url)
        log(f"  remote size: {exp}")
        ok = download(url, dest, exp)
        got = os.path.getsize(dest) if os.path.exists(dest) else 0
        status = "OK" if ok else "INCOMPLETE"
        log(f"  -> {status} {got}B")
        manifest.append(dict(name=name,family=fam,params_b=pb,developer=dev,
                             arch=arch,repo=repo,file=fn,url=url,
                             local=dest,expected=exp,got=got,ok=bool(ok)))
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/zoo_manifest.json"),"w") as f:
            json.dump(manifest,f,indent=2)
    log("ALL DOWNLOADS DONE")

if __name__=="__main__":
    main()
