#!/usr/bin/env python3
"""
Hardened single-entry runner for the model-fingerprinting study.

Root causes fixed vs. the Jun 25 stall:
  1. OOM: box has 7.8GB RAM, no swap. Old orchestrator probed a loaded model
     WHILE downloads streamed -> spike OOM-killed ollama at phi3.5-mini and
     took the whole run down. Fix: two strict phases. Download everything with
     ollama OFF; only then start ollama and probe ONE model at a time.
  2. 404: SmolLM2-360M repo only ships q8_0, not q4_k_m. Fixed in ZOO.
  3. Truncated transfers: resolve exact size + real filename from the HF API,
     resume, many retries, verify byte-exact before marking ok.

Crash isolation: each model's import+probe is wrapped. If ollama dies, we
detect it, restart it, and move on. One bad model never kills the batch.
Idempotent/resumable: re-running skips models already downloaded/probed.
"""
import json, os, subprocess, time, urllib.request, urllib.error, signal, sys

ROOT = os.path.expanduser("~/workspace/model-fingerprinting")
GGUF_DIR = f"{ROOT}/models/gguf"
MF_DIR   = f"{ROOT}/models/modelfiles"
RAW_DIR  = f"{ROOT}/raw_responses"
LOGS     = f"{ROOT}/logs"
MANIFEST = f"{ROOT}/models/zoo_manifest.json"
LOG      = f"{LOGS}/run_all.log"
OLLAMA   = os.path.expanduser("~/.local/bin/ollama")
STORE    = f"{ROOT}/models/ollama_store"
ENV = dict(os.environ,
           OLLAMA_MODELS=STORE,
           OLLAMA_MAX_LOADED_MODELS="1",
           OLLAMA_NUM_PARALLEL="1",
           OLLAMA_KEEP_ALIVE="0",
           PATH=os.path.expanduser("~/.local/bin")+":"+os.environ.get("PATH",""))
HF = "https://huggingface.co"
TRIALS = 5
N_PROBES = 16
EXPECTED_ROWS = TRIALS * N_PROBES
MEM_FLOOR_MB = 1500   # need at least this much free before loading a model

for d in (GGUF_DIR, MF_DIR, RAW_DIR, LOGS, STORE):
    os.makedirs(d, exist_ok=True)

# name, family, params_b, developer, arch, repo, file
ZOO = [
  ("qwen2.5-0.5b","qwen2.5",0.5,"Alibaba","qwen2","Qwen/Qwen2.5-0.5B-Instruct-GGUF","qwen2.5-0.5b-instruct-q4_k_m.gguf"),
  ("qwen2.5-1.5b","qwen2.5",1.5,"Alibaba","qwen2","Qwen/Qwen2.5-1.5B-Instruct-GGUF","qwen2.5-1.5b-instruct-q4_k_m.gguf"),
  ("qwen2.5-3b","qwen2.5",3.0,"Alibaba","qwen2","Qwen/Qwen2.5-3B-Instruct-GGUF","qwen2.5-3b-instruct-q4_k_m.gguf"),
  ("llama3.2-1b","llama3.2",1.0,"Meta","llama","bartowski/Llama-3.2-1B-Instruct-GGUF","Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
  ("llama3.2-3b","llama3.2",3.0,"Meta","llama","bartowski/Llama-3.2-3B-Instruct-GGUF","Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
  ("smollm2-360m","smollm2",0.36,"HuggingFace","llama","HuggingFaceTB/SmolLM2-360M-Instruct-GGUF","smollm2-360m-instruct-q8_0.gguf"),  # repo only ships q8_0
  ("smollm2-1.7b","smollm2",1.7,"HuggingFace","llama","HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF","smollm2-1.7b-instruct-q4_k_m.gguf"),
  ("gemma2-2b","gemma2",2.0,"Google","gemma2","bartowski/gemma-2-2b-it-GGUF","gemma-2-2b-it-Q4_K_M.gguf"),
  ("phi3.5-mini","phi3.5",3.8,"Microsoft","phi3","bartowski/Phi-3.5-mini-instruct-GGUF","Phi-3.5-mini-instruct-Q4_K_M.gguf"),
  ("tinyllama-1.1b","tinyllama",1.1,"community","llama","TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF","tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
  ("stablelm2-1.6b","stablelm2",1.6,"Stability","stablelm","second-state/stablelm-2-zephyr-1.6b-GGUF","stablelm-2-zephyr-1_6b-Q4_K_M.gguf"),
  ("granite3.1-2b","granite3.1",2.0,"IBM","granite","bartowski/granite-3.1-2b-instruct-GGUF","granite-3.1-2b-instruct-Q4_K_M.gguf"),
  ("mistral-7b","mistral",7.0,"Mistral","llama","bartowski/Mistral-7B-Instruct-v0.3-GGUF","Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
  # Phase 2: new families
  ("deepseek-r1-1.5b","deepseek-r1",1.5,"DeepSeek","qwen2","bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF","DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"),
  ("olmo2-1b","olmo2",1.0,"AI2","llama","allenai/OLMo-2-0425-1B-Instruct-GGUF","OLMo-2-0425-1B-Instruct-Q4_K_M.gguf"),
  # Phase 3: cross-generational
  ("qwen3-0.6b","qwen3",0.6,"Alibaba","qwen3","bartowski/Qwen3-0.6B-GGUF","Qwen3-0.6B-Q4_K_M.gguf"),
]

def log(m):
    line=f"[{time.strftime('%H:%M:%S')}] {m}"; print(line,flush=True)
    with open(LOG,"a") as f: f.write(line+"\n")

def free_mb():
    with open("/proc/meminfo") as f:
        mi={l.split(':')[0]:int(l.split()[1]) for l in f}
    return mi.get("MemAvailable",0)//1024

# ---------- HF metadata: resolve real filename + exact LFS size ----------
def hf_files(repo):
    url=f"https://huggingface.co/api/models/{repo}"
    try:
        with urllib.request.urlopen(url,timeout=30) as r:
            d=json.load(r)
        return [s["rfilename"] for s in d.get("siblings",[]) if s["rfilename"].endswith(".gguf")]
    except Exception as e:
        log(f"  hf_files {repo}: {e}"); return []

def exact_size(repo, fn):
    """LFS pointer metadata via the resolve HEAD; X-Linked-Size is the true blob size."""
    url=f"{HF}/{repo}/resolve/main/{fn}"
    try:
        req=urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req,timeout=30) as r:
            cl=r.headers.get("X-Linked-Size") or r.headers.get("Content-Length")
            return int(cl) if cl else None, url
    except Exception as e:
        log(f"  HEAD {fn}: {e}"); return None, url

def resolve(repo, fn):
    """Return (filename, size, url). Fall back to any q4_k_m / first gguf if 404."""
    size,url = exact_size(repo, fn)
    if size: return fn, size, url
    files=hf_files(repo)
    if fn in files:  # exists but no size header; keep it
        return fn, None, f"{HF}/{repo}/resolve/main/{fn}"
    # pick best alternative: prefer q4_k_m, else q8_0, else first
    pref = ([f for f in files if "q4_k_m" in f.lower()] or
            [f for f in files if "q8_0" in f.lower()] or files)
    if pref:
        alt=pref[0]; size,url=exact_size(repo,alt)
        log(f"  resolved {fn} -> {alt}")
        return alt, size, url
    return fn, None, url

def download(url, dest, expected, tries=12):
    for attempt in range(1,tries+1):
        have=os.path.getsize(dest) if os.path.exists(dest) else 0
        if expected and have==expected: return True
        if expected and have>expected:
            os.remove(dest); have=0
        mode="ab" if have else "wb"
        headers={"Range":f"bytes={have}-"} if have else {}
        try:
            req=urllib.request.Request(url,headers=headers)
            with urllib.request.urlopen(req,timeout=120) as r, open(dest,mode) as f:
                while True:
                    chunk=r.read(1<<20)
                    if not chunk: break
                    f.write(chunk)
        except Exception as e:
            log(f"  dl attempt {attempt} @ {os.path.getsize(dest) if os.path.exists(dest) else 0}B: {e}")
            time.sleep(min(2*attempt,15)); continue
        have=os.path.getsize(dest)
        if expected and have==expected: return True
        if not expected and have>0:  # no size known; accept a stable non-empty file
            time.sleep(2)
            if os.path.getsize(dest)==have: return True
        log(f"  dl attempt {attempt}: have {have}/{expected}")
        time.sleep(min(2*attempt,15))
    return bool(expected) and os.path.exists(dest) and os.path.getsize(dest)==expected

# ---------- PHASE 1: download all (ollama OFF) ----------
def phase_download():
    log("===== PHASE 1: DOWNLOAD (ollama off) =====")
    manifest=[]
    for name,fam,pb,dev,arch,repo,fn0 in ZOO:
        dest=os.path.join(GGUF_DIR,f"{name}.gguf")
        fn,size,url=resolve(repo,fn0)
        log(f"=== {name} ({fam}, {pb}B, {dev})  size={size}")
        ok=download(url,dest,size)
        got=os.path.getsize(dest) if os.path.exists(dest) else 0
        log(f"  -> {'OK' if ok else 'INCOMPLETE'} {got}B")
        manifest.append(dict(name=name,family=fam,params_b=pb,developer=dev,arch=arch,
                             repo=repo,file=fn,url=url,local=dest,expected=size,got=got,ok=bool(ok)))
        json.dump(manifest,open(MANIFEST,"w"),indent=2)
    okn=sum(1 for m in manifest if m["ok"])
    log(f"PHASE 1 DONE: {okn}/{len(manifest)} models downloaded")
    return manifest

# ---------- ollama lifecycle ----------
_serve=None
def start_ollama():
    global _serve
    stop_ollama()
    sl=open(f"{LOGS}/ollama-serve.log","a")
    _serve=subprocess.Popen([OLLAMA,"serve"],env=ENV,stdout=sl,stderr=subprocess.STDOUT)
    for _ in range(30):
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags",timeout=3); 
            log("  ollama up"); return True
        except Exception:
            time.sleep(1)
    log("  ollama failed to come up"); return False

def stop_ollama():
    global _serve
    try:
        subprocess.run(["pkill","-f","ollama serve"],capture_output=True)
    except Exception: pass
    _serve=None
    time.sleep(2)

def ollama_alive():
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags",timeout=3); return True
    except Exception: return False

def already_done(name):
    p=f"{RAW_DIR}/{name}.jsonl"
    if not os.path.exists(p): return False
    rows=[l for l in open(p) if l.strip()]
    if len(rows)<EXPECTED_ROWS: return False
    return sum(1 for l in rows if json.loads(l).get("error"))==0

def import_model(name):
    mf=f"{MF_DIR}/{name}.Modelfile"
    open(mf,"w").write(f"FROM {GGUF_DIR}/{name}.gguf\n")
    r=subprocess.run([OLLAMA,"create",name,"-f",mf],env=ENV,capture_output=True,text=True,timeout=900)
    ok=r.returncode==0
    if not ok: log(f"  import FAIL {name}: {(r.stdout+r.stderr)[-200:]}")
    return ok

def probe_model(name):
    r=subprocess.run(["python3",f"{ROOT}/scripts/probe.py",name,"--trials",str(TRIALS)],
                     env=ENV,capture_output=True,text=True,timeout=7200)
    log(f"  probe rc={r.returncode}: {(r.stdout+r.stderr).strip()[-180:]}")
    return r.returncode==0

# ---------- PHASE 2: probe serially (ollama on, one model at a time) ----------
def phase_probe(manifest):
    log("===== PHASE 2: PROBE (serial, mem-gated) =====")
    if not start_ollama():
        log("FATAL: ollama won't start"); return
    done=set(); 
    targets=[m for m in manifest if m["ok"] and os.path.exists(m["local"])]
    for m in targets:
        name=m["name"]
        if already_done(name):
            log(f"SKIP {name} (already probed)"); done.add(name); continue
        # memory gate: wait for headroom, restart ollama if it died
        for _ in range(15):
            if not ollama_alive():
                log(f"  ollama down before {name}; restarting"); start_ollama()
            if free_mb()>=MEM_FLOOR_MB: break
            log(f"  low mem {free_mb()}MB, waiting"); 
            subprocess.run([OLLAMA,"stop",name],env=ENV,capture_output=True); time.sleep(5)
        log(f"=== PROCESS {name} ({m['family']}, {m['params_b']}B)  free={free_mb()}MB")
        try:
            if not import_model(name):
                log(f"  -> import failed, skip"); continue
            ok=probe_model(name)
            subprocess.run([OLLAMA,"stop",name],env=ENV,capture_output=True)
            if ok and already_done(name):
                done.add(name); log(f"  -> DONE {name} ({len(done)}/{len(targets)})")
            else:
                log(f"  -> {name} incomplete; ollama_alive={ollama_alive()}")
                if not ollama_alive(): start_ollama()  # recover for next model
        except Exception as e:
            log(f"  -> EXCEPTION on {name}: {e}; recovering")
            if not ollama_alive(): start_ollama()
    stop_ollama()
    log(f"PHASE 2 DONE: {len(done)}/{len(targets)} models probed")
    return done

def main():
    t0=time.time()
    log(f"##### RUN START  free={free_mb()}MB #####")
    stop_ollama()  # ensure nothing loaded during downloads
    manifest=phase_download()
    done=phase_probe(manifest)
    # summary
    okn=sum(1 for m in manifest if m["ok"])
    log(f"##### RUN COMPLETE in {int(time.time()-t0)}s: {okn} downloaded, {len(done or [])} probed #####")
    summ=dict(ts=time.time(),downloaded=okn,probed=len(done or []),total=len(ZOO),
              models=[m["name"] for m in manifest if m["ok"]])
    json.dump(summ,open(f"{ROOT}/run_summary.json","w"),indent=2)

if __name__=="__main__":
    main()
